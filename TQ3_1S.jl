module TQ3_1S

using CUDA, Printf

export tq3_1s_compress, tq3_1s_decompress, TQ3_1S_BLOCK_SIZE, TQ3_1S_Ratio

const TQ3_1S_BLOCK_SIZE = 256
const TQ3_1S_BITS = 3
const TQ3_1S_MAX_VAL = 7
const TQ3_1S_BYTES_PER_BLOCK = 96
const TQ3_1S_Ratio = 5.22  # 512 / 98
const TQ3_1S_ZERO_POINT = 4.0f0 # Смѣщеніе для signed 3-bit (-4 to +3)

function tq3_1s_compress_kernel(out_data, out_scales, x, n_elements, n_blocks)
    bid = blockIdx().x
    if bid > n_blocks; return; end

    tid = threadIdx().x
    stride = blockDim().x
    base_idx = (bid - 1) * TQ3_1S_BLOCK_SIZE

    shmem_vals = @cuStaticSharedMem(Float16, TQ3_1S_BLOCK_SIZE)
    shmem_max = @cuStaticSharedMem(Float32, TQ3_1S_BLOCK_SIZE)
    shmem_q = @cuStaticSharedMem(UInt8, TQ3_1S_BLOCK_SIZE)

    local_max = -Inf32
    for i in tid:stride:TQ3_1S_BLOCK_SIZE
        idx = base_idx + i
        val_f16 = Float16(0.0)
        if idx <= n_elements
            @inbounds val_f16 = x[idx]
        end
        shmem_vals[i] = val_f16
        val_f32 = Float32(val_f16)
        if isfinite(val_f32)
            local_max = max(local_max, abs(val_f32))
        end
    end

    shmem_max[tid] = local_max
    sync_threads()

    len = TQ3_1S_BLOCK_SIZE
    while len > 1
        len = div(len + 1, 2)
        if tid <= len && tid + len <= TQ3_1S_BLOCK_SIZE
            shmem_max[tid] = max(shmem_max[tid], shmem_max[tid + len])
        end
        sync_threads()
    end

    scale_f32 = 0f0
    if tid == 1
        m = shmem_max[1]
        if !isfinite(m) || m <= 0f0; m = 1f-8; end
        # Дѣлимъ на 4 (максимальное отклоненіе отъ zero-point), а не на 7
        scale_f32 = m / 4.0f0 
        @inbounds out_scales[bid] = Float16(scale_f32)
        shmem_max[1] = scale_f32
    end
    sync_threads()
    scale_f32 = shmem_max[1]

    # Квантованіе съ zero-point (сдвигомъ)
    for i in tid:stride:TQ3_1S_BLOCK_SIZE
        val = shmem_vals[i]
        # Сдвигаемъ на +4, чтобы диапазонъ [-4, 3] перешелъ въ [0, 7]
        q = clamp(round(Int32, Float32(val) / scale_f32 + TQ3_1S_ZERO_POINT), 0, TQ3_1S_MAX_VAL)
        shmem_q[i] = UInt8(q)
    end
    sync_threads()

    if tid == 1
        data_offset = (bid - 1) * TQ3_1S_BYTES_PER_BLOCK
        byte_idx = 1
        current_byte = UInt8(0)
        bits_in_byte = 0

        for i in 1:TQ3_1S_BLOCK_SIZE
            q = shmem_q[i]
            remaining_bits = TQ3_1S_BITS
            while remaining_bits > 0
                space = 8 - bits_in_byte
                take = min(remaining_bits, space)
                mask = UInt8((1 << take) - 1)
                current_byte |= ((q & mask) << bits_in_byte)
                q >>= take
                bits_in_byte += take
                remaining_bits -= take
                if bits_in_byte == 8
                    @inbounds out_data[data_offset + byte_idx] = current_byte
                    byte_idx += 1
                    current_byte = UInt8(0)
                    bits_in_byte = 0
                end
            end
        end
        if bits_in_byte > 0 && byte_idx <= TQ3_1S_BYTES_PER_BLOCK
            @inbounds out_data[data_offset + byte_idx] = current_byte
        end
    end
    return nothing
end

function tq3_1s_dequant_kernel(out, data, scales, n_blocks, n_elements)
    bid = blockIdx().x
    if bid > n_blocks; return; end

    tid = threadIdx().x
    stride = blockDim().x
    base_idx = (bid - 1) * TQ3_1S_BLOCK_SIZE
    data_offset = (bid - 1) * TQ3_1S_BYTES_PER_BLOCK

    scale = Float16(@inbounds scales[bid])
    shmem = @cuStaticSharedMem(Float16, TQ3_1S_BLOCK_SIZE)

    for i in tid:stride:TQ3_1S_BLOCK_SIZE
        idx = base_idx + i
        if idx <= n_elements
            bit_pos = (i - 1) * TQ3_1S_BITS
            byte_idx = (bit_pos >> 3) + 1
            bit_offset = bit_pos & 0x07

            @inbounds b0 = data[data_offset + byte_idx]
            val = (b0 >> bit_offset) & 0x07
            if bit_offset > 5
                @inbounds b1 = data[data_offset + byte_idx + 1]
                val |= (b1 << (8 - bit_offset)) & 0x07
            end
            
            # Вычитаемъ zero-point при распаковкѣ
            shmem[i] = (Float16(val) - Float16(TQ3_1S_ZERO_POINT)) * scale
        end
    end
    sync_threads()

    for i in tid:stride:TQ3_1S_BLOCK_SIZE
        idx = base_idx + i
        if idx <= n_elements
            @inbounds out[idx] = shmem[i]
        end
    end
    return nothing
end

function tq3_1s_compress(x::CuArray{Float16})
    n = length(x)
    if n == 0; return CuArray{UInt8}(), CuArray{Float16}(), size(x); end
    n_blocks = cld(n, TQ3_1S_BLOCK_SIZE)
    data = CUDA.zeros(UInt8, n_blocks * TQ3_1S_BYTES_PER_BLOCK)
    scales = CuArray{Float16}(undef, n_blocks)
    @cuda threads=TQ3_1S_BLOCK_SIZE blocks=n_blocks tq3_1s_compress_kernel(data, scales, x, n, n_blocks)
    return data, scales, size(x)
end

function tq3_1s_decompress(data::CuArray{UInt8}, scales::CuArray{Float16}, shape::Dims)
    n = prod(shape)
    if n == 0; return CUDA.zeros(Float16, shape); end
    n_blocks = length(scales)
    out = CuArray{Float16}(undef, n)
    @cuda threads=TQ3_1S_BLOCK_SIZE blocks=n_blocks tq3_1s_dequant_kernel(out, data, scales, n_blocks, n)
    return reshape(out, shape)
end

end # module
