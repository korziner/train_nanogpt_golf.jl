# ============================================================
# Custom FP16 GEMM for NVIDIA CMP 50HX (Turing SM 7.5)
# Bypasses broken cuBLAS Tensor Core path entirely.
#
# CRITICAL: CUDA.jl device indexing is 1-based (unlike CUDA C which is 0-based).
# blockIdx().x, threadIdx().x ∈ {1, 2, 3, ...}
# So we subtract 1 for arithmetic, then use directly as Julia array index.
# ============================================================

using CUDA

"""
FP16 GEMM kernel: each thread computes one entire row.
C[row, :] = A[row, :] * B  (Float32 accumulation)
"""
function _fp16_gemm_kernel!(
    C::CuDeviceMatrix{Float16},
    A::CuDeviceMatrix{Float16},
    B::CuDeviceMatrix{Float16},
    M::Int32, N::Int32, K::Int32
)
    # CUDA.jl blockIdx/threadIdx are 1-based. Convert to 0-based for grid-stride.
    row0   = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    stride = blockDim().x * gridDim().x

    while row0 < M
        row = row0 + Int32(1)  # Back to 1-based for Julia arrays
        for col in Int32(1):N
            acc = Float32(0.0)
            for k in Int32(1):K
                acc += Float32(A[row, k]) * Float32(B[k, col])
            end
            C[row, col] = Float16(acc)
        end
        row0 += stride
    end
    return nothing
end

function custom_fp16_gemm!(C, A, B)
    M = Int32(size(A, 1)); K = Int32(size(A, 2))
    K2 = Int32(size(B, 1)); N = Int32(size(B, 2))
    @assert K == K2 "Dimension mismatch A($M×$K) * B($K2×$N)"

    threads = Int32(256)
    blocks  = max(cld(M, threads), Int32(1))

    @cuda blocks=blocks threads=threads _fp16_gemm_kernel!(C, A, B, M, N, K)
    CUDA.synchronize()
    return C
end

function custom_fp16_gemm(A, B)
    M = size(A, 1); N = size(B, 2)
    C = CUDA.zeros(Float16, M, N)
    return custom_fp16_gemm!(C, A, B)
end

function Base.:*(A::CuArray{Float16, 2}, B::CuArray{Float16, 2})
    return custom_fp16_gemm(A, B)
end

# ============================================================
# Batched GEMM (for attention)
# ============================================================
function _fp16_batched_gemm_kernel!(
    C::CuDeviceArray{Float16, 3},
    A::CuDeviceArray{Float16, 3},
    B::CuDeviceArray{Float16, 3},
    M::Int32, N::Int32, K::Int32, Batch::Int32
)
    # CUDA.jl blockIdx/threadIdx are 1-based. Convert to 0-based.
    idx0   = (blockIdx().x - Int32(1)) * blockDim().x + (threadIdx().x - Int32(1))
    stride = blockDim().x * gridDim().x
    total  = M * Batch

    while idx0 < total
        row = (idx0 % M) + Int32(1)
        bat = div(idx0, M) + Int32(1)

        for col in Int32(1):N
            acc = Float32(0.0)
            for k in Int32(1):K
                acc += Float32(A[row, k, bat]) * Float32(B[k, col, bat])
            end
            C[row, col, bat] = Float16(acc)
        end
        idx0 += stride
    end
    return nothing
end

function custom_batched_mul_fp16(A::CuArray{Float16, 3}, B::CuArray{Float16, 3})
    M = Int32(size(A, 1)); K = Int32(size(A, 2)); Batch = Int32(size(A, 3))
    K2 = Int32(size(B, 1)); N = Int32(size(B, 2))
    @assert K == K2
    @assert Batch == Int32(size(B, 3))

    C = CUDA.zeros(Float16, M, N, Batch)
    threads = Int32(256)
    blocks  = max(cld(M * Batch, threads), Int32(1))

    @cuda blocks=blocks threads=threads _fp16_batched_gemm_kernel!(C, A, B, M, N, K, Batch)
    CUDA.synchronize()
    return C
end

function NNlib.batched_mul(A::CuArray{Float16, 3}, B::CuArray{Float16, 3})
    return custom_batched_mul_fp16(A, B)
end

# ============================================================
# Verification — with SMALL matrices first for debugging
# ============================================================
function verify_custom_fp16_gemm()
    @info "🔧 Testing custom FP16 GEMM kernel..."

    # === Small test with identity matrix ===
    @info "--- Small test (4x4 identity × random) ---"
    Ms, Ns, Ks = Int32(4), Int32(4), Int32(4)
    As = CuArray(Float16[1 0 0 0; 0 2 0 0; 0 0 3 0; 0 0 0 4])
    Bs = CuArray(Float16[1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16])
    Cs = custom_fp16_gemm(As, Bs)
    
    As_cpu = Array(As); Bs_cpu = Array(Bs); Cs_cpu = Array(Cs)
    Cs_ref = Float16.(Float32.(As_cpu) * Float32.(Bs_cpu))
    
    @info "A = "; println(As_cpu)
    @info "B = "; println(Bs_cpu)
    @info "C (computed) = "; println(Cs_cpu)
    @info "C (expected) = "; println(Cs_ref)
    
    match_small = Cs_cpu == Cs_ref
    @info "Small test match: $match_small"
    
    if !match_small
        @warn "❌ Small test FAILED — kernel is broken"
        return false
    end
    
    # === Large test ===
    @info "--- Large test (2048x2048) ---"
    Nl = 2048
    Al = CUDA.rand(Float16, Nl, Nl)
    Bl = CUDA.rand(Float16, Nl, Nl)

    # Warmup
    Cl = custom_fp16_gemm(Al, Bl)
    CUDA.synchronize()

    # Benchmark
    t = @elapsed begin
        for _ in 1:10
            custom_fp16_gemm!(Cl, Al, Bl)
        end
        CUDA.synchronize()
    end

    # Correctness
    Al_cpu = Array(Al); Bl_cpu = Array(Bl); Cl_cpu = Array(Cl)
    Cl_ref = Float16.(Float32.(Al_cpu) * Float32.(Bl_cpu))

    max_err = maximum(abs.(Float32.(Cl_cpu) .- Float32.(Cl_ref)))
    rel_err = max_err / max(maximum(abs.(Float32.(Cl_ref))), 1f-8)
    tflops  = 10 * 2 * Nl^3 / t / 1e12

    @info "Large FP16 GEMM Results:" tflops=round(tflops, digits=1) rel_err=round(rel_err, digits=6) max_err=round(max_err, digits=6)

    if rel_err < 0.01
        @info "✅ Custom FP16 GEMM PASSED"
        return true
    else
        @warn "❌ Custom FP16 GEMM FAILED (relative error: $rel_err)"
        return false
    end
end

export custom_fp16_gemm, custom_fp16_gemm!, custom_batched_mul_fp16, verify_custom_fp16_gemm
