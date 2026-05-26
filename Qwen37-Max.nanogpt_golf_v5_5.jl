#!/usr/bin/env julia
# NanoGPT-Golf v5.5 — Hybrid tokenizer + Z-Loss + Master Weights + Hybrid Checkpoint
# Optimized for NVIDIA CMP 50HX (Turing SM 7.5) with broken Tensor Cores and FP32 FMA microcode bug

using Flux, NNlib, Optimisers, Zygote, Functors
using Flux: params
using ArgParse, JLD2, JSON3
using LinearAlgebra, Statistics, Random, Printf, Dates

# === 1. Проверка LLVM аргументов для запрета FMA contraction ===
# Флаг --fp-contract=off запрещает LLVM склеивать mul+add в fma инструкцию
# Это критично для CMP 50HX, где FP32 FMA микрокод сломан
if !haskey(ENV, "JULIA_LLVM_ARGS") || !occursin("fp-contract=off", ENV["JULIA_LLVM_ARGS"])
    @warn "⚠️ JULIA_LLVM_ARGS not set. To disable FMA contraction, run with:"
    @warn "   export JULIA_LLVM_ARGS=\"--fp-contract=off\""
else
    @info "✅ JULIA_LLVM_ARGS set: FMA contraction disabled"
end

using CUDA
using CUDA.CUBLAS

#!/usr/bin/env julia
# NanoGPT-Golf v5.5 — Optimized for NVIDIA CMP 50HX (Turing SM 7.5)

using Flux, NNlib, Optimisers, Zygote, Functors
using Flux: params
using ArgParse, JLD2, JSON3
using LinearAlgebra, Statistics, Random, Printf, Dates

using CUDA
using CUDA.CUBLAS

# === Принудительная установка PEDANTIC_MATH для cuBLAS ===
# Режим PEDANTIC_MATH (значение 2) обходит сломанные Tensor Cores и отключает TF32
function force_cublas_pedantic!()
    try
        handle = CUBLAS.handle()
        
        # В CUDA.jl 5.x cublasMath_t определён через @cenum
        pedantic_mode = if isdefined(CUBLAS, :CUBLAS_PEDANTIC_MATH)
            CUBLAS.CUBLAS_PEDANTIC_MATH
        elseif isdefined(CUBLAS, :cublasMath_t)
            CUBLAS.cublasMath_t(2)
        else
            Cint(2)
        end
        
        CUBLAS.cublasSetMathMode(handle, pedantic_mode)
        @info "✅ cuBLAS math mode set to PEDANTIC_MATH (bypasses broken TC/FMA on CMP 50HX)"
    catch e
        @warn "⚠️ Failed to set cuBLAS math mode: $e. Continuing with default."
    end
end

force_cublas_pedantic!()

# ... остальной код ...

# dtype = Float32
ENV["CUDA_DISABLE_TENSOR_CORES"] = "1"

CUDA.versioninfo()

# === 3. Верификация: проверка отсутствия FMA инструкций в PTX ===
# В CUDA.jl 5.x @cuda требует, чтобы функция была определена отдельно
function _fma_test_kernel(d, a, b, c, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        # Если бы работал FMA, здесь была бы одна инструкция
        # С флагом --fp-contract=off LLVM сгенерирует две: mul + add
        @inbounds d[i] = a[i] + b[i] * c[i]
    end
    return nothing
end


# === 4. Верификация производительности FP16 GEMM ===
function verify_gpu_math_performance()
    try
        N = 2048
        A = CUDA.rand(Float16, N, N)
        B = CUDA.rand(Float16, N, N)
        C = CUDA.zeros(Float16, N, N)
        
        # Warmup
        CUBLAS.gemm!('N', 'N', Float16(1.0), A, B, Float16(0.0), C)
        CUDA.synchronize()
        
        # Benchmark (20 iters)
        t = @elapsed for _ in 1:20
            CUBLAS.gemm!('N', 'N', Float16(1.0), A, B, Float16(0.0), C)
        end
        CUDA.synchronize()
        
        # Correctness check
        C_cpu = Array(C)
        correct = abs(C_cpu[1,1] - sum(Array(A[:,1]) .* Array(B[1,:]))) < 1f-2
        
        flops = 20 * 2 * N^3 / t / 1e12
        
        @info "GPU Math Verification" fp16_tflops=round(flops, digits=1) correctness=correct
        
        if !correct || flops < 12.0
            @warn "❌ FP16 GEMM suboptimal: $(flops) TFLOP/s. Expected >12 in PEDANTIC mode."
        else
            @info "✅ FP16 GEMM healthy: $(round(flops, digits=1)) TFLOP/s"
        end
    catch e
        @warn "⚠️ GPU performance verification failed: $e"
    end
end

verify_gpu_math_performance()

using NNkernels
using Zygote: @adjoint

const HAS_CUDA = try CUDA.functional() catch _ false end
DEV(x) = HAS_CUDA ? gpu(x) : x

# ============================================================
# HYBRID TOKENIZER (vocab = 385)
# ============================================================
#!/usr/bin/env julia
# NanoGPT-Golf v5.5 — Hybrid tokenizer + Z-Loss + Master Weights + Hybrid Checkpoint
# ============================================================
const BYTE_VOCAB  = 256
const CYR2_BASE   = BYTE_VOCAB + 1
const CYR2_STRIDE = 64
const CYR2_COUNT  = 2 * CYR2_STRIDE
const EOS_TOKEN   = CYR2_BASE + CYR2_COUNT
const VOCAB       = EOS_TOKEN

@inline is_cyr2_pair(b1::UInt8, b2::UInt8) =
    ((b1 == UInt8(0xD0) || b1 == UInt8(0xD1)) && (UInt8(0x80) <= b2 <= UInt8(0xBF)))

@inline function cyr2_token(b1::UInt8, b2::UInt8)::Int32
    lead_off = (b1 == UInt8(0xD0)) ? 0 : CYR2_STRIDE
    return Int32(CYR2_BASE + lead_off + (Int(b2) - 0x80))
end

@inline function token_to_cyr2(t::Integer)
    off = Int(t) - CYR2_BASE
    b1 = off < CYR2_STRIDE ? UInt8(0xD0) : UInt8(0xD1)
    b2 = UInt8(0x80 + (off % CYR2_STRIDE))
    return b1, b2
end

function encode_text_tokens(txt::AbstractString; add_eos::Bool=true)
    bs = collect(codeunits(txt))
    out = Int32[]
    sizehint!(out, length(bs) + 2)
    i = 1
    while i <= length(bs)
        if i < length(bs) && is_cyr2_pair(bs[i], bs[i+1])
            push!(out, cyr2_token(bs[i], bs[i+1]))
            i += 2
        else
            push!(out, Int32(bs[i]) + 1)
            i += 1
        end
    end
    add_eos && push!(out, Int32(EOS_TOKEN))
    return out
end

function decode_tokens(ts)
    buf = UInt8[]
    sizehint!(buf, length(ts) * 2)
    for t in ts
        if 1 <= t <= 256
            push!(buf, UInt8(t - 1))
        elseif CYR2_BASE <= t < EOS_TOKEN
            b1, b2 = token_to_cyr2(t)
            push!(buf, b1); push!(buf, b2)
        end
    end
    return String(buf)
end

decode_bytes(ts) = decode_tokens(ts)

function probe_tokenizer(path::AbstractString; max_lines::Int=200_000)
    raw_bytes = 0; toks = 0; fused = 0; lines = 0
    open(path, "r") do io
        for line in eachline(io)
            isempty(line) && continue
            lines += 1
            txt = try string(JSON3.read(line)[:text]) catch _; line end
            isempty(txt) && continue
            bs = collect(codeunits(txt))
            raw_bytes += length(bs)
            i = 1
            while i <= length(bs)
                if i < length(bs) && is_cyr2_pair(bs[i], bs[i+1])
                    toks += 1; fused += 1; i += 2
                else
                    toks += 1; i += 1
                end
            end
            toks += 1
            lines >= max_lines && break
        end
    end
    @printf("📐 Tokenizer probe: lines=%d raw_bytes=%d tokens=%d fused=%d shrink=%.3fx fused_share=%.2f%%\n",
            lines, raw_bytes, toks, fused, raw_bytes / max(toks, 1), 100 * fused / max(toks, 1))
    return (lines=lines, raw_bytes=raw_bytes, tokens=toks, fused=fused, shrink=raw_bytes/max(toks,1))
end

# ============================================================
# Z-LOSS & NLL
# ============================================================
const Z_LOSS_COEF = Ref{Float32}(0.0f0)

function nll_loss(logits::AbstractMatrix, targets::AbstractVecOrMat{<:Integer})
    logits32 = eltype(logits) === Float32 ? logits : Float32.(logits)
    flat_t = vec(targets)
    N = length(flat_t)
    V = size(logits32, 1)
    lp = Flux.logsoftmax(logits32; dims=1)
    idx = flat_t .+ (0:N-1) .* V
    base_loss = -mean(lp[idx])
    z_loss_part = 0.0f0
    if Z_LOSS_COEF[] > 0
        vars = var(logits32; dims=1)
        z_loss_part = -mean(log.(vars .+ Float32(1e-8)))
    end
    return base_loss + Z_LOSS_COEF[] * z_loss_part
end

# ============================================================
# Tree utils
# ============================================================
scalar_value(x) = x isa Number ? Float32(x) : Float32(Array(cpu(x))[1])

function tree_all_finite(x)::Bool
    ok = Ref(true)
    Functors.fmap(x) do v
        if v isa AbstractArray
            try; ok[] &= Bool(scalar_value(all(isfinite, v))); catch _; ok[]=false; end
        end
        v
    end
    ok[]
end

function tree_max_abs(x)::Float32
    mx = Ref(0f0)
    Functors.fmap(x) do v
        if v isa AbstractArray && length(v)>0
            try; m = scalar_value(maximum(abs, v)); mx[] = max(mx[], m); catch _; mx[]=Inf32; end
        end
        v
    end
    mx[]
end

function grad_l2norm(gs)::Float32
    s = Ref(0f0)
    Functors.fmap(gs) do g
        g isa AbstractArray && (s[] += scalar_value(sum(abs2, g)))
        g
    end
    sqrt(s[])
end

scale_grads(gs, scale::Float32) = Functors.fmap(gs) do g
    g isa AbstractArray ? g .* scale : g
end

function add_grads(a, b)
    Functors.fmap(a, b) do x, y
        (x === nothing || y === nothing) ? nothing : (x .+ y)
    end
end

div_grads(gs, denom) = Functors.fmap(gs) do g
    g isa AbstractArray ? g ./ denom : g
end

function ema_update!(m_ema, m, beta::Float32)
    Functors.fmap(m_ema, m) do a, b
        (a isa AbstractArray && b isa AbstractArray) &&
            (a .= beta .* a .+ (1f0-beta) .* eltype(a).(b))
        a
    end
    m_ema
end

# ============================================================
# Pretty printing & Hardware
# ============================================================
fmt_bytes(n::Integer) = begin
    n < 1024 && return "$(n) B"
    u = Float64(n); units = ["KiB","MiB","GiB","TiB"]; i=0
    while u >= 1024 && i < length(units); u /= 1024; i += 1; end
    @sprintf("%.2f %s", u, units[i])
end

bar_bytes(n::Integer; maxn::Integer=1, width::Int=28, full::Char='█', empty::Char='░') = begin
    maxn <= 0 && return string(empty)^width
    r = clamp(n / maxn, 0.0, 1.0); k = Int(round(r * width))
    string(full)^k * string(empty)^(width-k)
end

section(title::String) = begin
    println("┏", "━"^70)
    println("┃ ", title)
    println("┗", "━"^70)
end

struct GPUInfo
    ok::Bool; name::String; total_mem::Int; free_mem::Int
    l2_cache::Int; sm_count::Int; shared_mem_per_sm::Int
    shared_mem_per_block::Int; warp_size::Int
    cc_major::Int; cc_minor::Int
end

function detect_gpu()::GPUInfo
    !HAS_CUDA && return GPUInfo(false, "CPU", 0, 0, 0, 0, 0, 0, 0, 0, 0)
    dev = CUDA.device()
    name = CUDA.name(dev)
    total_mem = Int(CUDA.total_memory())
    free_mem  = Int(CUDA.available_memory())
    l2    = try Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)) catch _; 0 end
    sm    = try Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)) catch _; 0 end
    sh_sm = try Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)) catch _; 0 end
    sh_blk= try Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)) catch _; 0 end
    warp  = try Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE)) catch _; 32 end
    cc = CUDA.capability(dev)
    GPUInfo(true, name, total_mem, free_mem, l2, sm, sh_sm, sh_blk, warp, cc.major, cc.minor)
end

struct CPUCacheInfo
    ok::Bool; l1d::Int; l1i::Int; l2::Int; l3::Int; note::String
end

function detect_cpu_caches()::CPUCacheInfo
    base = "/sys/devices/system/cpu/cpu0/cache"
    !isdir(base) && return CPUCacheInfo(false,0,0,0,0,"sysfs not available")
    l1d=l1i=l2=l3=0; note="from sysfs"
    for entry in readdir(base; join=true)
        startswith(basename(entry),"index") || continue
        try
            lvl = parse(Int, strip(read(joinpath(entry,"level"),String)))
            typ = strip(read(joinpath(entry,"type"),String))
            sz  = parse_sysfs_size(strip(read(joinpath(entry,"size"),String)))
            if lvl==1 && typ=="Data";            l1d=max(l1d,sz)
            elseif lvl==1 && typ=="Instruction"; l1i=max(l1i,sz)
            elseif lvl==2 && typ in ("Unified","Data"); l2=max(l2,sz)
            elseif lvl==3 && typ in ("Unified","Data"); l3=max(l3,sz)
            end
        catch _; end
    end
    ok = (l1d+l1i+l2+l3)>0
    ok || (note = "could not parse sysfs cache sizes")
    CPUCacheInfo(ok,l1d,l1i,l2,l3,note)
end

function parse_sysfs_size(s::String)
    t = lowercase(strip(s))
    endswith(t,"k") && return parse(Int,t[1:end-1])*1024
    endswith(t,"m") && return parse(Int,t[1:end-1])*1024^2
    endswith(t,"g") && return parse(Int,t[1:end-1])*1024^3
    parse(Int,t)
end

function print_hw(gpu::GPUInfo, cpu_cache::CPUCacheInfo)
    section("Hardware & Cache hierarchy (detected)")
    if gpu.ok
        println("GPU: ", gpu.name, "  (cc ", gpu.cc_major, ".", gpu.cc_minor, ")")
        println("  VRAM total: ", fmt_bytes(gpu.total_mem), "  ", bar_bytes(gpu.total_mem; maxn=gpu.total_mem))
        println("  VRAM free : ", fmt_bytes(gpu.free_mem),  "  ", bar_bytes(gpu.free_mem;  maxn=gpu.total_mem))
        println("  L2 cache  : ", fmt_bytes(gpu.l2_cache))
        println("  SM count  : ", gpu.sm_count)
    else
        println("GPU: not available (CPU-only)")
    end
    println()
    println("CPU caches: ", cpu_cache.ok ? cpu_cache.note : "(unknown)")
    println()
end

# ============================================================
# CLI
# ============================================================
function parse_cmd()
    s = ArgParseSettings(description="NanoGPT-Golf v5.5 (vocab=$VOCAB)")
    @add_arg_table s begin
        "--data"; required=true
        "--save"; default="model_golf_v5_5.jld2"
        "--ckpt-dir"; default="checkpoints_v55"
        "--resume"; default=""
        "--ckpt-every-steps"; arg_type=Int; default=10
        "--keep-last"; arg_type=Int; default=6

        "--attn"; default="flash"
        "--layers"; arg_type=Int; default=11
        "--dim"; arg_type=Int; default=384
        "--heads"; arg_type=Int; default=6
        "--kv-heads"; arg_type=Int; default=3
        "--ff-mult"; arg_type=Int; default=3
        "--seq"; arg_type=Int; default=512
        "--batch"; arg_type=Int; default=4
        "--accum"; arg_type=Int; default=16

        "--iters"; arg_type=Int; default=25000
        "--lr"; arg_type=Float64; default=6e-4
        "--lr-min"; arg_type=Float64; default=1e-5
        "--lr-scheduler"; default="cosine_restarts"
        "--lr-T0"; arg_type=Int; default=4000
        "--lr-Tmult"; arg_type=Float64; default=2.0
        "--warmup"; arg_type=Int; default=500
        "--grad-clip"; arg_type=Float64; default=1.0

        "--resume-warmup"; arg_type=Int; default=300
        "--resume-lr-scale"; arg_type=Float64; default=0.25
        "--min-lr-scale"; arg_type=Float64; default=0.05
        "--lr-backoff-factor"; arg_type=Float64; default=0.5

        "--wd"; arg_type=Float64; default=0.01
        "--muon-beta"; arg_type=Float64; default=0.95
        "--muon-beta2"; arg_type=Float64; default=0.95
        "--muon-ns-steps"; arg_type=Int; default=5
        
        "--optimizer-8bit"; action=:store_true
        "--optimizer-8bit-scale-update-every"; arg_type=Int; default=16

        "--log-every"; arg_type=Int; default=10
        "--sample-every-steps"; arg_type=Int; default=200
        "--sample-tokens"; arg_type=Int; default=140
        "--seed"; arg_type=Int; default=1337
        "--dry-run"; action=:store_true
        "--print-hw"; action=:store_true
        "--quick-check"; action=:store_true
        "--probe-tokenizer"; action=:store_true

        "--sample-greedy"; action=:store_true
        "--sample-topk"; arg_type=Int; default=40
        "--sample-topp"; arg_type=Float64; default=0.9
        "--sample-temp"; arg_type=Float64; default=0.8

        "--min-healthy-loss"; arg_type=Float64; default=0.03
        "--max-healthy-loss"; arg_type=Float64; default=200.0
        "--loss-spike-factor"; arg_type=Float64; default=1.35
        "--loss-ema-beta"; arg_type=Float64; default=0.98
        "--param-check-every"; arg_type=Int; default=25
        "--max-param-abs"; arg_type=Float64; default=200.0
        "--bad-step-patience"; arg_type=Int; default=1
        "--bad-sample-patience"; arg_type=Int; default=1
        "--abort-on-bad-sample"; action=:store_true

        "--min-space-ratio"; arg_type=Float64; default=0.01
        "--max-top-token-ratio"; arg_type=Float64; default=0.65
        "--max-prefix-ratio"; arg_type=Float64; default=0.95
        "--max-repeat-run"; arg_type=Int; default=96
        "--min-unique-ratio"; arg_type=Float64; default=0.02
        "--min-sample-entropy"; arg_type=Float64; default=1.4
        "--min-bigram-diversity"; arg_type=Float64; default=0.15
        "--min-trigram-diversity"; arg_type=Float64; default=0.10

        "--rollback-on-bad-step"; action=:store_true
        "--stop-on-collapse"; action=:store_true

        "--autotune"; action=:store_true
        "--no-autotune"; action=:store_true
        "--autotune-benchmark"; action=:store_true
        "--no-autotune-benchmark"; action=:store_true
        "--autotune-max-seq"; arg_type=Int; default=8192
        "--autotune-max-batch"; arg_type=Int; default=256
        "--autotune-max-global-scale"; arg_type=Float64; default=4.00
        "--autotune-vram-reserve-gb"; arg_type=Float64; default=0.80
        "--autotune-vram-target-frac"; arg_type=Float64; default=0.85
        "--autotune-candidates"; arg_type=Int; default=10
        "--autotune-bench-iters"; arg_type=Int; default=3
        "--autotune-bench-runs"; arg_type=Int; default=3

        "--loader-cache-frac"; arg_type=Float64; default=0.25
        "--loader-min-mb"; arg_type=Int; default=4
        "--loader-max-mb"; arg_type=Int; default=64

        "--qat"; action=:store_true
        "--qat-bits-start"; arg_type=Int; default=8
        "--qat-bits-final"; arg_type=Int; default=6
        "--qat-alpha-mid"; arg_type=Float64; default=0.5
        "--qat-start-step"; arg_type=Int; default=-1
        "--qat-warmup"; arg_type=Int; default=500
        "--qat-freeze-alpha"; action=:store_true
        "--qat-alpha-target"; arg_type=Float64; default=1.0
        "--qat-scale-update-every"; arg_type=Int; default=16
        "--qat-per-row"; action=:store_true
        "--qat-per-tensor"; action=:store_true

        "--ema-beta"; arg_type=Float64; default=0.999
        "--fp16"; action=:store_true
        "--no-fp16"; action=:store_true
        "--z-loss"; arg_type=Float64; default=0.0

 "--skip-hybrid-checkpoint-test"; action=:store_true

    end

    args = parse_args(s)
    if !args["autotune"] && !args["no-autotune"]; args["autotune"] = false; end
    args["autotune"] && !args["no-autotune"] && (args["autotune"] = true)
    args["no-autotune"] && (args["autotune"] = false)

    if !args["autotune-benchmark"] && !args["no-autotune-benchmark"]
        args["autotune-benchmark"] = true
    end
    args["no-autotune-benchmark"] && (args["autotune-benchmark"] = false)

    args["lr-min"] = max(args["lr-min"], 1e-7)

    if !args["qat-per-row"] && !args["qat-per-tensor"]
        args["qat-per-row"] = true
    end
    args["qat-per-tensor"] && (args["qat-per-row"] = false)

    if !args["fp16"] && !args["no-fp16"]
        args["fp16"] = HAS_CUDA
    end
    args["no-fp16"] && (args["fp16"] = false)

    return args
end

# ============================================================
# QAT control
# ============================================================
const QAT_ON = Ref(false)
const QAT_ALPHA = Ref(0f0)
const QAT_BITS_CUR = Ref(8)
const QAT_STEP = Ref(0)
const QAT_PER_ROW = Ref(true)
const QAT_SCALE_UPDATE_EVERY = Ref(16)

function update_qat_control!(args, step::Int, start_step::Int)
    QAT_STEP[] = step
    if !args["qat"]
        QAT_ON[] = false; QAT_ALPHA[] = 0f0
        QAT_BITS_CUR[] = args["qat-bits-final"]; return
    end
    qstart = args["qat-start-step"] < 0 ? start_step : args["qat-start-step"]
    if step < qstart
        QAT_ON[] = false; QAT_ALPHA[] = 0f0
        QAT_BITS_CUR[] = args["qat-bits-final"]; return
    end
    QAT_ON[] = true
    QAT_SCALE_UPDATE_EVERY[] = args["qat-scale-update-every"]
    QAT_PER_ROW[] = args["qat-per-row"]
    if args["qat-freeze-alpha"]
        QAT_ALPHA[] = Float32(clamp(args["qat-alpha-target"],0,1))
    else
        ramp = max(1, args["qat-warmup"])
        t = clamp((step-qstart)/ramp, 0.0, 1.0)
        QAT_ALPHA[] = Float32(t)
    end
    α = QAT_ALPHA[]; mid = Float32(args["qat-alpha-mid"])
    QAT_BITS_CUR[] = α < mid ? args["qat-bits-start"] : args["qat-bits-final"]
end

const RECENT_GNORMS = Float32[]
const RECENT_GNORMS_MAX = 100

function push_gnorm!(v::Float32)
    push!(RECENT_GNORMS, v)
    while length(RECENT_GNORMS) > RECENT_GNORMS_MAX
        popfirst!(RECENT_GNORMS)
    end
end

# ============================================================
# ByteLoader
# ============================================================
mutable struct ByteLoader
    io::IO; buf::Vector{Int32}; pos::Int
    seq::Int; batch::Int; target_tokens::Int
end

function ByteLoader(path::String, seq::Int, batch::Int; target_tokens::Int=1_000_000)
    ByteLoader(open(path,"r"), Int32[], 1, seq, batch, target_tokens)
end

loader_available(ld::ByteLoader) = length(ld.buf) - ld.pos + 1

function maybe_compact!(ld::ByteLoader)
    if ld.pos > 1_000_000 || (ld.pos > length(ld.buf)÷2 && ld.pos > 10_000)
        ld.buf = ld.buf[ld.pos:end]; ld.pos = 1
    end
end

function refill!(ld::ByteLoader, need::Int)
    target = max(need, ld.target_tokens)
    while loader_available(ld) < target
        if eof(ld.io)
            try seekstart(ld.io) catch end
            break
        end
        
        local line
        read_ok = true
        try
            line = readline(ld.io)
        catch
            read_ok = false
        end
        
        if !read_ok
            break
        end
        
        isempty(line) && continue
        txt = try string(JSON3.read(line)[:text]) catch _; line end
        isempty(txt) && continue
        append!(ld.buf, encode_text_tokens(txt))
    end
end

function next_batch!(ld::ByteLoader)
    need = (ld.seq + 1) * ld.batch
    refill!(ld, need)
    if loader_available(ld) < need
        append!(ld.buf, fill(Int32(EOS_TOKEN), need - loader_available(ld)))
    end
    chunk = ld.buf[ld.pos:(ld.pos + need - 1)]
    ld.pos += need
    maybe_compact!(ld)
    reshape(chunk, ld.seq + 1, ld.batch)
end

function loader_state_for_save(ld::ByteLoader)
    tail = ld.pos <= length(ld.buf) ? ld.buf[ld.pos:end] : Int32[]
    return tail, 1
end

function loader_restore!(ld::ByteLoader, buf::Vector{Int32}, pos::Int)
    empty!(ld.buf); append!(ld.buf, Int32.(buf))
    ld.pos = max(1, pos)
end

# ============================================================
# Hybrid Checkpoint (PCIe offload)
# ============================================================
const OFFLOAD_BYTES_TO_CPU = Ref{Int64}(0)
const OFFLOAD_BYTES_TO_GPU = Ref{Int64}(0)
const OFFLOAD_TIME_CPU_S   = Ref{Float64}(0.0)
const OFFLOAD_TIME_GPU_S   = Ref{Float64}(0.0)
const OFFLOAD_COUNT        = Ref{Int64}(0)
const HYBRID_CKPT_ENABLED  = Ref{Bool}(true)
const HYBRID_CKPT_TESTED   = Ref{Bool}(false)

function reset_offload_stats!()
    OFFLOAD_BYTES_TO_CPU[] = 0
    OFFLOAD_BYTES_TO_GPU[] = 0
    OFFLOAD_TIME_CPU_S[]   = 0.0
    OFFLOAD_TIME_GPU_S[]   = 0.0
    OFFLOAD_COUNT[]        = 0
end

function print_offload_stats(step::Int; log_interval::Int=1)
    cnt = OFFLOAD_COUNT[]
    cnt == 0 && return
    mb_to_cpu = OFFLOAD_BYTES_TO_CPU[] / 1024^2
    mb_to_gpu = OFFLOAD_BYTES_TO_GPU[] / 1024^2
    bw_cpu = mb_to_cpu / max(OFFLOAD_TIME_CPU_S[], 1e-6)
    bw_gpu = mb_to_gpu / max(OFFLOAD_TIME_GPU_S[], 1e-6)
    @printf("📊 PCIe offload @ step %d: calls=%d │ GPU→CPU %.1f MB @ %.0f MB/s │ CPU→GPU %.1f MB @ %.0f MB/s │ Σ %.1f MB\n",
            step, cnt, mb_to_cpu, bw_cpu, mb_to_gpu, bw_gpu, mb_to_cpu + mb_to_gpu)
    reset_offload_stats!()
end

function hybrid_checkpoint(f, x)
    return f(x)
end

@adjoint function hybrid_checkpoint(f, x)
    if !HYBRID_CKPT_ENABLED[] || !(x isa CuArray)
        y_fb, pullback_fb = Zygote.pullback(f, x)
        return y_fb, Δ -> (nothing, pullback_fb(Δ)[1])
    end

    nbytes = sizeof(x)
    y = Zygote.ignore_derivatives() do
        f(x)
    end

    t0 = time_ns()
    x_cpu = Array(x)
    HAS_CUDA && CUDA.synchronize()
    dt = (time_ns() - t0) / 1e9
    OFFLOAD_BYTES_TO_CPU[] += nbytes
    OFFLOAD_TIME_CPU_S[]   += dt
    OFFLOAD_COUNT[]        += 1

    function back(Δ)
        t1 = time_ns()
        x_reloaded = CuArray(x_cpu)
        HAS_CUDA && CUDA.synchronize()
        dt_r = (time_ns() - t1) / 1e9
        OFFLOAD_BYTES_TO_GPU[] += nbytes
        OFFLOAD_TIME_GPU_S[]   += dt_r

        if HAS_CUDA
            try
                if any(!isfinite, x_reloaded)
                    @warn "hybrid_checkpoint: x_reloaded contains NaN/Inf, zeroing gradients"
                    return (nothing, zero(x))
                end
            catch e
                @warn "hybrid_checkpoint: check failed: $e"
            end
        end

        _, pullback = Zygote.pullback(f, x_reloaded)
        grads = pullback(Δ)

        if grads[1] !== nothing
            try
                if any(!isfinite, grads[1])
                    @warn "hybrid_checkpoint: gradients contain NaN/Inf, zeroing"
                    return (nothing, zero(x))
                end
            catch e
                @warn "hybrid_checkpoint: grad check failed: $e"
            end
        end

        return (nothing, grads[1])
    end

    return y, back
end

# ============================================================
# QAT globals
# ============================================================
round_ste(x::Real) = round(x)
@adjoint round_ste(x::Real) = round(x), Δ -> (Δ,)

qmax_int(bits::Int) = Float32(2^(bits-1)-1)

function compute_scale_sym(W; bits::Int, per_row::Bool, eps::Float32=1f-8)
    qm = qmax_int(bits)
    if per_row
        mx = vec(maximum(abs, Float32.(W); dims=2))
        return max.(mx ./ qm, eps)
    else
        mx = Float32(maximum(abs, W))
        return max(mx / qm, eps)
    end
end

function fake_quant_weight_sym(W; bits::Int, per_row::Bool)
    qm = qmax_int(bits)
    sc = Zygote.ignore_derivatives() do
        compute_scale_sym(W; bits=bits, per_row=per_row)
    end
    T = eltype(W)
    if per_row
        Wn = W .* reshape(T.(1f0 ./ sc), :, 1)
        Wq = clamp.(round_ste.(Wn), T(-qm), T(qm)) .* reshape(T.(sc), :, 1)
        return Wq
    else
        Wn = W .* T(1f0 / sc)
        Wq = clamp.(round_ste.(Wn), T(-qm), T(qm)) .* T(sc)
        return Wq
    end
end

function dense_qat_forward(W::AbstractMatrix, x)
    !QAT_ON[] && return W * x
    bits = QAT_BITS_CUR[]
    Wq = fake_quant_weight_sym(W; bits=bits, per_row=QAT_PER_ROW[])
    α = QAT_ALPHA[]
    T = eltype(W)
    Wmix = (T(1)-T(α)) .* W .+ T(α) .* Wq
    return Wmix * x
end

# ============================================================
# Model
# ============================================================
struct RMSNorm{W}; scale::W; end
Flux.@layer RMSNorm
RMSNorm(dim::Int) = RMSNorm(ones(Float32, dim))
(r::RMSNorm)(x) = x ./ sqrt.(mean(abs2, x; dims=1) .+ 1f-6) .* r.scale

relu2(x) = relu.(x) .^ 2

function make_rope_cache(hd::Int, maxseq::Int, rd::Int)
    d = min(rd - rd % 2, hd)
    θ  = 10000f0 .^ (-2f0 .* Float32.(0:d÷2-1) ./ d)
    ang = Float32.(0:maxseq-1) * θ'
    cc = permutedims(cos.(ang), (2,1))
    ss = permutedims(sin.(ang), (2,1))
    return cc, ss, d
end

function apply_rope(x, cc, ss, rd)
    h   = rd ÷ 2
    seq = size(x, 2)
    c = cc[:, 1:seq]; s = ss[:, 1:seq]
    x1 = x[1:h, :, :]; x2 = x[h+1:rd, :, :]
    r1 = x1 .* c .- x2 .* s
    r2 = x1 .* s .+ x2 .* c
    rd < size(x, 1) ? vcat(r1, r2, x[rd+1:end, :, :]) : vcat(r1, r2)
end

struct GQAttention{Q,K,V,O,C,S}
    Wq::Q; Wk::K; Wv::V; Wo::O
    cos_c::C; sin_c::S
    nq::Int; nkv::Int; hd::Int; rd::Int
    cap::Float32; mode::String
end

Flux.@layer GQAttention trainable=(Wq,Wk,Wv,Wo)
Functors.@functor GQAttention (Wq,Wk,Wv,Wo,cos_c,sin_c)

function GQAttention(dim::Int, nq::Int, nkv::Int, seq::Int; rd=16, cap=30f0, mode="flash")
    hd = div(dim, nq)
    cc, ss, rd_a = make_rope_cache(hd, seq, rd)
    GQAttention(
        Dense(dim => dim;      bias=false),
        Dense(dim => hd*nkv;   bias=false),
        Dense(dim => hd*nkv;   bias=false),
        Dense(dim => dim;      bias=false),
        cc, ss, nq, nkv, hd, rd_a, Float32(cap), mode)
end

function (a::GQAttention)(x)
    dim, seq, batch = size(x)
    hd, nq, nkv = a.hd, a.nq, a.nkv
    xf = reshape(x, dim, :)

    q = reshape(dense_qat_forward(a.Wq.weight, xf), hd, nq,  seq, batch)
    k = reshape(dense_qat_forward(a.Wk.weight, xf), hd, nkv, seq, batch)
    v = reshape(dense_qat_forward(a.Wv.weight, xf), hd, nkv, seq, batch)

    q = reshape(apply_rope(reshape(q, hd, seq, nq*batch),  a.cos_c, a.sin_c, a.rd), hd, nq,  seq, batch)
    k = reshape(apply_rope(reshape(k, hd, seq, nkv*batch), a.cos_c, a.sin_c, a.rd), hd, nkv, seq, batch)

    if a.mode == "flash"
        q4 = permutedims(q, (1,3,2,4))
        k4 = permutedims(k, (1,3,2,4))
        v4 = permutedims(v, (1,3,2,4))
        o4 = NNkernels.flash_attention(q4, k4, v4; causal=true)
        o  = permutedims(o4, (1,3,2,4))
        out = reshape(o, hd*nq, seq, batch)
    else
        g   = nq ÷ nkv
        k_e = repeat(k; outer=(1,g,1,1))
        v_e = repeat(v; outer=(1,g,1,1))
        Q   = reshape(permutedims(q,   (1,3,2,4)), hd, seq, nq*batch)
        K   = reshape(permutedims(k_e, (1,3,2,4)), hd, seq, nq*batch)
        sc  = NNlib.batched_mul(permutedims(Q,(2,1,3)), K) .* Float32(1/sqrt(hd))
        a.cap > 0f0 && (sc = a.cap .* tanh.(sc ./ a.cap))
        cmask = Zygote.ignore_derivatives() do
            m = triu(fill(Float32(-Inf), seq, seq), 1)
            HAS_CUDA && sc isa CuArray ? cu(m) : m
        end
        sc   = sc .+ reshape(cmask, seq, seq, 1)
        attn = softmax(sc; dims=2)
        V    = reshape(permutedims(v_e, (1,3,2,4)), hd, seq, nq*batch)
        out3 = NNlib.batched_mul(V, permutedims(attn, (2,1,3)))
        out  = reshape(permutedims(reshape(out3, hd, seq, nq, batch), (1,3,2,4)), hd*nq, seq, batch)
    end

    reshape(dense_qat_forward(a.Wo.weight, reshape(out, dim, :)), dim, seq, batch)
end

struct FFN{U,D}; up::U; down::D; end
Flux.@layer FFN
FFN(dim::Int, mult::Int) = FFN(Dense(dim => dim*mult; bias=false), Dense(dim*mult => dim; bias=false))

function (f::FFN)(x)
    d, s, b = size(x)
    y = dense_qat_forward(f.up.weight,   reshape(x, d, :))
    y = relu2(y)
    y = dense_qat_forward(f.down.weight, y)
    reshape(y, d, s, b)
end

struct TBlock{A,F,N1,N2}; attn::A; ffn::F; n1::N1; n2::N2; end
Flux.@layer TBlock
TBlock(dim::Int, nq::Int, nkv::Int, ff::Int, seq::Int, mode::String) =
    TBlock(GQAttention(dim, nq, nkv, seq; mode=mode), FFN(dim, ff), RMSNorm(dim), RMSNorm(dim))

function (b::TBlock)(x)
    if HYBRID_CKPT_ENABLED[] && x isa CuArray
        attn_fn = inp -> b.attn(b.n1(inp))
        ffn_fn  = inp -> b.ffn(b.n2(inp))
        h = hybrid_checkpoint(attn_fn, x)
        x_new = x .+ h
        out = hybrid_checkpoint(ffn_fn, x_new)
        result = x_new .+ out
        return result
    else
        h = x .+ b.attn(b.n1(x))
        return h .+ b.ffn(b.n2(h))
    end
end

struct GolfGPT{E,B,N}; embed::E; blocks::B; norm::N; end
Functors.@functor GolfGPT (embed, blocks, norm)
Flux.@layer GolfGPT trainable=(embed, blocks, norm)

function GolfGPT(; vocab=VOCAB, layers=5, dim=384, heads=6, kv=3, ff=3, seq=1024, mode="flash")
    blocks = Tuple(TBlock(dim, heads, kv, ff, seq, mode) for _ in 1:layers)
    embed_init = (dims...) -> 0.02f0 .* randn(Float32, dims...)
    embed = Embedding(vocab => dim; init=embed_init)
    GolfGPT(embed, blocks, RMSNorm(dim))
end

function (m::GolfGPT)(tokens)
    x = m.embed(tokens)
    for blk in m.blocks; x = blk(x); end
    x = m.norm(x)
    dim = size(x, 1)
    reshape(m.embed.weight' * reshape(x, dim, :), VOCAB, size(tokens)...)
end

function build_model_cpu(args; seq_override=nothing)
    seq = seq_override === nothing ? args["seq"] : Int(seq_override)
    GolfGPT(layers=args["layers"], dim=args["dim"], heads=args["heads"],
            kv=args["kv-heads"], ff=args["ff-mult"], seq=seq, mode=args["attn"])
end

# ============================================================
# Checkpoint Compatibility
# ============================================================
function compatible_state(dst, src)
    if dst isa NamedTuple && src isa NamedTuple
        names = keys(dst)
        vals  = map(names) do k
            haskey(src, k) ? compatible_state(getfield(dst,k), getfield(src,k)) : getfield(dst,k)
        end
        return NamedTuple{names}(Tuple(vals))
    elseif dst isa Tuple && src isa Tuple && length(dst)==length(src)
        return ntuple(i -> compatible_state(dst[i], src[i]), length(dst))
    elseif dst isa AbstractArray && src isa AbstractArray
        return size(dst)==size(src) ? src : dst
    else
        return src
    end
end

function load_compatible_model!(model, src_state)
    dst_state = Flux.state(model)
    Flux.loadmodel!(model, compatible_state(dst_state, src_state))
end

# ============================================================
# Optimizers (NorMuon)
# ============================================================
function zeropower_ns5(G; steps::Int=5)
    nd = ndims(G)
    nd < 2 && return G
    sz = size(G); m0, n0 = sz[nd-1], sz[nd]
    B  = nd > 2 ? prod(sz[1:nd-2]) : 1
    G_norm = sqrt.(sum(Float32.(G).^2; dims=(nd-1, nd)) .+ 1f-7)
    G_normalized = Float32.(G) ./ Float32.(G_norm)
    X  = reshape(Float16.(G_normalized), m0, n0, B)
    tr = m0 > n0
    if tr; X = permutedims(X, (2,1,3)); m0, n0 = n0, m0; end
    fn = sqrt.(sum(Float32.(X).^2; dims=(1,2)) .+ 1f-7)
    X .= X ./ Float16.(fn)
    a, b, c = Float16(3.4445), Float16(-4.7750), Float16(2.0315)
    for _ in 1:steps
        Xt = permutedims(X, (2,1,3))
        A  = NNlib.batched_mul(X, Xt)
        A2 = NNlib.batched_mul(A, A)
        X  = a .* X .+ NNlib.batched_mul(b .* A .+ c .* A2, X)
    end
    tr && (X = permutedims(X, (2,1,3)))
    result = Float32.(reshape(X, sz)) .* Float32.(G_norm)
    return result
end

struct NorMuon <: Optimisers.AbstractRule
    eta::Float32; beta1::Float32; beta2::Float32; ns_steps::Int; wd::Float32
end
NorMuon(; lr=6f-4, beta1=0.95f0, beta2=0.95f0, ns_steps=5, wd=0f0) =
    NorMuon(Float32(lr), Float32(beta1), Float32(beta2), Int(ns_steps), Float32(wd))

struct NMState{M,V}; m::M; v::V; end

function Optimisers.init(o::NorMuon, x::AbstractArray)
    nd  = ndims(x)
    vsz = nd >= 2 ? ntuple(i -> i==nd ? 1 : size(x,i), nd) : size(x)
    v   = zeros(eltype(x), vsz)
    HAS_CUDA && x isa CuArray && (v = cu(v))
    NMState(zero(x), v)
end

function Optimisers.apply!(o::NorMuon, st::NMState, x::AbstractArray, dx)
    dx === nothing && return st, zero(x)
    T  = eltype(x); dx = T.(dx); nd = ndims(x); η = T(o.eta)
    if nd < 2
        m_new = T(o.beta1) .* st.m .+ T(1 - o.beta1) .* dx
        upd   = m_new
        o.wd > 0f0 && (upd = upd .+ T(o.wd) .* x)
        return NMState(m_new, st.v), η .* upd
    end
    β1, β2, ε = T(o.beta1), T(o.beta2), T(1e-4)
    m_new = β1 .* st.m .+ (T(1) - β1) .* dx
    upd   = β1 .* dx  .+ (T(1) - β1) .* m_new
    osz = nd == 4 ? size(upd) : nothing
    osz !== nothing && (upd = reshape(upd, size(upd, 1), :))
    
    # === ДОБАВИТЬ: защита от переполнения в FP16 ===
    if T == Float16
        upd = clamp.(upd, Float16(-1000), Float16(1000))
    end
    # ==============================================
    


upd = T.(zeropower_ns5(upd; steps=o.ns_steps))
    nd2 = ndims(upd)
    vn    = sqrt.(sum(upd.^2; dims=(nd2-1, nd2)) .+ ε)
    vm    = mean(upd.^2; dims=nd2)
    v_new = β2 .* st.v .+ (T(1) - β2) .* vm
    upd  = upd ./ sqrt.(v_new .+ ε)
    vn2  = sqrt.(sum(upd.^2; dims=(nd2-1, nd2)) .+ ε)
    upd  = upd .* (vn ./ (vn2 .+ ε))
    upd  = upd .* T(sqrt(max(1f0, Float32(size(upd, nd2-1)) / Float32(size(upd, nd2)))))
    o.wd > 0f0 && (upd = upd .+ T(o.wd) .* x)
    osz !== nothing && (upd = reshape(upd, osz))
    NMState(m_new, v_new), η .* upd
end


function setup_optimizer(model, args; lr_override=nothing)
    base_lr = lr_override === nothing ? Float32(args["lr"]) : Float32(lr_override)
    wd = Float32(args["wd"])
    # Единый AdamW обходит проблемы совместимости деревьев и стабилен в FP32
    opt_rule = Optimisers.AdamW(base_lr, (0.9f0, 0.999f0), wd)
    return Optimisers.setup(opt_rule, model)
end


# LR schedule
# ============================================================
function lr_cosine(step::Int, base_lr::Float32, min_lr::Float32, total::Int)
    t = clamp(step / max(1,total), 0.0, 1.0)
    Float32(min_lr + 0.5*(base_lr-min_lr)*(1+cospi(t)))
end

function lr_cosine_restarts(step::Int, base_lr::Float32, min_lr::Float32, T0::Int, Tmult::Float64)
    step <= 0 && return base_lr
    t_curr = T0; s = step
    while s > t_curr; s -= t_curr; t_curr = Int(round(t_curr*Tmult)); end
    frac = (s-1) / max(1, t_curr-1)
    Float32(min_lr + 0.5*(base_lr-min_lr)*(1+cospi(frac)))
end

function lr_base(step::Int, args)::Float32
    base   = Float32(args["lr"]); min_lr = Float32(args["lr-min"])
    step <= args["warmup"] && return base * step / max(1, args["warmup"])
    if args["lr-scheduler"] == "cosine"
        return lr_cosine(step-args["warmup"], base, min_lr, args["iters"]-args["warmup"])
    else
        return lr_cosine_restarts(step-args["warmup"], base, min_lr, args["lr-T0"], args["lr-Tmult"])
    end
end

function lr_qat_aware(base_lr::Float32, qat_alpha::Float32)
    qat_alpha > 0.8f0 && return base_lr * 0.3f0
    qat_alpha > 0.5f0 && return base_lr * 0.5f0
    return base_lr
end

function resume_lr_multiplier(step::Int, start_step::Int, args)
    rw    = args["resume-warmup"]; scale = Float32(args["resume-lr-scale"])
    rw <= 0 && return 1f0
    local_step = step - start_step + 1
    local_step <= rw && return scale * Float32(local_step / rw)
    local_step2 = local_step - rw; local_step2 <= rw || return 1f0
    return scale + (1f0-scale) * Float32(local_step2/rw)
end

# ============================================================
# Checkpoints
# ============================================================
ckpt_latest(dir)      = joinpath(dir, "latest.jld2")
ckpt_latest_good(dir) = joinpath(dir, "latest_good.jld2")
ckpt_best(dir)        = joinpath(dir, "best.jld2")
ckpt_step(dir, step)  = joinpath(dir, @sprintf("step_%07d.jld2", step))
ckpt_step_good(dir, step) = joinpath(dir, @sprintf("step_%07d_good.jld2", step))
ckpt_step_bad(dir, step)  = joinpath(dir, @sprintf("bad_step_%07d.jld2", step))

function save_ckpt(path, model, opt, step::Int, best_loss::Float32, loader::ByteLoader, args; note="")
    mkpath(dirname(path))
    model_cpu = cpu(model)
    opt_cpu = Functors.fmap(opt) do v
        HAS_CUDA && v isa CuArray ? Array(v) : v
    end
    lbuf, lpos = loader_state_for_save(loader)
    JLD2.jldsave(path; 
        model_state = Flux.state(model_cpu),
        opt_state   = opt_cpu,
        step        = step,
        best_loss   = best_loss,
        loader_buf  = Vector{Int32}(lbuf),
        loader_pos  = Int(lpos),
        vocab       = VOCAB,
        note        = note,
        saved_at    = string(now())
    )
    return path
end

function save_latest!(dir, model, opt, step, best, loader, args; note="")
    p = ckpt_latest(dir); save_ckpt(p, model, opt, step, best, loader, args; note=note)
    cp(p, ckpt_step(dir, step); force=true); p
end
function save_good!(dir, model, opt, step, best, loader, args; note="")
    p = ckpt_latest_good(dir); save_ckpt(p, model, opt, step, best, loader, args; note=note)
    cp(p, ckpt_step_good(dir, step); force=true); p
end
save_best!(dir, model, opt, step, best, loader, args; note="") = save_ckpt(ckpt_best(dir), model, opt, step, best, loader, args; note=note)
save_bad!(dir, model, opt, step, best, loader, args; note="") = save_ckpt(ckpt_step_bad(dir, step), model, opt, step, best, loader, args; note=note)

function prune_ckpts!(dir, keep_last::Int)
    files = filter(f -> occursin(r"step_\d+(_good)?\.jld2$", basename(f)), readdir(dir; join=true))
    length(files) <= keep_last && return
    sort!(files)
    for f in files[1:end-keep_last]; rm(f; force=true); end
end

function load_ckpt_state(path)
    ck   = JLD2.load(path)
    st   = ck["model_state"]
    opt_state = haskey(ck, "opt_state") ? ck["opt_state"] : nothing
    lbuf = haskey(ck,"loader_buf") ? Vector{Int32}(ck["loader_buf"]) : Int32[]
    lpos = haskey(ck,"loader_pos") ? Int(ck["loader_pos"]) : 1
    step = Int(ck["step"])
    best = Float32(get(ck,"best_loss",Inf32))
    ckvocab = Int(get(ck, "vocab", 0))
    if ckvocab != 0 && ckvocab != VOCAB
        @warn "Checkpoint vocab=$ckvocab differs from current VOCAB=$VOCAB"
    end
    return st, opt_state, lbuf, lpos, step, best
end

function rollback_to_good!(dir, model_cpu, loader, args)
    path = ckpt_latest_good(dir)
    if !isfile(path)
        @warn "No latest_good checkpoint for rollback: $path"
        return nothing
    end
    st, _, lbuf, lpos, step, best = load_ckpt_state(path)
    load_compatible_model!(model_cpu, st)
    loader_restore!(loader, lbuf, lpos)
    return (step, best)
end

# ============================================================
# Sampling + health
# ============================================================
function sample_topk_topp(logits::Vector{Float32}; topk::Int=40, topp::Float64=0.9, temp::Float64=0.8, rng::AbstractRNG=Random.GLOBAL_RNG)
    V = length(logits); t = max(temp, 1e-5); lv = logits ./ Float32(t)
    if topk > 0 && topk < V
        thr = partialsort(lv, topk; rev=true); mask = lv .< thr; lv = copy(lv); lv[mask] .= -Inf32
    end
    m = maximum(lv); e = exp.(lv .- m); s = sum(e); s <= 0 && (return argmax(logits))
    p = e ./ s
    if topp > 0.0 && topp < 1.0
        ord = sortperm(p; rev=true); cum = 0.0; keep = falses(V)
        for i in ord; keep[i] = true; cum += p[i]; cum >= topp && break; end
        p = ifelse.(keep, p, 0f0); ssum = sum(p); ssum <= 0 && (return argmax(logits)); p ./= ssum
    end
    r = rand(rng); acc = 0.0
    for i in 1:V; acc += p[i]; if r <= acc; return i; end; end
    return V
end

function generate_sample(model, prompt::String; ctx_len::Int, max_new::Int, greedy::Bool=false, topk::Int=40, topp::Float64=0.9, temp::Float64=0.8)
    toks = encode_text_tokens(prompt; add_eos=false)
    for _ in 1:max_new
        ctx    = toks[max(1, length(toks)-ctx_len+1):end]
        x      = DEV(reshape(ctx, :, 1))
        logits = model(x)
        lv     = Array(cpu(vec(logits[:, end, 1])))
        if any(!isfinite, lv); push!(toks, Int32(EOS_TOKEN)); break; end
        nxt = greedy ? argmax(lv) : sample_topk_topp(Float32.(lv); topk=topk, topp=topp, temp=temp)
        push!(toks, Int32(nxt))
        nxt == EOS_TOKEN && break
    end
    toks, decode_tokens(toks)
end

generate_greedy(model, prompt::String; ctx_len::Int, max_new::Int) = generate_sample(model, prompt; ctx_len=ctx_len, max_new=max_new, greedy=true)

function max_repeat_run(ts)
    isempty(ts) && return 0; best = cur = 1
    for i in 2:length(ts); ts[i]==ts[i-1] ? (cur+=1; best=max(best,cur)) : (cur=1); end
    best
end

function hist_entropy_bits(freq, total)
    total <= 0 && return 0.0; h = 0.0
    for c in freq; c>0 && (p=c/total; h -= p*log2(p)); end; h
end

function ngram_diversity(text::String, n::Int)
    chars = collect(text); length(chars) < n && return 0.0
    ngrams = Set{String}()
    for i in 1:(length(chars) - n + 1); push!(ngrams, join(chars[i:i+n-1])); end
    return length(ngrams) / max(1, length(chars) - n + 1)
end

mutable struct SampleHealth
    total::Int; space_ratio::Float64; top_token::Int; top_ratio::Float64
    prefix_ratio::Float64; unique_ratio::Float64; entropy_bits::Float64
    max_run::Int; bigram_div::Float64; trigram_div::Float64
    ok::Bool; reason::String
end

function analyze_tokens_health(ts::Vector{Int32}, text::String, args)
    total = length(ts); freq = zeros(Int, VOCAB)
    for t in ts; 1<=t<=VOCAB && (freq[t]+=1); end
    top_token    = argmax(freq)
    top_ratio    = total>0 ? freq[top_token]/total : 1.0
    space_ratio  = total>0 ? freq[33]/total : 0.0
    cyr2_ratio   = total>0 ? count(t -> CYR2_BASE <= t < EOS_TOKEN, ts) / total : 0.0
    unique_ratio = total>0 ? count(>(0),freq)/total : 0.0
    entropy_bits = hist_entropy_bits(freq, total)
    mr           = max_repeat_run(ts)
    bigram_div   = ngram_diversity(text, 2)
    trigram_div  = ngram_diversity(text, 3)
    reasons = String[]
    space_ratio  < args["min-space-ratio"]      && push!(reasons, @sprintf("space %.2f%% < %.2f%%",100space_ratio,100args["min-space-ratio"]))
    top_ratio    > args["max-top-token-ratio"]   && push!(reasons, @sprintf("top token %d %.2f%% > %.2f%%",top_token,100top_ratio,100args["max-top-token-ratio"]))
    cyr2_ratio   > args["max-prefix-ratio"]      && push!(reasons, @sprintf("cyr2 ratio %.2f%% > %.2f%%",100cyr2_ratio,100args["max-prefix-ratio"]))
    mr           > args["max-repeat-run"]        && push!(reasons, "max repeat run $mr > $(args["max-repeat-run"])")
    unique_ratio < args["min-unique-ratio"]      && push!(reasons, @sprintf("unique ratio %.4f < %.4f",unique_ratio,args["min-unique-ratio"]))
    entropy_bits < args["min-sample-entropy"]    && push!(reasons, @sprintf("hist entropy %.2f < %.2f",entropy_bits,args["min-sample-entropy"]))
    bigram_div   < args["min-bigram-diversity"]  && push!(reasons, @sprintf("bigram div %.3f < %.3f",bigram_div,args["min-bigram-diversity"]))
    trigram_div  < args["min-trigram-diversity"] && push!(reasons, @sprintf("trigram div %.3f < %.3f",trigram_div,args["min-trigram-diversity"]))
    ok = isempty(reasons)
    SampleHealth(total, space_ratio, top_token, top_ratio, cyr2_ratio, unique_ratio, entropy_bits, mr, bigram_div, trigram_div, ok, ok ? "ok" : join(reasons,"; "))
end

function print_samples_and_health(model, args, step)
    prompts = ["Въ началѣ ", "На другой день ", "Старикъ сказалъ ", "Она отвѣчала, что ", "Въ Петербургѣ ", "Господинъ ", "Исторія эта "]
    println(); println("══════════ samples @ step $step ══════════")
    all_tokens = Int32[]; texts = String[]; greedy = args["sample-greedy"]
    for (i,p) in enumerate(prompts)
        toks, txt = generate_sample(model, p; ctx_len=args["seq"], max_new=args["sample-tokens"], greedy=greedy, topk=args["sample-topk"], topp=args["sample-topp"], temp=args["sample-temp"])
        append!(all_tokens, toks); push!(texts, replace(txt, '\n'=>' ')); println("[$i] ", texts[end])
    end
    all_text = join(texts," "); health = analyze_tokens_health(all_tokens, all_text, args)
    freq = zeros(Int, VOCAB); for t in all_tokens; 1<=t<=VOCAB && (freq[t]+=1); end
    total = length(all_tokens); space_count = freq[33]
    println()
    @printf("📊 Space token (33): %d / %d = %.2f%%\n", space_count, total, total>0 ? 100*space_count/total : 0.0)
    @printf("📊 Health: top=%d %.2f%% │ cyr2=%.2f%% │ unique=%.4f │ H=%.2f bits │ max_run=%d\n",
            health.top_token, 100health.top_ratio, 100health.prefix_ratio, health.unique_ratio, health.entropy_bits, health.max_run)
    @printf("📊 Char diversity: bigram=%.3f trigram=%.3f  (sampler: %s)\n",
            health.bigram_div, health.trigram_div, greedy ? "greedy" : "topk=$(args["sample-topk"]) topp=$(args["sample-topp"]) T=$(args["sample-temp"])")
    !health.ok && println("⚠️  Health BAD: ", health.reason)
    println("═══════════════════════════════════════════"); println()
    health
end

# ============================================================
# Health gates
# ============================================================
function loss_is_suspicious(avg_loss::Float32, loss_ema::Float32, best_loss::Float32, args)
    !isfinite(avg_loss) && return true, "loss is NaN/Inf"
    if QAT_STEP[] <= 100; return false, "grace period"; end
    avg_loss < Float32(args["min-healthy-loss"]) && return true, @sprintf("loss %.6f < min %.6f", avg_loss, args["min-healthy-loss"])
    avg_loss > Float32(args["max-healthy-loss"]) && return true, @sprintf("loss %.6f > max %.6f", avg_loss, args["max-healthy-loss"])
    if loss_ema > 0f0 && avg_loss > Float32(args["loss-spike-factor"]) * loss_ema
        return true, @sprintf("loss spike %.4f > %.2fx ema %.4f", avg_loss, args["loss-spike-factor"], loss_ema)
    end
    if isfinite(best_loss) && best_loss < 100f0 && avg_loss > Float32(args["loss-spike-factor"]) * best_loss
        return true, @sprintf("loss drift %.4f >  %.2fx best %.4f", avg_loss, args["loss-spike-factor"], best_loss)
    end
    return false, "ok"
end

function model_params_ok(model, args)
    st = Flux.state(model); !tree_all_finite(st) && return false, "params contain NaN/Inf"
    mx = tree_max_abs(st)
    mx > Float32(args["max-param-abs"]) && return false, @sprintf("max |param| %.3f > %.3f", mx, args["max-param-abs"])
    return true, @sprintf("max |param| %.3f", mx)
end

# ============================================================
# AUTOTUNE
# ============================================================
round_to_multiple(n::Integer, m::Integer) = max(m, ((n + m÷2) ÷ m) * m)

function suggest_loader_target_tokens(cpu_cache::CPUCacheInfo, args)
    l3 = cpu_cache.ok && cpu_cache.l3>0 ? cpu_cache.l3 : 8*1024*1024
    frac = args["loader-cache-frac"]; minb = args["loader-min-mb"] * 1024^2; maxb = args["loader-max-mb"] * 1024^2
    target_bytes  = Int(clamp(l3*frac, minb, maxb))
    target_tokens = max(10_000, target_bytes÷4)
    return target_tokens, target_bytes
end

function microprobe_tokps(model_dev, seq::Int, batch::Int; iters::Int=3)
    x = DEV(rand(Int32(1):Int32(VOCAB), seq, batch)); y = DEV(rand(Int32(1):Int32(VOCAB), seq, batch))
    old_qat = QAT_ON[]; QAT_ON[] = false
    l, grads = Zygote.withgradient(model_dev) do m; logits = m(x); nll_loss(reshape(logits,VOCAB,:), reshape(y,:)); end
    _ = grads[1]; HAS_CUDA && CUDA.synchronize(); t0 = time_ns()
    for _ in 1:iters
        l, grads = Zygote.withgradient(model_dev) do m; logits = m(x); nll_loss(reshape(logits,VOCAB,:), reshape(y,:)); end
        _ = grads[1]; HAS_CUDA && CUDA.synchronize()
    end
    t1 = time_ns(); QAT_ON[] = old_qat; seq*batch*iters / max((t1-t0)/1e9, 1e-9)
end

function microprobe_tokps_avg(model_dev, seq::Int, batch::Int; runs::Int=3, iters::Int=3)
    vals = Float64[]
    for _ in 1:runs; push!(vals, microprobe_tokps(model_dev, seq, batch; iters=iters)); GC.gc(false); HAS_CUDA && CUDA.reclaim(); end
    median(vals)
end

function autotune!(args, gpu::GPUInfo, cpu_cache::CPUCacheInfo)
    args["autotune-applied"] = false; !args["autotune"] && return
    section("AUTOTUNE (cache-aware)")
    target_tokens, target_bytes = suggest_loader_target_tokens(cpu_cache, args)
    println("ByteLoader target buffer: ", fmt_bytes(target_bytes), " (~", target_tokens, " Int32 tokens)")
    args["loader-target-tokens"] = target_tokens
    if !gpu.ok; println("No GPU detected: skipping GPU autotune."); args["autotune-applied"] = true; return; end

    G0 = args["seq"]*args["batch"]*args["accum"]
    println("Baseline tokens/update: G0 = $(args["seq"])*$(args["batch"])*$(args["accum"]) = $G0")
    max_seq = args["autotune-max-seq"]; max_batch = args["autotune-max-batch"]; seq0 = args["seq"]
    seq_candidates = unique([round_to_multiple(seq0, 128), round_to_multiple(Int(floor(seq0*1.2)), 128), round_to_multiple(Int(floor(seq0*1.6)), 128), round_to_multiple(Int(floor(seq0*2.0)), 128)])
    seq_candidates = [s for s in seq_candidates if 128<=s<=max_seq]
    isempty(seq_candidates) && (seq_candidates = [round_to_multiple(seq0,128)])
    base_batches = unique([args["batch"],1,2,4,8,16,32])
    batch_candidates = [b for b in base_batches if 1<=b<=max_batch]; sort!(batch_candidates)
    maxcand = args["autotune-candidates"]; pairs = Tuple{Int,Int,Int}[]
    for s in seq_candidates, b in batch_candidates
        a = max(1, Int(round(G0/(s*b)))); maxG = Int(floor(G0*args["autotune-max-global-scale"]))
        while s*b*a > maxG && a>1; a-=1; end; push!(pairs, (s,b,a))
    end
    score(p) = begin s,b,a=p; G=s*b*a; abs(G-G0)/max(G0,1)+0.15/b end
    sort!(pairs, by=score); pairs = pairs[1:min(length(pairs), maxcand)]
    println("Candidates to try:")
    for (s,b,a) in pairs; G = s*b*a; @printf("  • seq=%4d batch=%2d accum=%2d => tokens/update=%d (%.2fx)\n", s,b,a,G,G/max(G0,1)); end

    if !args["autotune-benchmark"]; s,b,a = pairs[1]; args["seq"],args["batch"],args["accum"] = s,b,a; args["autotune-applied"] = true; return; end

    reserve = Int(round(args["autotune-vram-reserve-gb"]*1024^3))
    vram_budget = max(0, gpu.free_mem-reserve)
    target_budget = Int(floor(vram_budget * args["autotune-vram-target-frac"]))
    println("VRAM budget for probe: free=", fmt_bytes(gpu.free_mem), " reserve=", fmt_bytes(reserve), " => budget≈", fmt_bytes(vram_budget))

    best_pair = nothing; best_tps = 0.0
    for (s,b,a) in pairs
        println(); @printf("Probe: seq=%d batch=%d accum=%d ...\n", s,b,a)
        model_cpu = build_model_cpu(args; seq_override=s)
        if args["fp16"] && HAS_CUDA; model_cpu = Functors.fmap(model_cpu) do v; v isa AbstractArray ? Float16.(v) : v; end; end
        model = model_cpu |> DEV
        tps = 0.0; used = 0; probe_ok = false
        try
            GC.gc(false); HAS_CUDA && CUDA.reclaim(); before = HAS_CUDA ? CUDA.available_memory() : 0
            tps = microprobe_tokps_avg(model, s, b; runs=args["autotune-bench-runs"], iters=args["autotune-bench-iters"])
            GC.gc(false); HAS_CUDA && CUDA.reclaim(); after = HAS_CUDA ? CUDA.available_memory() : 0
            used = max(0, before - after); probe_ok = true
        catch e
            println("  ✗ probe error: ", sprint(showerror, e))
        finally
            model = nothing; model_cpu = nothing; GC.gc(false); HAS_CUDA && CUDA.reclaim()
        end
        if !probe_ok; continue; end
        @printf("  tok/s ≈ %.0f │ mem_delta≈ %s\n", tps, fmt_bytes(used))
        if used > target_budget && target_budget > 0; println("  ✗ exceeds VRAM budget, skipping."); continue; end
        if tps > best_tps; best_tps = tps; best_pair = (s, b, a); end
    end
    if best_pair === nothing; println("No candidate succeeded; keeping original config."); return; end
    s,b,a = best_pair; println(); println("✅ AUTOTUNE selected:")
    @printf("  seq=%d batch=%d accum=%d  tok/s≈%.0f\n", s,b,a,best_tps)
    args["seq"],args["batch"],args["accum"] = s,b,a; args["autotune-applied"] = true
end

# ============================================================
# Smoke test & Unit tests
# ============================================================
function run_unit_tests(model, args)
    println("🔬 Unit tests..."); expected = log(Float32(VOCAB))
    test_ok = try
        seq_t = 4; batch_t = 2; x_t = DEV(ones(Int32, seq_t, batch_t)); logits_t = model(x_t)
        flat_logits = reshape(logits_t, VOCAB, :); zero_logits = zero(flat_logits)
        targets_t = DEV(ones(Int32, seq_t * batch_t)); loss_t = scalar_value(nll_loss(zero_logits, targets_t))
        abs_err = abs(loss_t - expected)
        if abs_err < 0.01; @printf("  ✅ logits=0 → loss=%.4f ≈ log(%d)=%.4f (err=%.5f)\n", loss_t, VOCAB, expected, abs_err); true
        else; @printf("  ❌ logits=0 → loss=%.4f expected≈%.4f (err=%.5f)\n", loss_t, expected, abs_err); false; end
    catch e; println("  ❌ unit test 1 failed: ", sprint(showerror, e)); false; end

    test2_ok = try
        seq_t = 8; batch_t = 2; x_t = DEV(rand(Int32(1):Int32(VOCAB), seq_t, batch_t)); y_t = DEV(rand(Int32(1):Int32(VOCAB), seq_t, batch_t))
        logits_t = model(x_t); loss_t = scalar_value(nll_loss(reshape(logits_t,VOCAB,:), reshape(y_t,:)))
        reasonable = isfinite(loss_t) && loss_t < 100.0f0
        if reasonable; @printf("  ✅ model init loss=%.4f (expected %.2f±2)\n", loss_t, expected); true
        else; @printf("  ❌ model init loss=%.4f outside reasonable range\n", loss_t); false; end
    catch e; println("  ❌ unit test 2 failed: ", sprint(showerror, e)); false; end

    test3_ok = try
        sample = "Въ началѣ было Слово. Привет 123!"; toks = encode_text_tokens(sample; add_eos=false); decoded = decode_tokens(toks)
        if decoded == sample; @printf("  ✅ tokenizer roundtrip OK (\"%s\" → %d tokens)\n", sample, length(toks)); true
        else; @printf("  ❌ tokenizer roundtrip FAILED:\n     in:  %s\n     out: %s\n", sample, decoded); false; end
    catch e; println("  ❌ unit test 3 (tokenizer) failed: ", sprint(showerror, e)); false; end
    return test_ok && test2_ok && test3_ok
end

function smoke_test!(model, args)
    println("🔥 Smoke test + unit tests...")
    
    # === Тест hybrid_checkpoint (только если не задан --skip-hybrid-checkpoint-test) ===
    if HAS_CUDA && HYBRID_CKPT_ENABLED[] && !get(args, "skip-hybrid-checkpoint-test", false)
        println("  🧪 Testing hybrid_checkpoint (PCIe offload)...")
        try
            seq_t = min(32, args["seq"]); batch_t = 2
            x_t = DEV(rand(Int32(1):Int32(VOCAB), seq_t, batch_t))
            y_t = DEV(rand(Int32(1):Int32(VOCAB), seq_t, batch_t))
            l, grads = Zygote.withgradient(model) do m
                logits = m(x_t)
                nll_loss(reshape(logits, VOCAB, :), reshape(y_t, :))
            end
            lf = scalar_value(l)
            if isfinite(lf) && tree_all_finite(grads[1])
                vram_before = CUDA.available_memory()
                for _ in 1:2
                    _, _ = Zygote.withgradient(model) do m
                        logits = m(x_t)
                        nll_loss(reshape(logits, VOCAB, :), reshape(y_t, :))
                    end
                end
                HAS_CUDA && CUDA.synchronize()
                vram_after = CUDA.available_memory()
                delta_mb = (vram_after - vram_before) / 1024^2
                @printf("  ✅ hybrid_checkpoint OK │ loss=%.4f │ VRAM Δ=%+.1f MB\n", lf, delta_mb)
                HYBRID_CKPT_TESTED[] = true
            else
                @warn "  ⚠️ hybrid_checkpoint gave NaN/Inf (loss=$lf) — falling back."
                HYBRID_CKPT_ENABLED[] = false
            end
        catch e
            @warn "  ⚠️ hybrid_checkpoint failed: $(sprint(showerror, e)) — falling back."
            HYBRID_CKPT_ENABLED[] = false
        finally
            GC.gc(false); HAS_CUDA && CUDA.reclaim()
        end
    elseif get(args, "skip-hybrid-checkpoint-test", false)
        println("  ⏭️ Skipping hybrid_checkpoint test (--skip-hybrid-checkpoint-test)")
    end

    # === Остальные юнит-тесты (без изменений) ===
    ut_ok = run_unit_tests(model, args)
    !ut_ok && error("Unit tests FAILED — fix before training.")
    
    # ... (остальной код функции)

    seq_s = min(32, args["seq"]); batch_s = 1
    ok = try
        x_s = DEV(rand(Int32(1):Int32(VOCAB), seq_s, batch_s)); y_s = DEV(rand(Int32(1):Int32(VOCAB), seq_s, batch_s))
        opt_s = setup_optimizer(model, args)
        l_s, grads = Zygote.withgradient(model) do m; logits = m(x_s); nll_loss(reshape(logits,VOCAB,:), reshape(y_s,:)); end
        gs_s = grads[1]; _, model = Optimisers.update(opt_s, model, gs_s); HAS_CUDA && CUDA.synchronize(); isfinite(scalar_value(l_s))
    catch e; println("❌ Smoke test FAILED:\n", sprint(showerror, e)); false; end
    ok || error("Smoke test fail."); println("✅ Smoke test passed"); GC.gc(false); HAS_CUDA && CUDA.reclaim(); model
end

# ============================================================
# Quick check
# ============================================================
function run_quick_check(model, args, loader)
    println(); section("QUICK CHECK (5 steps)"); println("Verifying: loss decreases, gnorm finite, no NaN...")
    opt = setup_optimizer(model, args); losses = Float32[]
    for step in 1:5
        chunk = DEV(next_batch!(loader)); x_in  = chunk[1:end-1, :]; y_tgt = chunk[2:end,   :]
        l, grads = Zygote.withgradient(model) do m; logits = m(x_in); nll_loss(reshape(logits,VOCAB,:), reshape(y_tgt,:)); end
        lf = scalar_value(l); push!(losses, lf); gs = grads[1]; gnorm = grad_l2norm(gs)
        @printf("  step %d/5 │ loss=%.4f │ gnorm=%.3f\n", step, lf, gnorm)
        if !isfinite(lf) || !isfinite(gnorm); println("❌ QUICK CHECK FAILED: NaN/Inf at step $step"); return false; end
        opt, model = Optimisers.update(opt, model, gs)
    end
    if losses[end] < losses[1]; @printf("✅ QUICK CHECK PASSED: loss %.4f → %.4f (↓)\n", losses[1], losses[end]); return true
    elseif losses[end] < losses[1] * 1.05; @printf("⚠️  QUICK CHECK OK (flat): loss %.4f → %.4f\n", losses[1], losses[end]); return true
    else; @printf("❌ QUICK CHECK FAILED: loss did not decrease: %.4f → %.4f\n", losses[1], losses[end]); return false; end
end

# ============================================================
# Adaptive grad clip
# ============================================================
function adaptive_clip_grads(gs, args)
    gnorm = grad_l2norm(gs); push_gnorm!(gnorm)
    clip = if length(RECENT_GNORMS) < 50; Float32(args["grad-clip"]); else; Float32(quantile(RECENT_GNORMS, 0.95) * 1.5); end
    if gnorm > clip; scale = clip / (gnorm + 1f-8); return scale_grads(gs, scale), clip, gnorm
    else; return gs, gnorm, gnorm; end
end

# ============================================================
# Training loop
# ============================================================
function train!(model_dev, model_cpu_ref, args, loader, start_step::Int, best::Float32; opt_loaded=nothing, model_master=nothing)
    if opt_loaded !== nothing
        opt = opt_loaded
        @info "✅ Loaded optimizer state from checkpoint"
    else
        opt = setup_optimizer(model_dev, args)
    end

    ema_beta = Float32(args["ema-beta"]); model_ema = deepcopy(model_dev)
    resumed = !isempty(args["resume"]); loss_ema = Inf32; lr_backoff = 1f0
    bad_step_count = 0; bad_sample_count = 0; log_t0 = time(); run_loss = 0f0; run_n = 0

    if resumed && !isfile(ckpt_latest_good(args["ckpt-dir"]))
        try; save_good!(args["ckpt-dir"], model_dev, opt, start_step-1, best, loader, args; note="initial_good"); catch e; @warn "Failed to save initial latest_good"; end
    end

    for step in start_step:args["iters"]
        update_qat_control!(args, step, start_step)
        base_lr = lr_base(step, args); base_lr = lr_qat_aware(base_lr, QAT_ALPHA[])
        resume_mult = resumed ? resume_lr_multiplier(step, start_step, args) : 1f0
        effective_lr = max(base_lr * resume_mult * lr_backoff, Float32(args["lr"]) * Float32(args["min-lr-scale"]))
        Optimisers.adjust!(opt, eta=effective_lr)

        gs_sum = nothing; step_loss = 0f0; local_bad = false; local_reason = ""
        for _ in 1:args["accum"]
            chunk = DEV(next_batch!(loader)); x_in = chunk[1:end-1, :]; y_tgt = chunk[2:end, :]
            l, grads = Zygote.withgradient(model_dev) do m; logits = m(x_in); nll_loss(reshape(logits,VOCAB,:), reshape(y_tgt,:)); end
            lf = scalar_value(l)
            if !isfinite(lf); local_bad=true; local_reason="micro loss NaN/Inf"; break; end
            g = grads[1]
            if !tree_all_finite(g); local_bad=true; local_reason="gradient NaN/Inf"; break; end
            step_loss += lf; gs_sum = gs_sum === nothing ? g : add_grads(gs_sum, g)
        end

        if local_bad
            bad_step_count += 1; lr_backoff = max(Float32(args["min-lr-scale"]), lr_backoff * Float32(args["lr-backoff-factor"]))
            @warn "Bad micro-step; skipping update" step reason=local_reason bad_step_count lr_backoff
            save_bad!(args["ckpt-dir"], model_dev, opt, step, best, loader, args; note=local_reason)
            if args["rollback-on-bad-step"] || bad_step_count >= args["bad-step-patience"]
                rb = rollback_to_good!(args["ckpt-dir"], model_cpu_ref, loader, args)
                if rb !== nothing
                    rb_step, rb_best = rb; best = rb_best; model_dev = model_cpu_ref |> DEV
                    opt = setup_optimizer(model_dev, args; lr_override=effective_lr); model_ema = deepcopy(model_dev)
                end
            end
            args["stop-on-collapse"] && bad_step_count>=args["bad-step-patience"] && return model_dev, model_ema, best
            continue
        end

        gs = div_grads(gs_sum, args["accum"]); avg = Float32(step_loss / args["accum"])
        suspicious, reason = loss_is_suspicious(avg, loss_ema, best, args)
        if suspicious
            bad_step_count += 1; lr_backoff = max(Float32(args["min-lr-scale"]), lr_backoff * Float32(args["lr-backoff-factor"]))
            @warn "Suspicious loss; skipping update" step avg loss_ema reason bad_step_count lr_backoff
            save_bad!(args["ckpt-dir"], model_dev, opt, step, best, loader, args; note=reason)
            if args["rollback-on-bad-step"] || bad_step_count >= args["bad-step-patience"]
                rb = rollback_to_good!(args["ckpt-dir"], model_cpu_ref, loader, args)
                if rb !== nothing
                    rb_step, rb_best = rb; best = rb_best; model_dev = model_cpu_ref |> DEV
                    opt = setup_optimizer(model_dev, args; lr_override=effective_lr); model_ema = deepcopy(model_dev)
                end
            end
            args["stop-on-collapse"] && bad_step_count>=args["bad-step-patience"] && return model_dev, model_ema, best
            continue
        end

        gs, gnorm_post, gnorm_pre = adaptive_clip_grads(gs, args)
        if step % 10 == 0
            @printf("  [DEBUG] gnorm_pre=%.3f gnorm_post=%.3f clip_applied=%s\n", gnorm_pre, gnorm_post, gnorm_pre != gnorm_post ? "YES" : "NO")
        end
        
        if !isfinite(gnorm_pre)
            bad_step_count += 1
            @warn "Gradient norm NaN/Inf; skipping update" step gnorm_pre
            save_bad!(args["ckpt-dir"], model_dev, opt, step, best, loader, args; note="gnorm_nan_inf")
            continue
        end

        # === Обновление весов ===
        opt, model_dev = Optimisers.update(opt, model_dev, gs)
        ema_beta < 1f0 && ema_update!(model_ema, model_dev, ema_beta)

        if step % args["param-check-every"] == 0
            ok_params, reason = model_params_ok(model_dev, args)
            if !ok_params
                bad_step_count += 1; lr_backoff = max(Float32(args["min-lr-scale"]), lr_backoff * Float32(args["lr-backoff-factor"]))
                @warn "Bad params after update" step reason bad_step_count lr_backoff
                save_bad!(args["ckpt-dir"], model_dev, opt, step, best, loader, args; note=reason)
                rb = rollback_to_good!(args["ckpt-dir"], model_cpu_ref, loader, args)
                if rb !== nothing
                    rb_step, rb_best = rb; best = rb_best; model_dev = model_cpu_ref |> DEV
                    opt = setup_optimizer(model_dev, args; lr_override=effective_lr); model_ema = deepcopy(model_dev)
                end
                args["stop-on-collapse"] && return model_dev, model_ema, best
                continue
            end
        end

        bad_step_count = 0
        if !isfinite(loss_ema); loss_ema = avg; else; β = Float32(args["loss-ema-beta"]); loss_ema = β*loss_ema + (1f0-β)*avg; end
        run_loss += avg; run_n += 1

        if step % args["log-every"] == 0
            nowt = time(); dt = nowt - log_t0; toks = args["seq"]*args["batch"]*args["accum"]*args["log-every"]
            tps = toks / max(dt, 1e-6); rl = run_loss / max(run_n, 1); bpb = rl / log(2)
            qat_s = args["qat"] ? @sprintf(" │ qat α=%.2f bits=%d", QAT_ALPHA[], QAT_BITS_CUR[]) : ""
            if HAS_CUDA
                @printf("step %5d/%d │ loss %.4f │ ema %.4f │ bpb %.3f │ lr %.2e │ rb %.2f │ gnorm %.2f→%.2f │ %6.0f tok/s │ VRAM %.2f GB free%s\n",
                    step, args["iters"], rl, loss_ema, bpb, effective_lr, lr_backoff, gnorm_pre, gnorm_post, tps, CUDA.available_memory()/1e9, qat_s)
            else
                @printf("step %5d/%d │ loss %.4f │ ema %.4f │ bpb %.3f │ lr %.2e │ rb %.2f │ gnorm %.2f→%.2f │ %6.0f tok/s%s\n",
                    step, args["iters"], rl, loss_ema, bpb, effective_lr, lr_backoff, gnorm_pre, gnorm_post, tps, qat_s)
            end
            run_loss=0f0; run_n=0; log_t0=nowt
            if HYBRID_CKPT_ENABLED[]; print_offload_stats(step); end
        end

        if step % args["sample-every-steps"] == 0
            health = print_samples_and_health(model_dev, args, step)
            if !health.ok
                bad_sample_count += 1; lr_backoff = max(Float32(args["min-lr-scale"]), lr_backoff * Float32(args["lr-backoff-factor"]))
                @warn "Bad sample health" step reason=health.reason bad_sample_count lr_backoff
                save_bad!(args["ckpt-dir"], model_dev, opt, step, best, loader, args; note="bad_sample: "*health.reason)
                if bad_sample_count >= args["bad-sample-patience"]
                    rb = rollback_to_good!(args["ckpt-dir"], model_cpu_ref, loader, args)
                    if rb !== nothing
                        rb_step, rb_best = rb; best = rb_best; model_dev = model_cpu_ref |> DEV
                        opt = setup_optimizer(model_dev, args; lr_override=effective_lr); model_ema = deepcopy(model_dev)
                    end
                    if args["abort-on-bad-sample"]; println("⛔ Aborting: sample health is bad."); return model_dev, model_ema, best; end
                end
            else
                bad_sample_count = 0; lr_backoff = min(1f0, lr_backoff / Float32(args["lr-backoff-factor"]))
            end
            HAS_CUDA && (GC.gc(false); CUDA.reclaim())
        end

        if step % args["ckpt-every-steps"] == 0
            latest = save_latest!(args["ckpt-dir"], model_dev, opt, step, best, loader, args; note="periodic")
            ok_params, reason = model_params_ok(model_dev, args)
            if ok_params && avg >= Float32(args["min-healthy-loss"]) && isfinite(avg)
                save_good!(args["ckpt-dir"], model_dev, opt, step, best, loader, args; note="good_periodic")
                if avg < best; best = avg; save_best!(args["ckpt-dir"], model_dev, opt, step, best, loader, args; note="best_training_loss"); end
                @printf("  💾 ckpt saved: %s │ good=yes │ best=%.4f\n", latest, best)
            else
                save_bad!(args["ckpt-dir"], model_dev, opt, step, best, loader, args; note="checkpoint_not_good: "*reason)
                @printf("  ⚠️ ckpt saved only as latest/bad │ best=%.4f │ %s\n", best, reason)
            end
            prune_ckpts!(args["ckpt-dir"], args["keep-last"]); HAS_CUDA && (GC.gc(false); CUDA.reclaim())
        end
    end
    return model_dev, model_ema, best
end

# ============================================================
# Internal preflight
# ============================================================
function internal_preflight!()
    required = [:nll_loss, :grad_l2norm, :scale_grads, :add_grads, :div_grads, :adaptive_clip_grads, :update_qat_control!, :smoke_test!, :train!,
        :setup_optimizer, :lr_base, :lr_qat_aware, :loss_is_suspicious, :model_params_ok, :encode_text_tokens, :decode_tokens, :is_cyr2_pair, :cyr2_token,
        :token_to_cyr2, :probe_tokenizer, :sample_topk_topp, :generate_sample]
    missing = Symbol[]
    for sym in required; isdefined(@__MODULE__, sym) || push!(missing, sym); end
    if !isempty(missing); error("Internal preflight failed: missing definitions: " * join(string.(missing), ", ")); end
    println("✅ Internal preflight: all required functions are defined"); return true
end

# ============================================================
# Main
# ============================================================
function main()
    args = parse_cmd()
    Z_LOSS_COEF[] = Float32(args["z-loss"]); Z_LOSS_COEF[] > 0 && println("z-loss active with coef=$(Z_LOSS_COEF[])")
    internal_preflight!()
    if args["probe-tokenizer"]; section("Tokenizer probe (vocab=$VOCAB)"); probe_tokenizer(args["data"]); return; end
    if haskey(args, "quick-check") && args["quick-check"]; args["autotune"] = false; args["autotune-benchmark"] = false; end
    Random.seed!(args["seed"])
    gpu = detect_gpu(); cpu_cache = detect_cpu_caches()
    
#if args["no-fp16"]; args["fp16"] = false; end

    if gpu.ok
        cc = CUDA.capability(CUDA.device())
        if cc.major == 6 && cc.minor == 1
            @info "🔧 Detected Pascal (SM 6.1). Forcing FP32 and Standard Attention."
            args["fp16"] = false; args["no-fp16"] = true; args["attn"] = "standard"
        elseif cc.major == 7 && cc.minor == 5
            @info "🔧 Detected Turing (SM 7.5). Forcing FP16 and FlashAttention."
      #      args["fp16"] = true; args["no-fp16"] = false; args["attn"] = "flash"
            ENV["CUDA_DISABLE_TENSOR_CORES"] = "0"
        end
    end

    hd = div(args["dim"], args["heads"])
    if args["attn"] == "flash"; ispow2(hd) || error("flash requires head_dim=dim/heads to be power-of-two. head_dim=$hd"); end
    args["heads"] % args["kv-heads"] == 0 || error("--heads must be divisible by --kv-heads")
    print_hw(gpu, cpu_cache); args["print-hw"] && return
    section("Tokenizer (vocab=$VOCAB)"); println("  hybrid: 256 raw bytes + 128 fused Cyrillic UTF-8 pairs (D0/D1 80..BF) + EOS")

    resume_state = nothing; resume_step = 0; best = Inf32; lbuf = Int32[]; lpos = 1; opt_loaded = nothing
    if !isempty(args["resume"])
        println("Resuming from: ", args["resume"])
        st, opt_loaded, lbuf, lpos, resume_step, best = load_ckpt_state(args["resume"])
        resume_state = st
        println("Resumed at step=$resume_step  best_loss=$best")

        # ⚠️ СБРОС СОСТОЯНИЯ ОПТИМИЗАТОРА
        # При переходе на гибридную схему (NorMuon + AdamW) старое дерево состояний
        # структурно несовместимо. Игнорируем загруженный opt_state для безопасного старта.
        if opt_loaded !== nothing
            @info "⚠️ Optimizer state discarded: switching to hybrid rule tree (NorMuon + AdamW)"
            opt_loaded = nothing
        end
    end

    model_cpu = build_model_cpu(args)
    if resume_state !== nothing; println("Loading resume weights compatibly..."); load_compatible_model!(model_cpu, resume_state); end
    
    if args["fp16"] && HAS_CUDA
        model_cpu = Functors.fmap(model_cpu) do v
            v isa AbstractArray ? Float16.(v) : v
        end
    end
    model = model_cpu |> DEV

    # Вычисляем количество параметров для вывода
    vecp, _ = Optimisers.destructure(model_cpu)
    np = length(vecp)

    section("Run configuration (final)")
    @printf("Params: %.2fM (%.2f MB FP16)\n", np/1e6, np*2/1e6)
    @printf("Config: %dL d=%d heads=%d kv=%d head_dim=%d seq=%d batch=%d accum=%d attn=%s vocab=%d\n",
        args["layers"], args["dim"], args["heads"], args["kv-heads"], hd, args["seq"], args["batch"], args["accum"], args["attn"], VOCAB)
    println("FP16: ", args["fp16"]); println("Loss: nll_loss + z-loss=$(Z_LOSS_COEF[])")
    println("Optimizer: NorMuon (FP16/FP32)")

    args["dry-run"] && (println("Dry-run done."); return)
    old_seq = args["seq"]; autotune!(args, gpu, cpu_cache); println("Autotune: applied=", get(args,"autotune-applied",false))
    if get(args,"autotune-applied",false) && args["seq"] != old_seq
        println("Seq changed by autotune, rebuilding model...")
        new_model_cpu = build_model_cpu(args)
        if args["fp16"] && HAS_CUDA; new_model_cpu = Functors.fmap(new_model_cpu) do v; v isa AbstractArray ? Float16.(v) : v; end; end
        load_compatible_model!(new_model_cpu, Flux.state(cpu(model))); model_cpu = new_model_cpu; model = model_cpu |> DEV
        GC.gc(false); HAS_CUDA && CUDA.reclaim()
    end

    tgt = haskey(args,"loader-target-tokens") ? Int(args["loader-target-tokens"]) : 1_000_000
    loader = ByteLoader(args["data"], args["seq"], args["batch"]; target_tokens=tgt)
    !isempty(lbuf) && loader_restore!(loader, lbuf, lpos)

    if args["quick-check"]
        ok = run_quick_check(model, args, loader)
        ok ? println("\n✅ Quick check passed.") : println("\n❌ Quick check FAILED.")
        return
    end

HYBRID_CKPT_ENABLED[] = true
#println("⚡ Hybrid checkpoint disabled for max throughput")

    smoke_test!(model, args)
    start_step = isempty(args["resume"]) ? 1 : (resume_step + 1)
    
    model, model_ema, best = train!(model, model_cpu, args, loader, start_step, best; opt_loaded=opt_loaded)

    state_live = Flux.state(cpu(model)); state_ema  = Flux.state(cpu(model_ema))
    jldsave(args["save"]; live=state_live, ema=state_ema, config=args, best_loss=best, vocab=VOCAB)
    println("Saved final model to: ", args["save"])
end

main()
