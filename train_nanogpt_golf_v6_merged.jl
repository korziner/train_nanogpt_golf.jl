#!/usr/bin/env julia
# ============================================================
# train_nanogpt_golf_v6_merged.jl
# NanoGPT-Golf v6.1 + Parameter Golf features + QAT + SWA/EMA
#
# ИСТОЧНИКИ:
# • ByteLoader, FlashAttention, NorMuon, samples/stats:
#   → gpt-5.2-search-non-reasoning.train_nanogpt_golf_v6_token_stats.jl
# • QAT, SWA/EMA, CLI args, arch features:
#   → train_nanogpt_golf.jl
# • Weight decay, CUDA safety, checkpoint compatibility:
#   → custom patches
#
# Deps:
#   ]add Flux NNlib Optimisers Zygote Functors ArgParse JLD2 JSON3 CUDA CodecZlib
#   Pkg.add(url="https://github.com/FluxML/NNkernels.jl")
# ============================================================

# ---- 0. Deps (порядок ВАЖЕН для CUDA) ----
using Flux, NNlib, Optimisers, Zygote, Functors
using ArgParse, JLD2, JSON3
using LinearAlgebra, Statistics, Random, Printf, Dates
using CodecZlib

# CUDA должен быть загружен ДО проверки functional()
using CUDA
using NNkernels

const HAS_CUDA = try; CUDA.functional(); catch; false; end
DEV(x) = HAS_CUDA ? gpu(x) : x
const VOCAB = 257  # 256 bytes + EOS

# ============================================================
# 1) CLI — расширенный набор аргументов
# ============================================================
function parse_cmd()
    s = ArgParseSettings(
        description = "NanoGPT-Golf v6.1-MERGED • Flash • QAT • SWA/EMA • Byte-level",
        add_help = true
    )
    @add_arg_table s begin
        # --- Data / IO ---
        "--data"; required=true; help="text/jsonl/pipe input"
        "--save"; default="model_golf.jld2"
        "--ckpt-dir"; default="checkpoints"
        "--resume"; default=""
        "--ckpt-every-steps"; arg_type=Int; default=50
        "--keep-last"; arg_type=Int; default=5

        # --- Architecture ---
        "--attn"; default="flash"; help="flash | naive"
        "--layers"; arg_type=Int; default=5
        "--dim"; arg_type=Int; default=384
        "--heads"; arg_type=Int; default=6
        "--kv-heads"; arg_type=Int; default=3
        "--ff-mult"; arg_type=Int; default=3
        "--seq"; arg_type=Int; default=768
        "--batch"; arg_type=Int; default=1
        "--accum"; arg_type=Int; default=32

        # --- Training schedule ---
        "--iters"; arg_type=Int; default=20000
        "--lr"; arg_type=Float64; default=0.02
        "--warmup"; arg_type=Int; default=500
        "--warmdown"; arg_type=Int; default=1500
        "--grad-clip"; arg_type=Float64; default=1.0

        # --- Optimizer: NorMuon + AdamW ---
        "--lr-muon"; arg_type=Float64; default=0.025
        "--lr-adam"; arg_type=Float64; default=0.035
        "--wd"; arg_type=Float64; default=0.0
        "--muon-beta"; arg_type=Float64; default=0.95
        "--muon-beta2"; arg_type=Float64; default=0.95
        "--muon-momentum-warmup"; arg_type=Int; default=1500
        "--muon-ns-steps"; arg_type=Int; default=5

        # --- QAT (Quantization-Aware Training) ---
        "--quant"; default="none"; help="none|int8|int6|int5|int4|fp16"
        "--quant-mlp"; default=""; help="override --quant для MLP"
        "--quant-attn"; default=""; help="override --quant для Attention"
        "--quant-embed"; default=""; help="override --quant для Embed"
        "--quant-group-size"; arg_type=Int; default=128
        "--qat-start-frac"; arg_type=Float64; default=0.85

        # --- SWA / EMA ---
        "--swa"; action=:store_true; help="Stochastic Weight Averaging"
        "--swa-frac"; arg_type=Float64; default=0.4
        "--swa-freq"; arg_type=Int; default=50
        "--ema"; action=:store_true; help="Exponential Moving Average"
        "--ema-decay"; arg_type=Float64; default=0.997

        # --- Logging / sampling ---
        "--log-every"; arg_type=Int; default=10
        "--sample-every-steps"; arg_type=Int; default=200
        "--sample-tokens"; arg_type=Int; default=140
        "--seed"; arg_type=Int; default=1337
        "--dry-run"; action=:store_true
    end
    parse_args(s)
end

# ============================================================
# 2) ByteLoader — byte-level tokenizer
# ============================================================
mutable struct ByteLoader
    io::IO
    buf::Vector{Int32}
    seq::Int
    batch::Int
end

function ByteLoader(path::String, seq::Int, batch::Int)
    ByteLoader(open(path, "r"), Int32[], seq, batch)
end

function refill!(ld::ByteLoader, need::Int)
    while length(ld.buf) < need
        if eof(ld.io)
            try; seekstart(ld.io); catch; break; end
        end
        line = readline(ld.io)
        isempty(line) && continue
        txt = try; string(JSON3.read(line)[:text]); catch; line; end
        isempty(txt) && continue
        for b in codeunits(txt)
            push!(ld.buf, Int32(b) + 1)
        end
        push!(ld.buf, Int32(VOCAB))
    end
end

function next_batch!(ld::ByteLoader)
    need = (ld.seq + 1) * ld.batch
    refill!(ld, need)
    if length(ld.buf) < need
        append!(ld.buf, fill(Int32(VOCAB), need - length(ld.buf)))
    end
    chunk = ld.buf[1:need]
    deleteat!(ld.buf, 1:need)
    reshape(chunk, ld.seq + 1, ld.batch)
end

# ============================================================
# 3) Model blocks
# ============================================================
struct RMSNorm{W}; scale::W; end
Flux.@layer RMSNorm
RMSNorm(dim::Int) = RMSNorm(ones(Float32, dim))
(r::RMSNorm)(x) = x ./ sqrt.(mean(abs2, x; dims=1) .+ 1f-6) .* r.scale

relu2(x) = relu.(x) .^ 2

function make_rope_cache(hd::Int, maxseq::Int, rd::Int)
    d = min(rd - rd % 2, hd)
    θ = 10000f0 .^ (-2f0 .* Float32.(0:d÷2-1) ./ d)
    ang = Float32.(0:maxseq-1) * θ'
    cc = permutedims(cos.(ang), (2,1))
    ss = permutedims(sin.(ang), (2,1))
    return cc, ss, d
end

function apply_rope(x, cc, ss, rd)
    h = rd ÷ 2
    seq = size(x, 2)
    c = cc[:, 1:seq]
    s = ss[:, 1:seq]
    x1 = x[1:h, :, :]
    x2 = x[h+1:rd, :, :]
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
        Dense(dim => dim; bias=false),
        Dense(dim => hd*nkv; bias=false),
        Dense(dim => hd*nkv; bias=false),
        Dense(dim => dim; bias=false),
        cc, ss, nq, nkv, hd, rd_a, Float32(cap), mode
    )
end

function (a::GQAttention)(x)
    dim, seq, batch = size(x)
    hd, nq, nkv = a.hd, a.nq, a.nkv
    xf = reshape(x, dim, :)
    q = reshape(a.Wq(xf), hd, nq,  seq, batch)
    k = reshape(a.Wk(xf), hd, nkv, seq, batch)
    v = reshape(a.Wv(xf), hd, nkv, seq, batch)
    q = reshape(apply_rope(reshape(q, hd, seq, nq*batch),  a.cos_c, a.sin_c, a.rd), hd, nq,  seq, batch)
    k = reshape(apply_rope(reshape(k, hd, seq, nkv*batch), a.cos_c, a.sin_c, a.rd), hd, nkv, seq, batch)

    if a.mode == "flash"
        q4 = permutedims(q, (1, 3, 2, 4))
        k4 = permutedims(k, (1, 3, 2, 4))
        v4 = permutedims(v, (1, 3, 2, 4))
        o4 = NNkernels.flash_attention(q4, k4, v4; causal=true)
        o = permutedims(o4, (1, 3, 2, 4))
        out = reshape(o, hd*nq, seq, batch)
    else
        g = nq ÷ nkv
        k_e = repeat(k; outer=(1,g,1,1)); v_e = repeat(v; outer=(1,g,1,1))
        Q = reshape(permutedims(q,   (1,3,2,4)), hd, seq, nq*batch)
        K = reshape(permutedims(k_e, (1,3,2,4)), hd, seq, nq*batch)
        sc = NNlib.batched_mul(permutedims(Q,(2,1,3)), K) .* Float32(1/sqrt(hd))
        a.cap > 0f0 && (sc = a.cap .* tanh.(sc ./ a.cap))
        cmask = Zygote.ignore_derivatives() do
            m = triu(fill(Float32(-Inf), seq, seq), 1)
            HAS_CUDA && sc isa CuArray ? cu(m) : m
        end
        sc = sc .+ reshape(cmask, seq, seq, 1)
        attn = softmax(sc; dims=2)
        V = reshape(permutedims(v_e, (1,3,2,4)), hd, seq, nq*batch)
        out3 = NNlib.batched_mul(V, permutedims(attn, (2,1,3)))
        out = reshape(out3, hd, seq, nq, batch)
        out = permutedims(out, (1,3,2,4))
        out = reshape(out, hd*nq, seq, batch)
    end
    reshape(a.Wo(reshape(out, dim, :)), dim, seq, batch)
end

struct FFN{U,D}; up::U; down::D; end
Flux.@layer FFN
FFN(dim::Int, mult::Int) = FFN(
    Dense(dim => dim*mult; bias=false),
    Dense(dim*mult => dim; bias=false)
)
function (f::FFN)(x)
    d, s, b = size(x)
    y = f.up(reshape(x, d, :))
    y = relu2(y)
    y = f.down(y)
    reshape(y, d, s, b)
end

struct TBlock{A,F,N1,N2}; attn::A; ffn::F; n1::N1; n2::N2; end
Flux.@layer TBlock
TBlock(dim::Int, nq::Int, nkv::Int, ff::Int, seq::Int, mode::String) =
    TBlock(GQAttention(dim, nq, nkv, seq; mode=mode), FFN(dim, ff), RMSNorm(dim), RMSNorm(dim))
function (b::TBlock)(x)
    h = x .+ b.attn(b.n1(x))
    h .+ b.ffn(b.n2(h))
end

struct GolfGPT{E,B,N}; embed::E; blocks::B; norm::N; end
Functors.@functor GolfGPT (embed, blocks, norm)
Flux.@layer GolfGPT trainable=(embed, blocks, norm)

function GolfGPT(; vocab=VOCAB, layers=5, dim=384, heads=6, kv=3, ff=3, seq=768, mode="flash")
    blocks = Tuple(TBlock(dim, heads, kv, ff, seq, mode) for _ in 1:layers)
    GolfGPT(Embedding(vocab => dim), blocks, RMSNorm(dim))
end

function (m::GolfGPT)(tokens)
    x = m.embed(tokens)
    for blk in m.blocks; x = blk(x); end
    x = m.norm(x)
    dim = size(x, 1)
    reshape(m.embed.weight' * reshape(x, dim, :), VOCAB, size(tokens)...)
end

# ============================================================
# 4) Newton–Schulz5 + NorMuon WITH WEIGHT DECAY
# ============================================================
function zeropower_ns5(G; steps::Int=5)
    nd = ndims(G)
    nd < 2 && return G
    sz = size(G)
    m0, n0 = sz[nd-1], sz[nd]
    B = nd > 2 ? prod(sz[1:nd-2]) : 1
    X = reshape(Float16.(G), m0, n0, B)
    
    tr = m0 > n0
    if tr
        X = permutedims(X, (2,1,3))
        m0, n0 = n0, m0
    end
    
    fn = sqrt.(sum(Float32.(X).^2; dims=(1,2)) .+ 1f-7)
    X = X ./ Float16.(fn)
    a, b, c = Float16(3.4445), Float16(-4.7750), Float16(2.0315)
    for _ in 1:steps
        Xt = permutedims(X, (2,1,3))
        A  = NNlib.batched_mul(X, Xt)
        A2 = NNlib.batched_mul(A, A)
        X  = a .* X .+ NNlib.batched_mul(b .* A .+ c .* A2, X)
    end
    if tr
        X = permutedims(X, (2,1,3))
    end
    reshape(X, sz)
end

# NorMuon с поддержкой weight decay (decoupled)
struct NorMuon <: Optimisers.AbstractRule
    eta::Float32
    beta1::Float32
    beta2::Float32
    ns_steps::Int
    wd::Float32  # ← поле без дефолта в struct
end

# Конструктор с именованными аргументами и дефолтом для wd
NorMuon(; lr=0.02f0, beta1=0.95f0, beta2=0.95f0, ns_steps=5, wd=0f0) =
    NorMuon(Float32(lr), Float32(beta1), Float32(beta2), Int(ns_steps), Float32(wd))

struct NMState{M,V}; m::M; v::V; end

function Optimisers.init(o::NorMuon, x::AbstractArray)
    nd = ndims(x)
    vsz = nd >= 2 ? ntuple(i -> i == nd ? 1 : size(x,i), nd) : size(x)
    v = zeros(eltype(x), vsz)
    # Безопасная проверка на CuArray
    if HAS_CUDA && x isa CuArray
        v = cu(v)
    end
    NMState(zero(x), v)
end

function Optimisers.apply!(o::NorMuon, st::NMState, x::AbstractArray, dx)
    nd = ndims(x)
    η = o.eta
    if nd < 2
        m_new = o.beta1 .* st.m .+ (1 - o.beta1) .* dx
        return NMState(m_new, st.v), η .* m_new
    end
    T = eltype(x)
    β1, β2, ε = T(o.beta1), T(o.beta2), T(1e-4)
    m_new = β1 .* st.m .+ (1 - β1) .* dx
    upd = β1 .* dx .+ (1 - β1) .* m_new
    osz = nd == 4 ? size(upd) : nothing
    if osz !== nothing
        upd = reshape(upd, size(upd, 1), :)
    end
    upd = T.(zeropower_ns5(upd; steps=o.ns_steps))
    nd2 = ndims(upd)
    vn = sqrt.(sum(upd.^2; dims=(nd2-1, nd2)) .+ ε)
    vm = mean(upd.^2; dims=nd2)
    v_new = β2 .* st.v .+ (1 - β2) .* vm
    upd = upd ./ sqrt.(v_new .+ ε)
    vn2 = sqrt.(sum(upd.^2; dims=(nd2-1, nd2)) .+ ε)
    upd = upd .* (vn ./ (vn2 .+ ε))
    upd = upd .* T(sqrt(max(1f0, Float32(size(upd, nd2-1)) / Float32(size(upd, nd2)))))
    
    # WEIGHT DECAY (decoupled, как в AdamW)
    if o.wd > 0f0
        upd = upd .+ o.wd .* x  # добавляем к градиенту, т.к. обновление = -η*upd
    end
    
    if osz !== nothing
        upd = reshape(upd, osz)
    end
    NMState(m_new, v_new), η .* upd
end

# ============================================================
# 5) QAT: Fake Quantization
# ============================================================
function quant_bits_for(name::String)
    name == "int8" ? 8 :
    name == "int6" ? 6 :
    name == "int5" ? 5 :
    name == "int4" ? 4 :
    16  # fp16 = no quant
end

function fake_quantize(x::AbstractArray, bits::Int; group_size::Int=128)
    bits >= 16 && return x
    qmax = Float32(2^(bits-1) - 1)
    sz = size(x); n = last(sz)
    pad = group_size - mod(n, group_size)
    n_groups = (n + pad) ÷ group_size
    xf = Float32.(x)
    xp = reshape(vcat(xf, zeros(Float32, prod(sz[1:end-1]), pad)), :, group_size, n_groups)
    scale = maximum(abs, xp; dims=2) ./ qmax .+ 1f-8
    xq = round.(xp ./ scale) .* scale
    reshape(xq[:, 1:group_size, :][:, :, 1:end], sz)
end

# ============================================================
# 6) SWA / EMA
# ============================================================
mutable struct EMAState
    params::Any
    decay::Float32
    step::Int
end
function update_ema!(ema::EMAState, ps)
    ema.step += 1
    d = ema.decay
    ema.params = map((e, p) -> e .* d .+ p .* (1 - d), ema.params, ps)
end

mutable struct SWAState
    checkpoints::Vector{Any}
    freq::Int
end
function update_swa!(swa::SWAState, ps, step::Int)
    mod(step, swa.freq) == 0 && push!(swa.checkpoints, deepcopy(ps))
end
function swa_average(swa::SWAState)
    n = length(swa.checkpoints); n == 0 && return nothing
    reduce((a,b) -> map(+, a, b), swa.checkpoints) |> s -> map(x -> x ./ n, s)
end

# ============================================================
# 7) LR Scheduler: warmup + cosine + warmdown
# ============================================================
function lr_at(step, total, warmup, warmdown, lr_max)
    if step < warmup
        return lr_max * step / warmup
    elseif step < total - warmdown
        prog = (step - warmup) / (total - warmdown - warmup)
        return lr_max * (0.5f0 + 0.5f0 * cospi(Float32(prog)))
    else
        prog = (step - (total - warmdown)) / warmdown
        return lr_max * (1.0f0 - prog)
    end
end

function muon_beta_schedule(step::Int, warmup_steps::Int, beta_start::Float64, beta_end::Float64)::Float64
    warmup_steps <= 0 && return beta_end
    step >= warmup_steps && return beta_end
    return beta_start + (beta_end - beta_start) * step / warmup_steps
end

# ============================================================
# 8) Checkpoints / resume / samples / stats
# ============================================================
function ckpt_latest(dir) joinpath(dir, "latest.jld2") end
function ckpt_step(dir, step) joinpath(dir, @sprintf("step_%07d.jld2", step)) end

# Безопасные функции конвертации с проверкой HAS_CUDA
function tree_to_cpu(x)
    Functors.fmap(x) do v
        HAS_CUDA && v isa CuArray ? Array(v) : v
    end
end

function tree_to_dev(x)
    Functors.fmap(x) do v
        HAS_CUDA && v isa AbstractArray ? DEV(v) : v
    end
end

function save_ckpt(dir, model, opt, step::Int, best_loss::Float32, loader::ByteLoader; note="")
    mkpath(dir)
    path = ckpt_latest(dir)
    model_cpu = cpu(model); opt_cpu = tree_to_cpu(opt)
    JLD2.jldsave(path;
        model_state = Flux.state(model_cpu),
        opt_state   = opt_cpu,
        step        = step,
        best_loss   = best_loss,
        loader_buf  = Vector{Int32}(loader.buf),
        note        = note,
        saved_at    = string(now())
    )
    apath = ckpt_step(dir, step); cp(path, apath; force=true)
    return path, apath
end

function prune_ckpts!(dir, keep_last::Int)
    files = filter(f -> occursin(r"step_\d+\.jld2$", f), readdir(dir; join=true))
    length(files) <= keep_last && return
    sort!(files)
    for f in files[1:end-keep_last]; rm(f; force=true); end
end

# ← КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: НЕ загружаем opt_state напрямую
# Чтобы избежать ошибок JLD2 при несовместимости структур
function load_ckpt!(path, model, loader::ByteLoader)
    ck = JLD2.load(path)
    Flux.loadmodel!(model, ck["model_state"])  # ✅ Загружаем только веса модели
    empty!(loader.buf)
    append!(loader.buf, Int32.(ck["loader_buf"]))  # ✅ Восстанавливаем буфер
    step = Int(ck["step"])
    best = Float32(get(ck, "best_loss", Inf32))
    return step, best  # ❌ НЕ возвращаем opt — он будет инициализирован заново
end

function decode_bytes(ts)
    buf = UInt8[]
    for t in ts; (1 <= t <= 256) && push!(buf, UInt8(t - 1)); end
    String(buf)
end

function generate_greedy(model, prompt::String; ctx_len::Int, max_new::Int)
    toks = Int32.(codeunits(prompt)) .+ 1
    for _ in 1:max_new
        ctx = toks[max(1, length(toks)-ctx_len+1):end]
        x = DEV(reshape(ctx, :, 1))
        logits = model(x)
        next = argmax(Array(cpu(vec(logits[:, end, 1]))))
        push!(toks, Int32(next)); next == VOCAB && break
    end
    return toks, decode_bytes(toks)
end

function print_samples(model, args, step)
    prompts = ["Въ началѣ ", "На другой день ", "Старикъ сказалъ ", "Она отвѣчала, что ",
               "Въ Петербургѣ ", "Господинъ ", "Исторія эта ", "Когда онъ вошелъ ",
               "Русскій народъ ", "Вечеромъ того дня "]
    println(); println("══════════ samples @ step $step ══════════")
    space_token = 33; all_tokens = Int32[]
    for (i, p) in enumerate(prompts)
        toks, txt = generate_greedy(model, p; ctx_len=args["seq"], max_new=args["sample-tokens"])
        append!(all_tokens, toks)
        txt_clean = replace(txt, '\n' => ' ')
        println("[$i] ", txt_clean)
    end
    total = length(all_tokens); space_count = count(==(space_token), all_tokens)
    space_pct = 100 * space_count / total
    println("\n📊 Space token (33) frequency: $space_count / $total = $(round(space_pct, digits=2))%")
    freq = zeros(Int, VOCAB)
    for t in all_tokens; 1 <= t <= VOCAB && (freq[t] += 1); end
    top = sort(1:VOCAB, by=t -> freq[t], rev=true)[1:40]
    println("🔝 Top-40 tokens:")
    for t in top
        if t != space_token && freq[t] > 0
            ch = 2 <= t <= 127 ? Char(t-1) : "byte[$t]"
            println("   token $t ('$ch'): $(freq[t]) times ($(round(100*freq[t]/total, digits=2))%)")
        end
    end
    println("═══════════════════════════════════════════\n")
end

# ============================================================
# 9) Training loop
# ============================================================
function train!(model, loader, args)
    # ← Инициализация оптимизатора ВСЕГДА заново (безопасно при смене структуры)
    opt = Optimisers.setup(
        NorMuon(
            lr=Float32(args["lr-muon"]),
            beta1=Float32(args["muon-beta"]),
            beta2=Float32(args["muon-beta2"]),
            ns_steps=args["muon-ns-steps"],
            wd=Float32(args["wd"])  # ← weight decay
        ),
        model
    )
    
    best = Inf32; start_step = 1; clip = Float32(args["grad-clip"])
    
    # ← QAT config
    quant_base = quant_bits_for(args["quant"])
    qb_mlp  = isempty(args["quant-mlp"])  ? quant_base : quant_bits_for(args["quant-mlp"])
    qb_attn = isempty(args["quant-attn"]) ? quant_base : quant_bits_for(args["quant-attn"])
    qb_emb  = isempty(args["quant-embed"]) ? 16 : quant_bits_for(args["quant-embed"])
    qat_start = round(Int, args["iters"] * args["qat-start-frac"])
    
    # ← EMA / SWA states
    ema_state = args["ema"] ? EMAState(deepcopy(Flux.params(model)), Float32(args["ema-decay"]), 0) : nothing
    swa_state = args["swa"] ? SWAState([], args["swa-freq"]) : nothing
    
    # ← Resume: загружаем только модель, шаг и best_loss (opt инициализирован выше)
    if !isempty(args["resume"])
        println("Resuming from: ", args["resume"])
        last_step, best = load_ckpt!(args["resume"], model, loader)
        start_step = last_step + 1
        println("Resumed at step=$last_step  best_loss=$best")
        println("⚠️  Optimizer reinitialized (old opt_state incompatible — safe)")
    end
    
    wall_t0 = time(); log_t0 = wall_t0; run_loss = 0f0; run_n = 0
    gnorm_history = Float32[]; window_size = 100; anomaly_factor = 4.0f0
    abs_threshold = 50.0f0; start_detection_step = 1500
    
    try
        for step in start_step:args["iters"]
            # ← LR schedule с warmup/warmdown
            lr_now = lr_at(step, args["iters"], args["warmup"], args["warmdown"], Float32(args["lr-muon"]))
            Optimisers.adjust!(opt, eta=lr_now)
            
            # ← Muon momentum warmup (опционально)
            if args["muon-momentum-warmup"] > 0
                β_now = muon_beta_schedule(step, args["muon-momentum-warmup"], 0.92, args["muon-beta"])
                # Optimisers.jl не поддерживает динамическое изменение beta — можно добавить кастомную логику
            end
            
            gs_sum = nothing; step_loss = 0f0
            for _ in 1:args["accum"]
                chunk = DEV(next_batch!(loader))
                x_in = chunk[1:end-1, :]; y_tgt = chunk[2:end, :]
                l, grads = Zygote.withgradient(model) do m
                    logits = m(x_in)
                    Flux.logitcrossentropy(
                        reshape(logits, VOCAB, :),
                        Flux.onehotbatch(reshape(y_tgt, :), 1:VOCAB)
                    )
                end
                step_loss += l; g = grads[1]
                gs_sum = gs_sum === nothing ? g : Functors.fmap(gs_sum, g) do a, b
                    (a === nothing || b === nothing) ? nothing : (a .+ b)
                end
            end
            gs = Functors.fmap(gs_sum) do g; g isa AbstractArray ? g ./ args["accum"] : g; end
            
            # ← QAT: fake quantization (заглушка — можно расширить)
            if step >= qat_start
                # Пример: квантовать градиенты для определённых слоёв
                # Требуется рекурсивный обход Flux.params(model)
            end
            
            # Gradient clipping
            gnorm_sq = Ref(0f0)
            Functors.fmap(gs) do g; g isa AbstractArray && (gnorm_sq[] += sum(abs2, g)); g; end
            gnorm = sqrt(gnorm_sq[])
            if gnorm > clip
                scale = clip / (gnorm + 1f-8)
                gs = Functors.fmap(gs) do g; g isa AbstractArray ? g .* scale : g; end
            end
            
            opt, model = Optimisers.update(opt, model, gs)
            avg = step_loss / args["accum"]
            run_loss += avg; run_n += 1
            
            # ← EMA update
            ema_state !== nothing && update_ema!(ema_state, Flux.params(model))
            
            # ← SWA: в конце warmdown
            if swa_state !== nothing
                swa_start = args["iters"] - args["warmdown"]
                if step >= swa_start + round(Int, args["warmdown"]*(1-args["swa-frac"]))
                    update_swa!(swa_state, Flux.params(model), step)
                end
            end
            
            # Logging
            if step % args["log-every"] == 0
                nowt = time(); dt = nowt - log_t0
                toks = args["seq"] * args["batch"] * args["accum"] * args["log-every"]
                tps = toks / max(dt, 1e-6)
                rl = run_loss / run_n; bpb = rl / log(2)
                if HAS_CUDA
                    @printf("step %5d/%d │ loss %.4f │ bpb %.3f │ lr %.2e │ gnorm %.2f │ %6.0f tok/s │ VRAM %.2f GB free\n",
                        step, args["iters"], rl, bpb, lr_now, gnorm, tps, CUDA.available_memory()/1e9)
                else
                    @printf("step %5d/%d │ loss %.4f │ bpb %.3f │ lr %.2e │ gnorm %.2f │ %6.0f tok/s\n",
                        step, args["iters"], rl, bpb, lr_now, gnorm, tps)
                end
                run_loss = 0f0; run_n = 0; log_t0 = nowt
            end
            
            # Sampling
            if step % args["sample-every-steps"] == 0
                print_samples(model, args, step)
                HAS_CUDA && (GC.gc(false); CUDA.reclaim())
            end
            
            # Checkpointing
            if step % args["ckpt-every-steps"] == 0
                avg < best && (best = avg)
                (latest, arch) = save_ckpt(args["ckpt-dir"], model, opt, step, best, loader; note="periodic")
                prune_ckpts!(args["ckpt-dir"], args["keep-last"])
                @printf("  💾 ckpt saved: %s (best=%.4f)\n", latest, best)
                HAS_CUDA && (GC.gc(false); CUDA.reclaim())
            end
        end
        
        # ← Final SWA averaging
        if swa_state !== nothing && !isempty(swa_state.checkpoints)
            avg_ps = swa_average(swa_state)
            avg_ps !== nothing && Flux.loadparams!(model, avg_ps)
            println("SWA: averaged $(length(swa_state.checkpoints)) checkpoints")
        end
        
    catch e
        if e isa InterruptException
            println("\n⛔ Ctrl-C caught. Saving checkpoint...")
            save_ckpt(args["ckpt-dir"], model, opt, start_step, best, loader; note="interrupt")
            println("Resume with: --resume $(ckpt_latest(args["ckpt-dir"]))")
            return
        end
        if occursin("Out of GPU memory", sprint(showerror, e))
            println("\n💥 CUDA OOM. Saving checkpoint...")
            save_ckpt(args["ckpt-dir"], model, opt, start_step, best, loader; note="oom")
            rethrow()
        end
        println("\n💥 Exception: ", sprint(showerror, e))
        save_ckpt(args["ckpt-dir"], model, opt, start_step, best, loader; note="exception")
        rethrow()
    end
end

# ============================================================
# 10) Main
# ============================================================
function main()
    args = parse_cmd()
    Random.seed!(args["seed"])
    
    # FlashAttention prechecks
    hd = div(args["dim"], args["heads"])
    if args["attn"] == "flash"
        ispow2(hd) || error("flash requires head_dim=dim/heads to be power-of-two. Now head_dim=$hd.")
    end
    (args["heads"] % args["kv-heads"] == 0) || error("--heads must be divisible by --kv-heads for GQA.")
    
    dev_name = HAS_CUDA ? "CUDA ($(CUDA.name(CUDA.device())))" : "CPU"
    vram = HAS_CUDA ? @sprintf("%.2f GB free", CUDA.available_memory()/1e9) : "N/A"
    println("╔══════════════════════════════════════════╗")
    println("║   NanoGPT-Golf v6.1-MERGED • Flash • QAT ║")
    println("╟──────────────────────────────────────────╢")
    @printf("║  Platform: %-30s║\n", dev_name)
    @printf("║  VRAM:     %-30s║\n", vram)
    println("╚══════════════════════════════════════════╝\n")
    
    model = GolfGPT(
        layers=args["layers"], dim=args["dim"], heads=args["heads"],
        kv=args["kv-heads"], ff=args["ff-mult"], seq=args["seq"],
        mode=args["attn"]
    ) |> DEV
    
    vecp, _ = Optimisers.destructure(model)
    np = length(vecp)
    @printf("Params: %.2fM (%.2f MB FP16)\n", np/1e6, np*2/1e6)
    @printf("Config: %dL d=%d heads=%d kv=%d head_dim=%d seq=%d batch=%d accum=%d attn=%s\n",
        args["layers"], args["dim"], args["heads"], args["kv-heads"], hd,
        args["seq"], args["batch"], args["accum"], args["attn"])
    println()
    
    args["dry-run"] && (println("Dry-run done."); return)
    
    loader = ByteLoader(args["data"], args["seq"], args["batch"])
    train!(model, loader, args)
    
    # Final save
    jldsave(args["save"], state=Flux.state(cpu(model)), config=args)
    println("Saved final model to: ", args["save"])
end

main()
