#!/usr/bin/env julia
# -*- coding: utf-8 -*-
#
# server_v5_5.jl — Серверъ инференса NanoGPT-Golf v5.5
# Исправлено: корректное завершѳніе chunked-потоковъ через HTTP.finishwrite().
#
# Запускъ:
#   $ julia --project=@server --threads=4 server_v5_5.jl --ckpt best.jld2 --port 8080 --host 0.0.0.0

using HTTP, JSON3, Base.Threads
using Flux, NNlib, Zygote, Functors
using JLD2, ArgParse
using LinearAlgebra, Statistics, Random, Printf, Dates
using CUDA
using NNkernels
using Logging

Logging.disable_logging(Logging.Warn)

# ============================================================
# Глобальная статистика (потокобезопасная)
# ============================================================
mutable struct ServerStats
    requests_total::Int
    requests_success::Int
    requests_error::Int
    tokens_generated::Int
    tokens_prompt::Int
    time_first_token_sum::Float64
    time_total_sum::Float64
    gpu_memory_used::Float64
    cpu_memory_used::Float64
    last_summary_time::Float64
    lock::ReentrantLock
end

const STATS = ServerStats(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, time(), ReentrantLock())

function record_request!(success::Bool, prompt_tokens::Int, gen_tokens::Int, 
                         time_first::Float64, time_total::Float64)
    lock(STATS.lock) do
        STATS.requests_total += 1
        success ? (STATS.requests_success += 1) : (STATS.requests_error += 1)
        STATS.tokens_prompt += prompt_tokens
        STATS.tokens_generated += gen_tokens
        STATS.time_first_token_sum += time_first
        STATS.time_total_sum += time_total
        
        # ✅ Исправлено: корректное полученіе статистики памяти
        if HAS_CUDA
            try 
                STATS.gpu_memory_used = (CUDA.total_memory() - CUDA.available_memory()) / 1e9 
            catch _ 
                STATS.gpu_memory_used = 0.0
            end
        end
        
        # ✅ Исправлено: gc_total_bytes требуетъ аргументъ
        gc_stats = Base.gc_num()
        STATS.cpu_memory_used = Base.gc_total_bytes(gc_stats) / 1e9
    end
end

function print_stats_summary()
    lock(STATS.lock) do
        now_t = time()
        elapsed = now_t - STATS.last_summary_time
        if elapsed < 60.0; return; end
        STATS.last_summary_time = now_t
        
        req_rate = STATS.requests_total > 0 ? (STATS.requests_success / max(STATS.requests_total, 1)) * 100 : 0.0
        avg_tps = STATS.tokens_generated > 0 ? STATS.tokens_generated / max(STATS.time_total_sum, 1e-6) : 0.0
        avg_ttft = STATS.requests_success > 0 ? STATS.time_first_token_sum / STATS.requests_success : 0.0
        avg_latency = STATS.requests_success > 0 ? STATS.time_total_sum / STATS.requests_success : 0.0
        
        @printf("\n📊 === СТАТИСТИКА СЕРВЕРА (за %.1f сек) ===\n", elapsed)
        @printf("   Запросы: всего=%d, успѣх=%.1f%%, ошибки=%d\n", 
                STATS.requests_total, req_rate, STATS.requests_error)
        @printf("   Токены: сгенерировано=%d, промптов=%d\n", 
                STATS.tokens_generated, STATS.tokens_prompt)
        @printf("   Скорость: %.1f ток/сек (средн.)\n", avg_tps)
        @printf("   Латентность: TTFT=%.2f сек, общая=%.2f сек (средн.)\n", 
                avg_ttft, avg_latency)
        @printf("   Память: GPU=%.2f GB, CPU=%.2f GB\n", 
                STATS.gpu_memory_used, STATS.cpu_memory_used)
        @printf("   Потоковъ: %d/%d активно\n", 
                Threads.nthreads(), length(Base.Threads.threadpoolsize()))
        println("   " * "─"^50 * "\n")
        
        STATS.requests_total = 0; STATS.requests_success = 0; STATS.requests_error = 0
        STATS.tokens_generated = 0; STATS.tokens_prompt = 0
        STATS.time_first_token_sum = 0.0; STATS.time_total_sum = 0.0
    end
end

# ============================================================
# Токенизаторъ (vocab = 385)
# ============================================================
const BYTE_VOCAB  = 256
const CYR2_BASE   = BYTE_VOCAB + 1
const CYR2_STRIDE = 64
const CYR2_COUNT  = 2 * CYR2_STRIDE
const EOS_TOKEN   = CYR2_BASE + CYR2_COUNT
const VOCAB       = EOS_TOKEN

function is_cyr2_pair(b1::UInt8, b2::UInt8)
    return (b1 == 0xD0 || b1 == 0xD1) && (0x80 <= b2 <= 0xBF)
end
function cyr2_token(b1::UInt8, b2::UInt8)::Int32
    lead = (b1 == 0xD0) ? 0 : CYR2_STRIDE
    return Int32(CYR2_BASE + lead + (Int(b2) - 0x80))
end
function token_to_cyr2(t::Integer)
    off = Int(t) - CYR2_BASE
    b1 = off < CYR2_STRIDE ? UInt8(0xD0) : UInt8(0xD1)
    b2 = UInt8(0x80 + (off % CYR2_STRIDE))
    return b1, b2
end
function encode_text_tokens(txt::AbstractString; add_eos::Bool=true)
    bs = collect(codeunits(txt)); out = Int32[]; sizehint!(out, length(bs) + 2); i = 1
    while i <= length(bs)
        if i < length(bs) && is_cyr2_pair(bs[i], bs[i+1])
            push!(out, cyr2_token(bs[i], bs[i+1])); i += 2
        else
            push!(out, Int32(bs[i]) + 1); i += 1
        end
    end
    if add_eos; push!(out, Int32(EOS_TOKEN)); end
    return out
end
function decode_tokens(ts)
    buf = UInt8[]; sizehint!(buf, length(ts) * 2)
    for t in ts
        if 1 <= t <= 256; push!(buf, UInt8(t - 1))
        elseif CYR2_BASE <= t < EOS_TOKEN; b1, b2 = token_to_cyr2(t); push!(buf, b1); push!(buf, b2)
        end
    end
    return String(buf)
end

# ============================================================
# Совмѣстимая загрузка состояній
# ============================================================
function compatible_state(dst, src)
    if dst isa NamedTuple && src isa NamedTuple
        keys_list = keys(dst)
        vals = map(k -> haskey(src, k) ? compatible_state(getfield(dst, k), getfield(src, k)) : getfield(dst, k), keys_list)
        return NamedTuple{keys_list}(Tuple(vals))
    elseif dst isa Tuple && src isa Tuple && length(dst) == length(src)
        return ntuple(i -> compatible_state(dst[i], src[i]), length(dst))
    elseif dst isa AbstractArray && src isa AbstractArray
        return size(dst) == size(src) ? src : dst
    else; return src; end
end
function load_compatible_model!(model, src_state)
    Flux.loadmodel!(model, compatible_state(Flux.state(model), src_state))
end

# ============================================================
# Модель
# ============================================================
const HAS_CUDA = try CUDA.functional() catch _; false; end
DEV(x) = HAS_CUDA ? gpu(x) : x

struct RMSNorm{W}; scale::W; end
Flux.@layer RMSNorm
RMSNorm(dim::Integer) = RMSNorm(ones(Float32, dim))
(r::RMSNorm)(x) = x ./ sqrt.(mean(abs2, x; dims=1) .+ 1f-6) .* r.scale
relu2(x) = relu.(x) .^ 2

function make_rope_cache(hd, maxseq, rd)
    d = min(rd - rd % 2, hd); θ = 10000f0 .^ (-2f0 * Float32.(0:d÷2-1) ./ d)
    ang = Float32.(0:maxseq-1) * θ'
    return permutedims(cos.(ang), (2, 1)), permutedims(sin.(ang), (2, 1)), d
end
function apply_rope(x, cc, ss, rd)
    h = rd ÷ 2; seq = size(x, 2); c = cc[:, 1:seq]; s = ss[:, 1:seq]
    x1 = x[1:h, :, :]; x2 = x[h+1:rd, :, :]
    r1 = x1 .* c .- x2 .* s; r2 = x1 .* s .+ x2 .* c
    return rd < size(x, 1) ? vcat(r1, r2, x[rd+1:end, :, :]) : vcat(r1, r2)
end

struct GQAttention{Q,K,V,O,C,S}
    Wq::Q; Wk::K; Wv::V; Wo::O; cos_c::C; sin_c::S
    nq::Int; nkv::Int; hd::Int; rd::Int; cap::Float32; mode::String
end
Flux.@layer GQAttention trainable=(Wq,Wk,Wv,Wo)
Functors.@functor GQAttention (Wq,Wk,Wv,Wo,cos_c,sin_c)
function GQAttention(dim::Integer, nq::Integer, nkv::Integer, seq::Integer; rd=16, cap=30f0, mode="flash")
    hd = div(dim, nq); cc, ss, rd_a = make_rope_cache(hd, seq, rd)
    return GQAttention(Dense(dim=>dim;bias=false), Dense(dim=>hd*nkv;bias=false),
                       Dense(dim=>hd*nkv;bias=false), Dense(dim=>dim;bias=false),
                       cc, ss, nq, nkv, hd, rd_a, Float32(cap), mode)
end
function (a::GQAttention)(x)
    dim, seq, batch = size(x); hd, nq, nkv = a.hd, a.nq, a.nkv; xf = reshape(x, dim, :)
    q = reshape(a.Wq(xf), hd, nq, seq, batch)
    k = reshape(a.Wk(xf), hd, nkv, seq, batch)
    v = reshape(a.Wv(xf), hd, nkv, seq, batch)
    q = reshape(apply_rope(reshape(q, hd, seq, nq*batch), a.cos_c, a.sin_c, a.rd), hd, nq, seq, batch)
    k = reshape(apply_rope(reshape(k, hd, seq, nkv*batch), a.cos_c, a.sin_c, a.rd), hd, nkv, seq, batch)
    if a.mode == "flash" && HAS_CUDA
        o = reshape(permutedims(NNkernels.flash_attention(permutedims(q,(1,3,2,4)), permutedims(k,(1,3,2,4)), permutedims(v,(1,3,2,4)); causal=true),(1,3,2,4)), hd*nq, seq, batch)
    else
        g = nq ÷ nkv; k_e = repeat(k; outer=(1,g,1,1)); v_e = repeat(v; outer=(1,g,1,1))
        Q = reshape(permutedims(q,(1,3,2,4)), hd, seq, nq*batch); K = reshape(permutedims(k_e,(1,3,2,4)), hd, seq, nq*batch)
        sc = NNlib.batched_mul(permutedims(Q,(2,1,3)), K) .* Float32(1/sqrt(hd))
        a.cap > 0f0 && (sc = a.cap .* tanh.(sc ./ a.cap))
        cmask = triu(fill(Float32(-Inf), seq, seq), 1); sc .= sc .+ reshape(cmask, seq, seq, 1)
        attn = softmax(sc; dims=2); V = reshape(permutedims(v_e,(1,3,2,4)), hd, seq, nq*batch)
        o = reshape(permutedims(reshape(NNlib.batched_mul(V, permutedims(attn,(2,1,3))), hd, seq, nq, batch),(1,3,2,4)), hd*nq, seq, batch)
    end
    return reshape(a.Wo(reshape(o, dim, :)), dim, seq, batch)
end

struct FFN{U,D}; up::U; down::D; end
Flux.@layer FFN
FFN(dim::Integer, mult::Integer) = FFN(Dense(dim=>dim*mult;bias=false), Dense(dim*mult=>dim;bias=false))
function (f::FFN)(x)
    d, s, b = size(x); y = f.up(reshape(x, d, :)); y = relu2(y)
    return reshape(f.down(y), d, s, b)
end

struct TBlock{A,F,N1,N2}; attn::A; ffn::F; n1::N1; n2::N2; end
Flux.@layer TBlock
TBlock(dim::Integer, nq::Integer, nkv::Integer, ff::Integer, seq::Integer, mode::String) = TBlock(
    GQAttention(dim, nq, nkv, seq; mode=mode), FFN(dim, ff), RMSNorm(dim), RMSNorm(dim))
function (b::TBlock)(x); h = x .+ b.attn(b.n1(x)); return h .+ b.ffn(b.n2(h)); end

struct GolfGPT{E,B,N}; embed::E; blocks::B; norm::N; end
Functors.@functor GolfGPT (embed, blocks, norm)
Flux.@layer GolfGPT trainable=(embed, blocks, norm)
function GolfGPT(; vocab=VOCAB, layers=5, dim=384, heads=6, kv=3, ff=3, seq=1024, mode="flash")
    blocks = Tuple(TBlock(dim, heads, kv, ff, seq, mode) for _ in 1:layers)
    return GolfGPT(Embedding(vocab => dim; init=(dims...) -> 0.02f0*randn(Float32, dims...)), blocks, RMSNorm(dim))
end
function (m::GolfGPT)(tokens)
    x = m.embed(tokens); for blk in m.blocks; x = blk(x); end
    x = m.norm(x); dim = size(x, 1)
    return reshape(m.embed.weight' * reshape(x, dim, :), VOCAB, size(tokens)...)
end

# ============================================================
# Сэмплингъ и Генераторъ
# ============================================================
function sample_topk_topp(logits::Vector{Float32}; topk=40, topp=0.9, temp=0.8, rng=Random.GLOBAL_RNG)
    V = length(logits); t = max(temp, 1e-5); lv = logits ./ Float32(t)
    if topk > 0 && topk < V; thr = partialsort(lv, topk; rev=true); lv[lv .< thr] .= -Inf32; end
    m = maximum(lv); e = exp.(lv .- m); s = sum(e); s <= 0.0f0 && return argmax(logits)
    p = e ./ s
    if topp > 0.0 && topp < 1.0
        ord = sortperm(p; rev=true); cum = 0.0; keep = falses(V)
        for i in ord; keep[i] = true; cum += p[i]; if cum >= topp; break; end; end
        p = ifelse.(keep, p, 0.0f0); ssum = sum(p); ssum <= 0.0f0 && return argmax(logits); p ./= ssum
    end
    r = rand(rng); acc = 0.0
    for i in 1:V; acc += p[i]; if r <= acc; return i; end; end
    return V
end

function generate_iterator_with_metrics(model, init_toks; ctx_len, max_new, temp, topp, topk, greedy, rng, stop_strings=String[])
    time_first_token = nothing
    tokens_count = 0
    start_time = time()
    ch = Channel{String}(0) do c
        toks = copy(init_toks)
        for _ in 1:max_new
            ctx = toks[max(1, length(toks)-ctx_len+1):end]
            x = DEV(reshape(ctx, :, 1)); logits = model(x)
            lv = Array(cpu(vec(logits[:, end, 1])))
            if any(!isfinite, lv); break; end
            nxt = greedy ? argmax(lv) : sample_topk_topp(Float32.(lv); topk=topk, topp=topp, temp=temp, rng=rng)
            if nxt == EOS_TOKEN; break; end
            push!(toks, Int32(nxt)); token_str = decode_tokens([Int32(nxt)])
            if time_first_token === nothing
                time_first_token = time() - start_time
            end
            tokens_count += 1
            if !isempty(stop_strings)
                recent = join(decode_tokens(toks[max(1,end-10):end]), "")
                if any(s -> occursin(s, recent), stop_strings); break; end
            end
            put!(c, token_str)
        end
        close(c)
    end
    return ch, () -> begin
        total_time = time() - start_time
        tf = time_first_token === nothing ? 0.0 : time_first_token
        tps = tokens_count > 0 ? tokens_count / total_time : 0.0
        return (tokens_count, tf, total_time, tps)
    end
end

# ============================================================
# API Обработчики (ИСПРАВЛЕНО: finishwrite для chunked)
# ============================================================
function build_prompt(req)
    if haskey(req, :messages) || haskey(req, "messages")
        msgs = get(req, :messages, get(req, "messages", []))
        return join(["$(get(m,:role,get(m,"role","user"))): $(get(m,:content,get(m,"content","")))\n" for m in msgs], "")
    elseif haskey(req, :prompt) || haskey(req, "prompt")
        return string(get(req, :prompt, get(req, "prompt", "")))
    end; return ""
end

function handle_chat_completions(http, model, args, body::String)
    req_start = time(); prompt_tokens = 0; gen_tokens = 0
    time_first = 0.0; time_total = 0.0; tps = 0.0; success = false
    
    req_json = try JSON3.read(body)
    catch e
        @error "Ошибка парсинга JSON" body exception=e
        resp = Dict("error" => Dict("message" => "Invalid JSON format", "type" => "invalid_request_error"))
        HTTP.setstatus(http, 400)
        HTTP.setheader(http, "Content-Type" => "application/json")
        HTTP.startwrite(http)
        HTTP.write(http, JSON3.write(resp))
        HTTP.finishwrite(http)  # ✅ Завершѳніе потока
        record_request!(false, 0, 0, 0.0, time() - req_start)
        return
    end

    stream = get(req_json, :stream, get(req_json, "stream", false))
    max_tok = Int(get(req_json, :max_tokens, get(req_json, "max_tokens", 256)))
    temp = Float64(get(req_json, :temperature, get(req_json, "temperature", 0.8)))
    topp = Float64(get(req_json, :top_p, get(req_json, "top_p", 0.9)))
    greedy = Bool(get(req_json, :greedy, get(req_json, "greedy", false)))
    stop = String[]; s_arr = get(req_json, :stop, get(req_json, "stop", nothing))
    s_arr isa String && push!(stop, s_arr); s_arr isa Vector && append!(stop, s_arr)
    
    prompt = build_prompt(req_json); toks = encode_text_tokens(prompt; add_eos=false)
    prompt_tokens = length(toks); rng = MersenneTwister(rand(UInt)); ctx_len = args["ctx-len"]
    
    try
        ch, get_metrics = generate_iterator_with_metrics(model, toks; ctx_len=ctx_len, max_new=max_tok, temp=temp, topp=topp, topk=40, greedy=greedy, rng=rng, stop_strings=stop)
        if stream
            # ✅ Потоковый режим: SSE + chunked
            HTTP.setheader(http, "Content-Type" => "text/event-stream")
            HTTP.setheader(http, "Cache-Control" => "no-cache")
            HTTP.setheader(http, "Connection" => "keep-alive")
            HTTP.startwrite(http)
            
            cmpl_id = "chatcmpl-$(randstring(10))"
            for token_str in ch
                chunk = Dict("id"=>cmpl_id,"object"=>"chat.completion.chunk",
                            "created"=>Dates.unix2datetime(time()),
                            "model"=>get(args,"model","golf-v5.5"),
                            "choices"=>[Dict("index"=>0,"delta"=>Dict("role"=>"assistant","content"=>token_str),"finish_reason"=>nothing)])
                HTTP.write(http, "data: $(JSON3.write(chunk))\n\n")
                flush(http)  # ✅ Немедленная отправка
            end
            HTTP.write(http, "data: [DONE]\n\n")
            flush(http)
            HTTP.finishwrite(http)  # ✅ Критично: завершѳніе chunked-потока
        else
            # ✅ Не-потоковый режим: обычный JSON отвѣтъ
            tokens_collected = String[]
            for t in ch; push!(tokens_collected, t); end
            gen_tokens, time_first, time_total, tps = get_metrics()
            
            resp = Dict("id"=>"chatcmpl-$(randstring(10))","object"=>"chat.completion",
                        "created"=>Dates.unix2datetime(time()),"model"=>get(args,"model","golf-v5.5"),
                        "choices"=>[Dict("index"=>0,"message"=>Dict("role"=>"assistant","content"=>join(tokens_collected,"")),"finish_reason"=>"stop")],
                        "usage"=>Dict("prompt_tokens"=>prompt_tokens,"completion_tokens"=>gen_tokens,
                                      "total_tokens"=>prompt_tokens+gen_tokens,
                                      "performance"=>Dict("tokens_per_second"=>round(tps,digits=2),
                                                         "time_to_first_token"=>round(time_first,digits=3),
                                                         "total_time"=>round(time_total,digits=3))))
            
            HTTP.setheader(http, "Content-Type" => "application/json")
            # ✅ Для не-потоковыхъ отвѣтовъ НЕ используем startwrite/finishwrite
            # HTTP.jl самъ управляетъ заголовками и передачей
            HTTP.write(http, JSON3.write(resp))
            flush(http)
        end
        gen_tokens, time_first, time_total, tps = get_metrics()
        @printf("🔹 %s: %d ток., %.1f т/с, TTFT=%.2fс, всего=%.2fс\n", stream ? "Stream" : "Non-stream", gen_tokens, tps, time_first, time_total)
        success = true
    catch e
        @error "Ошибка генераци" exception=(e, catch_backtrace()); rethrow(e)
    finally
        record_request!(success, prompt_tokens, gen_tokens, time_first, time_total)
        print_stats_summary()
    end
end

# ============================================================
# Загрузка модели
# ============================================================
function load_model_robust(args)
    println("⚙  Загрузка модели изъ: $(args["ckpt"]) ...")
    !isfile(args["ckpt"]) && error("Файлъ не найденъ: $(args["ckpt"])")
    ck = JLD2.load(args["ckpt"]); vocab = get(ck, "vocab", VOCAB); cfg = get(ck, "config", Dict())
    layers = get(cfg, "layers", args["layers"]); dim = get(cfg, "dim", args["dim"])
    heads = get(cfg, "heads", args["heads"]); kv = get(cfg, "kv-heads", args["kv-heads"])
    ff = get(cfg, "ff-mult", args["ff-mult"]); seq = get(cfg, "seq", args["seq"]); mode = get(cfg, "attn", args["attn"])
    println(@sprintf("Архитектура: %dL d=%d heads=%d kv=%d ff=%d seq=%d mode=%s", layers, dim, heads, kv, ff, seq, mode))
    model = GolfGPT(vocab=vocab, layers=layers, dim=dim, heads=heads, kv=kv, ff=ff, seq=seq, mode=mode)
    model_state = haskey(ck, "model_state") ? ck["model_state"] : haskey(ck, "live") ? ck["live"] : haskey(ck, "ema") ? ck["ema"] : error("Нѣтъ состоянія модели.")
    load_compatible_model!(model, model_state)
    dev_pref = args["device"] == "auto" ? (HAS_CUDA ? "cuda" : "cpu") : args["device"]
    if dev_pref == "cuda" && HAS_CUDA
        try println("Попытка переноса на GPU..."); model = model |> gpu; println("✅ Успѣшно размещено на GPU.")
        catch e
            @warn "Ошибка GPU ($(sprint(showerror,e))). Переключаемся на CPU."; model = model |> cpu
        end
    else; println("Используется CPU."); model = model |> cpu; end
    return Flux.testmode!(model)
end

# ============================================================
# CLI
# ============================================================
function parse_cmd()
    s = ArgParseSettings(
        description="""
Серверъ инференса NanoGPT-Golf v5.5 (OpenAI API-совмѣстимый)

╔══════════════════════════════════════════════════════════╗
║              ПРИМѢРЫ УПОТРЕБЛЕНІЯ (CURL)                 ║
╚══════════════════════════════════════════════════════════╝

1. Базовая генерация:
   \$ curl http://localhost:8080/v1/chat/completions \\
     -H "Content-Type: application/json" \\
     -d '{"messages":[{"role":"user","content":"Въ началѣ было слово"}], "max_tokens":128}'

2. Потоковая передача (Server-Sent Events):
   \$ curl -N http://localhost:8080/v1/chat/completions \\
     -H "Content-Type: application/json" \\
     -d '{"messages":[{"role":"user","content":"Сказка о рыбаке"}], "stream":true, "max_tokens":200}'

3. Настройка сэмплинга (temperature, top-p, top-k):
   \$ curl http://localhost:8080/v1/chat/completions \\
     -H "Content-Type: application/json" \\
     -d '{"messages":[{"role":"user","content":"Господинъ "}], "temperature":0.7, "top_p":0.9, "max_tokens":150}'

4. Проверка здоровья сервера:
   \$ curl http://localhost:8080/health

╔══════════════════════════════════════════════════════════╗
║           ИНТЕГРАЦІЯ СЪ PYTHON (OpenAI SDK)              ║
╚══════════════════════════════════════════════════════════╝
   from openai import OpenAI
   client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-dummy")
   resp = client.chat.completions.create(
       model="golf-v5.5",
       messages=[{"role":"user", "content":"Исторія эта"}],
       max_tokens=100, temperature=0.7, stream=True
   )
   for chunk in resp:
       print(chunk.choices[0].delta.content or "", end="", flush=True)

╔══════════════════════════════════════════════════════════╗
║                  ДОПОЛНИТЕЛЬНЫЯ СВЕДѢНІЯ                 ║
╚══════════════════════════════════════════════════════════╝
• Токенизаторъ поддерживаетъ дореформенную орѳографію (ѣ, і, ѳ, ѵ, конечный ъ).
• Поле `top_k` въ JSON-запросѣ является расширеніемъ данной реализаціи.
• Серверъ автоматически собираетъ метрики: токены/сек, TTFT, общую задержку, память.
• Статистика выводится въ консоль каждые 60 секундъ (--stats-interval).
• При нехваткѣ VRAM происходитъ автоматическій переходъ на CPU.
""",
        epilog="Для подробной справки см. документацию NanoGPT-Golf v5.5."
    )
    @add_arg_table s begin
        "--ckpt", "-c"; help="Путь къ файлу контрольной точки (.jld2)"; required=true; arg_type=String
        "--port"; help="Портъ сервера"; arg_type=Int; default=8080
        "--host"; help="Сѣтевой адресъ привязки"; arg_type=String; default="127.0.0.1"
        "--model"; help="Идентификаторъ модели для API"; arg_type=String; default="golf-v5.5"
        "--layers"; help="Число слоёвъ архитектуры"; arg_type=Int; default=11
        "--dim"; help="Размерность скрытаго слоя (d_model)"; arg_type=Int; default=384
        "--heads"; help="Число головокъ вниманія"; arg_type=Int; default=6
        "--kv-heads"; help="Число KV-головокъ (GQA)"; arg_type=Int; default=3
        "--ff-mult"; help="Множитель расширенія FFN"; arg_type=Int; default=3
        "--seq"; help="Длина контекста при обученіи"; arg_type=Int; default=512
        "--attn"; help="Режимъ вниманія: 'flash' или 'eager'"; arg_type=String; default="flash"
        "--ctx-len"; help="Максимальная длина контекста при инференсѣ"; arg_type=Int; default=512
        "--device"; help="Устройство вычисленій: 'auto', 'cuda' или 'cpu'"; arg_type=String; default="auto"
        "--stats-interval"; help="Періодъ вывода статистики (въ секундахъ)"; arg_type=Int; default=60
    end
    return parse_args(s)
end

function main()
    args = parse_cmd()
    model = load_model_robust(args)
    println("✅ Модель готова. Запускъ сервера на $(args["host"]):$(args["port"])")
    println("📊 Мониторингъ активированъ (интервалъ: $(args["stats-interval"]) сек)")
    
    @async begin
        while true
            sleep(args["stats-interval"])
            print_stats_summary()
        end
    end
    
    HTTP.listen(args["host"], args["port"]; reuseaddr=true) do http
        try
            req = http.message; path = req.target; meth = req.method
            if (path == "/v1/chat/completions" || path == "/v1/completions") && meth == "POST"
                handle_chat_completions(http, model, args, String(read(http)))
            elseif path == "/v1/models" && meth == "GET"
                HTTP.setheader(http, "Content-Type" => "application/json")
                HTTP.write(http, JSON3.write(Dict("object"=>"list","data"=>[Dict("id"=>"golf-v5.5","object"=>"model","created"=>Int(Dates.now(Dates.UTC).unix),"owned_by"=>"local")])))
                flush(http)
            elseif path == "/health" && meth == "GET"
                HTTP.setstatus(http, 200); HTTP.setheader(http, "Content-Type" => "application/json")
                HTTP.write(http, JSON3.write(Dict("status"=>"healthy","model"=>args["model"],"device"=>HAS_CUDA ? "cuda" : "cpu","threads"=>Threads.nthreads())))
                flush(http)
            else
                HTTP.setstatus(http, 404); HTTP.setheader(http, "Content-Type" => "application/json")
                HTTP.write(http, JSON3.write(Dict("error"=>"Not found"))); flush(http)
            end
        catch e
            if e isa EOFError; return; end
            @error "Request failed" exception=(e, catch_backtrace())
            try
                HTTP.setstatus(http, 500); HTTP.setheader(http, "Content-Type" => "application/json")
                HTTP.write(http, JSON3.write(Dict("error"=>"Internal server error"))); flush(http)
            catch _ end
        end
    end
end

main()
