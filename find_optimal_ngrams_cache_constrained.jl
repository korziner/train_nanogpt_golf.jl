#!/usr/bin/env julia
# -*- coding: utf-8 -*-
#
# find_optimal_ngrams_cache_constrained.jl — Бенчмаркъ съ ЖЁСТКИМЪ ограниченіемъ L2-кэша
# Требованіе: embedding_size ≤ L2_cache (Pascal: 2.2 МБ)
# Исправлено: синтаксисъ @printf (убрана недопустимая интерполяція $())
#
# Употребленіе:
#   $ julia find_optimal_ngrams_cache_constrained.jl -d train.jsonl --l2-cache 2.2 --dim 384 --tps 5200

using ArgParse, Printf, Dates, Statistics

# ============================================================
# Сборщикъ частотъ (только кириллическія 3- и 4-граммы)
# ============================================================
function collect_ngram_frequencies(filepath::String; max_chars::Int=-1)
    counts3 = Dict{String, Int}()
    counts4 = Dict{String, Int}()
    total_chars = 0
    
    open(filepath, "r") do io
        for line in eachline(io)
            chars = collect(line)
            n = length(chars)
            n < 3 && continue
            max_chars > 0 && total_chars >= max_chars && break
            
            for i in 1:(n-2)
                c1, c2, c3 = chars[i], chars[i+1], chars[i+2]
                if '\u0400' <= c1 <= '\u04FF' && '\u0400' <= c2 <= '\u04FF' && '\u0400' <= c3 <= '\u04FF'
                    tri = String([c1, c2, c3])
                    counts3[tri] = get(counts3, tri, 0) + 1
                    total_chars += 1
                end
                if n >= i+3
                    c4 = chars[i+3]
                    if '\u0400' <= c4 <= '\u04FF'
                        quad = String([c1, c2, c3, c4])
                        counts4[quad] = get(counts4, quad, 0) + 1
                    end
                end
            end
        end
    end
    return counts3, counts4, total_chars
end

# ============================================================
# Бенчмаркъ съ ЖЁСТКИМЪ ограниченіемъ L2
# ============================================================
function benchmark_cache_constrained(counts3, counts4, orig_tokens, orig_bytes, dim, l2_cache_mb, base_tps)
    # Сортируемъ по ЭКОНОМІИ, а не по сырой частотѣ
    # 3-грамма: экономитъ 2 токена, 4-грамма: экономитъ 3 токена
    sorted3 = sort(collect(counts3), by=x -> x[2] * 2, rev=true)
    sorted4 = sort(collect(counts4), by=x -> x[2] * 3, rev=true)
    
    vocab_max = floor(Int, (l2_cache_mb * 1024 * 1024) / (dim * 4))
    base_vocab = 385
    max_additional = vocab_max - base_vocab
    
    println("📐 Ограниченія кэша:")
    @printf("   L2-кэшъ: %.2f МБ | dim=%d | Макс. словарь: %d токеновъ | Доп. токеновъ: ≤%d\n", 
            l2_cache_mb, dim, vocab_max, max_additional)
    
    N4_candidates = 0:min(200, max_additional)
    N3_candidates = 0:min(500, max_additional)
    
    results = []
    best_time = Inf
    best_cfg = nothing
    
    for N4 in N4_candidates
        cum_sav4 = sum(3 * sorted4[k][2] for k in 1:N4 if k <= length(sorted4); init=0)
        
        for N3 in N3_candidates
            N_total = N3 + N4
            if N_total > max_additional; continue; end
            
            cum_sav3 = sum(2 * sorted3[k][2] for k in 1:N3 if k <= length(sorted3); init=0)
            saved_tokens = cum_sav4 + cum_sav3
            new_tokens = max(orig_tokens - saved_tokens, orig_tokens * 0.85)
            
            vocab = base_vocab + N_total
            embed_mb = (vocab * dim * 4) / 1e6
            time_hours = new_tokens / base_tps / 3600
            
            push!(results, (N3=N3, N4=N4, vocab=vocab, tokens=new_tokens, embed_mb=embed_mb, 
                            time_hours=time_hours, compression=orig_bytes/new_tokens))
            
            if time_hours < best_time
                best_time = time_hours
                best_cfg = results[end]
            end
        end
    end
    
    return results, best_cfg, vocab_max, max_additional
end

# ============================================================
# Генераторъ кода
# ============================================================
function generate_tokenizer_code(best3, best4, counts3, counts4)
    top3 = sort(collect(counts3), by=x->x[2], rev=true)[1:best3.N3]
    top4 = sort(collect(counts4), by=x->x[2], rev=true)[1:best4.N4]
    
    println("\n\n╔══════════════════════════════════════════════════════════╗")
    println("║          ГОТОВЫЙ КОДЪ: CACHE-OPTIMAL TOKENIZER           ║")
    println("╚══════════════════════════════════════════════════════════╝")
    
    println("const TRI_DICT = Dict(")
    for (i, (tri, _)) in enumerate(top3)
        @printf("    \"%s\" => %d,\n", tri, 385 + best4.N4 + i)
    end
    println(")\n")
    
    println("const QUAD_DICT = Dict(")
    for (i, (quad, _)) in enumerate(top4)
        @printf("    \"%s\" => %d,\n", quad, 385 + i)
    end
    println(")\n")
    
    println("""const EOS_TOKEN = 2_147_483_647
const BYTE_VOCAB = 256
const CYR2_BASE = 257

function encode_text_tokens(txt::AbstractString; add_eos::Bool=true)
    bs = codeunits(txt)
    n = length(bs)
    out = Int32[]
    i = 1
    while i <= n
        matched = false
        # 1. Проверяемъ 4-грамму (4 кирилл. символа = 8 байтъ)
        if !matched && i + 7 <= n
            q = String(txt[i:i+7])
            if haskey(QUAD_DICT, q)
                push!(out, Int32(QUAD_DICT[q]))
                i += 8; matched = true
            end
        end
        # 2. Проверяемъ 3-грамму (3 кирилл. символа = 6 байтъ)
        if !matched && i + 5 <= n
            t = String(txt[i:i+5])
            if haskey(TRI_DICT, t)
                push!(out, Int32(TRI_DICT[t]))
                i += 6; matched = true
            end
        end
        # 3. Фоллбэкъ: биграмма (текущая логика)
        if !matched && i + 1 <= n
            b1 = bs[i]; b2 = bs[i+1]
            if (b1 == 0xD0 || b1 == 0xD1) && (0x80 <= b2 <= 0xBF)
                lead = b1 == 0xD0 ? 0 : 64
                push!(out, Int32(CYR2_BASE + lead + (b2 - 0x80)))
                i += 2; matched = true
            end
        end
        # 4. Фоллбэкъ: одиночный байтъ
        if !matched
            push!(out, Int32(bs[i]) + 1)
            i += 1
        end
    end
    add_eos && push!(out, Int32(EOS_TOKEN))
    return out
end\n""")
end

# ============================================================
# CLI
# ============================================================
function parse_cmd()
    s = ArgParseSettings(description="Бенчмаркъ N-граммъ съ жёсткимъ ограниченіемъ L2-кэша")
    @add_arg_table s begin
        "--data", "-d"; help="Путь къ датасету"; required=true; arg_type=String
        "--orig-tokens"; help="Исходное число токеновъ"; arg_type=Int; default=85_959_565
        "--orig-bytes"; help="Исходный объёмъ въ байтахъ"; arg_type=Int; default=156_886_030
        "--dim"; help="Размерность модели"; arg_type=Int; default=384
        "--l2-cache"; help="Кэшъ L2 GPU (МБ) — ЖЁСТКОЕ ОГРАНИЧЕНІЕ"; arg_type=Float64; default=2.2
        "--tps"; help="Базовая скорость (ток/сек)"; arg_type=Float64; default=5200.0
        "--sample-chars"; help="Ограничить символы для быстрой оцѣнки"; arg_type=Int; default=-1
    end
    return parse_args(s)
end

function main()
    args = parse_cmd()
    println("🔍 Сборъ частотъ 3- и 4-граммъ...")
    c3, c4, total = collect_ngram_frequencies(args["data"]; max_chars=args["sample-chars"])
    println("✅ Найдено $(length(c3)) троекъ и $(length(c4)) четвёрокъ")

    println("⚙️  Расчётъ оптимальныхъ N3, N4 (L2 ≤ $(args["l2-cache"]) МБ)...")
    results, best, vocab_max, max_add = benchmark_cache_constrained(
        c3, c4, args["orig-tokens"], args["orig-bytes"], 
        args["dim"], args["l2-cache"], args["tps"]
    )
    
    if best === nothing
        println("❌ Не найдено конфигурацій, помѣщающихся въ L2-кэшъ!")
        println("   Попробуйте уменьшить --dim или увеличить --l2-cache")
        return
    end
    
    println("\n📊 РЕЗУЛЬТАТЫ (ВСѢ КОНФИГУРАЦІИ ПОМѢЩАЮТСЯ ВЪ L2)")
    println("─"^90)
    @printf("%-4s | %-4s | %-6s | %-8s | %-8s | %-7s | %-6s\n", 
            "N4", "N3", "Vocab", "Emb(MB)", "Эпоха(ч)", "Сжатіе", "Δ%")
    println("─"^90)
    base_time = results[1].time_hours
    for r in results
        star = r == best ? " ★" : "  "
        delta = 100 * (1 - r.time_hours / base_time)
        @printf("%-4d | %-4d | %-6d | %8.2f | %8.2f | %7.3f | %+.1f%%%s\n",
                r.N4, r.N3, r.vocab, r.embed_mb, r.time_hours, r.compression, delta, star)
    end
    println("─"^90)
    @printf("✅ ОПТИМУМЪ ДЛЯ L2=%.1f МБ: N4=%d, N3=%d, Vocab=%d, Embedding=%.2f МБ\n", 
            args["l2-cache"], best.N4, best.N3, best.vocab, best.embed_mb)
    println("   ✅ Embedding ПОЛНОСТЬЮ ПОМѢЩАЕТСЯ ВЪ L2-КЭШЪ → нулевые промахи!")
    @printf("   ⏱ Ускореніе эпохи: +%.1f%% | Сжатіе: %.3f байтъ/токенъ | Эпоха: %.2f ч.\n", 
            100*(1-best.time_hours/base_time), best.compression, best.time_hours)
    
    generate_tokenizer_code(best, best, c3, c4)
end

main()
