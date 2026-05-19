# train_nanogpt_golf.jl
Готовый скрипт NanoGPT-Golf + NorMuon проверен на T400 (безтензорный чип, как у популярной gtx1650).
Поддерживает дореформенную кириллицу (Byte-level, hybrid: 256 raw bytes + 128 fused Cyrillic UTF-8 pairs).
Автоматически распознает JSONL или Plain Text.
Поддерживает чтение из bash-pipes: <(zstdcat ...)

openai/parameter-golf tricks +NorMuon +FlashAttention via NNkernels.jl +Byte-level UTF-8 tokenizer (works for prereform Cyrillic)
https://github.com/openai/parameter-golf Находки топовые из не требовательных к железу портированы на Julia, оптимизатор: 
https://github.com/ShizukaKuze/NorMuon

GPU-poor optimized, e.g. 4GB GPU (no tensor cores 30 watt chip)
<img width="1116" height="535" alt="image" src="https://github.com/user-attachments/assets/2b7eb2be-c4d6-467a-9f11-d97402b71877" />

Пример запуска:
```
 time julia -t3 claude-opus-4-7-thinking.nanogpt_golf_v5_5.jl --data ../gemini31-pro-preview_corpus_cleaner/"tr0-9.awk500.clean.window500--min-valid484.52670=7.2%.txt" --layers 11 --dim 384 --heads 6 --kv-heads 3 --ff-mult 3   --seq 512 --batch 4 --accum 16   --iters 25000 --attn flash   --ckpt-dir ckpt_11l384_date189.yat7   --sample-tokens 200   --ema-beta 0.997  --sample-every-steps 100000 --grad-clip  2.5 --lr 0.0035 --warmup 1200 --lr-T0 6000 --wd 0.007 --z-loss 0.0003 --sample-topk 40 --sample-topp 0.9 --sample-temp 0.8 --resume ckpt_11l384_date189.yat7/latest_good.jld2
z-loss active with coef=0.0003


✅ Internal preflight: all required functions are defined
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃ Hardware & Cache hierarchy (detected)
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU: NVIDIA T400 4GB  (cc 7.5)
  VRAM total: 3.63 GiB  ████████████████████████████
  VRAM free : 3.07 GiB  ████████████████████████░░░░
  L2 cache  : 512.00 KiB
  SM count  : 6

CPU caches: (unknown)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃ Tokenizer (vocab=385)
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  hybrid: 256 raw bytes + 128 fused Cyrillic UTF-8 pairs (D0/D1 80..BF) + EOS
  log(VOCAB) ≈ 5.9532
Resuming from: ckpt_11l384_date189.yat7/latest_good.jld2
Resumed at step=1480  best_loss=4.9248047
Loading resume weights compatibly...
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃ Run configuration (final)
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Params: 14.75M (29.51 MB FP16)
Config: 11L d=384 heads=6 kv=3 head_dim=64 seq=512 batch=4 accum=16 attn=flash vocab=385
LR: base=0.0035 min=1.00e-05 sched=cosine_restarts T0=6000 Tmult=2.0 warmup=1200
FP16: true
Loss: nll_loss + z-loss=0.0003
Sampling: top-k=40 top-p=0.90 T=0.80
QAT: OFF

Autotune: applied=false
step  1490/25000 │ loss 4.9410 │ ema 4.9468 │ bpb 7.128 │ lr 1.75e-04 │ rb 1.00 │ gnorm 3.77→2.50 │    227 tok/s │ VRAM 0.02 GB free
  💾 ckpt saved: ckpt_11l384_date189.yat7/latest.jld2 │ good=yes │ best=4.9248
step  1500/25000 │ loss 4.9442 │ ema 4.9465 │ bpb 7.133 │ lr 1.75e-04 │ rb 1.00 │ gnorm 3.79→2.50 │    257 tok/s │ VRAM 0.05 GB free
  💾 ckpt saved: ckpt_11l384_date189.yat7/latest.jld2 │ good=yes │ best=4.9248
step  1510/25000 │ loss 4.9380 │ ema 4.9450 │ bpb 7.124 │ lr 1.75e-04 │ rb 1.00 │ gnorm 3.98→2.50 │    257 tok/s │ VRAM 0.05 GB free
  💾 ckpt saved: ckpt_11l384_date189.yat7/latest.jld2 │ good=yes │ best=4.9085
step  1520/25000 │ loss 4.9225 │ ema 4.9409 │ bpb 7.102 │ lr 1.75e-04 │ rb 1.00 │ gnorm 3.91→2.50 │    260 tok/s │ VRAM 0.05 GB free
  💾 ckpt saved: ckpt_11l384_date189.yat7/latest.jld2 │ good=yes │ best=4.9085
...
```
 
```usage: claude-opus-4-7-thinking.nanogpt_golf_v5_5.jl --data DATA
                        [--save SAVE] [--ckpt-dir CKPT-DIR]
                        [--resume RESUME]
                        [--ckpt-every-steps CKPT-EVERY-STEPS]
                        [--keep-last KEEP-LAST] [--attn ATTN]
                        [--layers LAYERS] [--dim DIM] [--heads HEADS]
                        [--kv-heads KV-HEADS] [--ff-mult FF-MULT]
                        [--seq SEQ] [--batch BATCH] [--accum ACCUM]
                        [--iters ITERS] [--lr LR] [--lr-min LR-MIN]
                        [--lr-scheduler LR-SCHEDULER] [--lr-T0 LR-T0]
                        [--lr-Tmult LR-TMULT] [--warmup WARMUP]
                        [--grad-clip GRAD-CLIP]
                        [--resume-warmup RESUME-WARMUP]
                        [--resume-lr-scale RESUME-LR-SCALE]
                        [--min-lr-scale MIN-LR-SCALE]
                        [--lr-backoff-factor LR-BACKOFF-FACTOR]
                        [--wd WD] [--muon-beta MUON-BETA]
                        [--muon-beta2 MUON-BETA2]
                        [--muon-ns-steps MUON-NS-STEPS]
                        [--log-every LOG-EVERY]
                        [--sample-every-steps SAMPLE-EVERY-STEPS]
                        [--sample-tokens SAMPLE-TOKENS] [--seed SEED]
                        [--dry-run] [--print-hw] [--quick-check]
                        [--probe-tokenizer] [--sample-greedy]
                        [--sample-topk SAMPLE-TOPK]
                        [--sample-topp SAMPLE-TOPP]
                        [--sample-temp SAMPLE-TEMP]
                        [--min-healthy-loss MIN-HEALTHY-LOSS]
                        [--max-healthy-loss MAX-HEALTHY-LOSS]
                        [--loss-spike-factor LOSS-SPIKE-FACTOR]
                        [--loss-ema-beta LOSS-EMA-BETA]
                        [--param-check-every PARAM-CHECK-EVERY]
                        [--max-param-abs MAX-PARAM-ABS]
                        [--bad-step-patience BAD-STEP-PATIENCE]
                        [--bad-sample-patience BAD-SAMPLE-PATIENCE]
                        [--abort-on-bad-sample]
                        [--min-space-ratio MIN-SPACE-RATIO]
                        [--max-top-token-ratio MAX-TOP-TOKEN-RATIO]
                        [--max-prefix-ratio MAX-PREFIX-RATIO]
                        [--max-repeat-run MAX-REPEAT-RUN]
                        [--min-unique-ratio MIN-UNIQUE-RATIO]
                        [--min-sample-entropy MIN-SAMPLE-ENTROPY]
                        [--min-bigram-diversity MIN-BIGRAM-DIVERSITY]
                        [--min-trigram-diversity MIN-TRIGRAM-DIVERSITY]
                        [--rollback-on-bad-step] [--stop-on-collapse]
                        [--autotune] [--no-autotune]
                        [--autotune-benchmark]
                        [--no-autotune-benchmark]
                        [--autotune-max-seq AUTOTUNE-MAX-SEQ]
                        [--autotune-max-batch AUTOTUNE-MAX-BATCH]
                        [--autotune-max-global-scale AUTOTUNE-MAX-GLOBAL-SCALE]
                        [--autotune-vram-reserve-gb AUTOTUNE-VRAM-RESERVE-GB]
                        [--autotune-candidates AUTOTUNE-CANDIDATES]
                        [--autotune-bench-iters AUTOTUNE-BENCH-ITERS]
                        [--autotune-bench-runs AUTOTUNE-BENCH-RUNS]
                        [--loader-cache-frac LOADER-CACHE-FRAC]
                        [--loader-min-mb LOADER-MIN-MB]
                        [--loader-max-mb LOADER-MAX-MB] [--qat]
                        [--qat-bits-start QAT-BITS-START]
                        [--qat-bits-final QAT-BITS-FINAL]
                        [--qat-alpha-mid QAT-ALPHA-MID]
                        [--qat-start-step QAT-START-STEP]
                        [--qat-warmup QAT-WARMUP] [--qat-freeze-alpha]
                        [--qat-alpha-target QAT-ALPHA-TARGET]
                        [--qat-scale-update-every QAT-SCALE-UPDATE-EVERY]
                        [--qat-per-row] [--qat-per-tensor]
                        [--ema-beta EMA-BETA] [--fp16] [--no-fp16]
                        [--z-loss Z-LOSS] [-h]

NanoGPT-Golf v5.5  (vocab=385 hybrid tokenizer)

optional arguments:
  --data DATA
  --save SAVE            (default: "model_golf_v5_5.jld2")
  --ckpt-dir CKPT-DIR    (default: "checkpoints_v55")
  --resume RESUME        (default: "")
  --ckpt-every-steps CKPT-EVERY-STEPS
                        (type: Int64, default: 10)
  --keep-last KEEP-LAST
                        (type: Int64, default: 6)
  --attn ATTN            (default: "flash")
  --layers LAYERS       (type: Int64, default: 5)
  --dim DIM             (type: Int64, default: 384)
  --heads HEADS         (type: Int64, default: 6)
  --kv-heads KV-HEADS   (type: Int64, default: 3)
  --ff-mult FF-MULT     (type: Int64, default: 3)
  --seq SEQ             (type: Int64, default: 512)
  --batch BATCH         (type: Int64, default: 4)
  --accum ACCUM         (type: Int64, default: 16)
  --iters ITERS         (type: Int64, default: 25000)
  --lr LR               (type: Float64, default: 0.0006)
  --lr-min LR-MIN       (type: Float64, default: 1.0e-5)
  --lr-scheduler LR-SCHEDULER
                        (default: "cosine_restarts")
  --lr-T0 LR-T0         (type: Int64, default: 4000)
  --lr-Tmult LR-TMULT   (type: Float64, default: 2.0)
  --warmup WARMUP       (type: Int64, default: 500)
  --grad-clip GRAD-CLIP
                        (type: Float64, default: 1.0)
  --resume-warmup RESUME-WARMUP
                        (type: Int64, default: 300)
  --resume-lr-scale RESUME-LR-SCALE
                        (type: Float64, default: 0.25)
  --min-lr-scale MIN-LR-SCALE
                        (type: Float64, default: 0.05)
  --lr-backoff-factor LR-BACKOFF-FACTOR
                        (type: Float64, default: 0.5)
  --wd WD               (type: Float64, default: 0.01)
  --muon-beta MUON-BETA
                        (type: Float64, default: 0.95)
  --muon-beta2 MUON-BETA2
                        (type: Float64, default: 0.95)
  --muon-ns-steps MUON-NS-STEPS
                        (type: Int64, default: 5)
  --log-every LOG-EVERY
                        (type: Int64, default: 10)
  --sample-every-steps SAMPLE-EVERY-STEPS
                        (type: Int64, default: 200)
  --sample-tokens SAMPLE-TOKENS
                        (type: Int64, default: 140)
  --seed SEED           (type: Int64, default: 1337)
  --dry-run
  --print-hw
  --quick-check
  --probe-tokenizer
  --sample-greedy
  --sample-topk SAMPLE-TOPK
                        (type: Int64, default: 40)
  --sample-topp SAMPLE-TOPP
                        (type: Float64, default: 0.9)
  --sample-temp SAMPLE-TEMP
                        (type: Float64, default: 0.8)
  --min-healthy-loss MIN-HEALTHY-LOSS
                        (type: Float64, default: 0.03)
  --max-healthy-loss MAX-HEALTHY-LOSS
                        (type: Float64, default: 200.0)
  --loss-spike-factor LOSS-SPIKE-FACTOR
                        (type: Float64, default: 1.35)
  --loss-ema-beta LOSS-EMA-BETA
                        (type: Float64, default: 0.98)
  --param-check-every PARAM-CHECK-EVERY
                        (type: Int64, default: 25)
  --max-param-abs MAX-PARAM-ABS
                        (type: Float64, default: 200.0)
  --bad-step-patience BAD-STEP-PATIENCE
                        (type: Int64, default: 1)
  --bad-sample-patience BAD-SAMPLE-PATIENCE
                        (type: Int64, default: 1)
  --abort-on-bad-sample
                        
  --min-space-ratio MIN-SPACE-RATIO
                        (type: Float64, default: 0.01)
  --max-top-token-ratio MAX-TOP-TOKEN-RATIO
                        (type: Float64, default: 0.65)
  --max-prefix-ratio MAX-PREFIX-RATIO
                        (type: Float64, default: 0.85)
  --max-repeat-run MAX-REPEAT-RUN
                        (type: Int64, default: 96)
  --min-unique-ratio MIN-UNIQUE-RATIO
                        (type: Float64, default: 0.02)
  --min-sample-entropy MIN-SAMPLE-ENTROPY
                        (type: Float64, default: 1.4)
  --min-bigram-diversity MIN-BIGRAM-DIVERSITY
                        (type: Float64, default: 0.15)
  --min-trigram-diversity MIN-TRIGRAM-DIVERSITY
                        (type: Float64, default: 0.1)
  --rollback-on-bad-step
                        
  --stop-on-collapse
  --autotune
  --no-autotune
  --autotune-benchmark
  --no-autotune-benchmark
                        
  --autotune-max-seq AUTOTUNE-MAX-SEQ
                        (type: Int64, default: 4096)
  --autotune-max-batch AUTOTUNE-MAX-BATCH
                        (type: Int64, default: 32)
  --autotune-max-global-scale AUTOTUNE-MAX-GLOBAL-SCALE
                        (type: Float64, default: 1.5)
  --autotune-vram-reserve-gb AUTOTUNE-VRAM-RESERVE-GB
                        (type: Float64, default: 0.8)
  --autotune-candidates AUTOTUNE-CANDIDATES
                        (type: Int64, default: 10)
  --autotune-bench-iters AUTOTUNE-BENCH-ITERS
                        (type: Int64, default: 3)
  --autotune-bench-runs AUTOTUNE-BENCH-RUNS
                        (type: Int64, default: 3)
  --loader-cache-frac LOADER-CACHE-FRAC
                        (type: Float64, default: 0.25)
  --loader-min-mb LOADER-MIN-MB
                        (type: Int64, default: 4)
  --loader-max-mb LOADER-MAX-MB
                        (type: Int64, default: 64)
  --qat
  --qat-bits-start QAT-BITS-START
                        (type: Int64, default: 8)
  --qat-bits-final QAT-BITS-FINAL
                        (type: Int64, default: 6)
  --qat-alpha-mid QAT-ALPHA-MID
                        (type: Float64, default: 0.5)
  --qat-start-step QAT-START-STEP
                        (type: Int64, default: -1)
  --qat-warmup QAT-WARMUP
                        (type: Int64, default: 500)
  --qat-freeze-alpha
  --qat-alpha-target QAT-ALPHA-TARGET
                        (type: Float64, default: 1.0)
  --qat-scale-update-every QAT-SCALE-UPDATE-EVERY
                        (type: Int64, default: 16)
  --qat-per-row
  --qat-per-tensor
  --ema-beta EMA-BETA   (type: Float64, default: 0.999)
  --fp16
  --no-fp16
  --z-loss Z-LOSS       (type: Float64, default: 0.0)
  -h, --help
  ```
## 🔍 Сравненіе токенизаторовъ: Гибридный (vocab=385) vs. Mainstream

| Характеристика | **Гибридный** (vocab=385) | **BPE** (GPT-2/Llama) | **Qwen** (tiktoken) | **RuAdapt** (LLaMa-ru) |
|:---|:---|:---|:---|:---|
| **Размѣръ словаря** | 385 | ~50,000 | ~151,643 | ~32,000–128,000 |
| **Базовые единицы** | Байты + слитыя кириллическія пары | Байты + частыя под-словныя послѣдовательности | Байты + многоязычныя единицы | Байты + русскія под-слова |
| **Сжатіе (русскій текстъ)** | **1.825 байтъ/токенъ** | ~2.5–3.0 | ~2.0–2.5 | ~1.9–2.2 |
| **Экономія vs побайтовый** | **45.2%** | ~25–35% | ~30–40% | ~35–45% |
| **Дореформенная орѳографія** | ✅ Отличная (автосліяніе `0xD0/0xD1` паръ) | ⚠️ Частичная (разбивка по байтамъ) | ⚠️ Частичная | ⚠️ Частичная |
| **OOV токены** | ❌ Невозможенъ | ❌ Невозможенъ | ❌ Невозможенъ | ❌ Невозможенъ |
| **Скорость кодированія** | ⚡ Очень высокая (~90 MB/s CPU) | 🐌 Средняя | 🐌 Низкая | 🐌 Средняя |
| **Память (embedding, d=384)** | ~0.6 MB | ~77 MB | ~230 MB | ~49 MB |
| **Лучшій сценарій** | Малые модели, дореформенный текстъ, CPU/слабый GPU | Универсальныя LLM | Многоязычныя коммерческія модели | Русскоязычныя адаптаціи большихъ моделей |

> **Примѣчаніе:** Гибридный токенизаторъ оптимизированъ для историческихъ и дореформенныхъ текстовъ. За счётъ принудительнаго сліянія UTF-8 паръ кириллицы (`D0/D1 80..BF`) достигается минимальный размѣръ словаря и высокая скорость обработки безъ потери покрытия (OOV = 0).


Data-driven perf:
```
julia -t3 find_optimal_ngrams_cache_constrained.jl -d my.jsonl --dim 384 --l2-cache 2
🔍 Сборъ частотъ 3- и 4-граммъ...
✅ Найдено 42150 троекъ и 294498 четвёрокъ
⚙️  Расчётъ оптимальныхъ N3, N4 (L2 ≤ 2.0 МБ)...
📐 Ограниченія кэша:
   L2-кэшъ: 2.00 МБ | dim=384 | Макс. словарь: 1365 токеновъ | Доп. токеновъ: ≤980

📊 РЕЗУЛЬТАТЫ (ВСѢ КОНФИГУРАЦІИ ПОМѢЩАЮТСЯ ВЪ L2)
──────────────────────────────────────────────────────────────────────────────────────────
N4   | N3   | Vocab  | Emb(MB)  | Эпоха(ч) | Сжатіе  | Δ%    
──────────────────────────────────────────────────────────────────────────────────────────
0    | 0    | 385    |     0.59 |     4.59 |   1.825 | +0.0%  
0    | 1    | 386    |     0.59 |     4.56 |   1.838 | +0.7%  
...
200  | 498  | 1083   |     1.66 |     3.90 |   2.147 | +15.0%  
200  | 499  | 1084   |     1.67 |     3.90 |   2.147 | +15.0%  
200  | 500  | 1085   |     1.67 |     3.90 |   2.147 | +15.0%  
──────────────────────────────────────────────────────────────────────────────────────────
✅ ОПТИМУМЪ ДЛЯ L2=2.0 МБ: N4=0, N3=41, Vocab=426, Embedding=0.65 МБ
   ✅ Embedding ПОЛНОСТЬЮ ПОМѢЩАЕТСЯ ВЪ L2-КЭШЪ → нулевые промахи!
   ⏱ Ускореніе эпохи: +15.0% | Сжатіе: 2.147 байтъ/токенъ | Эпоха: 3.90 ч.


╔══════════════════════════════════════════════════════════╗
║          ГОТОВЫЙ КОДЪ: CACHE-OPTIMAL TOKENIZER           ║
╚══════════════════════════════════════════════════════════╝
const TRI_DICT = Dict(
    "ост" => 386,
    "ств" => 387,
    "ені" => 388,
    "омъ" => 389,
    "енн" => 390,
    "аго" => 391,
    "ихъ" => 392,
    "про" => 393,
    "при" => 394,
    "что" => 395,
    "ста" => 396,
    "тор" => 397,
    "нія" => 398,
    "ніе" => 399,
    "етъ" => 400,
    "льн" => 401,
    "раз" => 402,
    "сто" => 403,
    "тел" => 404,
    "ест" => 405,
    "ать" => 406,
    "ото" => 407,
    "ыхъ" => 408,
    "оль" => 409,
    "акъ" => 410,
    "пре" => 411,
    "его" => 412,
    "оро" => 413,
    "ель" => 414,
    "нно" => 415,
    "сти" => 416,
    "тся" => 417,
    "ред" => 418,
    "ова" => 419,
    "нос" => 420,
    "емъ" => 421,
    "лен" => 422,
    "как" => 423,
    "овъ" => 424,
    "ані" => 425,
    "пол" => 426,
)

const QUAD_DICT = Dict(
)

const EOS_TOKEN = 2_147_483_647
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
end
```
