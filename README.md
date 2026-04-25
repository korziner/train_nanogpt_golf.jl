# train_nanogpt_golf.jl
Готовый скрипт NanoGPT-Golf + NorMuon проверен на T400 (безтензорный чип, как у популярной gtx1650).
Поддерживает дореформенную кириллицу (Byte-level, vocab=256).
Автоматически распознает JSONL или Plain Text.
Поддерживает чтение из bash-pipes: <(zstdcat ...)

openai/parameter-golf tricks +NorMuon +FlashAttention via NNkernels.jl +Byte-level UTF-8 tokenizer (works for prereform Cyrillic)
https://github.com/openai/parameter-golf Находки топовые из не требовательных к железу портированы на Julia, оптимизатор: 
https://github.com/ShizukaKuze/NorMuon

GPU-poor optimized, e.g. 4GB GPU (no tensor cores 30 watt chip)
<img width="1116" height="535" alt="image" src="https://github.com/user-attachments/assets/2b7eb2be-c4d6-467a-9f11-d97402b71877" />

Пример запуска:
```
time julia -t3 gpt-5.2-search-non-reasoning.train_nanogpt_golf_v6.jl --data  <(zstdcat ultra22.zst|paste -s|sed 's/¬ //g;s/- //g;s/— //g') --layers 5 --dim 384 --heads 6 --kv-heads 3 --seq 1280 --batch 1 --accum 32   --iters 20000 --lr 0.02 --ckpt-dir ckpt_5l384_ultra22   --ckpt-every-steps 50  --attn flash 
╔══════════════════════════════════════════╗
║   NanoGPT-Golf v6.1 • FlashAttention     ║
╟──────────────────────────────────────────╢
║  Platform: CUDA (NVIDIA T400 4GB)        ║
║  VRAM:     3.44 GB free                  ║
╚══════════════════════════════════════════╝

Params: 6.74M (13.48 MB FP16)
Config: 5L d=384 heads=6 kv=3 head_dim=64 seq=1280 batch=1 accum=32 attn=flash

step    10/20000 │ loss 147.9288 │ bpb 213.416 │ lr 4.00e-04 │ gnorm 328.15 │   1669 tok/s │ VRAM 1.34 GB free
step    20/20000 │ loss 131.1444 │ bpb 189.201 │ lr 8.00e-04 │ gnorm 280.53 │   2733 tok/s │ VRAM 1.14 GB free
step    30/20000 │ loss 102.0000 │ bpb 147.155 │ lr 1.20e-03 │ gnorm 215.07 │   2668 tok/s │ VRAM 0.86 GB free
step    40/20000 │ loss 66.2956 │ bpb 95.644 │ lr 1.60e-03 │ gnorm 171.37 │   2154 tok/s │ VRAM 0.63 GB free
step    50/20000 │ loss 28.7802 │ bpb 41.521 │ lr 2.00e-03 │ gnorm 140.57 │   2409 tok/s │ VRAM 0.05 GB free
  💾 ckpt saved: ckpt_5l384_ultra22/latest.jld2 (best=17.3711)
step    60/20000 │ loss 12.9573 │ bpb 18.693 │ lr 2.40e-03 │ gnorm 161.23 │   2423 tok/s │ VRAM 0.23 GB free
step    70/20000 │ loss 9.2352 │ bpb 13.324 │ lr 2.80e-03 │ gnorm 158.43 │   2563 tok/s │ VRAM 0.23 GB free
step    80/20000 │ loss 7.7036 │ bpb 11.114 │ lr 3.20e-03 │ gnorm 160.94 │   2631 tok/s │ VRAM 0.36 GB free
step    90/20000 │ loss 6.7853 │ bpb 9.789 │ lr 3.60e-03 │ gnorm 151.44 │   2399 tok/s │ VRAM 0.37 GB free
step   100/20000 │ loss 6.5490 │ bpb 9.448 │ lr 4.00e-03 │ gnorm 160.60 │   2549 tok/s │ VRAM 0.33 GB free
  💾 ckpt saved: ckpt_5l384_ultra22/latest.jld2 (best=6.6782)
...
```
Взрываются градиеты при дефолтных стартовых --lr 0.02
Взрыв не страшен, рестартуем со здоровых чекпонтов. 
Риск взрыва сохраняю, чтобы широкими шагами быстрее обучать в начале. Затем понижаем --lr 0.004:
```
time julia -t3 train_nanogpt_golf_v6_merged.jl --data  <(zstdcat ultra22.zst|paste -s -|sed 's/¬ //g;s/- //g;s/— //g') --layers 5 --dim 384 --heads 6 --kv-heads 3 --seq 1280 --batch 2 --accum 32   --iters 25000  --grad-clip 1.0 --ckpt-dir ckpt_5l384_ultra22   --ckpt-every-steps 50  --attn flash --resume ckpt_5l384_ultra22/latest.jld2 --lr 0.004 --wd 0.01

╔══════════════════════════════════════════╗
║   NanoGPT-Golf v6.1-MERGED • Flash • QAT ║
╟──────────────────────────────────────────╢
║  Platform: CUDA (NVIDIA T400 4GB)        ║
║  VRAM:     3.03 GB free                  ║
╚══════════════════════════════════════════╝

Params: 6.74M (13.48 MB FP16)
Config: 5L d=384 heads=6 kv=3 head_dim=64 seq=1280 batch=2 accum=32 attn=flash

Resuming from: ckpt_5l384_txt-1tr_ultra22/latest.jld2
Resumed at step=6600  best_loss=1.6541209
⚠️  Optimizer reinitialized (old opt_state incompatible — safe)
step  6610/25000 │ loss 1.7954 │ bpb 2.590 │ lr 2.09e-02 │ gnorm 3.59 │   2219 tok/s │ VRAM 0.04 GB free
step  6620/25000 │ loss 1.7874 │ bpb 2.579 │ lr 2.09e-02 │ gnorm 3.47 │   2695 tok/s │ VRAM 0.03 GB free
step  6630/25000 │ loss 1.7811 │ bpb 2.570 │ lr 2.09e-02 │ gnorm 3.65 │   2277 tok/s │ VRAM 0.04 GB free
step  6640/25000 │ loss 1.7685 │ bpb 2.551 │ lr 2.09e-02 │ gnorm 3.71 │   2276 tok/s │ VRAM 0.04 GB free
step  6650/25000 │ loss 1.7774 │ bpb 2.564 │ lr 2.08e-02 │ gnorm 3.18 │   2277 tok/s │ VRAM 0.04 GB free
  💾 ckpt saved: ckpt_5l384_txt-1tr_ultra22/latest.jld2 (best=1.6541)
...
```
v6.1-MERGED • Flash • QAT • SWA/EMA, --wd и --quant int6 добавляет новая версия с полным мёржем всех фич.
```
julia -t3 train_nanogpt_golf_v6_merged.jl --help
usage: train_nanogpt_golf_v6_merged.jl --data DATA [--save SAVE]
                        [--ckpt-dir CKPT-DIR] [--resume RESUME]
                        [--ckpt-every-steps CKPT-EVERY-STEPS]
                        [--keep-last KEEP-LAST] [--attn ATTN]
                        [--layers LAYERS] [--dim DIM] [--heads HEADS]
                        [--kv-heads KV-HEADS] [--ff-mult FF-MULT]
                        [--seq SEQ] [--batch BATCH] [--accum ACCUM]
                        [--iters ITERS] [--lr LR] [--warmup WARMUP]
                        [--warmdown WARMDOWN] [--grad-clip GRAD-CLIP]
                        [--lr-muon LR-MUON] [--lr-adam LR-ADAM]
                        [--wd WD] [--muon-beta MUON-BETA]
                        [--muon-beta2 MUON-BETA2]
                        [--muon-momentum-warmup MUON-MOMENTUM-WARMUP]
                        [--muon-ns-steps MUON-NS-STEPS]
                        [--quant QUANT] [--quant-mlp QUANT-MLP]
                        [--quant-attn QUANT-ATTN]
                        [--quant-embed QUANT-EMBED]
                        [--quant-group-size QUANT-GROUP-SIZE]
                        [--qat-start-frac QAT-START-FRAC] [--swa]
                        [--swa-frac SWA-FRAC] [--swa-freq SWA-FREQ]
                        [--ema] [--ema-decay EMA-DECAY]
                        [--log-every LOG-EVERY]
                        [--sample-every-steps SAMPLE-EVERY-STEPS]
                        [--sample-tokens SAMPLE-TOKENS] [--seed SEED]
                        [--dry-run] [-h]

NanoGPT-Golf v6.1-MERGED • Flash • QAT • SWA/EMA • Byte-level

optional arguments:
  --data DATA           text/jsonl/pipe input
  --save SAVE            (default: "model_golf.jld2")
  --ckpt-dir CKPT-DIR    (default: "checkpoints")
  --resume RESUME        (default: "")
  --ckpt-every-steps CKPT-EVERY-STEPS
                        (type: Int64, default: 50)
  --keep-last KEEP-LAST
                        (type: Int64, default: 5)
  --attn ATTN           flash | naive (default: "flash")
  --layers LAYERS       (type: Int64, default: 5)
  --dim DIM             (type: Int64, default: 384)
  --heads HEADS         (type: Int64, default: 6)
  --kv-heads KV-HEADS   (type: Int64, default: 3)
  --ff-mult FF-MULT     (type: Int64, default: 3)
  --seq SEQ             (type: Int64, default: 768)
  --batch BATCH         (type: Int64, default: 1)
  --accum ACCUM         (type: Int64, default: 32)
  --iters ITERS         (type: Int64, default: 20000)
  --lr LR               (type: Float64, default: 0.02)
  --warmup WARMUP       (type: Int64, default: 500)
  --warmdown WARMDOWN   (type: Int64, default: 1500)
  --grad-clip GRAD-CLIP
                        (type: Float64, default: 1.0)
  --lr-muon LR-MUON     (type: Float64, default: 0.025)
  --lr-adam LR-ADAM     (type: Float64, default: 0.035)
  --wd WD               (type: Float64, default: 0.0)
  --muon-beta MUON-BETA
                        (type: Float64, default: 0.95)
  --muon-beta2 MUON-BETA2
                        (type: Float64, default: 0.95)
  --muon-momentum-warmup MUON-MOMENTUM-WARMUP
                        (type: Int64, default: 1500)
  --muon-ns-steps MUON-NS-STEPS
                        (type: Int64, default: 5)
  --quant QUANT         none|int8|int6|int5|int4|fp16 (default:
                        "none")
  --quant-mlp QUANT-MLP
                        override --quant для MLP (default: "")
  --quant-attn QUANT-ATTN
                        override --quant для Attention (default: "")
  --quant-embed QUANT-EMBED
                        override --quant для Embed (default: "")
  --quant-group-size QUANT-GROUP-SIZE
                        (type: Int64, default: 128)
  --qat-start-frac QAT-START-FRAC
                        (type: Float64, default: 0.85)
  --swa                 Stochastic Weight Averaging
  --swa-frac SWA-FRAC   (type: Float64, default: 0.4)
  --swa-freq SWA-FREQ   (type: Int64, default: 50)
  --ema                 Exponential Moving Average
  --ema-decay EMA-DECAY
                        (type: Float64, default: 0.997)
  --log-every LOG-EVERY
                        (type: Int64, default: 10)
  --sample-every-steps SAMPLE-EVERY-STEPS
                        (type: Int64, default: 200)
  --sample-tokens SAMPLE-TOKENS
                        (type: Int64, default: 140)
  --seed SEED           (type: Int64, default: 1337)
  --dry-run
  -h, --help
  ```
