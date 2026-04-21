# train_nanogpt_golf.jl
# 
# Готовый скрипт NanoGPT-Golf + NorMuon проверен на T400 (безтензорный чип, как у популярной gtx1650).
# Поддерживает дореформенную кириллицу (Byte-level, vocab=256).
# Автоматически распознает JSONL или Plain Text.
# Поддерживает чтение из bash-pipes: <(zstdcat ...)

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
