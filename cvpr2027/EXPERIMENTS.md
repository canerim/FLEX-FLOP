# Experiment plan — DCVC-UF only

✓ = sonuç var, güncel checkpoint (RECIPE512 e15) ile · ◐ = sonuç var ama eski checkpoint / eski protokol, yeniden ölçülmeli · ○ = yapılmadı

## A. Ana sonuçlar (paper'ın gövdesi)

| # | Deney | Durum | Kaynak |
|---|---|---|---|
| A1 | RD–compute: released vs routed, 0.05–0.3 dB bütçe, CTC 53 kare × 5 qp, gerçek rANS bpp | ✓ | `flexplus/results/eval_rules_ctc_e15.json` |
| A2 | BD-rate (rate-equivalent PCHIP + klasik iki kübik) vs saving | ✓ | `docs/figures/eval2/uf_analysis.json` |
| A3 | Sınıf başına (HEVC B/C/D/E, UVG, MCL-JCV) ve qp başına | ✓ | fig7, fig11 |
| A4 | Released codec'lerle RD referansı: VTM-intra, ELIC, TCM vb. aynı eksende (DCVC-UF'un yeri + bizim nokta) | ○ | — |
| A5 | Kodak / CLIC'e transfer (image benchmark) | ◐ | `results/kodak_*`, ablations.json `kodak` |

## B. Baseline'lar (hakemlerin ilk soracağı)

| # | Deney | Durum |
|---|---|---|
| B1 | Uniform (tek exit tüm kare) — içerikten bağımsız | ✓ |
| B2 | Dither (Bayer, içerikten bağımsız, aynı ortalama maliyet) | ✓ |
| B3 | Aynı MAC'te sabit küçük decoder: kanal-dar / blok-az DCVC-UF, aynı adım sayısıyla distill | ◐ (`rd_baseline_singleexit_ep0`, `static_*`) |
| B4 | Tek-exit'li ağlar ayrı ayrı eğitilmiş (ladder'ın paylaşımlı ağırlık maliyeti) | ○ |
| B5 | Literatürden: ClassSR-tarzı ayrı dallar, slimmable/width-scalable decoder | ◐ (`classsr_router`) / ○ |

## C. Yöntem ablasyonları

| # | Soru | Durum | Kaynak |
|---|---|---|---|
| C1 | Exit sayısı K (3 / 6 / 12) | ◐ (FLEX-FLOP) | config docstring |
| C2 | Split depth j (tam-kare stem uzunluğu) | ◐ | `seam_vs_split`, ablations `clamp` |
| C3 | Karo boyutu (64 / 128 / 256 px) | ◐ | `granularity*`, `iso_*` |
| C4 | Halo: latent halo 0/1/2, trunk halo 0 vs clamped (0 ULP, ×2.25) | ◐ | `ceiling_e15_halo*`, `exact_tiling*` |
| C5 | Grid seam repair açık/kapalı, gate init | ✓/◐ | `seam_audit_ctc`, `seam_gate`, `seam_module` |
| C6 | Tam-kare head vs karolu head | ◐ | `head_term*` |
| C7 | Adapter: yok / 1×1 / 3×3 / closed-form refit | ◐ | `adapter_ablation`, ablations `calex_refit` |
| C8 | Eğitim: deploy edilen karolu yoldan vs tam-kare; rastgele karo derinliği vs uniform | ◐ | model.py notları |
| C9 | Padding modu | ◐ | `padding_ablation` |
| C10 | Epoch / eğitim süresi | ◐ | ablations `epochs` |

## D. Exit seçimi

| # | Soru | Durum |
|---|---|---|
| D1 | Oracle vs router vs dither vs uniform, bütçe süpürme | ✓ |
| D2 | Router girdileri (latent / qp / scales / stem) ve modelleri | ✓ (`router_ablation_*`) |
| D3 | Mean-budget vs per-frame guarantee | ✓ |
| D4 | Harita kodlama maliyeti (KT-4), bit/kare | ✓ |
| D5 | Oracle arama: bisection adımı vs kalite vs encoder süresi | ✓ (runtime2) |
| D6 | Oracle ile router arasındaki boşluk: nerede kaybediliyor (sınıf, qp) | ◐ |

## E. Sistem

| # | Deney | Durum |
|---|---|---|
| E1 | Encoder süresi R1–R5, 1080p, 53×5 | ✓ |
| E2 | Decoder duvar saati vs çözünürlük (crossover 720p) | ✓ |
| E3 | Batch, CPU, güç/enerji | ◐ (`supp_latency_*`, `supp_power`) |
| E4 | 4K çözünürlük | ○ |
| E5 | Farklı GPU (A6000 dışında) | ○ |
| E6 | Bit-exactness 0 ULP, TF32 kapalı | ✓ |

## F. Kalite / görsel

| # | Deney | Durum |
|---|---|---|
| F1 | Exit haritası örnekleri + kırpıntılar | ✓ (fig6, `supp_exitmap_*`) |
| F2 | Seam görünürlüğü, en kötü karolar | ◐ (`supp_seam_problem_*`) |
| F3 | Algısal metrik: MS-SSIM / LPIPS kaybı aynı bütçede | ◐ (`budget_perceptual`) |
| F4 | Güven aralıkları (kare bazlı bootstrap), seed tekrarı | ○ |

## Öncelik (yeni iş)

1. A4 — released codec'lerle aynı RD grafiği
2. B3/B4 — aynı MAC'te sabit decoder baseline'ı
3. C-grubunun e15 ile tek tabloda yeniden ölçümü (K, j, karo, halo, seam, head, adapter)
4. F4 — güven aralıkları
5. E4 — 4K
