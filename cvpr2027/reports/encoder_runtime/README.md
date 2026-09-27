# Encoder and decoder runtime of exit-map selection

**Measurement-scope correction, 27 September 2026.** This is a historical
stage-accounting report. The inspected benchmark replays encoder-captured
symbols into neural decoding, estimates map length instead of parsing map
bytes, and labels the fine-tuned e15 full path as “released”. The table below
does not establish standalone bitstream-to-image latency or final quality
for each regime. See [the runtime audit](../../RUNTIME_AUDIT_TR.md) and the
current `main.tex` / `supplement.tex` for the corrected interpretation.

DCVC-UF: all 53 CTC intra frames × qp [0, 16, 32, 48, 63] (265 frame-qp), NVIDIA RTX A6000 (shared; medians). Budget 0.1 dB per frame. Full report: `report.pdf`.

| Regime | Encoder 1080p (ms) | × plain encode | extra, in synthesis passes | Decoder 1080p (ms) | of released | side info (% of bits) | saving % | worst dB | over budget |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Released (no selection) | 447 | 1.00 | 0.0 | 605 | 100 % | 0.000 | 0.0 | 0.000 | 0 |
| Oracle, as first shipped | 2,974 | 6.78 | 6.4 | 540 | 86 % | 0.066 | 29.3 | 0.139 | 1 |
| Oracle, fast | 1,030 | 2.25 | 1.4 | 540 | 86 % | 0.066 | 29.3 | 0.139 | 1 |
| Router, β fixed per qp | 447 | 1.00 | 0.0 | 543 | 88 % | 0.000 | 27.4 | 0.278 | 120 |
| Router, β searched on the table | 1,191 | 2.60 | 1.8 | 543 | 88 % | 0.014 | 27.7 | 0.139 | 2 |
| Router, β searched by decoding | 4,093 | 8.95 | 9.0 | 543 | 88 % | 0.014 | 27.6 | 0.139 | 2 |

| Image decoder, set | images | table decodes | shipped search | fast search | decodes shipped / fast | saving shipped / fast |
|---|---:|---:|---:|---:|---|---|
| NoGAN-MS, Kodak-24 | 24 | 19 | 14.1 s (101 decodes' time) | 8.1 s (58) | 19+14 / 19+1.2 | 18.2 / 15.4 |
| NoGAN-MS, CLIC-41 | 41 | 19 | 26.2 s (52 decodes' time) | 16.2 s (32) | 19+14 / 19+1.7 | 33.3 / 28.1 |
| ELIC, Kodak-24 | 24 | 23 | 17.0 s (199 decodes' time) | 11.2 s (131) | 23+14 / 23+1.1 | 14.6 / 14.4 |
| ELIC, CLIC-41 | 41 | 23 | 28.2 s (97 decodes' time) | 19.1 s (66) | 23+14 / 23+2.6 | 20.8 / 17.5 |

Figures (`figs/`, PDF + PNG): `rt_fig1_timeline_1080p`, `rt_fig2_resolution`, `rt_fig3_tradeoff`, `rt_fig4_image_decoders`, `rt_figS1_jitter`. Numbers: `runtime_numbers.json`.
Code: `flexplus/runtime_full_uf.py`, `evc/flexevc/encoder_runtime.py`; figures and report from `scratchpad/rep/runtime_full.py`, `runtime_report.py`.
