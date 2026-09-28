# Altı saatlik DCVC-UF araştırma oturumu

Başlangıç: 27 Eylül 2026 15:37:26 UTC / 17:37:26 Berlin.
En erken bitiş: 21:37:26 UTC / 23:37:26 Berlin.

## 15:37 UTC — başlangıç

Mevcut üç eğitim ve dondurulmuş resmî tarif korunuyor. İlk inceleme,
nihai ölçümün önündeki en kritik açığın küçük derinliklerde gerçek
bitstream decode yolu olduğunu gösteriyor. Mevcut otomatik değerlendirme
entropy tahmini ve neural modül süresi üretiyor; bunlar gerçek coded
byte veya uçtan uca süre değil. İlk çalışma bu açığın CPU üzerinden
doğrulanabilecek kısmına ve eş epoch/rate kalite analizine odaklanıyor.

## 16:05 UTC — ilk doğrulamalar ve geniş validation başlangıcı

- 18/19/20. epoch kayıtları aynı adım ve crop/QP üzerinden arşivlendi. Epoch20
  üç modelin FP32 checkpoint bütünlüğü ve beklenen parametre/blok sayısı doğrulandı.
- Yeni öğrenme analizi 61 kaynak dosyanın hash'ini kaydediyor. Aynı QP32'de
  D6–D2 farkı 0,0711 dB; görüntü bazında D2/QP32 bitrate'ine eşlenince 0,1896 dB.
  Sabit 0,2 bpp'de fark 0,2811 dB. Bu iki eşleme farklı hedefler kullanıyor.
  Lineer/PCHIP duyarlılığı en fazla 0,0304 dB; istatistiksel güven aralığı değil.
  İki figür görsel olarak incelendi; kaynak ve rapor `learning_epoch020/` altında.
- Gerçek rANS payload kullanan CPU FP32 araştırma codec'i eklendi. Decode,
  yalnız bitstream ve checkpoint alıyor; encoder'ın indekslerine ihtiyaç duymuyor.
  Modelin kaynak-görüntü analiz yolları bağımsız decoder sürecinde yasaklandı.
- 36/36 mühendislik vakası geçti: D2/D4/D6 ve released D12, QP0/32/63,
  64² / 65×97 / 512² boyutlar. Bağımsız süreçte latentler, entropy indeksleri
  ve reconstruction tam aynı; resmî FP32 forward ile de tam eşleşme var.
  Bozuk/yanlış modele ait container ve int8 taşmaları reddediliyor.
- İnceleme sırasında 1×1 hyperlatent tensor stride'ının CPU convolution
  seçimini ve yuvarlama hatasını değiştirdiği bulundu. Decoder'da açık
  contiguous-format clone ile düzeltildi ve 36 vaka yeniden çalıştırıldı.
- Format `FUFREF1`, 88 byte başlık içeriyor; bunun 64 byte'ı kimlik/bütünlük hash'i.
  Released CUDA formatı veya CPU/GPU bit-exact uyumluluk iddiası yok.
- Tüm 100 DIV2K validation görüntüsünün merkez512 RGB crop'u hash'leriyle
  hazırlandı. 16:05 UTC'de epoch20 D2/D4/D6 + released D12 için 2000 gerçek
  encode/bağımsız decode değerlendirmesi CPU'da başlatıldı (PID3192869).
  İlk vakalar geçti. Payload, container ve entropy tahmini ayrı tutuluyor.
- Eğitimler canlı; D2/D4 tamamlanan22, D6 tamamlanan20 epoch. Nonfinite=0,
  watcher alarmı yok. GPU'lara ek iş verilmedi, eğitim kodu değiştirilmedi.

Sunucudaki ham kanıt kökü:
`/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research/`.
Doğrulama: `reference_verification_epoch020/verification.json`.
Geniş validation: `div2k100_reference_epoch020/progress.json` ve case/stream dosyaları.

## 16:24 UTC — kaynak görüntüsüz kalibrasyon ve patch kontrolü

- Sabit router/dither/uniform için 53 sequence'in beş QP'sini birlikte tutan
  beş katlı kontrol kalibrasyonu tamamlandı. 9.540 held-out policy sonucu,
  tüm kontrol değerleri ve gruplar `crossfit_control/analysis.json` içinde.
  Router'ın 0,1 dB ortalama hedefi 122/265 aşım üretirken Q90 hedefinde 32/265
  aşım var; karşılık gelen MAC tasarrufu %27,379 ve %17,586. Bu, kaynak-MSE
  tablo teşhisi; final mixed reconstruction veya dış test iddiası değil.
- İki yeni vector figür seti üretildi. Ortalama ile kare-başına sınır
  arasındaki fark `CALIBRATION_RESEARCH_TR.md` içinde deney kararı olarak
  yazıldı. Conformal risk control, Learn then Test ve 2026 nonmonotonic-risk
  çalışmaları birincil kaynaklardan kontrol edildi. Henüz garanti teoremi
  veya yeni yöntem başarısı iddia edilmiyor.
- 100 görüntü validation için analiz aracı hazırlandı: actual payload /
  entropy estimate / container ayrı, per-image ortak rate desteği,
  lineer–PCHIP duyarlılığı, ortak PSNR aralığında BD-rate ve görüntü bazlı
  paired bootstrap. Analitik %10 rate ölçekleme ve destek dışı reddetme
  kontrolleri geçti. Tam 2.000 vaka olmadan analiz başlamıyor.
- Sabit-depth patchleme deneyi hazırlandı. Önceden belirlenen 16 görüntüde
  full512 / dört256 / halo32 karşılaştırması yapılacak; D2/D6/releasedD12,
  beş QP. 256 ve 288 girişleri D2/QP32'de bağımsız decoder preflight'ını
  geçti. Seam-mask alanı ve sabit-hata metriği analitik doğrulandı.
  Yoğun CPU kullanımı nedeniyle deney geniş validation sonrasına bırakıldı.

## 16:29 UTC — ilk yayın kaydı ve ortak epoch21

Araştırma araçları ve ilk dört figür seti `e8e60f9` commit'iyle
`canerim/FLEX-FLOP` deposunun `flex` dalına gönderildi; uzak SHA doğrulandı.
Makale ekine sequence-disjoint kalibrasyon bölümü ve figürü eklendi.
Ana PDF 8 metin + 1 kaynakça sayfasında kaldı; ek 12 sayfa olarak
derlendi. Undefined reference veya overfull box yok. Figürlerin altı
PDF/SVG/PNG çıktısı araştırma klasörüyle byte düzeyinde eşleşiyor.

D6 epoch21 validation'ını tamamladı; `watch/epoch021_matched_observation.json`
arşivlendi. Ortak 0,2 tahmini bpp'de D2/D4/D6 =
34,02897 / 34,22206 / 34,25315 dB (dört crop). D6−D2 = 0,22418 dB.
Geniş validation başlangıçta sabitlenen epoch20 ile devam ediyor; yeni
checkpoint'e geçilerek farklı epoch'lar karıştırılmıyor.

## 16:38–16:58 UTC — yayın doğrulaması ve native yürütme hazırlığı

- Makale `ece71676b7bc2eb8d618eb632ec9ce8f7f98a961` subtree commit'iyle
  `canerim/cvpr2027:main` dalına gönderildi. Araştırma deposu `3b258874`.
  Uzak SHA'lar kontrol edildi. Temiz git archive kopyasında ana makale ve
  ek yeniden derlendi; PDF metinleri ve yeniden üretilen kalibrasyon
  figürlerinin hash'leri eşleşti. Overleaf arayüzünde pull yapılmış olduğu
  iddia edilmiyor; GitHub tarafı güncel.
- Geniş validation tamamlanınca analiz, kaynak-feature ilişkisi ve patch
  kontrolünü sırayla çalıştıran CPU kuyruğu başlatıldı. Kod hash'leri
  sabitlendi; hata halinde kuyruk durur. Eğitim GPU'larına dokunmaz.
- 100 crop için önceden belirlenen RGB std, gradient ve Laplacian enerji
  özellikleri çıkarıldı. Kaliteyle ilişki analizi henüz çalışmadı; ileride
  üretilecek ilişki yalnız keşifsel olacak, router başarısı sayılmayacak.
- Bağımsız decoder doğruluğunu anlatan vektör figür görsel olarak kontrol
  edildi; ilk sürümdeki yazı çakışması giderildi. 36/36 taze süreç testi
  figürde açıkça CPU FP32 ve özel research formatı olarak etiketli.
- Native decoder'ın dört ayrı yerde 12 bloğu sabitlediği doğrulandı.
  Ayrı kaynak kopyası için 2/4/6/8/10/12 blok desteği patch'i hazırlandı;
  CPU C++ anahtar kontrolünde altı geçerli derinlik ve 58 hatalı state
  sınandı. Henüz CUDA doğruluğu veya hız sonucu yok.
- Released `compress` reconstruction üretirken CPU entropy coding ile
  synthesis'i örtüştürüyor; encoder toplam süresi parça sürelerinin
  toplamından çıkarılamaz. Native buffer'lar batch1; expert batching
  mevcut işlev olarak sunulamaz. Bulgular encoder muhasebesine eklendi.
- Microsoft'un belirttiği CUTLASS v4.4.1 ayrı dizine indirildi
  (`4370102f9dacab813282e1d67722fceb0b90a019`). SM86 için GPU sorgusuz,
  MAX_JOBS=1 ve nice19 ile ayrı native derleme başlatıldı. Kurulum yok;
  CUDA12.1 derleyici/PyTorch cu126 farkı kaydedildi. Eğitim checkout'u
  ve ortamı değişmedi. Derleme başarısı GPU parity sayılmayacak.
- 16:56 UTC: D2 step580177 (24 epoch tamam), D4 step561528 (23 tamam),
  D6 step519430 (21 tamam); üçü canlı, nonfinite0, alarm yok.

## 17:16 UTC — ortak epoch22 ve aritmetik payda düzeltmesi

- Ortak epoch22 validation arşivlendi. Dört monitor crop'ta0,2 tahmini bpp
  için D2/D4/D6 =33,99101 /34,20393 /34,35236dB; D6−D2 =0,36135dB.
  Eğitimler sağlıklı; geniş gerçek-stream değerlendirmesi sabit epoch20'de.
- Altı mimarinin Conv2d MAC izi meta tensor ile çıkarıldı. D2/64×64 izi
  gerçek CPU forward'ıyla katman bazında birebir eşleşti; spatial prior'ın
  üç tekrarının tümü sayıldı.512×512'de sabit neural entropy recovery
  31,079GMac. D12→D2 synthesis azalması%74,53; recovery dahil neural
  decoder azalması%48,21. Encoder+reconstruction Conv2d azalması daha küçük.
  Bu oranların hiçbiri wall-time ölçümü değil.
- Makalenin eski `decoder MAC` etiketleri, gerçek kapsam olan `synthesis
  MAC` olarak düzeltildi. Ölçülmüş arşiv sayıları değişmedi. Entropy recovery
  ağlarının ana saving paydasına dahil olmadığı yöntem bölümünde açıklandı.
  Yeni iki panelli vektör figür supplement'e eklendi; ana makale8+1 sayfa,
  ek12 sayfa. Render'da görülen alt satır kırpma izi düzeltildi.
- D4 seçeneğini kaldıran, aynı interpolated payload rate ve toplam neural
  decoder MAC bütçesinde çalışan whole-crop allocation analizi hazırlandı.
  Exact dynamic programming, üç görüntüde exhaustive aramayla doğrulandı.
  Henüz veri analizi çalışmadı; bu kaynak bilgili üst sınırdır, MLP veya
  görüntü-içi routing sonucu değildir.

## 17:30–17:36 UTC — shared-exit renk uzayı hatası bulundu ve doğrulandı

Eski `eval_rules_ctc_e15` içindeki `db_rgb` alanı RGB dönüşümü yapmıyor.
CTC okuyucusunun centred YCbCr4:4:4 çıktısından doğrudan MSE alınıyor;
`flexuf.eval` içindeki kaynak tabloları da aynı uzayda. Böylece önceki
makaledeki RGB kaybı etiketinin yanlış olduğu saptandı.

İki QP32 sabit router haritası, `9e17209` arşiv kodu ve temiz eski DCVC
`819c219b` ile CPU'da yeniden çalıştırıldı; checkpoint strict yüklendi.
BasketballPass ve BQMall'ın YCbCr444 kayıpları arşivdeki `db_rgb` alanını
sırasıyla2,69e-7 ve1,04e-6dB farkla yeniden üretti. Açık RGB dönüşümü
farklı değer veriyor. Kanıt `shared_metric_audit/analysis.json` içinde.

Ana metin ve13 temel vektör figürün renk uzayı etiketleri düzeltildi;
ham gözlemler değişmedi. `METRICS.md`, eski JSON isimlerinin düzeltilmiş
anlamını kaydediyor. İki replay, tüm265 çiftin RGB değerlendirmesi diye
sunulmuyor. Yeni bağımsız depth validation'ının explicit RGB dönüşümü
zaten doğru; bu iki deneyin metrikleri karıştırılmıyor.

Bu düzeltme öncesindeki maliyet ara sürümü araştırma `9e17209`, makale
`db3cca5` olarak GitHub'a gönderildi ve uzak makale SHA'sı doğrulandı.
Renk uzayı düzeltmesi takip eden commit ile yayımlanacak.

## 17:43–17:45 UTC — metric correction and native compilation

- The corrected manuscript builds to 8 body pages plus references; supplement 12 pages. All 52 bundled vector-figure artifacts reproduce byte-for-byte; PDF/bundle verification passes. Reviewed rendered pages and delivered-quality figure.
- Isolated SM86 native depth extension compiled successfully at 17:36:56 UTC (exit 0), without GPU use or modifying the training source/environment. GPU numerical parity and latency remain pending.
- Expanded epoch-20 reference validation reached 1,400/2,000 cases at 17:42:56 UTC. No partial-cohort performance conclusion is drawn.

## 17:47–17:52 UTC — fixed-control reconstruction replay

- Started CPU-only QP32 actual mixed-image replay for all53 archived CTC first frames, mean/Q90 cross-fit calibration × router/dither/uniform (318 policy cases). The nominal target is fixed at0.1dB before replay.
- Every map was frozen from disjoint-sequence controls and stored logits before any source image was opened; map construction also passes with M/R fields removed. Padded table loss and MAC accounting reproduce the stored cross-fit outcomes. No threshold filtering or per-source fallback.
- Replay uses the immutable9e17209 shared decoder snapshot and strict e15 checkpoint loading; reports both corrected444-MSE and explicitRGB loss plus the separate611 metric. This is a neural reconstruction diagnostic, not a new bitstream or runtime result.
- Metric-corrected publication pushed and remote-verified at4489e75cdf74799acf8961f81a5fb618f7b1d34d.
- Added plotting scripts for the pending complete patch-control and whole-crop allocation analyses; no synthetic outcomes plotted.

## 17:53–17:56 UTC — before-outcome halo64 amendment

- Found a geometry-specific opportunity: on the2×2 grid, halo32 yields288-square windows padded to320; halo64 yields320-square windows without padding. Both have1.5625×full512 coded area, but different useful context and hyperlatent-grid alignment. No equal-rate or equal-runtime claim follows.
- No patch output directory existed. Stopped only the owned waiting research queue after checking it had no child/completed commands; archived its state and original preflight. The2000-case reference evaluation and three trainings were untouched.
- Added halo64 to the still-unstarted patch protocol. Analysis will retain common support across full/halo0/halo32/halo64 and show counts. Original primary geometry rationale remains documented.
- Fresh engineering preflight independently decoded256/288/320 D2/QP32 streams exactly, checked seam areas/constant MSE and full stitching coverage; all passed. This smooth-input engineering check is not image-quality evidence.
- Restarted the sequential queue with the new evaluator hash4cfea6600f0280ec41a405a7c82c02d568e803db143f7a78563a0c4069db6d60.
- Started separate unmodified D12 native compilation with identical setup/toolchain/SM86 flags to support later stock-versus-patched GPU parity. NoGPU queried or used.

## 17:58–18:03 UTC — geometry and representation-boundary audit

- Verified the encoder's structural source support: one latent position depends on a136-pixel interval[16i−64,16i+71]. Seven depthwise3×3 blocks plus stride2 projection after unshuffle8 give this bound; a realCPU gradient probe reproduces bbox[32,167] in both axes. This is not the full-codec or trained effective receptive field.
- Exact area-scaled Conv2d accounting verified by meta traces at256/320 forD2/D12. Four halo32/64 D2 patches use80.92% of full512 D12 neural-decoder MACs, but100.60% of encoder-with-reconstruction MACs. D6 uses111.05%/122.86%. These are architectural ratios, not matched-quality results or wall time.
- Produced and visually reviewed the padding/context/MAC figure; corrected dimension-marker and data-label overlaps.
- Related-work reinspection identified an important qualification: Spatial Competition (arXiv2605.13243v1,section2.2) processes same-mode connected regions continuously. Our fixed-depth patch reset control deliberately forces separate streams and must not be treated as an unavoidable cost of every possible bank implementation. Region coalescing is a required additional control, not a novel concept claimed here.

## 18:05–18:21 UTC — resource contention and native padding distinction

- GPU6 gained another user's9.6GiB process at17:39:06UTC. D6 median window time increased121.8→179.2ms across fixed before/after windows (+47.2% step duration); D2/D4 increased10.0%/5.3%. This is an observational association, not randomized causal proof. No foreign process was modified; incident data archived.
- Important scope correction: native Microsoft test_video requests image padding to16, while DMCIProxy pads only the hyperanalysis latent to4. FUFREF1 CPU research coding pads the whole image to64. The earlier equal-area halo32/64 calculation applies toFUFREF1, not native halo32. Updated research documentation and figure scope. The completed512 reference cohort is64-aligned, so this distinction does not change its geometry.
- Added layer-by-layer native-shaped Conv2d traces and a realCPU D2/288 trace match. Native-shaped D2 halo32 costs67.82% neural-decoder and83.22% encoder-with-reconstruction MACs relative to full512 D12; D6 costs92.22%/101.25%. No CUDA timing or matched-quality claim.
- Derived separateFUFREF2 sources without editing any runningFUFREF1 file: image-pad16 and replicate latent-pad4. Passed45/45 fresh-process decoding cases acrossD2/D6/releasedD12,3 QPs and5 geometries;27 aligned cases also reproduceFUFREF1 payload and reconstruction exactly. This is CPU FP32 engineering evidence only.
- Queued a paired288-window padding-policy evaluation on the identical240-case patch cohort, to run after the primaryCPU-pad64 study. It reuses aligned full/core/halo64 evidence and measures only the changedhalo32 path; no image selection from outcomes.

## 18:27 UTC — compilation pair and common epoch23

- The unmodified stockD12 extension also compiled successfully with exactly the same isolated setup/toolchain/CUTLASS/SM86 configuration as the patched extension. Both binaries remain uninstalled; GPU parity and timings are still pending.
- Training snapshot: D2 step627675 (completed26 epochs), D4 step608226 (completed25), D6 step556328 (completed23). All live, no nonfinite skips or watcher alerts. Archived the common completed epoch23 validation.
- The large reference study reached1891/2000; actual-image shared-control replay reached25/53 sequences. Native-padding paired evaluator is waiting for the primary patch study.

### 18:50 UTC — Complete 2,000-case actual-byte depth evaluation

All 100 DIV2K centre512 crops × four depths × five QPs completed at18:37:09UTC. Exact independent-process reconstruction/latent/entropy-trace checks passed. On common actual-payload support at0.2bpp (99images), D4−D2=0.120711dB[0.101843,0.142020], D6−D2=0.183445[0.157211,0.211492], releasedD12−D2=0.698583[0.598878,0.811246]. These are epoch20/105 shallow models against a separately trained released anchor, not final or causal depth-only gaps. Full curves/coverage/BD-rate, feature associations and exact whole-crop allocation analyses are complete; all prespecified combinations retained. At a uniform-D4 arithmetic budget, middle-expert option value is0.01321/0.00857/0.00611dB at0.1/0.2/0.4bpp; this is an optimistic whole-crop diagnostic, not a spatial MLP result.

The sequential queue completed reference and feature analysis, then the primary patch run stopped on case0801/QP16. The metric recomputation reduced channels then space whereas the frozen reference reduced all elements together. Independent re-decoding established identical latent and a−7.077e−7dB FP32 reduction-order difference. The corrected direct reduction matches the reference exactly, without relaxing parity checks. The failed1-case output and its manifest remain archived under `patch_control_epoch020_reduction_order_attempt`. The dependent native-padding waiter also stopped as designed and its failed state was archived. Both corrected scripts restarted at18:39UTC; no codec/reference/training code changed. Primary patch PID3737251 has passed beyond the failed case; native-padding waits on its completion. Regression proof is in `patch_reduction_fix/analysis.json`.

New measured plots have been visually reviewed and made portable. Main paper now distinguishes interim actual-byte evidence from final pending bank evaluation. Supplement is16pages; main remains8body+references. Bundle checks pass with23vector figure sets. Standalone figure regeneration is running. No new GPU inference; official training remains healthy.

##2026-09-28 00:03–00:43UTC — recovery and figure extension

The prior CPU research processes were absent at00:02UTC after records stopped near18:58UTC. This gap is not continuous researcher activity. Official GPU training remained live. After checking process identity, the three predeclared CPU jobs resumed from immutable completed cases in detached sessions. The fixed-map region bank was queued after a10-profile independent-decoder preflight passed. The user then extended work until at least03:06:18UTC and requested redesign of every figure except Figure1.

Figure1 is protected by SHA25645f15d5951b72c1ac8da6c34fd6593d271615a1922b7c3ebfc22b331fec09cf8. A new conceptual Figure2 uses built-in image generation with two saved prompts; the final edit moves beta after MLP scores and retains all four exit-to-canvas paths. Quantitative figures remain generated from recorded data. Common typography, palette, compact canvases and layered vector glyphs were applied to all23 vector figure sets. The initial77-artifact clean reproduction was byte-identical; final small label/connectivity edits are being rechecked. Main text remains8pages plus references; the compact supplement is15pages.

Training at00:23UTC: D2step821267/currentepoch35; D4step795618/currentepoch34; D6step664724/currentepoch28; no nonfinite values or alerts. Four-crop health readings remain separate from full100-image epoch20 actual-payload evidence. The region analyzer averages complementary phases within image and requires the complete80×10 profile grid before producing comparisons.

##2026-09-28 00:48–00:54UTC — two complete outcome studies

Primary patch control completed240/240 with independent stream-decoder checks. AtD6/0.2payloadbpp, n16, core-only−0.06110dB, halo32−1.29140dB, halo64−1.66658dB versus same-model full512. SameQP32 halo32 adds39.72% payload with+0.00198dB; halo64 adds57.62% with−0.00374dB. Header-inclusive comparisons are separate and may have different common support. This exposes a context-transmission cost, not adaptive-router performance. Native-padding prerequisite automatically began at00:48:42UTC; region merge remains queued.

Fixed-control actual image replay completed53sequences/318policycases atQP32. Mean router/dither saving25.675/23.802%, RGB loss0.08679/0.08814dB; Q90saving16.741/14.204%, RGBloss0.05684/0.05636. Paired saving CIs include zero under both controls:1.87pp[−1.60,5.19],2.54pp[−0.42,5.54]. No resolved positive routing premium or equal-quality frontier is claimed. Actual padded-table discrepancy maxabs0.00039555dB. Allcases retained, development-corpus scope.

Region rectangular-cost audit also completed, two real-CPU/meta checks passed. Samepixeldepths with4→2regions reduces neuraldecoderMACs12.43835%, encoder+reconstruction11.98843%; no latency claim. Native correctness harness CPU preflight passed4strictmodels with CUDAuninitialized; GPU branch remains untested and was not launched.

Firstfigurepublication: parentd7b33e0, public6dcb7da6d87a21468848fbabc6ba27115a2c3e85, both remote-confirmed. Clean archive builds and checks passed. A subsequent review found two obsolete LaTeX trim settings from the old canvases; they are removed in the pending result update.

## 28 September, 01:20–01:32 UTC: completed controls and publication update

All 240 native-shaped padding cases completed. D6 at 0.2 payload bpp, all16 common images: moving image padding to the latent improves RGB PSNR by0.26522dB [0.23540,0.29515]; the native-shaped halo32 path remains1.02617dB below full-frame D6. AtQP32 it reduces payload by6.0279% versus image-pad64. No CUDA output or latency equivalence is asserted.

Fresh CPU router inference completed all53 QP32 sequences: both frozen mean/Q90 controls reproduce all1765 tile decisions; maximum absolute log-probability difference from the historical GPU archive is0.0226574. Executed Conv2d/Linear router cost is289.71484375MAC/padded pixel, excluding non-convolutional work and source encoding.

The supplementary actual-reconstruction section reports unresolved paired router–dither margins, independent-region RD costs, padding location, and source-derived native stream dependencies. Main abstract/intro/conclusion explicitly distinguish source-calibrated allocation from the unresolved cross-fit margin. Obsolete LaTeX figure trims removed; an old Figure2/example cross-reference corrected to the actual depth-map figure. Native correctness matrix extended to43cases per model including288×512/512×288; CPU checkpoint/CDF/binary preflight rerun, no GPU execution.

Fixed-map region coalescing started after the completed prerequisite. A separately predeclared53-frame/QP32 frozen-component2×2 intervention is running on CPU: original, repair identity, adapters identity, both identity, with unchanged Q90 router maps. It verifies original-output agreement and exact cached repair/head replay. This is not a retrained component ablation; no partial-outcome claims made.

Portable figure suite:29 vector sets,99 byte-identically reproduced PDF/SVG/PNG artifacts and two separately identified AI illustrations. Figure1 checksum remains protected. Main9pages including references; supplementary pagination is checked after each content update.

## 28 September, 01:53–02:11 UTC: region execution and component sensitivity

The fixed-map D2/D6 region study completed all 800 outputs. With identical pixel-depth assignments and halo32, coalescing four regions into two improves matched-payload RGB PSNR at 0.2 bpp by 0.53274 dB [0.43913, 0.62911] for vertical and 0.49589 [0.41207, 0.57485] for horizontal layouts (16 images, phases averaged within image). At QP32, mean payload changes are −13.0531% / −12.3941%, while PSNR changes are −0.000447 / −0.0009999 dB. Container overhead falls from 380 to 196 bytes. This is a real-byte CPU research-format result, not a native latency result.

The 53-frame fixed-Q90 component experiment completed 212 reconstructions. Removing repair loses 0.002084 dB [0.000600, 0.003628]; adapters 1.23091 dB [0.87647, 1.65824]; both 1.25384 [0.88940, 1.69151]. All maps and weights stay fixed. Eight deepest-only frames remain included and their inactive-adapter output is exactly unchanged. The experiment measures checkpoint sensitivity, not retrained architectural necessity. Mean boundary/interior relative MSE increases for repair are 0.192%/0.039%; the effect is small and does not supply a runtime justification.

A separate immutable epoch30 validation clone passed 36/36 pipeline checks using epoch20 weights explicitly as an engineering canary. At 02:08:39 UTC, a finite CPU-only queue (PID895582) began waiting for all three fixed epoch30 snapshots and their completed milestone validation files. D2/D4 are ready; D6 is not. It will verify epoch30 independently, then evaluate all 100 images × 5 QPs × 4 models with actual bytes and a source-free decoder. No quality-based checkpoint selection, GPU allocation, training change or automatic publication. The queue expires after 48 hours waiting and stops on errors.

## 28 September, 02:15–02:19 UTC: released anchor and typography

The complete released-reference replay finished at 02:14:55 UTC: all 53 QP32 frames and 318 fixed-policy outputs. Stock released D12 mean cropped RGB PSNR is 34.71763 dB; e15 full-frame is 34.70855 dB. Paired e15 loss is 0.009080 dB [0.007851, 0.010335]. All 255 front-end/entropy tensors match before latent reuse, and stock/remapped released synthesis passed the exact canary. Under the released RGB anchor, mean-control router/dither loss is 0.09587/0.09722 dB (20/20 frames above 0.1); Q90 is 0.06592/0.06544 dB (7/5 above). Same outputs and costs, no map reselection; reference shifts cannot change paired policy differences. The shared replicate-to256 CPU contract is not native pad16 CUDA output.

The new mechanism illustration was reviewed at printed manuscript width. Its first typography enlargement lost one shared trunk slab and was rejected. A local correction restored exactly one upsampling slab plus four shared trunk slabs; all four two-block increments and exit-to-canvas routes remain intact. Final file is copied directly from built-in image generation, with all prompts and SHA provenance retained. Figure1 remains byte-identical. A stale caption referring to a removed third panel in the delivered-cap plot was corrected after checking all caption panel letters against figure layout records.

Common epoch29 health observations were archived without inference: at 0.2 estimated bpp over four fixed crops D2/D4/D6 are 34.19700/34.34945/34.40186 dB. These are not the 100-image actual-byte results. All official jobs remain live and nonfinite counts are zero.

## 28 September, 02:28–02:32 UTC: shared-exit research-stream preflight

A separately predeclared two-source/QP32 engineering preflight completed all8 cases at02:29:18UTC. FUFEXIT1 carries valid dimensions, e15 checkpoint identity and two-bit reachable-exit indices around the unmodified FUFREF1 entropy implementation. BasketballPass and BQMall each use fixed historical mean/Q90 router maps, all-deep, and cyclic6/8/10/12. A separate stream/output-path-only decoder with source analysis and hyperanalysis disabled reproduces final cropped tensors, latents, symbols and entropy indexes exactly. Seven malformed-input checks pass. Four maps per source retain identical inner entropy bytes. Payloads2198/10954bytes, maps1/2bytes, explicit research headers176bytes; this is not minimum signalling, full53-frame native bitstream coverage, autonomous selection or latency evidence. Code and protocol are frozen under `experiments/dcvcuf_shared_stream_20260928`.

The final32-vector-figure suite initially reproduced111artifacts byte-identically. Printed-page review prompted one shorter cross-fit y-axis label; the portable rerun covers that final typography. Caption panel-letter checks now cover all28 vector inclusions in the manuscript and supplement. The two-page Turkish brief builds without overfull boxes after simplifying one compound phrase.

## 28 September, 02:43–02:47 UTC: content-overlap screening

An outcome-blind first-frame audit compares all1378 pairs in the53-sequence archived corpus. Thresholds were fixed before fingerprint computation:63-bit DCT pHash Hamming<=6 and centred64×64 luma correlation>=0.98. Two candidates cross the existing sequence folds: RaceHorses832/416 (folds0/1, distance0, correlation0.992669) and videoSRC17/Kimono1 (folds0/2, distance2, correlation0.980390). Raw byte hashes are distinct. These screening flags do not establish exhaustive video-source identity; conservatively grouping both pairs yields51 proposed future content groups. Existing folds, controls and outcomes are unchanged. Main discussion and supplementary protocol now explicitly distinguish sequence separation from content independence; no positive deployment/generalization claim is strengthened by this audit.

Publication9e3e02cc95f1e711ac9a0825bc44b7e009071595 was pushed and remote-verified after a clean archive build. This subsequent scope clarification is being prepared as a follow-up publication commit; figure data and measured performance are unchanged.

The paired saving sensitivity with51 conservative content clusters retains the same point estimates and gives95% intervals[−1.9088,5.5184] and[−0.6394,5.6156] under mean/Q90. Both include zero. Five thousand group draws, seed20260928, include all members of each sampled group; no policy/fold re-selection. Component and released-anchor cluster sensitivities are also retained in the full audit.
