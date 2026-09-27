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
