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
