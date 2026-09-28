# CTC ilk-frame içerik benzerliği taraması

53 sequence içindeki 1.378 çift, kalite/PSNR/router çıktısı kullanılmadan tarandı. Önceden sabit eşik: DC hariç63-bit düşük-frekans DCT fingerprint Hamming ≤6 ve64×64 merkezlenmiş luma korelasyonu ≥0.98. Bu bir aday taraması; farklı frame, crop veya aynalama aynı kaynağı gizleyebilir. Eşiğin altında aday bulunmaması içerik bağımsızlığı kanıtı değildir.

| İlk sequence | İkinci sequence | Fold’lar | Hamming | Luma korelasyonu |
|---|---|---|---:|---:|
| videoSRC17_1920x1080_24.yuv | Kimono1_1920x1080_24.yuv | 0 / 2 | 2 | 0.980390 |
| RaceHorses_832x480_30.yuv | RaceHorses_416x240_30.yuv | 0 / 1 | 0 | 0.992669 |

Her iki aday çift karşı fold’larda. Ham dosya hash’lerinin farklı olması, aynı görsel içeriğin farklı çözünürlük veya kodlama kopyalarını dışlamıyor. RaceHorses ad ailesi ayrıca çözünürlükten bağımsız olarak aynı gruba alınmalı.

Mevcut53-sequence ölçümleri ve kontrolleri değiştirilmedi. Bunlar sequence-disjoint geliştirme-kümesi sonuçları olarak kalır; content-disjoint veya dokunulmamış dış test başarısı diye yorumlanmaz. Gelecek split, en az bu iki aday çiftin bağlı bileşenlerini birlikte tutmalı; video kökeni/temporal benzerlik denetimi de gerekir. Bu tarama outcomes sonrasında yapılan betimsel bir veri-kökeni kontrolüdür, yeni bir önceden kayıtlı performans testi değildir.

## İçerik gruplarıyla bootstrap duyarlılığı

Politika veya fold yeniden seçilmeden, 51 grubu üye sequence’leriyle birlikte 5.000 kez örnekliyoruz (seed20260928). Nokta tahmini 53 sequence’e eşit ağırlık vermeyi koruyor. Bu kontrol kalibrasyondaki içerik örtüşmesini gidermez.

| Ölçü | Ortalama | 51-gruplu %95 aralık |
|---|---:|---|
| mean_router_minus_dither_saving_points | +1.87261 | [-1.90877, +5.51836] |
| mean_router_minus_dither_cropped_rgb_loss_db | -0.00135 | [-0.01726, +0.01408] |
| q90_router_minus_dither_saving_points | +2.53690 | [-0.63945, +5.61555] |
| q90_router_minus_dither_cropped_rgb_loss_db | +0.00048 | [-0.01105, +0.01354] |
| e15_full_rgb_loss_vs_released_db | +0.00908 | [+0.00784, +0.01040] |
| repair_identity_psnr_loss_db | +0.00208 | [+0.00065, +0.00364] |
| adapters_identity_psnr_loss_db | +1.23091 | [+0.86338, +1.65070] |
| both_identity_psnr_loss_db | +1.25384 | [+0.87853, +1.68522] |

Aynı duyarlılık kontrolü kaynak-bilgili delivered-cap analizinin altı cap değerine de uygulandı. Her sequence’in beş QP’si birlikte tutuldu; mevcut aday seçimi değiştirilmedi.

| Cap | Router–dither puan farkı | 51-gruplu %95 aralık |
|---|---:|---|
| 0.05 | +2.02425 | [+1.27713, +2.77545] |
| 0.1 | +2.93449 | [+2.10650, +3.78984] |
| 0.15 | +1.98843 | [+1.47023, +2.57453] |
| 0.2 | +1.24245 | [+0.78358, +1.75952] |
| 0.3 | +0.34928 | [+0.11170, +0.65478] |
| 0.5 | +0.01969 | [+0.00000, +0.06138] |
