# Aynı derinlik haritasında bölge birleştirme

16 görüntü ×5 QP ×10 sabit profil. D2/D6 epoch20, yüzde50/yüzde50 piksel dağılımı. Aynı haritanın dört parçalı ve iki birleşik bölgeli uygulamaları; CPU FP32 FUFBNK1 araştırma bitstream'i.

| Harita | Payload bpp | Ortak görüntü | Birleştirme RGB PSNR farkı (dB) |
|---|---:|---:|---:|
| vertical | 0.1 | 13/16 | +0.52038 |
| vertical | 0.2 | 16/16 | +0.53274 |
| vertical | 0.4 | 16/16 | +0.56885 |
| horizontal | 0.1 | 13/16 | +0.50746 |
| horizontal | 0.2 | 16/16 | +0.49589 |
| horizontal | 0.4 | 16/16 | +0.51792 |

Tamamlayıcı iki faz görüntü içinde ortalanır;32 bağımsız örnek gibi sayılmaz. Header dahil oranlar ve PCHIP duyarlılığı analysis.json içinde. Checkerboard profilleri korunur, fakat birleşebilir komşuları olmadığı için merged kontrastı yoktur.
Bu, adaptif seçimin başarısını veya runtime kazancını ölçmez. Context ve entropy reset sayısı birlikte değişir. Connected-region uygulaması önceki çalışmalarda vardır; tek başına yenilik iddiası değildir.
