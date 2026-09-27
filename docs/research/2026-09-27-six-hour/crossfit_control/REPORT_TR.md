# Kaynak görüntüsüz kontrol: sequence-disjoint cross-fit teşhisi

Sabit e15 router logits ve padded tile-MSE tabloları kullanıldı. Yeni reconstruction veya hız deneyi değildir.
53 sequence hash ile beş gruba bölündü; aynı sequence'in beş QP'si aynı grupta. Her QP için kontrol diğer dört grupta seçildi.
Bu veri daha önce geliştirmede incelendiği için dış test başarısı iddia edilmiyor. Test kaynağı karar anında kullanılmıyor.

| Kalibrasyon | Hedef dB | Policy | MAC tasarrufu % | Ortalama kayıp dB | Q90 kayıp dB | Hedef aşımı /265 | Kalibrasyon uygunsuz /265 |
|---|---:|---|---:|---:|---:|---:|---:|
| mean | 0.05 | router | 10.223 | 0.0503 | 0.0769 | 122 | 32 |
| mean | 0.05 | dither | 7.200 | 0.0503 | 0.0765 | 120 | 32 |
| mean | 0.05 | uniform | 1.845 | 0.0413 | 0.0656 | 70 | 32 |
| mean | 0.10 | router | 27.379 | 0.1006 | 0.1571 | 122 | 0 |
| mean | 0.10 | dither | 24.608 | 0.0997 | 0.1533 | 123 | 0 |
| mean | 0.10 | uniform | 18.855 | 0.0747 | 0.1245 | 54 | 0 |
| mean | 0.15 | router | 34.130 | 0.1462 | 0.2466 | 106 | 0 |
| mean | 0.15 | dither | 32.200 | 0.1460 | 0.2312 | 115 | 0 |
| mean | 0.15 | uniform | 27.199 | 0.1104 | 0.1790 | 50 | 0 |
| mean | 0.20 | router | 37.701 | 0.1797 | 0.2914 | 95 | 0 |
| mean | 0.20 | dither | 36.263 | 0.1786 | 0.2926 | 102 | 0 |
| mean | 0.20 | uniform | 30.743 | 0.1328 | 0.2255 | 39 | 0 |
| mean | 0.30 | router | 39.067 | 0.2032 | 0.3401 | 40 | 0 |
| mean | 0.30 | dither | 39.125 | 0.2045 | 0.3496 | 44 | 0 |
| mean | 0.30 | uniform | 39.125 | 0.2045 | 0.3496 | 44 | 0 |
| mean | 0.50 | router | 39.067 | 0.2032 | 0.3401 | 5 | 0 |
| mean | 0.50 | dither | 39.125 | 0.2045 | 0.3496 | 5 | 0 |
| mean | 0.50 | uniform | 39.125 | 0.2045 | 0.3496 | 5 | 0 |
| q90 | 0.05 | router | 2.730 | 0.0409 | 0.0660 | 70 | 159 |
| q90 | 0.05 | dither | 0.320 | 0.0396 | 0.0636 | 67 | 159 |
| q90 | 0.05 | uniform | -0.951 | 0.0379 | 0.0628 | 61 | 159 |
| q90 | 0.10 | router | 17.586 | 0.0663 | 0.1035 | 32 | 0 |
| q90 | 0.10 | dither | 15.558 | 0.0667 | 0.1018 | 29 | 0 |
| q90 | 0.10 | uniform | 10.940 | 0.0571 | 0.0928 | 18 | 0 |
| q90 | 0.15 | router | 26.895 | 0.0983 | 0.1558 | 33 | 0 |
| q90 | 0.15 | dither | 24.380 | 0.0983 | 0.1523 | 29 | 0 |
| q90 | 0.15 | uniform | 19.742 | 0.0768 | 0.1272 | 9 | 0 |
| q90 | 0.20 | router | 31.733 | 0.1279 | 0.2054 | 30 | 0 |
| q90 | 0.20 | dither | 30.230 | 0.1306 | 0.2031 | 30 | 0 |
| q90 | 0.20 | uniform | 23.063 | 0.0911 | 0.1520 | 10 | 0 |
| q90 | 0.30 | router | 37.155 | 0.1755 | 0.2844 | 16 | 0 |
| q90 | 0.30 | dither | 36.051 | 0.1754 | 0.2844 | 18 | 0 |
| q90 | 0.30 | uniform | 33.162 | 0.1498 | 0.2568 | 8 | 0 |
| q90 | 0.50 | router | 39.067 | 0.2032 | 0.3401 | 5 | 0 |
| q90 | 0.50 | dither | 39.125 | 0.2045 | 0.3496 | 5 | 0 |
| q90 | 0.50 | uniform | 39.125 | 0.2045 | 0.3496 | 5 | 0 |

Ortalama hedef kontrolü tek görüntü garantisi değildir. Q90 kontrolü de %100 güvence sağlamaz; başarısız kalibrasyonlar filtrelenmedi.
Eş MAC veya eş gerçekleşen kalite karşılaştırması yapılmadan tasarruf farkı doğrudan router üstünlüğü diye okunmamalı.
Sonraki aşama: kontrol değerlerini dondurup final mixed reconstruction üzerinde, eğitimde/kalibrasyonda görülmeyen ayrı veriyle ölçmek.
