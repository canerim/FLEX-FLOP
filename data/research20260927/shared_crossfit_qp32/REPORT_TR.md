# Cross-fit kontrolü gerçek görüntüye taşıma

QP32, nominal0.1dB,53 CTC ilk frame; her birinde mean/Q90 × router/dither/uniform. Hiçbir case elenmedi; haritalar kaynak hata etiketleri kullanılmadan, başka sequence'lerde seçilmiş sabit kontrollerle üretildi.

| Kalibrasyon | Kural | Synthesis MAC tasarrufu % | Final444 kayıp dB | Final RGB kayıp dB | 444/RGB hedef aşımı |
|---|---|---:|---:|---:|---:|
| mean | router | 25.675 | 0.08945 | 0.08679 | 17/17 /53 |
| mean | dither | 23.802 | 0.09079 | 0.08814 | 17/16 /53 |
| mean | uniform | 19.784 | 0.07974 | 0.07753 | 15/14 /53 |
| q90 | router | 16.741 | 0.05865 | 0.05684 | 5/4 /53 |
| q90 | dither | 14.204 | 0.05779 | 0.05636 | 2/2 /53 |
| q90 | uniform | 13.027 | 0.05289 | 0.05161 | 2/2 /53 |

Bu tablo aynı gerçekleşen kaliteye eşlenmiş bir üstünlük karşılaştırması değildir. Ortalama/Q90 kalibrasyonu tek-frame garantisi değildir. Yöntem karar anında testM/R kullanmaz ama corpus daha önce geliştirmede görüldüğü için dış test başarısı iddia edilmez.
Padded released-anchor tablo hedefi ile cropped e15-reference raporlama farklıdır. Aynı sayısal0.1 eşik üzerindeki sayılar tek başına hangi farkın sorumlu olduğunu açıklamaz.
Gerçek RGB dönüşümü bu tekrarda açıkça yapılır. Eski265-pair arşivinin db_rgb alanı ise444-MSE idi; bu53-sequence/QP32 tekrarı bütün eskiRGB deneylerinin yeniden yapılması değildir.
CPU neural görüntü üretildi; entropy stream, native GPU latency veya otonom deployment benchmarkı yapılmadı. Paired bootstrap aralıkları ve bütün ham sonuçlar analysis.json içinde.
