# D4 gerçekten ayrı bir seçenek olarak değer katıyor mu?

Bu analiz, aynı payload rate'e interpolate edilen epoch20 D2/D4/D6 sonuçlarından bir allocation üst sınırı çıkarır. Seçim biriminin512×512 crop olması nedeniyle bunu görüntü-içi patch router sonucu saymıyoruz.
Amaç ortalama MSE; gösterilen pooled-crop PSNR, önceki mean-image PSNR tablolarıyla aynı istatistik değildir.

| Payload bpp | n | Ortalama en fazla4 blokta D4 seçenek değeri (dB) | Aynı bütçede placement premium (dB) |
|---:|---:|---:|---:|
| 0.1 | 95 | 0.01321 | 0.02392 |
| 0.2 | 99 | 0.00857 | 0.03025 |
| 0.4 | 93 | 0.00611 | 0.02994 |

D4 seçenek değeri: D2/D6 ile ulaşılabilen en iyi sonuç ile D2/D4/D6 optimumu arasındaki fark. Placement premium: aynı evaluation cohort'unda en iyi histogramın kör permütasyon beklentisi ile kaynak bilgili optimum arasındaki fark.
Her iki kontrol de aynı izin verilen toplam MAC bütçesini kullanır; kullanılmayan bütçe zorla tüketilmez. Kör histogramın kendisi kaynakla kalibre edilir, dolayısıyla deployed kontrol değildir.
Gerçek MLP eğitimi, test ayrımı, patch entropy reset/seam maliyetleri ve latency bu analizde yok. Sonuç, sonraki deneyin kapasite seçeneklerini gerekçelendirmek içindir.
