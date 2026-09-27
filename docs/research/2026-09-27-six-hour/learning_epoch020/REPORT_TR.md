# 20. epoch: eşlenmiş derinlik ve bitrate analizi

Bu sonuçlar dört sabit DIV2K merkez crop üzerinden, devam eden eğitim sırasında hesaplandı. Nihai kalite, codec hızı veya router başarısı değildir.

| Model | 0,1 tahmini bpp | 0,2 tahmini bpp | 0,4 tahmini bpp |
|---|---:|---:|---:|
| D2 | 31.3960 | 34.0128 | 36.5449 |
| D4 | 31.5450 | 34.2327 | 36.5726 |
| D6 | 31.6194 | 34.2939 | 36.7908 |
| Released D12 | 32.2822 | 35.0098 | Kapsam dışı (3/4) |

PSNR, her crop için log(tahmini bpp) ekseninde ayrı enterpole edilip dört crop üzerinden ortalanır. Ekstrapolasyon yoktur. Released D12 eğitim geçmişi farklı olduğu için bu fark saf kapasite cezası değildir.

## Aynı QP neden yanıltıyor?

Aşağıdaki karşılaştırmada her görüntünün hedef rate değeri o görüntünün D2/QP32 bitrate'idir; tablodaki sabit 0,2 bpp karşılaştırmasıyla aynı nokta değildir.

| Fark | Aynı QP32 PSNR farkı | Rate eşleme düzeltmesi | Eş rate toplam fark |
|---|---:|---:|---:|
| D4 − D2 | 0.0662 | 0.0691 | 0.1353 |
| D6 − D2 | 0.0711 | 0.1184 | 0.1896 |

Lineer yerine PCHIP kullanıldığında incelenen altı depth–rate farkının en büyük değişimi 0.0304 dB. Bu bir güven aralığı değil, enterpolasyon duyarlılığıdır. Beş QP arasındaki ara değerler doğrudan ölçüm değildir.

## Bundan sonra

- Dört crop üzerinden nihai derinlik sıralaması veya anlamlılık iddiası kurma; görüntü bazlı farkları koru.
- Ayrı, daha geniş validation değerlendirmesinde aynı checkpoint/epoch ve aynı rate desteğini kullan.
- Gerçek entropy payload, container başlığı ve entropy tahminini ayrı raporla.
- Yakın PSNR'yi aynı çıktı/model hatası diye yorumlamadan önce blok yürütmesi ve optimizer kapsamı kontrolünü kullan; 20. epoch checkpoint bütünlük denetimi üç modelde geçti.

[Öğrenme ve crop farkları](fig_learning_matched_rate.pdf) · [QP ve enterpolasyon etkisi](fig_qp_and_interpolation.pdf)

[PCHIP yöntem kaynağı](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).

Ham hesaplar ve kaynak SHA-256 değerleri `analysis.json` dosyasındadır.
