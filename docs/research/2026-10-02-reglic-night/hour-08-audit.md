# RegLIC gece deneyi: sekizinci saat kanıt denetimi

2 Ekim 2026, yaklaşık 11:38 Europe/Berlin. Bu kayıt mevcut eğitim ve doğrulama dosyalarından CPU ile çıkarıldı; yeni GPU işi başlatılmadı. `night_monitor.py` sekiz saatlik salt-okunur dönem boyunca 15 dakikalık durum örnekleri ve saatlik özetler yazdı. Bu izleme literatür okuması veya deney yürütmesi değildir; gece boyunca yalnız ilk 20 makalelik literatür turu tamamlandı.

| Derinlik | Güncel durum | Son görülen epoch | Sayısal skip | Son adım süresine göre kalan süre |
|---:|---|---:|---:|---:|
| D2 | tamam | 105 | 0 | — |
| D4 | tamam | 105 | 0 | — |
| D6 | çalışıyor | 99 | 0 | yaklaşık 16 saat |
| D8 | çalışıyor | 17 | 0 | **en az** yaklaşık 81 saat |
| D10 | çalışıyor | 12 | 0 | **en az** yaklaşık 86 saat |
| D12 | GPU 7'de çalışıyor | 7 | 0 | **en az** yaklaşık 95 saat |

Tahmin `kalan batch × son pencere saniye/batch` hesabıdır; D8/D10/D12 henüz 90. epoch'ta başlayan 512 piksel fazına ulaşmadığı için bu üç süre **alt sınır gibi** okunmalı, bitiş taahhüdü değildir. D6 zaten 512 piksel fazında ve yaklaşık 0,41 saniye/adım görüyor. Tüm derinlikler aynı 24.049 batch/epoch tarifindedir.

## D6 için erken uyarı: yalnız sağlık doğrulaması

DIV2K doğrulamanın ilk **dört** görüntüsündeki merkez 512 kırpımları, D6 epoch 99 ile D4 epoch 105 arasında görüntü başına log(tahmini bpp)–PSNR interpolasyonuyla kıyaslandı. 0,2 tahmini bpp'de D6−D4 ortalaması RGB **−0,0194 dB**, YUV 6:1:1 **−0,0082 dB** (n=4). 0,1 bpp'de sırasıyla +0,0096 ve +0,0150 dB (n=4); 0,4 bpp'de −0,0349 ve −0,0221 dB (n=3 ortak destek). Görüntü düzeyindeki işaretler karışık. Bunlar gerçek bitstream değil, checkpoint seçiminde kullanılmayan dört sağlık kırpımıdır; farklı epoch'lar ve örnek azlığı nedeniyle D6'nın nihai faydasına ilişkin çıkarım yapılamaz.

D6 tamamlandığında otomatik 24 görüntü × 5 QP tam çözünürlük Kodak değerlendirmesi beklenmeli ve D2/D4 ile eş tahmini bit hızında, görüntü eşlemeli kıyas yapılmalı. D4−D2 final Kodak'ta 0,2 bpp'de +0,1175 dB RGB bulunmuştu; D6'nın dört kırpımdaki küçük/negatif farkı bu sonucun doğrudan devamı sayılmaz. D6−D4 Kodak farkı sıfıra yakın veya negatif çıkarsa kapasite–kalite hikâyesini ve router için beklenen altı uzman değerini yeniden sınamak gerekir. D6 bitmeden ve Kodak kapsamı oluşmadan bunu makaleye sonuç olarak eklememek doğru.

## Yeni yakın literatür kontrolü

[MoECodec (arXiv:2606.21033)](https://arxiv.org/abs/2606.21033), komşu token'ların aynı uzmana gitmesini teşvik etmek için uzamsal total-variation cezası ve hafif bir expert-choice routing kullanıyor. Görevi insan + makine algısı için transformer tabanlı codec ve bizim bağımsız DCVC-UF derinlik bankamızla aynı düzenek değil; yine de ileride router eğitildiğinde **uzamsal tutarlılık cezası açık/kapalı** ablasyonu için yakın öncül. Bu deneyi mevcut ana eğitimlerden önce başlatmak veya router performansı varmış gibi yazmak için gerekçe sağlamıyor.
