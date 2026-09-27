# Önemli metrik düzeltmesi: eski `db_rgb` RGB değil

Shared-exit değerlendirmesinin kaynak yolu incelendi: CTC okuyucusu YCbCr
4:2:0 görüntüyü 4:4:4'e yükseltiyor ve 0,5 çıkarıyor. Decoder aynı uzayda
çıktı veriyor. `db_rgb` hesaplanırken RGB dönüşümü yapılmıyor; üç kanalın
eşit ağırlıklı, clipping öncesi MSE'leri oranlanıyor.

Dolayısıyla doğru tanım **YCbCr 4:4:4 MSE-oran kaybı**, yani
`10 log10(MSE_mixed / MSE_reference)`. Padded `M` ve `R` tabloları da
aynı uzayda. Ayrı `db_611` ise 4:2:0 plane PSNR'larının 6:1:1 ağırlığı.

İki gerçek QP32/router haritası CPU'da tekrar çalıştırıldı. Checkpoint
strict olarak yüklendi; eksik veya atlanan öğrenilmiş tensor yok.

| Kare | Arşiv `db_rgb` | CPU YCbCr444 | Açık dönüşüm ve clipping sonrası RGB |
|---|---:|---:|---:|
| BasketballPass | 0,06475068 | 0,06475041 | 0,05785548 |
| BQMall | 0,07743738 | 0,07743635 | 0,07739626 |

Eski alanı yeniden üreten doğrudan YCbCr hesabıdır. Makaledeki eski RGB
etiketleri bu nedenle düzeltiliyor; ham sayı ve anahtarlar korunuyor.
İki kontrol, 265 çiftin tamamı için yeni RGB sonuç üretmez. Metrik
değişince cap'i sağlayan adaylar da değişebilir; eski 2,93 MAC-puanı sonucu
RGB sınırı altında ölçülmüş gibi sunulamaz.

Yeni bağımsız D2/D4/D6/released D12 değerlendirmesi açık RGB dönüşümü
kullanıyor ve ayrı protokolle çalışıyor. Bu düzeltme, o deneyin `psnr_rgb`
alanının yeniden adlandırılmasını gerektirmiyor. İki deneyin değerleri
birleştirilmeyecek.

Kod, kaynak frame byte'ları ve ağırlık hash'leri `analysis.json` içinde.
Orijinal working-tree değişikliklerine dokunulmadı; e15 replay için
`9e17209` commit'inin kodu ayrı klasöre arşivlendi. Eski DCVC kaynağı
`819c219b24db34310bbd15c51a720aaaf5eb2e7d`, temiz kaynak kontrolü yapıldı.
