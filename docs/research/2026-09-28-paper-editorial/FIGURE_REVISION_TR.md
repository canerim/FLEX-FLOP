# Figür 3–7 ve tablosuz makale revizyonu

- Figür 1/2 dosyaları ve LaTeX figür blokları d404b42 ile birebir aynı.
- Figür 3: gerçek 5×8 haritanın eğik özellik düzlemleri; 40/27/7/3 aktif tile, 13/20/4/3 çıkış. D12 yolu identity. 77/160 sayısı yalnız suffix tile–blok çifti işi, runtime değil.
- Figür 4a: dört politika ve altı ölçülmüş bütçe noktasının ortografik 3B gösterimi. Politika ekseni kategorik; perdeler ara ölçüm veya fit değil. Figür 4b eşleştirilmiş belirsizlik aralıklarını koruyor.
- Figür 5: eski cap tablosu gerçek kayıp–MAC düzlemine taşındı; alt panel cap boyunca router farkını koruyor.
- Figür 6: sabit sınıf kotası ve isim hash'i ile sekiz kaynak, kaynak başına altı görünüm. Üç router bütçesi, bir dithering ve bir source-search koşulu. 40 harita konumundan ikisi infeasible ve N/A; görüntü yeniden seçilmedi. Kaynak luma görüntüleri kalite/reconstruction sonucu olarak sunulmuyor.
- Figür 7: sabit mean/Q90 kontroller için gerçek RGB kayıp–MAC grafiği; nokta etiketleri 0.1 dB'yi aşan kare sayısı.
- Ana metin ve ek materyalde typeset tablo yok. Metrik kontrolü, resmî LR takvimi ve epoch 20/30 farkları grafiğe dönüştürüldü; kapasite için mevcut altı-derinlik parametre grafiği kullanıldı. Sayısal audit exportları korunuyor.

## Kontrol

38 vektör figür seti; 130 PDF/SVG/PNG/atlas dosyası yalnız paketlenmiş veri ve kodla temiz geçici klasörde byte-identical üretildi. Altı türetilmiş sayısal audit dosyası da aynı. Deney ağırlığı, GPU veya dış dataset gerekmiyor; gerçek luma örnekleri pakette.

Ana metin 8 sayfa, kaynakça 9. sayfa. Ek materyal 24 sayfa. Tanımsız referans, overfull kutu veya gömülmemiş font yok. Grafik metni exportta en az 7 pt; CVPR ölçeğinde en az 6.55 pt. Sayfa 4, 6–8 ve yeni bağımsız grafikler görsel olarak incelendi. Bu kontroller dergi kabulü veya genel estetik üstünlük iddiası değil.

## Bu revizyonda makaleye giren tamamlanmış kontroller

QP32 sabit-Q90 yerleşim replay'i 424 map case tamamladı. Aynı derinlik histogramı ve geçerli tile alanı altında, üç permütasyonun ortalama RGB MSE'sine göre router yerleşim kazancı 0.01393 dB; dithering 0.00036 dB. Aralıklar bu üç permütasyona koşullu ve veri geliştirmede kullanılmış durumda.

Uniform-context kontrolü iki DIV2K kaynağı × dört derinlikte tamamlandı: eşleşen global sınır geometrisi ve yeterli suffix halo ile sekiz kontrolün özellik/çıktı maksimum mutlak farkı sıfır. Bu küçük CPU sayısal kontrol, genel karma-harita kalite veya latency sonucu değildir.
