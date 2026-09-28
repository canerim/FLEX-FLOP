# Sonraki native deney: ölçüm sözleşmesi

28 Eylül 2026. Bu belge gelecekteki ölçümün planıdır; yeni GPU doğruluğu veya hız sonucu içermez. Resmî eğitimler sürerken native benchmark başlatılmadı.

## Önce hangi sistemin ölçüldüğünü sabitle

Shared early exit ve bağımsız codec bankası iki ayrı yürütme yoludur. Shared deneyde tek entropy payload'ı, ortak stem ve exit haritası var. Bankada her expert kendi analysis/entropy parametrelerine sahip; bir expert'in latentini ötekine vermek geçerli bir baseline değildir. İlk bankalı pilot yalnız gerçekten mevcut D2/D4/D6/released-D12 checkpoint'lerini kullanabilir. D8/D10 sonuçları eğitilmeden doldurulmaz.

Doğruluk fixture'ları sabit epoch20 checkpoint'lerine bağlı kalabilir. Final RD değerlendirmesi ise ayrıca sabitlenmiş final checkpoint kimlikleriyle yapılmalıdır. Released D12 ve aynı tarifli D12-scratch farklı soruları cevaplar; bunları tek bir eş-eğitimli derinlik kontrolü olarak adlandırma.

## GPU doğruluğu için geçiş sırası

1. Hazır 43-case/model harness ile aynı binary/toolchain altında stock D12–patched D12 payload ve reconstruction eşleşmesini kontrol et. D2/D4/D6 bağımsız encoder ve decoder süreçleri, shape/QP değişimleri ve ilk örneğe dönüş dahil. CPU preflight geçti; bu CUDA aşaması henüz çalışmadı.
2. Aynı GPU üzerinde birden fazla resident model arasında **seri** geçişi ayrı test et. Her çıktıyı sonraki çağrıdan önce bağımsız belleğe kopyalayarak global buffer-pool aliasing'ini gizleme. İzole model çıktıları referans olsun. Seri geçiş doğruluğu eşzamanlı stream güvenliği anlamına gelmez.
3. Gerçek router haritalarıyla grouping ve assembly'yi ekle. Grup anahtarı yalnız expert değil; codec'in gerektirdiği şekil, QP ve runtime koşullarını da içermeli. Birbirinden farklı dikdörtgenleri varsayımsal tek batch olarak zamanlama.

İki-frame CPU FUFEXIT1 kontrolünün 8/8 eşleşmesi bu sıranın yerine geçmez: o araştırma formatı FP32'dir, açık map taşır ve native CUDA wire formatı değildir.

## Primer süreyi doğru sınırla

- **Encoder:** gereken source-feature extraction, seçim, grouping, analysis, entropy işçisi, synthesis ve assembly dahil; CPU bitstream hazır ve GPU reconstruction tamamlanmış olmalı. Çağrı öncesi ve sonrası gereken CUDA senkronizasyonu ile host wall-clock ölç. Birden fazla stream'e dağılan işleri tek CUDA-event aralığıyla eksik sayma.
- **Decoder:** yalnız gerçek byte dizisi, iletilmiş metadata ve resident checkpoint'ler giriş olsun. Kaynak görüntü, encoder latent'i, kaynak hata tablosu veya encoder'ın entropy index'leri decoder'a verilmez.
- **Çıktı sözleşmesi:** GPU-ready ve host-ready çıktıları ayrı adlandır. Host'a görüntü aktarımı gerekiyorsa onu primer süreden çıkarma. Doğruluk için alınan ek debug kopyasını ise hangi ölçüme dahil ettiğini açıkça belirt.
- **Stage süreleri:** teşhis içindir. Native entropy worker ve synthesis örtüşebildiği için aşamaları toplayıp toplam süre üretme. Soğuk başlatma, ağırlık yükleme, graph capture ve steady-state sonuçları ayrı raporla.

Hedef GPU, saat/güç koşulları, precision, extension hash'i, thread sayısı, çözünürlük, batch/group büyüklükleri ve model residency aynı karşılaştırmada sabit olmalı. Başka işin bulunduğu GPU'da kıyas başlatma; yöntem sırasını bloklar içinde değiştirerek ısınma/sıra etkisini kontrol et. Tek frame'i çok kez çalıştırmak çok sayıda bağımsız örnek üretmez: hem frame'ler arası dağılımı hem repeat içi değişkenliği sakla.

## Kalite ve rate eşlemesi

Aynı QP, aynı bitrate veya aynı gerçekleşmiş kalite demek değildir. Payload rate ile map/header dahil container rate ayrı raporlanmalı. RGB, 444 ve 6:1:1 metrikleri açıkça adlandırılmalı; referans, crop, padding ve clipping aynı kontrastta sabit tutulmalı. QP32'de ölçülen +0.00908 dB released/e15 farkını diğer QP'lere veya native FP16'ya aktarma.

Uniform, Bayer, küçük predictor ve kaynak-bilgili search aynı checkpoint/geometry altında kıyaslanmalı. Kaynak-bilgili seçim yapılmışsa adayların üretilme ve reddedilme süresi de ilgili encoder sonucuna dahil edilmeli. Deployment kıyasındaki kontrol değeri test çıktısına bakılarak yeniden seçilmez; achieved-quality eğrileri ve eşik aşımı birlikte raporlanır.

## En ayırt edici ablasyonlar

| Kontrol | Sabit tutulan | İzole etmeye çalıştığı etki |
|---|---|---|
| Uniform / Bayer / router | Codec, QP, padding, tamamlanmış çıktı kontratı | Ucuz derinliğe karşı içerik seçiminin net maliyeti |
| Aynı histogram, farklı yerleşim | Expert sayımları ve nominal aritmetik | Mekânsal yerleşim, grouping ve sınır etkisi |
| Aynı map, dört bölge / birleşmiş bölgeler | Her pikselin expert'i | Yürütme biçimi; context, header ve entropy reset birlikte değişir |
| Adapter/repair eş eğitimli karşılaştırma | Veri, update sayısı, seed, loss ve schedule | Kapatma hassasiyetinden farklı olarak yeniden optimizasyonla gerekli kapasite |
| Expert çıkarma | Kalibrasyon/test ayrımı ve toplam sözleşme | Altı modelin gerçekten ek frontier sağlaması |

Padding için +0.265 dB ile merging için yaklaşık +0.50–0.53 dB **toplanamaz**: farklı kontrol, model bileşimi ve referanslara ait ölçümlerdir. Mevcut korelasyon taramasının önerdiği 51 içerik grubu, gelecekteki ayrım için korumacı bir başlangıçtır; bütün video kökenlerini veya dokunulmamış test verisini kanıtlamaz.

Bir sonraki temel karar, predictor'ı büyütmekten önce aynı gerçekleşmiş RD koşulunda toplam sürenin azalıp azalmadığıdır. Bu soruya mevcut MAC sayıları veya araştırma formatının CPU diagnostic süreleri cevap vermez.
