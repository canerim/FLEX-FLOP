# DCVC-UF derinlik ve yönlendirme çalışması: öncelikli ablasyon planı

27 Eylül 2026 · Yeni GPU/eğitim deneyleri öneridir; tamamlanan CPU tablo
kontrolleri üçüncü bölümde açıkça ayrılmıştır.

**Ana soru:** Aynı kalite ve gerçek bitrate altında, içerikten hangi modelin
çalıştırılacağını öğrenmek, en iyi sabit veya içerikten bağımsız seçime göre
uçtan uca süreyi azaltıyor mu? D2/D4/D6 eğitimi bu sorunun kapasite eksenini
kuruyor. Tek başına üç modelin PSNR farkı, bir router katkısını kanıtlamıyor.

## 1. Önce iki sistemi birbirinden ayırıyoruz

| Deney ailesi | Şu anki durum | Ortak olan | Router'ın yeri | Doğru iddia |
|---|---|---|---|---|
| Shared-latent e15 | Arşivlenmiş 53×5 değerlendirme var | Encoder, entropy ve latent | Latent/stem sonrasında tile derinliği | Ortak sentez gövdesinde bölgesel early exit |
| D2/D4/D6 codec bankası | Resmî tarifle eğitim sürüyor | Mimari ailesi ve eğitim protokolü | RGB patch'ten önce model seçimi | Bağımsız codec'ler arasında derinlik seçimi |
| D8/D10 | Henüz bu deneyde eğitilmedi | Planlanan aynı protokol | Bankaya ek expert | Sonuç henüz yok |
| Released D12 | Yayımlanmış referans mevcut | Dağıtılan orijinal codec | Bankanın tam kapasiteli seçeneği | Dağıtım referansı; eş eğitimli derinlik kontrolü değil |

Bağımsız modellerin encoder ve entropy ağırlıkları da öğreniliyor. Bu nedenle
bir modelin latentini diğer decoder'a vermek geçerli bir bankalı codec tasarımı
değil. Ortak encoder isteyen alternatif için encoder/entropy dondurulmalı ve
decoder'lar ortak latent üzerinde ayrıca eğitilmeli. Bunlar ayrı deneylerdir.

## 2. Ana deneyin nedensel kontrolü

**En önemli eksik kontrol: aynı protokolle sıfırdan eğitilmiş D12.** Released
D12 ile fark, derinliği ve eğitim geçmişini birlikte değiştiriyor. Released
modeli koruyup yanında D12-scratch raporlamak gerekir. Aynı veri listesi,
105 epoch, başlangıç seed'i, optimizer, QP/lambda örneklemesi, precision ve
eğitim bütçesiyle D2/D4/D6/D12 karşılaştırması derinlik etkisini ayırır.
Bu yeni bir uzun eğitimdir; mevcut üç işi kesmek veya tarifini kısaltmak
anlamına gelmez. D8/D10'dan önce önceliklendirilmesi daha değerlidir.

| Kod / öncelik | Tek değişken ve kontrol | Sabit tutulacaklar | Birincil çıktı | Hangi yorumu sınar? |
|---|---|---|---|---|
| D0 / P0 | D2, D4, D6, D12-scratch | Resmî tarif, veri, QP, seed, tüm eğitim adımları | Tam RD eğrisi; aynı bitrate PSNR; aynı kartta encoder/decoder süreleri | Azalan decoder derinliği uçtan uca öğrenmede ne kaybettiriyor? |
| D1 / P0 | D12-scratch ve released D12 | Değerlendirme ve çalışma ortamı | RD farkı ve neural runtime | Eğitim/provenance farkı ne kadar? |
| D2 / P1 | Encoder/entropy ortak-dondurulmuş veya uçtan uca öğrenilmiş | Decoder derinliği ve karşılaştırılabilir eğitim bütçesi | RD ve toplam parametre/bellek | Ortak latent zorunluluğu ne kadar maliyetli? |
| D3 / P1 | Sıfırdan veya released prefix'ten başlatma | İki kola da aynı güncelleme sayısı, aynı veri | Öğrenme eğrisi ve nihai RD | Sığ model kapasitesi mi, optimizasyonu mu sınırlı? |
| D4 / P2 | Öğretmensiz veya D12 distillation | Öğrenci mimarisi, veri, update sayısı | RD; ek eğitim maliyeti | KD kaliteyi kurtarıyor mu? Resmî ana tarife ayrı varyanttır. |
| D5 / P2 | D8/D10 ekleme | Mevcut değerlendirme | Pareto eğrisine eklenen yeni bölgeler | Altı expert gerekli mi, ara derinlikler baskılanıyor mu? |

Bir seed ile başlamak kapasite taraması için makul; genelleme iddiası için
seçilmiş kritik çiftlerde bağımsız seed tekrarları gerekir. Görüntü bootstrap'ı
seed tekrarının yerine geçmez. Aynı seed ve aynı başlangıç sırası eşlemeyi
iyileştirir, fakat farklı GPU/kernel yollarında bit düzeyinde aynı öğrenme
gürültüsünü garanti etmez.

## 3. Router'ın kendisine ait kazancı ayıran deney

Her codec için QP başına gerçek `(D, R, T)` tablosu oluşturulmalı. Farklı
codec'lerde aynı QP aynı bitrate değildir. Router'ın kalite kıyasını yalnız
QP eşleyerek yapmak yanıltır; RD desteğinin ortak aralığında karşılaştırma
gerekir. Destek dışına extrapolation yapılmamalı.

**Kontrol sırası:** released tam görüntü → aynı patch protokolünde sabit D12 →
en iyi sabit expert → blind karışım → basit içerik eşiği → MLP → kaynak bilgili
arama. İlk iki basamak, patchleme ve entropy reset maliyetini model seçimine
yazmamamızı sağlar. En iyi sabit expert ve eşik validation üzerinde seçilir;
test kümesinin ortalamasından seçilen “en iyi” kontrol ayrıca oracle kontrol
diye adlandırılır.

| Kod / öncelik | Karşılaştırma | Ölçüm | Önemli kontrol |
|---|---|---|---|
| R0 / P0 | Uniform / Bayer / histogramı koruyan shuffle / MLP / source-informed | Aynı gerçekleşen RD altında toplam süre; paired fark | Bütün yöntemlerde aynı padding, patch ve bitstream formatı |
| R1 / P0 | Sabit held-out kontrol / kare başına source-calibrated kontrol | Encoder ve decoder maliyeti ayrı; side bits; hedef aşımı | Test kaynak görüntüsünün decoder policy kalibrasyonuna sızmaması |
| R2 / P0 | Exit-label CE / maliyet-ağırlıklı regret / `(D,R)` tahmini | Ek latency kazancı, RD regret, kalibrasyon | Eşleşme doğruluğu tek başına başarı ölçütü değil |
| R3 / P1 | QP-only / varyans+kenar / düşük çözünürlüklü RGB / öğrenilmiş feature | Girdi üretimi dahil süre ve RD | Codec seçilmeden mevcut olmayan latent/scales girdisi ücretsiz sayılamaz |
| R4 / P1 | D2+D12 / D2+D6+D12 / tam banka | RD-time frontier, ağırlık belleği ve expert kullanım oranı | Ara expertlerin yalnız kullanılması değil, frontiere katkısı |
| R5 / P1 | MAC etiketi / ölçülmüş latency etiketi | Farklı çözünürlük ve microbatch boyutunda frontier | Latency label'ı cihaz ve batch koşuluna bağlıdır |
| R6 / P2 | Tek seçim / yalnız belirsiz patch'lerde top-2 doğrulama | Arama maliyeti dahil encode süresi ve regret | İkinci denemenin reconstruction maliyetini dahil et |
| R7 / P0 | Bisection / karar eşiği aralıklarının tam taraması | Aynı tablo ve aynı policy ailesinde kaçan uygulanabilir plan | Router ve dither için kalite monotonluğu varsayma; ikisini de güçlendir |

Bu revizyonda R7'nin **CPU tablo kontrolü** tamamlandı: router'ın fiyat
yolunda 265 frame–QP çiftinin 227'sinde, dithering yolunda 232'sinde en az
bir kalite terslenmesi var. Buna rağmen 0,1 dB'de uygulanabilir eski 263
plan içinde router'da yalnız iki, dither'da yalnız bir plan iyileşiyor.
Ortalama ek MAC kazancı sırasıyla 0,0277 ve 0,0028 puan. Bu yeni haritalar
GPU'da yeniden decode edilmedi; ana reconstruction eğrileri değiştirilmedi.
Hedef gevşedikçe monotonluk problemi bu kayıtlarda sonucu değiştirmiyor.

Histogram kontrolünün CPU kısmı da tamamlandı. Kaydedilmiş her haritanın
exit sayıları korunarak rastgele yer değiştirme altındaki **beklenen MSE**
analitik hesaplandı. 0,1 dB'de router'ın yerleşim kazancı ortalama 0,0213 dB,
dither'ınki 0,0007 dB. Router'ın aynı histogramda mümkün olan en iyi tablo
yerleşimine uzaklığı 0,0044 dB. Bunlar kaynak tablo değerleridir; final
crop/repair altında kaliteyi ispatlamaz. Bir sonraki somut ablasyon,
aynı histogramlı shuffle ve optimum haritaları gerçekten decode etmek.

## 4. Denemeye değer somut yöntem: kazanç ve maliyeti tahmin eden router

“Karmaşık görüntü → derin model” etiketi yerine MLP, her expert için ucuz
referansa göre distortion ve rate farkını tahmin etsin. Seçim skoru
`J_k = D_hat_k + lambda * R_hat_k + mu * T_k(batch, shape)` olsun. Burada
`T_k` aynı kartta ölçülen yürütme tablosundan gelir; feature çıkarımı,
sıralama ve birleştirme de toplam süreye eklenir. Distortion/R'nin birimleri
sabitlenmeli; PSNR dB değerleri patch'ler arasında doğrudan toplanmamalı.

Eğitimde yanlış expert etiketi kadar, yanlış seçimin neden olduğu gerçek
`J` artışına ağırlık vermek anlamlıdır. Çok yakın iki expert'i ayırt
edememekle kötü bir expert seçmek aynı hata değildir. Karşılaştırılacak
en basit yöntem yine sıradan CE olmalı; karmaşık hedefin yararı ölçülmeli.

İkinci aşama, **belirsiz seçimlerde sınırlı doğrulama** olabilir. En iyi iki
tahmini skor arasındaki marj küçükse yalnız bu iki codec denenir. Eşik ve
maksimum doğrulama payı ayrı validation üzerinde seçilir. Bu yaklaşımın amacı
tam altı-model aramanın kazancına daha az encoder işiyle yaklaşmak; başarı
henüz ölçülmedi. Büyük marj otomatik kalite garantisi değildir. Gerçek bir
kare garantisi istenirse son reconstruction aynı referans/metrikle
doğrulanmalı ve gerekli fallback'in maliyeti raporlanmalıdır.

Offline etiket üretimi de kaydedilecek: kaç patch × expert × QP, GPU-saat,
depolama ve amortizasyon. Bu eğitim maliyeti inference ms içine eklenmez,
fakat “ücretsiz oracle” gibi sunulmaz.

## 5. Patch ve yürütme ablasyonları

| Kod | Deney | Kaydedilecekler |
|---|---|---|
| S0 / P0 | Tam görüntü D12 ve patch D12 | Salt parçalamanın RD, header, entropy reset ve boundary maliyeti |
| S1 / P0 | Patch başına seri yürütme ve expert'e göre gruplanmış microbatch | Encoder/decoder ayrı medyan ve p95; peak VRAM; kullanılan expert sayısı |
| S2 / P0 | 128 / 256 / 512 RGB patch | Gerçek payload+header+map bpp; latency; crop sonrası distortion; tile sayısı |
| S3 / P1 | Halo 0 / küçük / büyük; aynı sınır birleştirme | Halo'nun gerçek hesap ve bit maliyeti; sınır/interior hata profili |
| S4 / P1 | Aynı exit histogramı, farklı mekânsal yerleşim | Locality ve fragmented batching maliyeti; halo/başlık değişimi |
| S5 / P1 | Model ağırlıkları resident veya kapasiteye göre yüklenmiş | Banka belleği, transfer dahil soğuk ve sıcak latency |

Shared-latent ailede bunlara ayrı olarak split depth, pointwise adapter
kapasitesi, grid repair ve full-frame head ablasyonları eklenir. Mevcut
e15 modelinin repair'ını inference sırasında kapatmak bir hassasiyet
deneyidir; repair'sız yeniden eğitim ile eşdeğer değildir. Aynı ayrım
adapter kaldırma için de geçerlidir. Ana tabloda “retrained ablation” ve
“inference intervention” karıştırılmamalı.

## 6. Değerlendirme sözleşmesi

1. **Referans:** deployed released D12 ve eş eğitimli D12-scratch açık adlarla
   tutulur. Shared-latent e15 full-depth üçüncü, ayrı bir referanstır. Hem
   seçme hem raporlama aynı referans, crop ve metrik kullanır.
2. **Veri ayrımı:** expert eğitimi, router etiket eğitimi, kontrol kalibrasyonu
   ve final test görüntü düzeyinde ayrıdır. Aynı görüntünün patch'leri farklı
   split'lere dağılmaz. Final Kodak/CTC sonuçlarına bakarak eşik seçilmez.
3. **Validation:** dört crop yalnız sağlık göstergesidir. Aynı epoch'taki
   modeller ve released referans için 100 DIV2K validation görüntüsü daha
   güçlü ara kontrol; final için tam Kodak-24 ve ilan edilen CTC/CLIC kapsamı.
   Bu genişletilmiş validation henüz çalıştırılmış sonuç değildir.
4. **Bitrate:** `(payload + hyperprior + model ID + QP/control + headers)`
   toplam bit / orijinal piksel sayısı. Altı expert için 3 bit kimlik alt
   düzeyde bir bileşendir; tam stream maliyeti değildir.
5. **Kalite:** RGB PSNR ve seçilmiş YUV ağırlığı açıkça yazılır; MSE valid
   piksellerde toplanır. LPIPS/MS-SSIM ayrıca raporlanır. PSNR toleransından
   görünmezlik sonucu çıkarılmaz. En kötü görüntüler saklanır.
6. **Süre:** aynı GPU, aynı precision/kernel modu, warmup, dönüşümlü yöntem
   sırası ve ham tekrarlar. Encoder, decoder ve toplam yol ayrı. Modül
   CUDA-event zamanı ve uçtan uca wall-clock aynı metrik değildir.
7. **İstatistik:** eşlenik görüntü/dizi farkları, bootstrap CI ve seed
   varyasyonu ayrı. Aynı diziye ait QP'ler birlikte resample edilir. Tüm
   yöntemlerin ortak RD desteği ve infeasible/fallback sayıları yayımlanır.
8. **Başarı ölçütü:** öğrenilmiş router'ın, en iyi validation-seçilmiş blind
   kontrol üzerinde ortak gerçekleşen RD bölgesinde pozitif net latency
   avantajı. MAC kazancı veya oracle-label accuracy tek başına yeterli değil.

## 7. Deney bütçesini patlatmadan sıra

**Şimdi:** üç resmî eğitimi değiştirmeden tamamla; immutable checkpoint ve
geniş değerlendirme manifestini hazırla. D2/D4/D6 final frontier'ı görmeden
router aramasıyla tüm kombinasyonları çoğaltma.

**İlk boş kaynakta:** D12-scratch kontrolü; mevcut modellerde S0 ve S1.
Bu aşama model bankasının patchleme/operasyon maliyetinde baştan kaybedip
kaybetmediğini gösterir. Kaybediyorsa MLP kapasitesini büyütmek öncelik değildir.

**Sonraki küçük tarama:** üç-expert banka, tek patch boyutu, R0/R1/R2.
Önce ucuz özellikli küçük MLP. Aynı kalibrasyon ayrımıyla anlamlı net kazanç
varsa expert sayısı, tile boyutu ve regret/top-2 kolunu genişlet.

**Makale için son aşama:** kazanan yapılandırmada kritik baseline ve seed
tekrarları; en kötü içerik, çözünürlük, perceptual quality ve belleğin raporu.
Resmî reçeteyi kısaltarak “aynı tarif” iddiasını sürdürmek bu plana dahil değil.

## 8. Literatürün katkı iddiasına etkisi

[ClassSR](https://openaccess.thecvf.com/content/CVPR2021/html/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.html)
ve [APE](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2021_ECCV_2022_paper.php)
bölgesel kapasite ve patch early exit için doğrudan öncüller.
[AdaNIC](https://openaccess.thecvf.com/content/ICCV2023/html/Tao_AdaNIC_Towards_Practical_Neural_Image_Compression_via_Dynamic_Transform_Routing_ICCV_2023_paper.html)
compression içinde blok kapasitesi yönlendirmesini çalışıyor.
[cgSlimDecoder](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Complexity-Guided_Slimmable_Decoder_for_Efficient_Deep_Video_Compression_CVPR_2023_paper.html)
kompleksite kontrollü decoder genişliği için kıyas gerektiriyor.

[Spatial Competition (2026)](https://arxiv.org/abs/2605.13243), bölgeye göre
bağımsız codec seçimi ve mode map ile bankalı tasarımımızın yakın komşusu.
“Patch'e göre codec seçen ilk sistem” iddiası uygun değil. Bizim sınanacak
ayrımımız heterojen derinliklerin **ölçülmüş süre ve RD maliyetiyle** seçilmesi,
tam aramayı azaltan router ve bunun güçlü blind kontroller üzerindeki net
kazancı. Bunlar şimdilik araştırma hipotezi, doğrulanmış yenilik değil.

[Shallow Decoders](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Computationally-Efficient_Neural_Image_Compression_with_Shallow_Decoders_ICCV_2023_paper.html)
encoder–decoder hesap asimetrisini;
[DCVC-RT](https://arxiv.org/abs/2502.20762) operasyonel maliyeti göz ardı
etmememizi gerektiriyor. Yalnız derinlik/GMAC tablosu bu karşılaştırmaları
karşılamaz.

[What Matters in Practical Learned Image Compression](https://arxiv.org/abs/2605.05148)
cihaz süresi ve algısal kaliteyi birlikte gözeten güncel bir tasarım örneği.
Bizim çıkarımımız: router maliyetini teorik MAC ile etiketlemek yerine,
hedef cihazdaki patch/batch koşullarında ölçmek ve sınır artefaktlarını ayrı
incelemek gerekli. Bu çalışma bizim modele aktarılmış bir sonuç değil.
