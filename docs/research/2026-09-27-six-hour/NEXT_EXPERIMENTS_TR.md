# FLEX-UF: hangi deney hangi iddiayı taşıyacak?

27 Eylül 2026, devam eden araştırma oturumu. Bu belge bir deney protokolüdür; aşağıda önerilen MLP, GPU hız ve model-bank sonuçları tamamlanmış deney olarak sunulmaz. Resmî D2/D4/D6 eğitimleri değişmeden sürüyor.

## Ana hikâyenin omurgası

**Adaptasyonun codec içinde nerede yapıldığı, hangi hesabın ve hangi komşuluk bilgisinin paylaşılabileceğini belirliyor.** Shared early exit tek representation, encoder ve entropy yolunu paylaşır; synthesis içindeki hesap derinliğini değiştirir. Bağımsız model bankası ise encoder, entropy modeli ve reconstruction ağı farklı olan bir codec seçer. İki sistem aynı kaynak dağıtma sorusuna cevap verir ama aynı bitstream'in farklı decoderları değildir.

Bu nedenle paper'ın en savunulabilir omurgası “altı modelimiz var” değil, şu deney dizisidir: ek derinliğin faydasını ölç; bu faydanın nerede değiştiğini göster; seçimin bağımsız basit kontrollerden kazancını ölç; seçimin ve sınır işlemlerinin maliyetini toplam codec hesabına geri koy. D8/D10 tamamlanmadan altı derinlikli RD sonucu, native ölçüm olmadan hız sonucu yazılmaz.

## Öncelik sırası

| Öncelik | Deney | Kontrol ettiği soru | Protokol ve karar ölçüsü |
|---|---|---|---|
| P0 | Tam recipe D2/D4/D6; aynı recipe D12 scratch kontrolü | Fark gerçekten derinlikten mi, training provenance'tan mı geliyor? | Aynı105 epoch, train0/1/2, crop/LR takvimi, precision ve seed politikası. Released D12 ayrı pratik referans; scratch D12 nedensel derinlik kontrolü. Ara epoch20 sonucu final değildir. |
| P0 | Gerçek byte + bağımsız decode | Rate tahmini gerçek coder davranışını temsil ediyor mu? | Her stream source erişimi olmayan decoder sürecinde açılır; payload/header ayrılır. RGB dönüşümü, clipping, görüntü desteği ve referans dondurulur. CPU araştırma formatı ile native format ayrı raporlanır. |
| P0 | Native stock D12 / patched D12 eşleşmesi | Hız ölçtüğümüz kod aynı codec mi? | Aynı compiler/SM86 kurulumu; coded payload ve reconstruction karşılaştırması; yeni süreçte decoder-only başlangıç. Sonra gerçek D2/D4/D6 yolu ve shape/QP graph recapture. Derleme başarısı sayısal doğrulama değildir. |
| P0 | Uniform, Bayer, histogram-permutation, öğrenilmiş router | Kazanç daha ucuz ortalama derinlikten mi, doğru konum seçiminden mi? | Aynı candidate set, aynı kalite referansı, aynı toplam bütçe ve aynı sample cohort. Kör histogramın kendisi evaluation kaynaklarıyla optimize edilmişse bunu açıkça söyle. |
| P0 | Kaynak erişimsiz kontrol / kaynakla arama | Encoder'da best-match maliyeti ne? | Ham tekrarlı arama, shared-prefix/all-exit cache ve tek router ayrı baseline. Karar oluşturma, source quality check, encode, entropy, decode ve assembly dahil wall time. Cache baseline'ını atlamak router kazancını şişirebilir. |
| P1 | Halo0/32/64 ve bağımsız patch stream | Patchleme ne kadar RD kaybettiriyor? |16 önceden seçilmiş DIV2K crop;5 QP; D2/D6 epoch20 ve released D12. Full/halo0/halo32/halo64 ortak rate desteği; actual byte, PSNR, seam ve interior MSE. Halo64, hiçbir patch sonucu görülmeden eklendi. CPU image-pad64 ve native image-pad16/latent-pad4 ayrı protokoller. |
| P1 | Aynı expert komşuları birleştirme | Halo ve stream-reset maliyeti azaltılabilir mi? | Sabit map üzerinde ayrı256 stream, bitişik aynı-expert dikdörtgenlerini birleştirme ve whole-image uniform kontrol. Yeniden encode gerekir; concatenation veya batch çağrısı aynı deney değildir. |
| P1 | D4 seçeneğini kaldırma; sonra D8/D10 çıkarma | Her ara expert gerçekten gerekli mi? | Aynı bütçede tam choice set ve leave-one-depth-out set. Epoch20 D2/D4/D6 için exact whole-crop bound hazırlanmıştır; gerçek MLP ve görüntü-içi routing sonucu sayılmaz. |
| P1 | Mutlak complexity / marjinal fayda hedefi | “Zor patch” doğru label mı? | Aynı küçük router ile mutlak hata, komşu derinlikler arası hata farkı ve rate–distortion–cost regret hedeflerini karşılaştır. QP girdisi ayrı ablation. Özellik–gain korelasyonu tek başına prediction başarısı değildir. |
| P1 | Whole-image seçim /256 patch seçim | Daha ince adaptasyon kendi maliyetini geri kazanıyor mu? | Aynı modeller ve aynı bağımsız test görüntüleri. Whole-image seçim patch entropy reset, halo ve map maliyetinden kaçınan zorunlu ucuz kontrol olmalı. |
| P2 | Seed ve exit yerleşimi | Sonuç training şansına veya merdivene duyarlı mı? | Önce uç derinliklerde ek seed; sonra2-blok vs4-blok exit aralığı. Image-bootstrap ile training-seed değişkenliğini aynı güven aralığı diye sunma. |
| P2 | Quantized router / serialized map | Decoder seçimi encoder ile tekrarlanabilir mi? | Decoder yeniden hesaplıyorsa integer/fixed-point karar ve tie politikası; map aktarılıyorsa gerçek serialized byte ve parse testleri. Sadece float logits eşitliği wire determinism kanıtı değildir. |

## Router için önerilen somut hedef

Bağımsız bankada kaynak-only küçük bir ağ, her expert ve QP için distortion ve rate'i tahmin etsin. Seçim skoru örneğin `D_hat(k,q) + lambda_R R_hat(k,q) + lambda_C C_profile(k,geometry)` olabilir. Buradaki cost, ölçülmüş deployment profiline dayanmalıdır; sadece blok sayısı yeterli değildir. Çıktı için tüm expert encoderlarını çalıştırmak, “encode öncesi ucuz seçim” iddiasını ortadan kaldırır.

Label üretimi sırasında pahalı bütün-expert değerlendirmesi yapılabilir; inference sözleşmesi bunu içermemelidir. Regresyon hatası yerine doğru kararı teşvik etmek için, seçilen expert'in en iyi seçenekten RD–cost farkını ölçen regret hedefi de karşılaştırılmalı. Bu, önerilen bir ablation'dır; yeni veya işe yaradığı gösterilmiş bir yöntem olarak yazılmamalı.

Shared early exit için ortak stem zaten hesaplandığından latent/stem istatistikleri kullanılabilir. Burada ucuz regressor'ın tahmin edeceği şey, sonraki blok çiftinin sağlayacağı ek reconstruction faydasıdır. QP değiştikçe entropy kaynaklı hata ile synthesis kaynaklı hata oranı değişebilir; QP'siz tek complexity sınıfı bunu kaçırabilir.

Her iki yolda da sabitβ ile tek nokta yerine bütçe eğrisi raporlanmalı. Daha yüksekβ'nin her görüntüde monoton kalite vermesi varsayılmamalı; adaylar ve tiebreak kuralı önceden belirlenmeli. Kalite garantisi denilecekse ayrı calibration split, açık risk tanımı ve uygun istatistiksel kontrol gerekir. Evaluation görüntüsünün gerçek hatasına bakarak map değiştiren sonuç, source-free deployment sonucu değildir.

## Bölge birleştirme: uygulanabilir sistem fikri

Önce source-only router bütün haritayı üretir. Ardından aynı expert'e atanmış bitişik core'lar dikdörtgen bölgelerde birleştirilir; encoder yalnız seçilen bölgeleri işler. Bu, model değiştirme çağrılarını, tekrar kodlanan context'i ve stream header sayısını azaltabilir. Aynı expert'in dağınık patchlerini batch'e koymakla eşdeğer değildir: spatial context ve entropy reset sınırları da değişir.

En küçük kontrol2×2 grid üzerindedir: tek expert seçilmiş dört core full512 olarak; iki aynı-expert şerit512×256 bölgeler olarak; checkerboard ise ayrı core'lar olarak çalıştırılır. Halo64,64-aligned boyutlar için kodlanan alan oranları sırasıyla1,1.25 ve1.5625'tir. Bunlar geometrik değerlerdir; eşit kalite veya hız sonucu değildir. Birleştirme politikasının distortion etkisi yeniden encode edilerek ölçülmelidir.

Boundary-length regularizer, MLP'yi tutarlı bölgeler seçmeye teşvik edebilir. Ancak salt pürüzsüz harita daha iyi kalite anlamına gelmez. Bu regularizer'ın kazancı, değişen expert histogramından ayrılmalı; aynı histogramlı permutation ve aynı map üzerinde merge/no-merge kontrolleri gerekir. Bu aşama mevcut105-epoch resmî eğitim tarifine ek loss sokmaz; codec eğitiminden sonra router/scheduler araştırmasıdır.

## Veri ayrımı ve metrik sözleşmesi

Bu oturumdaki100 DIV2K validation crop ve53 CTC sequence araştırma kararlarında incelendi. Sonraki MLP için bunlara “hiç görülmemiş dış test” denmemeli. Training-label, calibration ve final test listeleri görüntü/sequence kimliğiyle ayrı ve hash'li tutulmalı. Aynı görüntünün farklı crop/QP'leri farklı splitlere dağılmamalı. Yeni test benchmarkı seçilmeden “genelleme” sonucu tamamlanmış sayılmaz.

Final tablo için minimum sütunlar: RGB PSNR, actual payload bpp, container/mode-map bpp, encoder wall time, decoder wall time, peak memory ve başarısız decode sayısı. Zamanlar aynı GPU, precision, batch/stream politikası, input geometry ve warmup protokolünde ölçülmeli. Resmî native encoder reconstruction ile entropy worker'ı örtüştürebilir; ayrı süreleri toplayıp wall time türetmek doğru değildir.

## Yakın literatürden alınan dersler

[APE, ECCV2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780286.pdf) mutlak patch zorluğu yerine katmanlar arası ek faydayı tahmin eder; exit aralığı ve patch/stride kontrolleri içerir. Bu, bizim marginal-benefit ve exit-spacing ablation'larımız için doğrudan ilgili bir öncüldür. SR metriklerini compression rate veya codec byte garantisine taşımıyoruz.

[Spatial Competition,2026](https://arxiv.org/html/2605.13243v1) codec seçimini RD maliyetiyle yapar, mode map aktarır ve aynı mode bölgelerini sürekli işler. Bu yüzden content-adaptive codec seçimini veya region coalescing'i yeni fikir diye sunamayız. Bizim bankamızdaki soru farklı synthesis derinlikleri ve pahalı aramanın source-only seçimle ne ölçüde karşılanabileceğidir; ikisi de ölçüm gerektirir.

[ClassSR, CVPR2021](https://arxiv.org/abs/2103.04039) farklı kapasitedeki ağlar ve bir sınıflandırıcıyla patch bazında SR hesabını dağıtır. Bu yararlı bir sistem analojisidir. Compression'da expert'in analiz ve entropy parametrelerinin de değişmesi ek rate/bitstream sorularını doğurur; analoji bu soruları çözmüş sayılmaz.

## Padding kapsamı düzeltmesi,18:12UTC

İlk halo geometri incelemesi FUFREF1 CPU image-pad64 düzenine aitti. Native CUDA image-pad16 ve hyperanalysis öncesinde latent-pad4 kullanır. Native halo32 görüntü penceresi288 kalırken bazı hyperprior ağları320 eşdeğerinde çalışır; tek alan çarpanı bütün codec maliyetini temsil etmez. Katman bazında native-shape Conv2d izi ve ayrıFUFREF2 CPU decode kontrolü hazırlanıyor. CPU-pad64 sonucunu native sisteme genellemiyoruz.
