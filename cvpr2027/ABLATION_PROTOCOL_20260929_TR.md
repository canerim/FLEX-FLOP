# DCVC-UF: literatürden uygulanabilir ablasyonlara

29 Eylül 2026. **Ana araştırma sorusu:** içerik bilgisi, kör bir derinlik
karışımının zaten sağladığı tasarrufa, karar ve yürütme maliyetleri ödendikten
sonra ne ekliyor? Önce bunu tek modelde early exit için göstereceğiz.
Bağımsız D2/D4/D6/D8/D10/D12 bankası aynı sorunun ikinci uygulaması;
aynı router veya aynı deney değildir.

Bu belge yeni deneyleri sonuç gibi sunmaz. Durumlar: **tamamlandı** =
mevcut çıktılardan doğrulanmış analiz; **hazır** = dondurulmuş girdiler;
**öneri** = uygulama/eğitim/değerlendirme gerektiriyor. İlgili kaynakların
okunan bölümleri ve yöntemimize aktarım sınırları
[literatür notunda](LITERATURE_ABLATIONS_20260929_TR.md).

## 1. Yeni analiz, önceliği nasıl değiştiriyor?

Epoch 20 ve 30'un 2.000'er gerçek-byte vakasını yeniden analiz ettim.
Ortak 99 DIV2K merkez-512 crop, **0,2 payload bpp**, log-rate doğrusal
interpolasyon, YUV611 **4:4:4** için:

- Epoch 30'da en iyi sığ seçenek D2/D4/D6 sırasıyla **6/45/48** görüntüde
  kazanıyor. Ortalama D6−D4 **−0,00415 dB**; ortalamaların yakınlığı,
  görüntü bazında aynı sıralama anlamına gelmiyor.
- Kaynağın gerçek hatasını bilen görüntü başına seçim, bu kümeden seçilen
  en iyi sabit modele **0,02816 dB** ekliyor. Bu, hesap bütçesi olmayan,
  test sonuçlarını bilen iyimser bir teşhis; patch router sonucu değildir.
- **45/99** görüntünün kazananı epoch 20→30 arasında değişiyor. Eski
  checkpoint'in kazananlarını epoch 30'da kullanmak, o epoch'un kaynak
  bilgili seçimine göre **0,02420 dB** kaybettiriyor. Etiketleri erken
  dondurmak yanlış uzmanlaşmayı öğretebilir.
  Bunun **29** tanesinde eski ve yeni kazanan arasındaki fark iki
  epoch'ta da kendi kazananı lehine 0,01 dB'den büyük; bütün değişimler
  sayısal olarak neredeyse eşit seçeneklerden kaynaklanmıyor. Yine de
  epoch30'un 18 görüntüsünde birinci–ikinci farkı ≤0,01 dB; hard label
  yerine regret/soft target kontrolünü bu nedenle önceliklendiriyoruz.
- RGB ve YUV kazananı epoch 30'da **18/99** görüntüde farklı; RGB seçimini
  kullanmanın ortalama YUV regret'i **0,00311 dB**. Metrik çoğu ortalamayı
  değiştirmese de karar etiketlerini değiştirebiliyor.

**Karar:** bitmemiş modellerde büyük bir selector mimarisi taraması yapma.
Şimdi veri/label üretimini ve gerçek codec yürütmesini doğrula. Nihai
bankada kaliteyi en yükseğe çıkarma yerine **kalite bütçesinde süreyi
azaltma** iddiasını test et. Kaynaklar, görüntü kimlikleri ve 2.264 RGB
interpolasyon eşdeğerlik kontrolü:
[readiness_audit.json](data/ablation20260929/readiness_audit.json).

## 2. Ana hikâye için altı belirleyici deney

### EE-A — Adaptasyon gerçekten gerekiyor mu? Öncelik P0

**Hipotez:** içerik tabanlı yerleştirme, aynı gerçekleşmiş kalite altında
uniform ve Bayer karışımından daha düşük toplam decode süresi sağlar.

Kontroller: released D12; aynı e15 ağırlıklarıyla full-frame D12;
aynı tiled yolda uniform 6/8/10/12; Bayer; exit MLP;
kaynak bilgili seçim. Bir ek kontrol, pooled stem özellikleriyle
görüntünün tümüne tek derinlik veren **image-adaptive uniform** selector.
Böylece budget-only, görüntüye adaptasyon ve görüntü içi yerleştirme
ayrılır; ELFIC'in instance-level fikrinin ötesinde ne kazanıldığı ölçülür.
Global ve patch selector aynı frozen codec'i kullanır; bu ek küçük
predictor henüz eğitilmedi. Son kaynak bilgili kol deploy edilebilir predictor değil,
karar aramasının maliyeti de ayrıca ölçülecek üst sınırdır.
Tüm kollar aynı padding/crop, entropy ve çıktı dönüşümünü kullanır.

Önce QP32, 0,1 dB hedef ve sabit e15 ile yürütme doğrulaması. Sonra
QP16/32/48 ve YUV kayıp hedefleri 0,05/0,1/0,3 dB; calibration'da
dondurulan kontroller testte değiştirilmez. Bunlar yeni YUV kontrolleridir;
eski Δ444 eşikleri yeniden etiketlenmez. Tam-derin fallback'ın dense
anchor'a eşitliği doğrulanmadan “garantili sıfır kayıp” denmez.

Çıktı: görüntü bazında YUV kaybı, hedef aşım oranı, actual container bpp,
toplam encode/decode ms, p50/p90, peak memory. Synthesis MAC ikincil.
**Karar:** kalite/rate koşullarında kör kontrolü geçemiyorsa ana iddiayı
“içerik router'ı şart” yerine “esnek derinlik ve basit bütçe kontrolü”
olarak daralt. Mevcut cross-fit CI'larının sıfırı içermesini gizleme.

### EE-B — Yerleştirme mi, histogram mı, parçalanma mı? P0

**Hipotez:** içerikle hizalanma kaliteyi; bölgesel parçalanma ise context,
paketleme ve yürütme maliyetini farklı mekanizmalarla etkiler.

İlk kontrolün gerçek-output sürümü zaten tamamlandı: aynı histogramla
üç rastgele permutation. Yeni kontrol **aynı geçerli tile boyutu içinde**
etiketleri değiştirerek sınır uzunluğunu azaltan/artıran haritalar üretir.
Derinlik sayıları ve her derinliğin geçerli piksel alanı birebir korunur.
Harita üretimi görüntüye, PSNR'a veya timing sonucuna bakmaz.

Hazır: **53 kaynak × 2 politika × 3 harita = 318**;
106 politika–kaynak çiftinin 92'sinde sınır yoğunluğu kontrastı var.
Sabit haritalar da tutulur. Orijinal/lower-interface/higher-interface
adları yerel arama sonucudur; global minimum/maksimum iddiası yok.
Dosya: [fragmentation_maps.json](data/ablation20260929/fragmentation_maps.json).
Arşiv exit kimlikleri **2/3/4/5**, toplam derinlikleri **6/8/10/12**;
bu kimlikler 0/1/2/3 diye yeniden numaralanmaz.

Bu kontrol iki aşamalı: önce sabit histogramda gerçek çıktı, sonra aynı
cihazda paketleme/toplama + suffix + repair/head süresi. Mevcut EE yolu
tile'ları derinliğe göre topluca işler; bu nedenle sınır sayısı arttığında
sürenin mutlaka artmasını **beklemiyoruz**. Sıfır timing etkisi anlamlı bir
sonuçtur. Bankada connected-region merging farklı yürütme oluşturur;
orada aynı harita için merge açık/kapalı ayrı bir faktördür.

### EE-C — Predictor “zor patch” mi, ek hesap faydası mı öğrenmeli? P0

**Hipotez:** en iyi exit sınıfını tahmin etmektense ek bloğun beklenen
kalite kazancını öğrenmek, benzer kalitedeki seçenekler arasında daha iyi
hesap tahsisi sağlar. APE'den esinlenen bu uyarlama yeni sonuç değildir.

Üç loss kolu: mevcut soft-label CE; signed ek hata azalması regresyonu;
regret ağırlıklı seçim. **Aynı stem özelliği, MLP genişliği, QP/bütçe
girdisi, veri sırası, update sayısı ve optimizer** korunur. Negatif blok
faydası sıfıra kırpılmaz. APE'deki gibi her exit'te yeni feature okuyan
sequential policy ayrı P1 deneyidir; loss kontrolüne karıştırılmaz.

Label tarafında patch başına Y/U/V MSE vektörlerini sakla; ağırlıklı
PSNR'ları patch'ler arasında toplayıp görüntü PSNR'ı sanma. Düzeltme/head
etkileşimi nedeniyle son kalite her zaman gerçek full-image çıktıda
ölçülür. İlk tarama üç kol × seed42; yalnız seçilen kritik çift seed43/44
ile tekrar edilir. Selector test sonucu üzerinden kol/epoch seçilmez.

Ana çıktı top-1 doğruluk değil: aynı rate ve gerçekleşmiş kalite altında
net ms, kaynak bilgili frontier'a regret ve hedef aşımı. Eşit kalite
veren yanlış sınıf, büyük kalite kaybettiren yanlış sınıfla eş tutulmaz.

### EE-D — Öğrenilmiş router her bütçede çalışmalı mı? P0

**Hipotez:** gevşek hedeflerde kör karışım yeterlidir; predictor maliyeti
tasarrufu tüketebilir. Kontroller: always-MLP, always-Bayer ve yalnız
**QP + talep edilen bütçeyi** kullanan validation-fixed policy gate.
Gate test görüntüsünün ölçülmüş hatasına bakmaz.

MixCompress'in rate-only routing'i bu kontrolün önemini artırıyor:
QP/bütçe tablosu zaten yeterliyse içerik predictor'ının varlığı tek başına
değer katmış sayılmaz. Bu bir DCVC-UF içi kontrol; MixCompress'in
sayısal reproduction'ı değildir.

Bu, yeni büyük bir ağ değil: calibration'da her çalışma noktası için
uygun uniform/Bayer/MLP ailesi dondurulur. Gate seçimini yapan veriyle
nihai risk sertifikasyonu aynı veri üzerinde uyarlamalı yapılmaz; ya
ayrı tuning/calibration ayrımı ya da tüm sabit adaylara eşzamanlı sınır
kullanılır. Kaynak bilgili “en iyi politika” ayrıca oracle diye etiketlenir.

**Başarı:** MLP avantajının küçük olduğu noktalarda aynı kaliteyi daha
az toplam süreyle sağlamak. Gate işe yaramazsa basit always-MLP/Bayer
korunur; yeni bileşen eklemek kendi başına contribution değildir.

### EE-E — Adapter ve repair tasarım mı, eğitimin bağımlılığı mı? P1

Tamamlanan sabit-ağırlık identity kontrolünde adapter etkisi yaklaşık
1,23 dB, repair etkisi 0,00208 dB. Bu, ayrı optimize edilmiş tasarımların
kıyaslaması değildir.

Minimum retraining tasarımı dört kol: mevcut yapı; adapter identity;
repair identity; her ikisi identity. Aynı released başlangıcı, frozen
encoder/entropy, veriler, update sayısı ve multi-exit loss. İki rapor
ver: aynı dondurulmuş map'te mekanizma; calibration'da yeniden ayarlanan
policy ile sistem frontier'ı. İkinci rapor birincinin yerine geçmez.

Boundary bandı ve interior hata ayrı; final YUV, deepest drift,
repair/head ms birlikte. Repair'siz kol aynı frontier'ı daha hızlı
sağlıyorsa repair kaldırılmalı. Sonraki P2 seçenek yalnız boundary loss
eklemek; DCVC-UF ana resmî training recipe'sine sessizce eklenmez.

### EE-F — Bağlam kaybını early exit kaybından ayır. P0

Uniform her derinlik için full-frame, zero-halo tiled ve doğru sınır
kurallı yeterli-halo tiled yollarını aynı ağırlıklarla karşılaştır.
Repair iki karşılaştırma kolunda da aynı durumda olmalı. Daha önceki
8/8 exact-context smoke yalnız uniform-depth ve repair-disabled
koşulunu doğrular; farklı komşu derinliklerde eşdeğerlik kanıtı değildir.

Ölç: output max-abs, boundary/interior MSE, halo MAC/bellek ve gerçek
süre. Yeterli halo kaliteden kazanırken süre avantajını kaldırıyorsa
bu trade-off gösterilir. Karma haritalarda “full-frame eşdeğeri”
tanımlanmadan bit-exact iddia edilmez. Ayrıntı:
[CONTEXT_CONTROL_TR.md](CONTEXT_CONTROL_TR.md).

## 3. Bağımsız model bankasına ait ayrı deneyler

**BANK-A / P0 — Eş eğitimli D12.** Aynı resmî Open Images train_0/1/2,
105 epoch, optimizer/QP/init politikası ile D12-scratch, released D12'nin
yanına eklenmeli. Released model dağıtım anchor'ıdır; eğitim geçmişi
eşleşmediği için tek başına derinliğin nedensel kontrolü değildir.
Mevcut D2/D4/D6 işlerini kesme veya tariflerini değiştirme. Cihaz açıldığında
önce bu kontrol; sonra D8/D10. GPU-saat tahmini, aynı kartta ölçülen
256/512 crop throughput'u olmadan verilmez.

**BANK-B / P0 — Ucuz model seçici vs encoder-side best match.**
QP/bütçe-only; varyans+kenar; küçük RGB MLP; bütün adaylarda gerçek
encode–reconstruct–RD araması. Dondurulmuş codec/QP adayları aynı.
Arama sonucunu hesaplamak için bütün adayların analysis, entropy,
synthesis ve karar maliyetini öde. Deneme bitstream'leri tekrar
kullanılıyorsa bunu açık yaz; kullanılmıyorsa son encode'u da say.
“MLP maliyeti” girdinin oluşturulmasını da içerir. Aday codec seçilmeden
elde edilemeyen latent feature bedava girdi değildir.

P1 uzantı: yalnız belirsiz patch'lerde top-2 dene. Belirsizlik eşiği
calibration'da sabitlenir; ikinci denemenin bütün maliyeti eklenir.
Bu kollar için henüz eğitilmiş bir bank selector veya hız sonucu yok.

Akıllı ama sınanabilir seçenek: salt softmax belirsizliği yerine eğitim
verisinde **ikinci denemenin beklenen RD-regret azalmasını** tahmin et;
yalnız bu kazanç ek encode zamanını karşılıyorsa top-2 çalıştır.
Kontroller aynı ortalama ikinci-deneme sayısında random, confidence ve
predicted-regret gate. Kazanç kadar selector ve ikinci candidate maliyeti
de sayılır. Bu P1 öneridir; uygulanmış yeni yöntem veya yenilik iddiası değil.

**BANK-C / P1 — Altı expert gerçekten gerekli mi?**
{2,12}, {2,6,12}, {2,4,6,8,10,12}; sonra yalnız tam bankanın
frontier'ında kullanılan ara expertler için leave-one-out.
Aynı QP numarasıyla kalite kıyaslamak yerine common actual-rate desteği
ve aynı kalite/süre hedefi kullan. Model-weight belleği, yükleme zamanı,
expert başına batch doluluğu dahil. Sadece seçilme oranı katkı kanıtı değil.
Eşit kullanım zorlayan ClassSR tarzı yardımcı loss, dondurulmuş uzmanlar
için zorunlu varsayılmamalı: no-balance / balance ancak ayrı selector
ablasyonu. Kullanılmayan bir expert bazen doğru optimumdur.

**BANK-D / P0 — Model seçimi mi, blok sınırı mı?**
Full-frame D12 → sabit D12 patch codec → heterojen patch codec;
son ikisinde aynı geometri, header ve entropy reset sözleşmesi.
Sabit map'te merge/no-merge, sonra aynı selector'la tekrar. Haritayı
değiştirip birleştirme etkisi ölçülmez. Her modelin encoder/entropy'si
ayrı; başka modelin latentini decoder'a geçirerek hız karşılaştırılmaz.

## 4. Ölçüm ve istatistik sözleşmesi

**Kalite:** kullanıcı önceliği YUV PSNR. Bankanın mevcut kaydı
YUV611 4:4:4, shared-exit kaydı YUV611 4:2:0'dır. Her aile içinde
aynı evaluator; aileler arası yeni kıyasta renk matrisi/range, chroma
downsample, clipping, crop ve ağırlıklar tek sürümde dondurulmalı.
Eski Δ444 bulguları korunur. Ayrıntı [METRICS.md](METRICS.md).

**Rate:** rANS payload ve container/header/map dahil total bpp birlikte;
entropy estimate yalnız teşhis. Ortak RD desteğinde log-rate
interpolasyon, extrapolation yok. Her noktada n ve eksik destek listesi.
Ana uygulama kararı total bpp üzerinde; eski payload analizleri açık
etiketleriyle kalır. Beş-QP gridinin interpolasyon hassasiyeti için
linear/PCHIP kontrolü; arşivde olmayan sayıyı ölçülmüş QP noktası diye çizme.

**Süre:** aynı boş GPU, aynı power/precision/compile ayarı, kaynak ve
harita dondurulmuş. İlk geçiş/compile ve warm steady-state ayrı.
30 warm-up, 100 tekrar × 3 ölçüm oturumu başlangıç protokolü;
model sırası oturumda randomize, CUDA sync toplam ölçüm çevresinde.
İç aşamaların senkronize ölçümü ayrı profiling geçişi; profiling toplamı
örtüşme nedeniyle gerçek end-to-end süreye eşit sayılmaz. Tek-görüntü
latency ile batch throughput ayrı. Başka eğitimle aynı GPU'da ölçülen
süre publication sonucu olarak kullanılmaz.

**Bütçe/fallback:** encode karar süresi + gerçek encode, decode entropy
+ shared stem/predictor + packing/suffix + repair/head + assembly.
Yükleme/transfer dahil cold ölçüm ayrı. Decoder tarafında çalışan
exit router ile encoder'ın source-informed araması aynı gözlem setine
sahip değildir; bu fark açıkça raporlanır.

**Split:** resmî codec eğitim setinin tamamı korunur. Router label
üretimi eğitim verisinden yapılabilir; hyperparameter tuning,
risk calibration ve final test kaynak aileleri ayrılır. Aynı video,
farklı çözünürlükleri, komşu kareleri ve near-duplicate görüntüleri
tek grup say. Mevcut 53 kaynak/51 aday grup ve DIV2K100 zaten
incelendi: bunları sonradan untouched test diye adlandırma. Yeni
external cohort ancak outcome-blind provenance/duplicate audit ve
dosya-hash split manifest'i tamamlanınca kilitlenir; şu an hazır değil.

**İstatistik:** tekrarların birimi bağımsız görüntü/kaynak grubu;
patch veya aynı görüntünün QP'leri bağımsız n değildir. Paired grup
bootstrap, seed20260929, 5.000 resample, %95 CI; QP'ler ayrı ve ortak
gruplar korunarak pooled ikincil sonuç. Eğitim seed etkisi için ayrı
tekrarlar gerekir. Çok sayıda kolu tarayıp en iyi test sonucuna sıradan
CI vermek selection bias'ı çözmez.

**Risk:** ortalama kayıp ve aşım olasılığı farklı iddialardır. Yeni
kontrolde bounded indicator `1[YUV_loss > target]` ve önceden sabit
sonlu politika gridine exact binomial/Bonferroni sınırı uygulanabilir;
monoton kalite varsayımı gerekmez. Ancak bağımsız calibration gerekir.
%95 güvenle aşım olasılığı ≤%5 için, sıfır ihlal görülse bile tek
politika **59**, altı aday **94**, 60 aday **139** bağımsız örnek ister.
Bu en iyi durum hesabıdır, mevcut deneyde sıfır ihlal iddiası değildir.
51 geliştirme grubu bu sertifikasyonu sağlayamaz. Adil karşılaştırmada
MLP ve Bayer aynı risk prosedürüne tabi tutulur.

## 5. Çalıştırma ve teslim sırası

1. **Tamamlandı:** mevcut gerçek-byte sonuçlarından bankanın etiket
   kararlılığı, metric sensitivity ve calibration örnek sayısı analizi.
2. **Hazır:** 318 EE fragmentation girdisi. **Tamamlandı:** iki küçük
   kaynakta 12-vaka CPU smoke; dört orijinal politika çıktısının RGB/YUV
   baseline'ı yeniden üretildi, CUDA açılmadı. Kayıt:
   [fragmentation_smoke.json](data/ablation20260929/fragmentation_smoke.json).
   Full-cohort decode ve native timing henüz yok.
3. **İlk yeni ölçüm:** EE-A/EE-F aynı çıktı ve gerçek süre yolu;
   ardından EE-B'nin 318 vakası. Bunlar codec retraining gerektirmez.
4. **Sonraki küçük eğitim:** EE-C'nin üç predictor loss'u ve EE-D gate;
   ancak split, label checkpoint'i ve ölçüm yolu dondurulduktan sonra.
5. **Uzun eğitim:** mevcut üç resmî işi bitir; D12-scratch, D8/D10;
   adapter/repair dört kolunu kaynaklar uygunken ayrı çalıştır.

Tekrarlanabilir komutlar (Python + NumPy):

```bash
python scripts/ablation_readiness_20260929.py
python scripts/prepare_fragmentation_ablation_20260929.py --source-cases /path/to/shared_crossfit_qp32/cases
```

Bu iki komut `cvpr2027/` içinden veya script'in tam yolu ile çalışır;
ilki bundle içindeki verilerle taşınabilir, ikincisi hash'leri kaydedilmiş
53 raw case dosyasını ister. CPU smoke sürücüsü ana repo'da
`scripts/check_fragmentation_ablation_20260929.py`; pinned checkpoint,
kaynak frame'ler ve frozen source snapshot gerektirir. Çalışmış sonuç
üzerine yazmayı reddeder. GPU benchmark veya yeni training başlatmaz.

Makale için önerilen üç yeni panel: eş kalite/rate altında **net süre
frontier'ı**; aynı histogramda **yerleştirme–parçalanma kontrolü**;
**bütçeye göre MLP/Bayer/gate kazanımı**. Bankanın model sayısı paneli
D8/D10 olmadan altı modelin sonucu gibi çizilmez. Bu tur Fig1/2 veya
mevcut ölçülmüş figürler değiştirilmedi.
