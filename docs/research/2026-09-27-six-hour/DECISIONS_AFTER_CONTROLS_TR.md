# Yeni kontrollerden sonra araştırma kararları

28 Eylül 2026. Tamamlanmış ara kontrollerden çıkan kararlar; önerilen yöntemlerin başarısı henüz ölçülmedi. Resmî D2/D4/D6 codec eğitimlerine müdahale yok.

**Ana iddiamız adaptif synthesis; bunu taşıyan soru ek derinliğin nerede yararlı olduğu ve o kararı uygulamanın toplam maliyetidir.** Altı model sayısı veya MLP kullanımı tek başına katkı değildir. Shared early exit ile bağımsız codec bankasını, farklı bilgi paylaşımı olan iki uygulama olarak karşılaştıracağız.

## Şu anda kanıt ne söylüyor?

| Gözlem | Çıkardığımız karar | Çıkarmadığımız sonuç |
|---|---|---|
| D6 epoch20, 0.2 actual payload bpp: core-only −0.0611 dB; halo32 −1.2914 dB; halo64 −1.6666 dB, aynı model full frame'e göre, 16 görüntü | Bağımsız bankada context'in tekrar kodlanmasını açık maliyet olarak modele kat | Her halo/native uygulaması aynı RD kaybını verir |
| D6/QP32 halo32: +39.72% payload, yalnız +0.0020 dB ortalama PSNR; seam excess azalıyor | Sınır kalitesini ve equal-rate global kaliteyi ayrı raporla | Güzel görünen seam daha iyi codec demektir |
| Sabit map'te dört bölgeden iki bölgeye geçiş: neural decoder Conv2d MAC −12.44% | Merge/no-merge aynı map üzerinde temel sistem ablation'ı olsun | Runtime aynı yüzde hızlanır veya kalite korunur |
| Source-disjoint control replay, QP32: router−dither mean saving +1.87 pp, CI [−1.60, 5.19]; Q90 +2.54 pp, CI [−0.42, 5.54] | Ucuz baselineları ciddi rakip kabul et; dış testte paired üstünlük göster | Router'ın pozitif marjı bu replay ile kanıtlandı |
| Aynı bütçede D4 seçeneğinin whole-crop koşullu değeri yaklaşık 0.009 dB (0.2 bpp) | D4/8/10 leave-one-out; seçenek sayısının faydasını ölç | Altı expert mutlaka üçten iyi deployment sistemi olur |

## En değerli yeni tasarım: fayda tahmini + maliyetli bölge uygulaması

Codec'ler eğitildikten sonra, source-only küçük bir ağ her core ve QP için her expert'in beklenen distortion/rate değerini tahmin edebilir. Etiket “texture sınıfı” yerine RD ve ek compute faydası olur. Basit ilk sürüm `D_hat + lambda R_hat + mu C` ile seçim yapar. İkinci sürüm komşu farklı uzman sınırlarına bir maliyet ekler; ardından aynı uzman komşularını birleştirir. Sınır cezasını test görüntüsünün gerçek hatasına bakarak ayarlamayız.

Mevcut e15 router zaten yumuşatılmış cost-aware hedef ve regret terimi kullanıyor. Dolayısıyla “ilk kez regret eklemek” yeni fikir değildir. Buradaki öneri, bağımsız codec bankasında gerçek coded rate ve bölge uygulama maliyetini de tahmin edilen hedefe katmak; bunun mevcut soft-label sınıflandırmaya göre farkını ölçmektir.

Bu fikrin önceki çalışmalardan farkını varsaymayacağız. [APE](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780286.pdf) katmanların ek faydasını tahmin eder; [ClassSR](https://arxiv.org/abs/2103.04039) patch'leri farklı kapasitelere yönlendirir. [Spatial Competition §2.2](https://arxiv.org/html/2605.13243v1#S2.SS2) aynı-mode komşularını sürekli işler ve mode map taşır. Bizim araştırma sorumuz, DCVC-UF derinlik ailesinde pahalı kaynak aramasının ne kadarının ucuz seçimle karşılanabildiği ve bunun gerçek codec maliyetine yansıyıp yansımadığıdır.

| Ablation | Sabit tutulan | Değişen | Karar ölçüsü |
|---|---|---|---|
| Regret/benefit hedefi vs expert-ID cross entropy | Ağ boyutu, eğitim/veri/seed, expert ailesi | Etiket/loss | Actual RD–cost regret; sınıflandırma accuracy ikincil |
| QP var/yok | Ağ kapasitesi, source features | QP bilgisi | Her QP'de equal-rate kalite ve maliyet |
| Bağımsız karar vs sınır cezası | Predictor ve candidate costs | Spatial regularization | Önce histogram farkı; sonra aynı-histogram kontrolü |
| Aynı map, merge kapalı/açık | Her pikselin expert'i | Bölge uygulaması | Actual payload, header, PSNR, encoder/decoder wall time |
| Üç expert vs altı expert vs leave-one-out | Eğitim protokolü ve bütçe | Erişilebilir kapasite seçenekleri | Kalite–maliyet eğrisi, resident memory, grouping overhead |
| Uniform / Bayer / histogram-shuffle / MLP | Candidate set, metrik ve reference | Konum bilgisi | Aynı achieved rate/quality'de paired compute ve latency farkı |
| Shared prefix search vs tekrarlı tüm-exit arama vs MLP | Aynı candidate outputs ve kalite | Karar bilgisini edinme | Encoder toplam wall time; pahalı baseline'ı seçerek avantaj üretme yok |

## İkinci fikir: bütçeye göre router'ı devreden çıkarmak

Mevcut source-calibrated eğrilerde gevşek bütçede ek routing marjı küçülüyor. Bu yüzden sabit bir bütçe bölgesinde doğrudan uniform/Bayer seçen, daha sıkı bölgede MLP çalıştıran bir kontrol denenebilir. Eşik yalnız calibration verisinde seçilir ve final testten önce dondurulur. Baseline her zaman MLP çalıştıran aynı sistemdir. Her iki kolda achieved-quality dağılımı ve toplam karar/dispatch süresi ölçülür. Bu bir kalite garantisi veya doğrulanmış hızlanma değildir.

Bu bypass'ın önemi MAC oranından tahmin edilmemeli: küçük MLP aritmetiği bile GPU kernel/dispatch gecikmesi yaratabilir; tersine selector maliyeti ihmal edilecek kadar küçük de çıkabilir. Native ölçüm belirleyici olacak.

## Sonuç kabul etmeden önce

1. Final codec checkpoint'lerini sabitle; released D12'yi pratik anchor, aynı-recipe scratch D12'yi depth kontrolü olarak ayır.
2. Router training, calibration ve test kimliklerini ayır. İncelediğimiz 100 DIV2K validation görüntüsü ve 53 CTC sequence artık dokunulmamış test değildir. Aynı görüntünün crop/QP'lerini farklı splitlere bölme.
3. Önce bağımsız decode ve native stock/patched D12 eşleşmesi; sonra latency. Üç eğitim GPU'su doluyken native ölçümü çalıştırma.
4. Aynı donanım/precision/shape/warmup ile paired wall time; codec encoding içinde GPU synthesis ile entropy worker örtüşmesini koru. Parça sürelerini toplayıp toplam süre üretme.
5. Etki küçükse CI'yi saklama. QP'leri bağımsız görüntü gibi sayma; görüntü/sequence düzeyinde eşleştirme ve çoklu seed sonuçlarını ayrı göster.

Native-padding kontrolü tamamlandı: D6/0.2 payload bpp/16 görüntüde +0.2652 dB geri kazanım var, ancak düzeltilmiş halo32 hâlâ full-frame D6'dan 1.0262 dB geride. QP32'de padding yeri değişimi payload'ı %6.03 azaltıyor. Bu CPU geometri sonucu native CUDA output/timing eşdeğerliği değildir. Region-merge ve frozen-component kontrollerinin tamamı bitmeden bu kolların beklenen kazancı sonuç olarak yazılmaz.
