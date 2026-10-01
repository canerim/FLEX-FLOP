# DCVC-UF derinlik bankası: ölçülen durum ve router tasarımı

**Veri kesiti:** 1 Ekim 2026, yaklaşık 23:47 Berlin. Bu dizindeki dört figür PDF ve SVG olarak vektördür; PNG önizlemeleri de vardır. [Kaynak ve SHA-256 kaydı](figure_evidence.json) ile [yalnız CPU kullanan çizim betiği](make_figures.py) yeniden üretimi sağlar. Betik çalışan eğitimlere dokunmaz.

| Figür | İçerik | Vektör | Önizleme |
|---|---|---|---|
| 01 | Tamamlanan D2/D4 ve released D12: Kodak RD, eşlenmiş fark ve görüntü-bootstrap aralığı | [PDF](fig01_kodak_final_depths.pdf) · [SVG](fig01_kodak_final_depths.svg) | [PNG](fig01_kodak_final_depths.png) |
| 02 | D2/D4/D6/D8 izleme eğrileri; D10/D12 kapsamı ve altı modelin parametre ayak izi | [PDF](fig02_training_and_capacity.pdf) · [SVG](fig02_training_and_capacity.svg) | [PNG](fig02_training_and_capacity.png) |
| 03 | Kaynak tarafı MLP, altı bağımsız codec, bitstream ve decoder akışı | [PDF](fig03_router_system.pdf) · [SVG](fig03_router_system.svg) | [PNG](fig03_router_system.png) |
| 04 | Offline altı-aday etiketleme ile online tek-aday çalıştırmanın ayrımı | [PDF](fig04_router_learning.pdf) · [SVG](fig04_router_learning.svg) | [PNG](fig04_router_learning.png) |

## Ölçülen bulgular

Altı derinliğin **aynı tarifi kullanan bağımsız codec** olması bu işin temelidir. D2 ve D4 105 epoch tamamladı; D6 yaklaşık 94,9, D8 3,5, D10 0,2 epoch'taydı; D12-scratch sıradaydı. D12 için Kodak grafiğinde görünen çizgi **Microsoft'un yayımlanmış checkpoint'idir**; sıradaki D12-scratch eğitimi değildir. D2/D4/D6/D8/D10 Xavier seed 42 ile sıfırdan, encoder ve entropy dahil uçtan uca eğitildi/eğitiliyor. Distillation yok.

**Figür 01:** 24 tam çözünürlüklü Kodak görüntüsünde, aynı **0,2 tahmini bpp** ve görüntü başına log-rate enterpolasyonuyla D2, released D12'nin **0,277 dB**; D4 **0,157 dB** gerisinde. Görüntü-bootstrap %95 aralıkları sırasıyla **[−0,326, −0,232]** ve **[−0,185, −0,131] dB**. 0,1 ve 0,4 bpp bulguları figürde ve `figure_evidence.json` içindedir; 0,4'te üç modelin ortak desteği 23 görüntüdür. Bootstrap eğitim seed belirsizliğini kapsamaz. Bitrate **deterministik entropy tahminidir; gerçek payload/container baytı değildir.** Bu farklar router kazancı ya da BD-rate değildir.

**Figür 02:** Diğer derinliklerin kalite eğrileri yalnız dört sabit DIV2K merkez-512 crop'undan gelen eğitim izlemesidir. D6'nın 90. epoch'tan sonra düşüşü, 512 piksel crop ve yüksek öğrenme oranı geçişiyle aynı zamana denk gelir; D2/D4 de bu geçişten sonra düşüp toparlandı. D8 çok erken aşamadadır. D10'un mevcut ilk kaydı 0,2 bpp'yi kapsamıyor; bu yüzden kalite noktası çizilmedi. Figür 01 ile Figür 02'nin dB değerleri **farklı görüntü kümeleri** olduğundan çıkarılıp karşılaştırılamaz.

Parametrelerin 27,948 milyonu her derinlikte decoder dışındaki modüllerde kalır; decoder 3,845 milyondan (D2) 14,232 milyona (D12) büyür. Toplam 31,792–42,179 milyon parametredir. D2 toplam parametrede D12'ye göre **%24,6 daha küçük**; bu değer **%24,6 hız kazanımı anlamına gelmez**. Altı modelin FP32 ağırlıkları birlikte yaklaşık **0,83 GiB** tutar; çalışma alanı, entropy tabloları ve patch batch'leri bu sayıya dahil değildir. Her codec kendi encoder ve entropy ağırlıklarını taşır.

## Ne tür router?

Burada eğitilecek router **altı bağımsız codec arasında patch seçicidir**. Paylaşılan-latent early-exit router'ı başka bir deneydir. Bankadaki `Dk`, yalnız k bloklu decoder demek değildir: her Dk kendi encoder, hyperprior, entropy ve decoder parametreleriyle baştan eğitilir. Dolayısıyla bir D2 encoder'ın latenti D8 decoder'a verilemez. Decoder, seçilen codec kimliğini bitstream'den öğrenmelidir.

Pratik akış: görüntüyü sabit tile'lara böl → kaynak tile ve hedef bütçeden ucuz özellikler hesapla → MLP ile her derinliğin beklenen bayt, Y/U/V hata ve süre değerini tahmin et → bütçeye uyan model haritasını seç → tile'ları seçilen modele göre grupla ve **yalnız seçilen codec'i** çalıştır → model kimliği/QP/uzunluk bilgisiyle stream'i yaz → decoder bu kimliği okuyup ilgili codec'i çalıştırır → tile'ları birleştirir. Bir tile için diğer beş codec'in encoder'ını çalıştırmak online ana yöntem değildir.

Başlangıç MLP'si somut olarak `8×8` luma havuzlaması (64 özellik), YUV ortalama/standart sapma (6), gradyan yön/enerji özeti (8), yüksek frekans/saturasyon özeti (4) ve QP (1) alabilir: **83 özellik → 128 → 64 → 6×5** çıktı, yaklaşık 20 bin ağırlık. Çıktılar her expert için `log(bytes)`, `log(SSE_Y)`, `log(SSE_U)`, `log(SSE_V)` ve kalibre edilmiş süre/hız tahminidir. Router bütçeyi sonuç başlığına katmak yerine karar aşamasında kullanır; böylece bir MLP farklı kalite/hız bütçelerine hizmet eder. Bu mimari **öneridir, eğitilmiş veya ölçülmüş bir router sonucu değildir**. Daha basit varyans/LUT özellikleri ve daha küçük ağlar ablasyonla karşılaştırılır. Özellik çıkarma süresi ayrıca ölçülür.

### Eğitim hedefi ve gerçek kalite hesabı

Etiket üretiminde router eğitim görüntülerinin her tile'ı altı frozen codec ve seçilmiş QP'ler ile **offline** kodlanır/çözülür. Her aday için gerçek stream baytı, Y/U/V toplam kare hatası ve encode/decode süresi tutulur. Tile geometrisi, halo, padding, overlap/blending, entropy reset ve container biçimi tüm adaylarda aynıdır. Bitstream kimliği için altı expert en az 3 bit/tile gerektirir; gerçek toplam oran tile payload, başlık, offset ve olası padding ile birlikte ölçülür.

Tile PSNR'larını aritmetik ortalamak yanlış görüntü PSNR'ı verir. Kullanacağımız rapor metriği görüntü birleştirildikten sonra, her düzlemin **tüm geçerli pikselleri** üzerinde hesaplanır:

`Q611 = [6·PSNR(Y) + PSNR(U) + PSNR(V)] / 8`.

Görüntü bazında `SSE_Y/U/V` toplanıp sonra dB'ye dönüştürülür; overlap karışımı varsa assembled görüntü yeniden ölçülür. Bu çalışmada YUV **4:4:4**; eski early-exit yolundaki YUV420 sayılarıyla doğrudan aynı metrik değildir. Router kaybı `log(bytes)` ve `log(SSE_c)` için sağlam regresyon ve adayların **fayda sıralaması** için yardımcı regret kaybından oluşur. Sınıf doğruluğu tek başına karar ölçüsü değildir: yakın adaylar arasında yanlış seçim önemsiz, büyük kalite kaybı yaratan yanlış seçim pahalıdır.

Asıl değerlendirme şu kısıtlı karardır: gerçekleşen görüntü kalitesini yükseltirken `actual_bpp ≤ B` ve ölçülmüş toplam/decoder gecikmesi `≤ T` koşullarına uymak. İlk çözüm için aday skoru `D̂ + λR̂ + βT̂ + γ·boundary_cost` kullanılır; `λ, β` kalibrasyon kümesinde bütçe taramasıyla seçilir. `D̂`, görüntü boyunca düzlem kare hatalarından türeyen kalite kaybı için yaklaşımdır; doğrudan tile dB ortalaması değildir. `boundary_cost`, komşu tile'larda farklı expert kullanımının gerçek harita/header/stitching maliyetini temsil eder. GPU'da gruplandırma ve batch boyutu sürede doğrusal davranmayacağı için `T̂` cihazda ölçülen grup boyu lookup tablosuyla kalibre edilir. Bütün kararlar gerçek encode sonrası raporlanır; tahmin edilen bütçenin tutması başarı sayılmaz.

### Veri bölme ve karşılaştırmalar

Open Images `train_0/1/2` uzman codec eğitimidir. Router için **ayrı görüntü kimlikleri**, görüntü seviyesinde train/calibration/test bölmesi ve içerik hash'iyle örtüşme denetimi gerekir. Mevcut dört DIV2K monitor görüntüsü ve geliştirme sırasında bakılmış Kodak sonucu bağımsız, hiç görülmemiş test kanıtı diye sunulmaz. Yayımlanacak ana iddia için router, QP/bütçe ızgarası ve eşikler dondurulduktan sonra yeni test kümesi gerekir.

Karşılaştırma sırayla: (1) full-frame released D12, (2) aynı geometriyle **fixed D12 tile** (yalnız tiling cezası), (3) her Dk sabit tile, (4) bütçe/QP-only LUT, (5) varyans eşiği ve Bayer/random yerleşim, (6) görüntü seviyesinde tek codec, (7) patch MLP, (8) tüm adayları görmüş offline oracle. Sabit modellerin sonucu gerçek bitrate üzerinde enterpole edilir; aynı QP'nin aynı bitrate olduğu varsayılmaz. Oracle'ın tüm aday maliyeti online kullanılırsa hız hesabına altı encode/decode'un tamamı girer. Ana nokta **aynı gerçek bpp ve benzer YUV kalite altında uçtan uca süre**; ayrıca encode, entropy, router, transfer, decode, stitch ve peak memory ayrı raporlanır. Bütçe ihlal oranı ve zor görüntülerdeki kalite kaybı ortalama kazancın yanında verilir.

### Yapılacak işler ve risk kapıları

1. D6/D8/D10/D12-scratch'in aynı 105 epoch tarifini tamamlaması; bağımsız bitstream yolunda her derinlik için encode→decode doğrulaması. Mevcut Kodak final dosyaları entropy **tahmini** içerir; gerçek bpp diye kullanılamaz. Upstream'in bazı inference yardımcıları 12 bloğu varsaydığından küçük derinlikleri sessizce yanlış çözen yol reddedilir.
2. Bütün frozen modellerde aynı tile/halo/QP taramasıyla tam görüntü reconstruction, actual bytes ve encode/decode timing tablosu. Tile sınırları ve çoklu model haritasının paketlenmesi birlikte ölçülür.
3. Ayrı router görüntülerinde altı-aday offline hedefler, küçük MLP eğitimi, ayrı kalibrasyonda bütçe seçimi. Önce basit LUT/eşik karşısında üstünlük aranır; artış yoksa karmaşık router ana yöntem yapılmaz.
4. Dondurulmuş bütçe ızgarasında gerçek bitstream ve cihaz süresiyle final karşılaştırma; expert sayısı 2/3/6, MLP vs fayda/regret hedefi, patch 256/512, halo, model-haritası sıkıştırma ve grouped execution ablasyonları.

## Literatür konumu

[ClassSR (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.html) içerik bazlı patch model seçimini süper çözünürlükte gösterdi; bitstream ve entropy maliyeti orada yok. [AdaNIC (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Tao_AdaNIC_Towards_Practical_Neural_Image_Compression_via_Dynamic_Transform_Routing_ICCV_2023_paper.pdf) image codec içinde uzamsal dönüşüm kapasitesi yönlendirir. [Spatial Competition (2026)](https://arxiv.org/abs/2605.13243) farklı codec'ler ve mod haritasını ele alır. [MixCompress (2026)](https://arxiv.org/abs/2607.14334) MoE/MoD ile kapasiteyi rate'e göre değiştirir. Bu nedenle “ilk adaptive codec” iddiası doğru değildir. Burada savunulabilir katkı, **aynı resmî tarifle eğitilmiş 2–12 bloklu bağımsız DCVC-UF bankasının** gerçek rate/kalite/süre altında, codec kimliğini bitstream'e taşıyan **kaynak ve bütçe koşullu seçim** ile sistematik değerlendirilmesidir. Bu katkı, yukarıdaki gerçek sonuçlar gelince test edilebilir; şu an mimari hipotezdir.
