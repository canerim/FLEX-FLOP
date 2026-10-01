# Altı bağımsız DCVC-UF codec için router eğitim protokolü

**1 Ekim 2026 — uygulanacak ilk sürüm.** Codec bankası D2/D4/D6/D8/D10/D12-scratch'tir. Released D12 dış referanstır. Codec'ler kendi encoder/entropy/decoder parametreleriyle 105 epoch resmî tarifte eğitilir; router eğitimi onların ağırlıklarını değiştirmez. Bu belge **önerilen ve henüz yürütülmemiş** router deneyini tarif eder. D6/D8/D10/D12 ve gerçek bitstream doğrulaması tamamlanmadan altı-yollu etiket üretimi başlatılmaz. Kaynak kodun `code_snapshot_v2` sürümü, tile geometrisi ve modellerin SHA-256 özeti etiketlerden önce dondurulur.

## 1. Karar birimi ve veri

İlk uygulamada bütün görüntüler **256×256 geçerli tile grid** üzerinde bölünür. Eksik kenar tile'ı deterministik pad/crop ve geçerli alan maskesiyle ele alınır. İlk kontrollü sürüm overlap/halo olmadan çalışır; `halo=16/32` ve 512 tile ayrıca ölçülür. Geometri ve birleşim kuralı **codec adayı değiştirilmeden önce** sabitlenir. Her tile'a tek model kimliği `k∈{2,4,6,8,10,12}` verilir. Girdide kaynak RGB'ye erişim yalnız encoder tarafında vardır. Decoder bitstream'deki `k`, QP, tile konumu/şekli ve payload uzunluğunu okuyup ilgili codec'i çalıştırır. Bağımsız modellerin latentleri birbirine takılmaz.

Codec uzmanlarının `train_0/1/2` görüntüleri router veri havuzuna girmez. Router için örtüşmeyen Open Images görüntü kimlikleri (`train_3` veya doğrulanmış validation kaynağı) kullanılır; dosya/decoded-pixel hash'iyle tekrarlar taranır. **Bölme birimi kaynak görüntü**, tile veya QP değildir. İlk pilot: 512 adet 512×512 görüntü crop'u × 4 tile × 6 codec × QP `{0,16,32,48,63}` = **61.440 aday codec çağrısı**. Önce tek bir adayın gerçek süre ve disk maliyeti ölçülür; pilot başarılıysa image-level train/dev/calibration/test bölmeleriyle birkaç bin görüntüye genişletilir. Test görüntülerine göre özellik, loss katsayısı veya bütçe ızgarası seçilmez. Mevcut dört DIV2K monitor crop'u router testine dahil edilmez.

Bir tile `i`, expert `k`, QP `q` için offline arşivlenen gerçek hedefler:

- `Bᵢₖq`: tam kodlanmış tile payload baytı; ayrı olarak container/header ve mode-map baytı.
- `Sᵢₖq,c`: decoded tile'ın **geçerli** piksellerinde kanal `c∈{Y,U,V}` için toplam kare hatası; `Nᵢ` geçerli piksel sayısı.
- `τₖ(b,g,q)`: aynı cihazda tile şekli `b`, expert grubu boyu `g`, QP `q` için encode/entropy/decode/transfer süre tablosu. Her çağrının duvar süresi tanısal tutulur, ancak asenkron GPU ve grup etkileşimi yüzünden MLP'ye ham tek-tile süre etiketi verilmez.
- `codec_hash`, `tile_geometry`, `source_image_hash`, `q`, `status`; başarısız/bit-exact olmayan adaylar sessizce atılmaz, `invalid` olarak kayıtlanır.

Öğretmen bütün altı adayın sonuçlarını **yalnız offline** görür. Bu 6× maliyet online sistemin hızı diye raporlanmaz.

## 2. Girdi, ağ ve tahmin

Küçük özellik vektörü: 8×8 pooled luma (64), Y/U/V mean ve std (6), 8-bin gradient özeti (8), dört yüksek-frekans/renk özeti (4), normalize QP (1): **83 sayı**. Bütün normalizasyon ortalama/std'leri yalnız router-train bölmesinden hesaplanıp dondurulur. Çıktı budget'ten bağımsız tutulur; budget aşağıdaki karara girer.

İlk ağ: `83 → 128 → 64 → 24`, ara aktivasyon GELU, girişte LayerNorm, gizli katmanlar arasında dropout 0,1. Her altı expert için dört çıkış vardır:

`ẑᵣ = log(predicted payload bpp)`, `ẑY = log(predicted MSE_Y)`, `ẑU = log(predicted MSE_U)`, `ẑV = log(predicted MSE_V)`.

Pozitif değer log uzayında üretilir, karar için `exp` ile geri alınır. Hedef log değerler expert/kanal bazında train istatistikleriyle standartlaştırılır; tahminler fiziksel birime ters dönüştürülmeden expertler karşılaştırılmaz. Model yaklaşık **20 bin ağırlık** içerir. Özellik çıkarma, normalizasyon, MLP ve model-haritası üretme süresi uçtan uca encode zamanına eklenir. Daha küçük `83→64→24` ve bütçe/QP-only LUT kontrolü aynı split üzerinde çalıştırılır.

## 3. Loss fonksiyonu

`mᵢₖq,c = Sᵢₖq,c / Nᵢ` ve `rᵢₖq = 8 Bᵢₖq / Nᵢ`. Log hedeflerde sıfır için yalnız ölçüm çözünürlüğünden gelen küçük sabit `ε` kullanılır; `ε` ve normalizasyon eğitimden önce kaydedilir. Standartlaştırılmış tahmin/gerçek farkına `Huber₁(·)` uygulanır. Tüm geçerli expert/patch/QP adayları üzerinde ilk regresyon terimi:

`L_pred = 0.5·Huber(log r) + 0.5·[0.75·Huber(log m_Y) + 0.125·Huber(log m_U) + 0.125·Huber(log m_V)]`.

Kanal ağırlıkları YUV 6:1:1 rapor metriğiyle uyumludur. Regresyondaki 0,5/0,5 rate–distortion başlangıç ağırlığı **hiperparametredir**, fizik kanunu değildir; `0.25/0.75` ve `0.75/0.25` dev setinde önceden tanımlı ablasyondur. Tek başına minimum tahmin hatası doğru model seçimini garanti etmez. İkinci terim karar pişmanlığını hedefler.

Her minibatch'te geliştirme verisinden **önceden sabitlenmiş** rate/zaman multipliers ızgarasından `(λ,β)` örneklenir. Gerçek aday yerel skoru

`Uᵢₖ = 0.75 log mᵢₖ,Y + 0.125 log mᵢₖ,U + 0.125 log mᵢₖ,V + λ rᵢₖ + β τₖ`

ve MLP çıktılarından aynı biçimde `Ûᵢₖ` hesaplanır. `τₖ` tek patch zamanı değil, hedef cihaz ve beklenen grup boyu için lookup değeridir. Softmax sıcaklığının `(λ,β)` birimlerinden etkilenmemesi için önce `Û` expertler arası sağlam aralığına bölünür: `p̂ᵢₖ = softmax(−Ûᵢₖ / [T_soft·max(IQRⱼ(Ûᵢⱼ),s₀)])`. İlk `T_soft=0,25`; `s₀`, train setindeki pozitif expert-IQR değerlerinin 10. yüzdelik dilimidir ve sıfıra yakın eşit adaylarda patlamayı önler. `T_soft` yalnız train/dev üzerinde ayarlanır. Regret terimi:

`L_regret = Σₖ p̂ᵢₖ · [Uᵢₖ − minⱼ Uᵢⱼ] / max(IQRⱼ(Uᵢⱼ), s₀)`.

Başlangıç birleşimi **`L = L_pred + 0.5·L_regret`**. `0.5` için `{0, 0.25, 0.5, 1}` önceden tanımlı dev ablasyonu yapılır. `α=0` yalnız sonuç regresyonu; `α>0` büyük seçim hatalarını daha güçlü cezalandırır. Uzman kullanımını eşitleyen auxiliary loss, teacher distillation veya codec ağırlığı güncellemesi ilk tarifte **yoktur**; uzmanların eşit seçilmesi hedef değildir. “En karmaşık patch en derin codec ister” gibi sabit sınıf etiketi de yoktur. Yakın iki expert'in yer değiştirmesi, büyük utility farkı olan yanlış seçimden daha az cezalandırılır.

**Önemli matematik sınırı:** Bu `Uᵢₖ` tile bazlı diferansiyellenebilir eğitim yaklaşımıdır. Gerçek makale kalitesi tile dB'lerinin ortalaması değildir. Assembled görüntü için her Y/U/V düzleminin toplam SSE'si ve geçerli piksel sayısı kullanılarak `Q611=[6·PSNR_Y+PSNR_U+PSNR_V]/8` hesaplanır. Halo/overlap eklenirse local SSE toplamı dahi assembled hatayı tam vermez; o durumda tam görüntü yeniden ölçülür. Öğretmen map ve budget kalibrasyonu bu **tam görüntü** ölçümüne dayanır.

## 4. Eğitim adımları ve model seçimi

1. **Codec doğrulaması:** D2…D12 tüm checkpoint'ler frozen; her derinlikte actual encode→decode ve model-ID round trip. Eksik bitstream yolu varken tahmini entropy bitlerini gerçek `r` etiketi olarak kullanma.
2. **Pilot etiketler:** Dört 256 tile'lı 512 crop, altı codec, beş QP. Önce 20 örnekte bayt, YUV, zaman, tiling ve hash bütünlüğü; ardından 512-image pilot. Ham aday kayıtları append-only ve kaynak SHA-256 ile tutulur.
3. **MLP:** AdamW `lr=3e−4`, `weight_decay=1e−4`, batch 512 tile/QP karar örneği, maksimum 80 epoch, dev regret+RD hatasında patience 10, gradient clipping 1,0. Üç router seed `{42,43,44}` raporlanır; codec seed 42 sabittir. Bütçe noktaları train/dev önceden belirlenir, test sonuçlarına bakılarak değiştirilmez.
4. **Bütçe kalibrasyonu:** Ağ dondurulur. Ayrı calibration görüntülerinde gerçek byte/time bütçeleri için `λ,β`, QP ve gerekirse güvenlik marjı seçilir. Map fragmentation/bitstream header ve model-gruplu batch lookup burada kalibre edilir. Batch boyu haritaya bağlı olduğundan en fazla iki ucuz harita→grup-boyu→yeniden-skorlama iterasyonu kullanılır; codec henüz çalıştırılmaz ve bu karar süresi ölçülür. Sistem gerçek bpp ve süreyi kaçırırsa ihlal oranı açıkça raporlanır; gizli çoklu encode ile düzeltilmez.
5. **Kör test:** Tek kodlama/çözme akışı, gerçek container baytı, YUV 4:4:4 PSNR, encode/decode/toplam wall-time, peak VRAM ve model-ID overhead. Karşılaştırma aynı actual bitrate ve kalite bölgesinde yapılır. Hem ortalama hem görüntü başına kötü dilim/ihlaller gösterilir.

İlk sürümde global QP, development RD LUT ile hedef bitrate'e göre seçilir; MLP QP koşullu **derinliği** seçer. Her tile için QP'yi ayrıca optimize etmek daha büyük bir arama uzayıdır ve ayrı ablasyon olur. Online bir bütçe düzeltmesi yeniden kodlama gerektirirse onun bütün süresi encode hanesine yazılır. Model seçicinin tahmini ile gerçekleşen bpp arasındaki fark da ayrıca raporlanır.

## 5. Ablasyon ve go/no-go

Sabit full-frame D12 → sabit tiled D12 → her sabit tiled Dk sırası **tiling cezasını** adaptasyon kazancından ayırır. Kontroller: bütçe/QP-only LUT, varyans eşiği, image-level tek model seçimi, Bayer/random harita, MLP yalnız `L_pred`, MLP `L_pred+L_regret`, altı adayı bilen offline oracle. Tümü aynı tile/halo, container ve gerçek bitrate protokolünde değerlendirilir. Expert sayısı 2/3/6, tile 256/512, halo 0/16/32, map sıkıştırma ve grouped execution ayrı faktörlerdir. Eğer offline oracle'ın hız/kalite alanında anlamlı bir aralığı yoksa, MLP'yi büyütmek yerine bu negatif sonucu raporlarız.

Router başarısının kanıtı yalnız `top-1` expert doğruluğu değildir: **aynı actual bpp ve YUV kalitesinde uçtan uca süre**, bütçe ihlali ve kalite regret'i. Çalışma başlatılmadan test ızgarası ve birincil karşılaştırma dondurulur. Altı adaylı oracle yalnız üst sınırdır; online uygulanırsa bütün adayların encode/decode zamanı ücrete girer. Gerekirse belirsiz tile'da top-2 ikinci aşama olarak sınanır ve ek süre açıkça eklenir.

Literatür sınırı: [ClassSR (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.html), [AdaNIC (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Tao_AdaNIC_Towards_Practical_Neural_Image_Compression_via_Dynamic_Transform_Routing_ICCV_2023_paper.pdf), [Spatial Competition (2026)](https://arxiv.org/abs/2605.13243) ve [MixCompress (2026)](https://arxiv.org/abs/2607.14334) yüzünden genel “ilk adaptif codec” iddiası kullanılmaz. Buradaki sınanabilir ayrım, aynı resmî eğitimdeki bağımsız DCVC-UF derinliklerinin gerçek container/rate/time altında kaynak ve bütçe koşullu seçimi olacaktır.
