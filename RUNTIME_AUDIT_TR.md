# Runtime iddiası: şu an ne ölçülü, neyi yeniden ölçmeliyiz?

27 Eylül 2026. Bu not, arşivdeki runtime sonuçlarını mevcut kodla eşleştiren
bir incelemedir. Yeni GPU benchmark'ı yapılmadı. Sağlıklı D2/D4/D6 eğitimleri
kesilmedi.

**Mevcut makalede güvenle söylenebilen:** kaynak hata tablosunu elde etmenin
bir hesap maliyeti var; ortak trunk geçişini yeniden kullanmak arşivlenen
tablo üretim süresini azaltmış. Gerçek uçtan uca codec hızlanması için ayrı,
nedensel bir bitstream→reconstruction ölçümü gerekiyor.

## 1. Aynı “router” adı iki farklı encoder maliyetini saklıyor

Sabit, ayrı validation üzerinde ayarlanmış beta kullanan router, kendi karar
girdileri decoder'da mevcutsa her görüntü için kaynak hata tablosu istemez.
Latent, entropy scales ve ortak stem'den hesap yapmak yine ücretsiz değildir;
ancak ortak stem'i yeniden kullanmak mümkündür.

Makalede çizilen source-calibrated sweep ise beta'yı test görüntüsünün hata
tablosuyla seçiyor. MLP'nin hızlı olması bu tablonun oluşturulmasını ortadan
kaldırmaz. Encoder'da aday reconstruction'ları üretme, kontrol arama ve gerekirse
son görüntüyü doğrulama giderleri ayrı kaydedilmelidir. Kaynağı bilen search
ile sabit router ancak bu bilgi rejimleri açıklandığında karşılaştırılabilir.

## 2. Eski “full runtime” kaydının sınırı

İncelenen `flexplus/runtime_full_uf.py` gerçek rANS çağrıları yapıyor; fakat
sonraki neural decoder'a bu çağrıların çıktısını bağlamıyor:

| Kod yolu | Somut durum | Çıkarım |
|---|---|---|
| `decode_y(idxs[k])` | İndeksler encoder tarafında üretilmiş listeden geliyor | Decoder entropy parametrelerini kendi decoded geçmişinden üretmiyor |
| `get_decoded_tensor()` | Dönen semboller okunup atılıyor | Bu akışın görüntüyü gerçekten yeniden kurduğu gösterilmiyor |
| `decoder_side(net, e, qp)` | `e` içindeki encoder'ın sakladığı `z_hat` ve `y_qs` kullanılıyor | Neural aşama, kodlanmış sembollerin bir yeniden yürütümü |
| `map_parse` | `kt_bits` ile ideal kod uzunluğu yeniden hesaplanıyor | Gerçek byte parser süresi değil |
| `dec_synth_released` | Yüklü e15 modelinin `forward_full` yolu çalışıyor | Released checkpoint latency'si diye adlandırılmamalı |
| `db_R1` | R2'nin değeri kopyalanıyor; diğer `db` alanları tablodan | Her rejimin delivered kalite kontrolü değil |

Bu bulgular modül zamanlarını yok saymayı gerektirmiyor. Ancak onların
toplamını bağımsız bitstream çözümünün uçtan uca süresi gibi sunmak doğru
değil. Eski `reports/encoder_runtime` tablosu bu nedenle tarihsel kayıt olarak
işaretlendi; yeni ana makalede doğrulanmış codec hızlanması diye kullanılmıyor.
Kod hash'leri bu inceleme anına ait; eski benchmark yürütme anında tutulmamış.

## 3. Router overhead'ine ne kadar yer var?

Kaydedilen synthesis MAC saving oranlarına `S_router` ve `S_dither` diyelim.
**Yalnızca süre MAC ile orantılı varsayılırsa**, router'ın ek karar maliyeti
full-frame synthesis süresinin `(S_router - S_dither)/100` oranını aşmamalıdır.
Bu varsayım altında dither synthesis süresine göre başa baş eşiği:

`ek maliyet / T_dither < (S_router - S_dither) / (100 - S_dither)`.

| Nominal hedef | Ek MAC tasarrufu (puan) | Dither synthesis süresine göre varsayımsal maliyet payı |
|---|---:|---:|
| 0,05 dB | 2,256 | %2,65 |
| 0,10 dB | 2,538 | %3,40 |
| 0,15 dB | 2,019 | %2,96 |
| 0,20 dB | 1,307 | %2,01 |
| 0,30 dB | 0,383 | %0,62 |
| 0,50 dB | 0,020 | %0,03 |

Bu tablo latency tahmini değil. Model maliyetiyle kurulmuş bir hassasiyet
hesabı: gevşek bütçede küçük bir karar/scheduling overhead'i bile router
avantajını tüketebilir. GPU kernel verimliliği, batch doluluğu ve memory
hareketi bu orantıyı değiştirebilir. Bu yüzden yalnız MLP FLOP sayısı yeterli
değerlendirme değil.

Kaynak bilgili arama, bir defa encode edilip çok defa decode edilen kullanımda
başka bir maliyet dengesine sahip olabilir. Ek encoding işi `ΔE`, her decode'da
gerçek tasarruf `ΔD>0` ise toplam iş açısından başa baş nokta `N>ΔE/ΔD` olur.
Bu ifade gerçek ölçümlerle doldurulmalı; server ve client cihaz sürelerini
aynı amaç fonksiyonunda toplamak ayrıca kullanım senaryosu gerektirir.

## 4. Sonraki benchmark'ın veri akışı

Encoder yalnız gerçek byte stream, gerekli shape/QP/model kimliği ve açık
metadata üretmeli. Ayrı decoder sürecine encoder tensor sözlüğü verilmemeli.
Decoder z sembollerini çözmeli, kendi entropy parametrelerini üretmeli,
y gruplarını gerekli sırada çözmeli ve reconstruction'ı bu sembollerden
oluşturmalı. Kodlanan değerlerde gerçekleşen aralık taşmaları veya quantize
edilmiş kontrol parametreleri gerçek kalite değerlendirmesine yansımalı.

Sabit dither, sabit/held-out router ve source-informed seçimin her biri için
aynı GPU, precision, kernel modu, warmup ve alternatif çalışma sırası
kullanılmalı. Ham tekrarlar saklanmalı; median ve p95 ile encoder, decoder ve
toplam süre ayrı raporlanmalı. Source-informed yolun tüm adayları, router'ın
feature üretimi, gerçek map/control encode/decode'u ve fallback'i ölçüme dahil.

Mevcut eğitim sonrası `paired_runtime.py` bu uçtan uca testi çözmüyor: o
script açıkça analysis/synthesis modüllerinin neural süresini ölçmek için
hazırlandı. Yararlı bir kapasite kontrolü, fakat bankalı codec'in son hız
iddiası için yukarıdaki byte akışı yine gerekli.

## 5. 28 Eylül: gerçek yürütülen router ve native encoder bağımlılıkları

53 CTC/QP32 frame üzerinde frozen checkpoint'ten yeniden hesaplanan router,
mean ve Q90 kontrollerinin her birinde 1.765 tile kararının tamamını tekrar
üretti. CPU/GPU log olasılıkları birebir eşit değil; en büyük mutlak fark
0,02266. Dolayısıyla yalnız bu iki kontrolün karar eşitliği doğrulandı.

Forward hook'ları gerçek yürütülen `proj_stem` ile üç Linear katmanını saydı:
289,715 MAC/padded-image-pixel. Latent/scale ve bit feature'ları bu run'da
görüntüye bağlı bilgi taşımıyor; checkpoint mimarisindeki sabit girdiler
MLP boyutunda korunuyor. Pooling, LayerNorm, aktivasyon, karar, taşıma ve
dispatch bu sayıya dahil değil. Küçük MAC sayısı, küçük duvar saati maliyetini
tek başına kanıtlamaz. Ayrıntı: `data/research20260927/shared_crossfit_qp32/router_logit_audit.json`.

Pinned native DCVC-UF kaynak kodunda latent hazır olduğunda kaydedilen CUDA
event'i entropy worker'ını serbest bırakıyor. Ana akışta synthesis yürürken
worker kendi CUDA stream'inde sembolleri topluyor, host'a aktarıyor ve CPU'da
rANS bitstream'ini bitiriyor. `compress` worker'ı bekliyor; GPU reconstruction'ın
tamamlanmasını ölçmek için çağıran tarafta CUDA synchronization da gerekiyor.
Bu bağımlılıklar örtüşmeye izin verir; gerçekleşen örtüşme henüz ölçülmedi.
Bu nedenle modül sürelerini toplayarak encoder latency üretmeyiz.
Kaynak diyagramı: `figs/research20260927/native_execution/fig_native_encoder_dependencies.pdf`.

Native correctness harness'inin dört checkpoint/CDF/binary CPU preflight'ı
geçti. İki görüntü × yedi geometri × üç QP ve bir tekrar, model başına
43 GPU doğruluk vakası olarak hazır. 288×512 ve 512×288 birleşik-region
boyutları da dahil. GPU doğruluğu, resident-expert switching ve timing
henüz çalıştırılmadı; resmî eğitimlerin GPU'larına müdahale edilmedi.


## 28 Eylül: shared-exit için sınırlı CPU bitstream doğruluğu

Yeni FUFEXIT1 araştırma kabı, e15 kimliği ve iki-bit exit map ile FUFREF1 ortak entropy payload'ını birleştiriyor. Önceden seçilmiş BasketballPass/BQMall ilk frame/QP32 × dört sabit harita = 8/8 bağımsız decode kontrolü geçti. Kaynak encoder ve hyper-encoder çağrıları decoder sürecinde kapalı; reconstruction, latent, sembol ve index izleri tam eşleşti. Map değişince inner entropy byte dizisi değişmedi. İki frame'in map maliyeti 1/2 byte; dış ve iç kimlik başlıkları toplam176 byte. Bu açık araştırma formatı minimum signalling değildir. Native CUDA formatını, 53-frame kapsamını, predictor'ın otonom kalibrasyonunu veya performansını doğrulamaz; çalışma içinde tutulan CPU diagnostic süreleri yayınlanmış runtime sonucu olarak kullanılmaz.
