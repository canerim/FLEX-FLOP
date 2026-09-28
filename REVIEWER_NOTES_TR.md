# 28 Eylül 2026: tamamlanan kontrollerin yorumu

Bu güncelleme, aşağıdaki 27 Eylül araştırma notunu yeni ölçümlerle tamamlar.

- **Early-exit ile router katkısı ayrılıyor.** Kaynak-kalibre arşivdeki 2.93 puanlık delivered-cap marjı yerinde duruyor. Fakat 53-frame/QP32 gerçek replay’de sequence-disjoint mean/Q90 kontrolün router–dither MAC farkları +1.87 [−1.60, 5.19] ve +2.54 [−0.42, 5.54] puan. Pozitif deployable marj henüz çözülmüş değil; aynı nominal hedef aynı gerçekleşmiş kalite değil.
- **Released referans boşluğu QP32’de kapandı.** Aynı valid-pixel RGB ve shared-pad256 CPU protokolünde released/e15 PSNR 34.71763/34.70855 dB. e15 drift’i +0.00908 [0.00785, 0.01033] dB. Aynı router çıktısının released’e göre mean/Q90 kaybı +0.09587/+0.06592 dB. Bu tek-QP karşılaştırma, bütün arşivi RGB veya native bitstream sonucuna dönüştürmez.
- **Adapter etkisi büyük, repair etkisi küçük.** 212 sabit-ağırlık çıktısında adapter kapatmak +1.23091 dB, repair kapatmak +0.00208 dB kaybettiriyor. Sekiz yalnız-D12 frame dahil bütün örnekler tutuldu. Mimari gereklilik için eş eğitimli yeniden optimizasyon hâlâ gerekli.
- **Bankanın maliyeti yalnız model seçimi değil.** Tam 800-region kontrolünde aynı piksel derinlikleri korunup komşu bölgeler birleştirildiğinde 0.2 payload bpp’de +0.5327/+0.4959 dB geliyor. QP32 payload azalması %13.05/%12.39. Bu yürütme biçimini kontrol etmeden router kapasitesi taramak yanlış değişkeni optimize edebilir. Coalescing’in kendisi yeni bir yöntem iddiası değil.
- **Hız iddiası henüz yok.** 43-case/model native correctness matrisi hazırlanıp CPU checkpoint/CDF/binary preflight’ı geçti; CUDA tarafı meşgul eğitim GPU’larında çalıştırılmadı. CPU research byte-decode kanıtı, native GPU eşdeğerliği veya toplam süre kanıtı değildir.

**İçerik ayrımı için ek bulgu:** 1.378 ilk-frame çiftinin kalite sonuçlarını kullanmayan fingerprint taraması, karşı fold’larda iki aday çift buldu: RaceHorses’ın iki çözünürlüğü ve videoSRC17–Kimono1. Ham hash’ler farklı olsa da benzer içerik kalabiliyor. Gelecek split için bu çiftleri korumacı biçimde birleştiren 51 içerik grubu önerildi; mevcut fold’lar ve sonuçlar değiştirilmedi. Sequence-disjoint demek content-independent demek değil.

Makalenin en önemli sonraki üç deneyi: aynı codec ve gerçekleşmiş kalite altında native toplam süre; aynı eğitim bütçesinde adapter/repair karşılaştırması; dokunulmamış calibration/test ayrımında router–dither marjı. D8/D10 sayısını büyütmek bu soruların yerine geçmez.

---

> Metrik düzeltmesi (27 Eylül): eski `db_rgb`, unclipped YCbCr 4:4:4
> eş-kanal MSE oranıdır. RGB etiketi düzeltilmiştir; iki CPU replay kanıtı
> ve ham anahtarların yorumu `METRICS.md` içinde. Aşağıdaki eski sayıların
> tamamı kendi açıklanmış metrik/protokolüne koşulludur.

# Revizyonun araştırma değerlendirmesi

27 Eylül 2026. Ana makale DCVC-UF intra early exit üzerinde kuruldu. Bu not,
metnin güçlü yanını ve nihai deneylerin hangi soruları çözmesi gerektiğini
ayırır; eksik deneyler için sonuç üretmez.

## Makalenin savunulabilir ana iddiası

DCVC-UF'nin tek latentini ve ortak sentez prefix'ini koruyarak, kalan sentez
derinliğini tile bazında değiştiren gerçek bir yürütme yolu var. Pointwise
adapter, grid repair ve ortak head bu yolu tanımlıyor. Arşivde final mixed
rekonstrüksiyonlar mevcut. Aynı kayıtlardan uniform, dither ve learned routing
karşılaştırması yapıldığında içerik yönlendirmesinin bütçeye bağlı ek aritmetik
kazancı görülebiliyor.

“Early exit'i ilk biz bulduk”, “her görüntüde 0,1 dB garantisi”, “%28 gerçek
codec hızlanması” veya “altı expert tamamlandı” bu kanıtların desteklediği
iddialar değil. Ana metin bunları söylemiyor.

## Bir hakemin soracağı altı soru

**1. APE veya ClassSR'den ayrım nerede?** Spatial kapasite seçimi önceden var.
Bizim anlatımız, bunun coded latent üzerinde nested synthesis olarak kurulması,
exit-feature uyumu ve sınır bağlamının birlikte ele alınması. Bilimsel ayrımı
güçlendirecek kontrol, adapter/repair/mixed-tile eğitim ablasyonlarıdır.
Bunların etkisini henüz ölçülmüş gibi göstermiyoruz.

**2. Router gerçekten gerekli mi?** Early exit tasarrufunun çoğunu uniform
ve dither sağlayabilir. Bu nedenle ana sonuç hem toplam tasarrufu hem dither
üzerindeki ek payı veriyor. Nominal 0,1 dB'de ek pay 2,54 MAC puanı; aynı
ölçülmüş YCbCr 4:4:4 cap altında sınırlı aday havuzunda 2,93 puan. Gevşek bütçede pay
küçülüyor. Sonraki karşılaştırmada ucuz içerik/entropy proxy'si ve router'ın
kendi zamanı mutlaka bulunmalı.

**3. Kalite kaybı hangi referansa göre?** Uniform-exit profilinin kaynağı padded
YCbCr 4:4:4 ve released ağırlıklı full-frame output. Mixed-output tablosunun referansı
crop edilmiş fine-tuned e15 full-frame output. Bunları tek RD eğrisiymiş gibi
birleştirmiyoruz. Yeni replay iki referansı aynı valid-pixel desteğinde ölçmeli.
398 inherited warm-start tensor'ünün resmî release ile eşleşmesi, checkpoint
kimliğini destekliyor; cropped kalite farkını hesaplamanın yerine geçmiyor.

**4. Calibration test kaynağını görüyor mu?** Şimdiki sweep evet: kontrol
parametresi kaynak hata tablosuyla frame başına seçiliyor. Bunu decoder-local
held-out policy başarısı diye adlandırmıyoruz. Predictor eğitimi, kontrol
kalibrasyonu ve test ayrımını yeniden kurmak, deployment deneyinin P0 işi.

**5. MAC neden ms değil?** Conditional tile yürütmesi gerçek olsa da entropy,
routing, packing ve GPU doluluk etkileri MAC toplamına indirgenmiyor. Eski
runtime kaydının bazı aşamaları encoder tensor'lerinin yeniden yürütümüydü.
Yeni codec ölçümünde decoder görüntüyü yalnız bytestream'den çözmeli. Mevcut
rakamlar bu eksikliği kapatan bir uçtan uca benchmark gibi kullanılmıyor.

**6. Sığ bağımsız modeller early exit ile aynı mı?** Hayır. D2/D4/D6 bütün
codec ağırlıklarını sıfırdan öğreniyor; e15 ortak latent üzerinde fine-tuned
bir sistem. D12-scratch aynı eğitimli derinlik kontrolünü sağlar. Ağırlık
paylaşımının nedensel etkisi için ayrıca iki ailenin training/data/init
koşullarını eşlemek gerekir. Altı bağımsız expert toplam 221.91 M parametre,
yalnız FP32 ağırlık depolaması yaklaşık 846.5 MiB tutar; aktivasyon/runtime
belleği buna eklenir. Bu deterministik kapasite hesabı, inference ölçümü değil.
Altı model CPU'da ayrı ayrı oluşturularak sayımlar da doğrulandı; korunan
başlangıç tensor'leri aynı seed'li D12 ile birebir eşleşiyor. D12→D2,
decoder parametrelerini %73,0 azaltıyor fakat bütün codec'te azalma %24,6:
encoder ve entropy ağırlıkları aynı kapasitede kaldığı için modelin tümü
altıda bire inmiyor. Parametre azalması runtime azalmasıyla eş tutulmamalı.

## En küçük güçlü deney paketi

1. Aynı e15 checkpoint'inde released/e15 full-frame ve tiled-deepest referansları;
   aynı crop, aynı metrik, gerçek byte akışı.
2. Aynı codec üzerinde uniform, dither, ucuz içerik kuralı, held-out router,
   kaynak bilgili search; hem kalite-cap hem net süre değerlendirmesi.
3. Eş eğitimli adapter ve repair kontrolleri. Bu iki kol, ana tasarımın
   katkısını salt truncation'dan ayırır.
4. D2/D4/D6 tamamlandıktan sonra aynı tarifli D12; ardından faydalı ara
   derinlikler ve bankadan expert çıkarma testi.

İlk üç madde early-exit makalesini güçlendirir. D8/D10 ve büyük bir router
hiperparametre taraması, ölçüm yolunun doğruluğundan önce gelmemeli.

## Sonuç birkaç içerikten mi geliyor?

Ek CPU duyarlılık kontrolü aynı gerçekleşmiş kalite sınırındaki kayıtlı
çıktıları kullanıyor. 0,1 dB'de router 170 frame–QP çiftinde daha fazla MAC
kazandırıyor, 24'ünde daha az, 71'inde eşit. Dizi başına beş QP ortalamasında
43/53 dizi router lehine, 5 dizi dither lehine, 5 dizi eşit. Her seferinde
bir diziyi tüm QP'leriyle çıkarınca ortalama fark 2,80–3,12 puan arasında
kalıyor. Bu aralık güven aralığı değil, tek diziye duyarlılık kontrolüdür.
Beş QP'nin her birindeki ortalama fark da pozitif.

İki yöntemden birinde fallback olan üç çifti çıkarınca fark 262 çiftte
2,91 puan; ana sonuçtaki fark yalnız fallback sayısından kaynaklanmıyor.
Bu altküme ana değerlendirme yerine geçmez. Buna karşılık 0,5 dB'de
265 çiftin 262'si eşit; kalan üç olumlu fark tek bir diziden geliyor.
Gevşek bütçedeki 0,02 puanı genel bir router başarısı olarak anlatmamak gerekir.

Tekrar üretim: `python scripts/audit_cap_influence.py`; tam kayıtlar
`data/refresh20260927/cap_influence_audit.json`. Hiçbir yeni GPU decode,
zamanlama veya held-out calibration sonucu üretilmedi.

## İlgili çalışmalardan hangi kontrolü almalıyız?

[APE](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2021_ECCV_2022_paper.php),
patch'in ek derinlikten göreceği faydayı tahmin ederek durma kararı veriyor.
Bu nedenle “karmaşık patch derine gitsin” tek başına yeni katkı sayılmaz.
DCVC-UF üzerinde exit-label CE'ye karşı marjinal distortion kazancı tahmini
somut bir yöntem ablasyonu olabilir; sonuç henüz yok.

[AdaNIC](https://openaccess.thecvf.com/content/ICCV2023/html/Tao_AdaNIC_Towards_Practical_Neural_Image_Compression_via_Dynamic_Transform_Routing_ICCV_2023_paper.html)
blok bazlı uzamsal kanal kapasitesi ve öğrenilmiş routing kullanıyor.
Bu yakınlık nedeniyle “compression içinde ilk spatial adaptation” iddiası
kurmuyoruz. Bizim sınanacak tasarım ayrımımız ortak latent üstünde nested
derinlik, pointwise exit uyumu ve full-frame repair/head birlikteliği.
Adapter/repair retraining kontrolleri bu ayrımı ölçülebilir kılar.

[Spatial Competition](https://arxiv.org/abs/2605.13243) ise bölgeye göre
bağımsız codec seçimi ve iletilen mode map kullanıyor. Altı-model bankası
bu literatürle karşılaştırılmalı; bugünkü shared-exit mimarisinin sonucu diye
sunulmamalı. Bunlar kaynaklardan çıkardığımız deney tasarımı önerileridir,
başka codec'ler üzerinde bu makaleye eklenmiş sayısal deneyler değildir.
