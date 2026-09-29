# Ablasyon odaklı birincil literatür okuması

29 Eylül 2026. Amaç başka codec'lerin sayılarını ana deneye eklemek değil;
DCVC-UF'ta hangi karşılaştırmanın hangi iddiayı gerçekten sınadığını
belirlemek. Aşağıdaki “bizim kontrol” maddeleri kaynakların sonuçları değil,
bu projeye önerilen uyarlamalardır. Sayısal çalışma kolları DCVC-UF içinde
kalır. [Yürütme protokolü](ABLATION_PROTOCOL_20260929_TR.md) ayrı dosyada.

## En yakın mekanizma çalışmaları

**ClassSR — CVPR 2021.** §3.6–3.7 ve §4.3 okundu.
Farklı kapasiteli SR ağlarına patch atar; Average-Loss, eğitimin en büyük
dala yığılmasını önlemek için dal kullanımını dengeler. Bu, bizim bağımsız
codec bankamıza yakın; ortak-latent early exit değildir.
Bizim kontrol: no-balance / ClassSR tarzı balance / açık maliyet bütçesi.
Dondurulmuş codec'lerde eşit kullanım otomatik olarak doğru hedef değil;
kalite-süre frontier'ında gereksiz expert'in kullanılmaması kabul edilir.
[Birincil makale](https://openaccess.thecvf.com/content/CVPR2021/papers/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.pdf).

**APE — ECCV 2022.** §3.1–3.3, deney protokolü okundu.
Ortak reconstruction tail ve artımlı kapasite regresyonu kullanır;
hedefi ardışık çıkışların PSNR farkından türetilir ve negatif faydaya
izin verir. Regressor mevcut katmanın feature'ını gözler.
Bizim kontrol EE-C: aynı stem gözlemiyle CE / signed-gain / regret.
Sequential feature erişimini loss değişikliğiyle aynı anda eklememek
gerekir. Ortak head veya patch early exit için ilk olma iddiası yapılmaz.
[Birincil makale](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780286.pdf).

**AdaNIC — ICCV 2023.** §3.4–3.5 ve §4.2–4.4 okundu.
Blockwise kapasite/width kontrolü, RD bozulumundan routing hedefleri,
öğretmende routing + cost başlıkları ve hafif öğrenci kullanır.
Input çözünürlüğü ve predictor maliyeti ayrıca ablate edilir.
Bizim kontrol: budget-only / ucuz RGB özellikleri / MLP; predictor
feature üretimini timing'e dahil et. Label accuracy yanında RD regret
ve hedef aşımını ölç. DCVC-UF depth routing'in width routing'den farkını
açık yaz; “ilk spatial adaptive codec” denmez.
[Birincil makale](https://openaccess.thecvf.com/content/ICCV2023/papers/Tao_AdaNIC_Towards_Practical_Neural_Image_Compression_via_Dynamic_Transform_Routing_ICCV_2023_paper.pdf).

**cgSlimDecoder — CVPR 2023.** Abstract ve yöntem/complexity objective
bölümleri incelendi. Decoder modüllerinin kanal genişliklerini complexity–
rate–distortion hedefiyle ayarlar; entropy skipping ayrı mekanizmadır.
Bizim kontrol: synthesis-only MAC yerine entropy dahil tüm decode
maliyeti; width/depth ve entropy değişimlerini aynı ablasyonda karıştırma.
Bu makalenin video modül sonuçları intra DCVC-UF derinlik sonucu değildir.
[Birincil makale](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Complexity-Guided_Slimmable_Decoder_for_Efficient_Deep_Video_Compression_CVPR_2023_paper.pdf).

**AdaRevD — CVPR 2024.** Bu tur başlık/abstract doğrulandı; ayrıntılı
reproduction taraması yapılmadı. Adaptive patch exiting'i deblurring
decoder'ında kullanır. Related work'te task farkıyla kalmalı; sayısal
DCVC-UF kontrolü değildir. APE ile aynı yöntemin adı gibi kullanılmaz.
[Birincil makale](https://openaccess.thecvf.com/content/CVPR2024/papers/Mao_AdaRevD_Adaptive_Patch_Exiting_Reversible_Decoder_Pushes_the_Limit_of_CVPR_2024_paper.pdf).

## Banka, bağlam ve gerçek maliyet

**ELFIC — ACM MM 2023.** Yazar kurumundaki abstract/bibliyografik kayıt
doğrulandı; bu tur full-paper ablation ayrıntıları doğrulanmadı.
Instance-aware decoding complexity allocation ve rate–distortion–complexity
optimizasyonu doğrudan önceki çalışmadır. ELIC ve ELF-VC ile isim
benzerliği nedeniyle karıştırılmamalı. Bizim tam-image sabit derinlik
kontrolümüz, patch yerleştirmenin instance-level seçime ne eklediğini
göstermeli. Related work'e bu sınırlı, doğrulanmış tanımla eklendi.
[Yazar kurumu kaydı](https://scholars.cityu.edu.hk/en/publications/elfic-a-learning-based-flexible-image-codec-with-rate-distortion-/).

**BaSIC — ECCV 2024; ABC — arXiv:2506.15228, 2025.**
BaSIC'in birincil abstract'ı; ABC'nin §III-C, §V-C ve §VI-E bölümleri
incelendi. Backbone ile autoregressive bileşenlerin hesap yapısını
birlikte kontrol ederler; ABC budget, içerik ve task girdilerini
birleştiren adaptif modül ekler. Bizim kontrol: sabit entropy maliyeti
ve synthesis-only tasarrufu ayrı; budget-only vs content+budget;
aynı checkpoint'in tam yolunda kalite drift'i. Bu kaynaklar “içerik
ve cihaz bütçesini birlikte kullanma” fikrinin de yeni olmadığını gösterir.
[BaSIC birincil makale](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03640.pdf),
[ABC birincil metin](https://arxiv.org/html/2506.15228v1).

**MixCompress — ECCV 2026 kabul bilgisi, arXiv:2607.14334.**
§3.3, §4.2, §4.6 ve §5 okundu. MoD kolunda farklı derinlikte expert'ler
var; router yalnız rate embedding'e bağlı, görüntü içeriğine değil.
Bu nedenle expert ataması önceden hesaplanabiliyor. Bizim ayrı kontrolümüz
QP/bütçe-only lookup vs içerik gören selector; içerik bilgisinin
marjinal değerini ölçmek. Aynı zamanda related work'e eklendi.
Bu çalışmanın varlığı nedeniyle “codec'te ilk depth adaptation” iddiası
yapılmamalı. Bizim bankamız bağımsız bütün codec'leri seçer;
MixCompress'in expert'leri codec içindeki modüllerdir.
[Birincil tam metin](https://arxiv.org/html/2607.14334v1),
[kabul/sürüm kaydı](https://arxiv.org/abs/2607.14334).

**Spatial Competition — arXiv:2605.13243, ICIP 2026 kabul bilgisi.**
§2.2–2.3, §3 ve sınırlılıklar okundu. Aynı mimarili codec'leri
assignment/update döngüsüyle uzmanlaştırır; bölgesel RD araması,
mode map ve bağlı aynı-mode bölgelerin ortak işlenmesi vardır.
Aday aramasının encoder maliyeti açıkça formüle edilir; tek seçilen
codec'in decode edilmesi, tüm ağırlıkların bellekte ücretsiz olduğu
anlamına gelmez.
Bizim kontroller BANK-B/C/D: heterojen derinlik, encode öncesi seçici,
gerçek arama maliyeti ve fixed-map merge. Mevcut D2/D4/D6 eğitimimiz
bu çalışmadaki uzmanlaşma döngüsünü kullanmıyor. Ortalama kalite yakınlığı
veya farklı kazananlar, böyle bir uzmanlaşmayı kanıtlamaz.
[Birincil tam metin](https://arxiv.org/html/2605.13243v1),
[sürüm/kabul kaydı](https://arxiv.org/abs/2605.13243).

**Block-based Learned Image Compression without Blocking Artifacts —
CVPR 2026.** §3–4 overlap/boundary varsayımları ve deney açıklamaları
incelendi. Convolution ve transpose-convolution boyunca gerekli overlap'ı
analitik izler; CNN block processing için sınır kuralları verir.
Bizim kontrol EE-F: uniform-depth yeterli-halo eşdeğerliği. Bu sonuç
heterojen derinlikli komşulara, global operator'lara veya bağımsız
entropy modellerine otomatik taşınmaz. Öğrenilmiş repair ile exact
context'in farklı tasarım seçenekleri olduğunu açıklamak gerekir.
[Birincil makale](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_Block-based_Learned_Image_Compression_without_Blocking_Artifacts_CVPR_2026_paper.pdf).

**DCVC-RT — CVPR 2025.** Operasyonel maliyet vurgusu ve birincil kaynak
yeniden kontrol edildi: aritmetik kadar memory access ve function-call
maliyeti de önemlidir. Bizim kontrol EE-A/B: aynı MAC histogramında
packing/suffix/head süresi; gerçek full-codec ölçüm. Farklı GPU'da ölçülmüş
paper FPS'ini kendi intra decoder'ımızın zamanı gibi kullanmayacağız.
[Birincil makale](https://openaccess.thecvf.com/content/CVPR2025/papers/Jia_Towards_Practical_Real-Time_Neural_Video_Compression_CVPR_2025_paper.pdf).

**What Matters in Practical Learned Image Compression — CVPR 2026.**
§3.2, §4.3 ve §5.2 incelendi; Apple'ın yayın kaydı kontrol edildi.
On-device runtime ile mimari filtreleme ve tile artifact'lerine yönelik
ayrı loss kontrolleri sunar. Bizim çıkarım: MAC'e dayalı kazanç etiketi
yerine ölçülmüş maliyet kontrolü; küçük repair kazancı için seam/interior
ve repair'siz retraining karşılaştırması. Perceptual/GAN eğitimini bizim
resmî MSE recipe'sine eklemek bu kontrolün parçası değil.
[Birincil makale](https://arxiv.org/abs/2605.05148),
[yazar kurumu yayın kaydı](https://machinelearning.apple.com/research/compression).

## Karar kalitesi ve risk

**Fast yet Safe: Early-Exiting with Risk Control — NeurIPS 2024.**
§3 ve risk-control varsayımları okundu. Empirical threshold, expectation
control ve high-probability control farklı sözleşmelerdir. CRC/UCB
uyarlamalarında bounded loss ve marginal monotonicity koşulları vardır;
makale monotonicity gerektirmeyen Learn-then-Test'i de tartışır.
Bizim kontrol EE-D: önceden sabit aday gridinde aşım indicator'ı,
eşzamanlı risk sınırı ve aynı prosedürü alan Bayer. Unbounded PSNR
kaybına formülü doğrudan uygulayıp guarantee yazılamaz; mevcut nonmonotone
policy tabloları dikkate alınmalıdır.
[Birincil makale](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea5a63f7ddb82e58623693fd1f4933f7-Paper-Conference.pdf).

**Rethinking Calibration for Early-Exit Neural Networks — ICML 2026
yazar kaydı; arXiv:2508.21495.** Abstract, EEFP motivasyonu ve §5'in
calibration/cost kontrolleri incelendi. Klasifikasyonda confidence
calibration'ın tek başına early-exit cost–accuracy davranışını yeterince
anlatmadığını gösterir; devam etmenin faydasını dikkate alır.
Bizim çıkarım EE-C: router accuracy/ECE yerine regret ve gerçekleşen
quality–time frontier. Classification EEFP formülü compression PSNR
ölçümü diye kullanılmaz; bu kavramsal bir uyarlamadır.
[Birincil makale](https://arxiv.org/abs/2508.21495),
[yazarın konferans kaydı](https://fszatkowski.github.io/publication/failure-prediction-early-exit/).

**Learn then Test — arXiv:2110.01052.** Abstract ve Fast yet Safe'daki
ilgili yöntem tartışması incelendi; bu tur tam ispat taraması yapılmadı.
Risk kontrolünü çoklu hipotez testi olarak kurar. Bizim hesap bağımsız
binomial örnekler için standart exact bound + Bonferroni'dir; yeni bir
teorem veya LTT implementasyonunun bütünü değildir. Hesaplanan 59/94/139
örnek gereksinimleri policy/bütçe ailesinin büyüklüğüne bağlıdır.
[Birincil preprint](https://arxiv.org/abs/2110.01052).

## Eğitim için ikincil seçenek

**Knowledge Distillation for Learned Image Compression — ICCV 2025.**
Abstract ve problem/method özeti incelendi; tüm ek deneyleri bu tur
okunmadı. Stage-wise modular distillation ve teacher–student mimari
uyumunu ele alır. Bizim öneri P2: temel resmi eğitim bittikten sonra
eş budget'lı KD/no-KD kontrolü. KD ana recipe'ye eklenirse “Microsoft
recipe'sine birebir uyan ana deney” olmaktan çıkar; ayrı varyant kalmalı.
[Birincil makale](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Knowledge_Distillation_for_Learned_Image_Compression_ICCV_2025_paper.pdf).

**Microsoft DCVC-UF resmî kodu/recipe.** Mevcut koşuların pinned kaynak
ve manifest'i başlangıç sözleşmesidir; değişen main branch değil.
Image105 schedule, video training schedule'ıyla karıştırılmaz. Bankada
train_0/1/2 tamamı korunur; bu tur optimizer, dataset veya eğitim adımı
değiştirilmedi. Uzun yeni eğitim kontrolü D12-scratch'tır.
[Resmî depo](https://github.com/microsoft/DCVC),
[eğitim açıklaması](https://github.com/microsoft/DCVC/blob/main/training.md).

## Dahil etmediklerimiz ve yenilik sınırı

- Aramada çıkan ISER, test-time fine-tuning'i durdurmaya yöneliktir;
  decoder early exit için doğrudan baseline sayılmadı.
- Yayın tarihi 9 Ekim 2026 görünen “When to stop” kaydı, mevcut
  29 Eylül tarihinden sonraya ait olduğundan bu plana dayanak yapılmadı.
- Genel LLM/diffusion MoE işleri, codec gözlemleri ve bitstream maliyeti
  farklı olduğu için deney listesine otomatik eklenmedi.
- GLIC/CVPR2026'nin abstract'ı kontrol edildi: adaptif graph connectivity
  ve receptive field tasarımı, burada ablate ettiğimiz nested synthesis
  exit'i değildir. Ayrı backbone deneyi kapsamına alınmadı.
  [Birincil makale](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Adaptive_Learned_Image_Compression_with_Graph_Neural_Networks_CVPR_2026_paper.pdf).
- İlk patch exit, ilk shared tail, ilk spatial codec selection veya ilk
  region merging iddiası yok. Savunulabilir hedef: **DCVC-UF'ta ortak
  representation üzerinde depth adaptation'ın ne zaman kör karışımdan
  daha değerli olduğunu, context ve karar maliyetleriyle birlikte ölçmek.**

İndirilen sekiz birincil PDF'nin SHA256'ları
[source_downloads.json](data/ablation20260929/source_downloads.json)'da.
PDF'lerin kendileri repoya eklenmedi. İndirilme, baştan sona okunma
anlamına gelmez; yukarıdaki bölüm/okuma kapsamı esas alınır.
