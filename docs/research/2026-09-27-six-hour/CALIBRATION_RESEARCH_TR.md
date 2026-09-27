# Router için araştırma kararı: hedefi açık tanımla, kontrolü ayrı veride seç

27 Eylül 2026. Bu belge ölçülen CPU teşhislerini ve henüz denenmemiş yöntem
önerisini ayırır. Ana mekanizma DCVC-UF spatial early exit olarak kalıyor.

## Yeni kanıt ne söylüyor?

53 sequence × 5 QP arşivinde router ağırlıkları ve tahminleri sabit tutuldu.
Sequence adlarından belirlenen beş grupta, aynı sequence'in bütün QP'leri
birlikte ayrıldı. Her QP'nin scalar kontrolü diğer dört grupta seçildi.
Test grubunun kaynak hatası karar için kullanılmadı. Bu veri geliştirme
sırasında incelenmiş olduğu için bağımsız dış test sayılmaz.

| 0,1 dB kalibrasyon hedefi | Router MAC tasarrufu | Dither MAC tasarrufu | Router hedef aşımı | Dither hedef aşımı |
|---|---:|---:|---:|---:|
| Eğitim gruplarında ortalama kayıp | %27,379 | %24,608 | 122/265 | 123/265 |
| Eğitim gruplarında Q90 kayıp | %17,586 | %15,558 | 32/265 | 29/265 |

İlk satırda ortalama kayıplar router/dither için 0,1006/0,0997 dB;
ikinci satırda 0,0663/0,0667 dB. Dolayısıyla router'ın katkısı bu tabloda
tamamen kaybolmuyor; fakat sert kalite sınırı ile ortalama hedef birbirinin
yerine kullanılamıyor. Q90 kontrolü de bir kare garantisi değil.
0,05 dB Q90 hedefi, her policy'de 159 test vakasını kapsayan kalibrasyon
gruplarında sağlanamıyor. Bu vakalar atılmadı; en düşük eğitim riski veren
kontrol seçilip uygunsuzluk kaydedildi.

Bu ölçümler **padded tile-MSE tabloları** üzerinden. Full-frame repair,
crop ve e15/released referans farkı çözülmeden final görüntü için aynı
kapsama iddiası kuramayız. Önceki kaynakla ayarlanan, gerçek reconstruction
frontier'ının yerine bu sayıları koymak doğru olmaz.

## Önerilen deney: kontrollü bir durma politikası

1. Mevcut MLP'yi sabitle. Ayrı bir geliştirme kümesinde makul küçük bir
   kontrol listesi belirle; final değerlendirme verisine bakarak listeyi genişletme.
2. Her kontrol için **son birleştirilmiş RGB görüntüyü** decode et. Aynı e15
   full-frame referansıyla MSE/PSNR farkını, actual bytes ve tüm decoder
   süresini kaydet. Router, dither ve uniform aynı referansı kullansın.
3. Ayrı kalibrasyon kümesinde hangi başarı ölçütünü istediğini belirt:
   ortalama PSNR kaybı, kayıp eşiğini aşma olasılığı veya encoder'da her
   kareyi doğrulayan sert sınır. Bunların maliyetleri ve garantileri farklıdır.
4. Kontrol değerini dondur. Dokunulmamış test kümesinde ortalama, Q90/Q95,
   aşım oranı, en kötü vaka ve net süreyi beraber raporla. Sequence, örnekleme
   birimi olsun; aynı görüntünün beş QP'sini bağımsız beş örnek sayma.
5. Belirsiz ya da dağılım dışı içerikte daha derin varsayılan politika ayrı
   bir ablasyon olsun. Sert encoder doğrulaması kullanılırsa reconstruction
   denemeleri ve fallback maliyeti encode süresine dahil edilsin.

Bu çalışma için ilk kontrollü iddia, **kaynak görüntüsünü kullanmadan
ortalama kalite–hesap dengesi** olmalı. Her kare için sınır sağlayan ayrı
encoder doğrulamalı varyant daha pahalı olabilir; bunu ölçmek gerekir.

## İstatistiksel kalibrasyon neden doğrudan eklenemez?

Klasik conformal risk control, scalar parametreye göre monoton kayıp için
beklenen riski kontrol eder. Bizim arşivde router'ın ve dither'ın kalite
yollarında terslenmeler var; maliyetin monoton olması kaliteyi monoton
yapmıyor. Bu nedenle standart prosedüre yalnız bir conformal düzeltme
ekleyip garanti iddia edemeyiz.
[Conformal Risk Control, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html).

Learn then Test yaklaşımı kalibrasyonu çoklu hipotez testi olarak ele alır.
Küçük, önceden sabitlenmiş policy listesindeki eşiği aşma olayları için
ayrı kalibrasyon ve çoklu seçim düzeltmesi, burada incelenebilir bir yol.
Bu, mevcut çapraz-katlama sonucuna geriye dönük bir garanti vermez.
[Learn then Test](https://arxiv.org/abs/2110.01052).

2026 tarihli bir çalışma monoton olmayan kayıplar için algoritma
kararlılığına bağlı risk sınırları geliştiriyor. Bu, araştırılabilecek
bir teori yönü; kendi seçim algoritmamızın gerekli varsayımları sağladığını
kanıtlamadan yöntemimize ait teorem diye kullanılamaz.
[Conformal Risk Control for Non-Monotonic Losses](https://arxiv.org/abs/2602.20151).

## Ana makaleye etkisi

Contribution, “her patch'i sınıflandıran MLP”den daha güçlü tanımlanmalı:
aynı coded representation üzerinde hesaplamanın nerede değerli olduğunu
ölçmek, bunu gerçekten atlanan nested synthesis ile yürütmek ve kazancı
aynı gerçekleşen kalite altında blind mixtures'a karşı ayırmak.
Kaynakla seçilmiş kontrolün sonucu, autonomous router sonucu diye
sunulmamalı. Makaledeki mevcut ayrım korunmalı; yeni cross-fit teşhisi
önce supplement veya araştırma raporuna eklenmeli.

Bağımsız 2/4/6/8/10/12 bankasında aynı kontrol ayrıca rate'i de etkiler.
Bankanın encoder seçimi ile shared-latent decoder durma kuralı farklı bilgi
setleri kullanır. Bu nedenle tek policy'nin iki sisteme ücretsiz taşındığı
izlenimi verilmemeli.

Ham veri ve seçilmiş kontroller: [crossfit_control/analysis.json](crossfit_control/analysis.json).
Figürler: [budget ve risk](crossfit_control/fig_crossfit_budget_risk.pdf),
[kayıp dağılımları](crossfit_control/fig_crossfit_loss_distribution.pdf).
