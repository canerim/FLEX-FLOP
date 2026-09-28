# Early exit'te bağlamı hangi kontrol ayırır?

28 Eylül 2026 · Mimari inceleme ve deney tasarımı; yeni kalite veya süre sonucu değildir.

Mevcut sistemde ortak stem tam görüntü üzerinde çalışır. Bu yüzden tile
özellikleri başlangıçta zaten görüntü bağlamı taşır. Fakat conditional
suffix'in her 3×3 depthwise işlemi, komşu tile'ın o derinlikteki özelliği
yerine replicate padding görür. Pointwise adapter bu uzamsal desteği
büyütmez. Sonradan ortak canvas üzerinde çalışan repair ve head, full-frame
suffix ile matematiksel eşdeğerlik sağlamaz; öğrenilmiş bir telafi yoludur.

Bu ayrım önemlidir: “aynı derinlikte tüm tile'lar” ile “tam görüntü forward”
aynı işlem değildir. Mevcut uniform-depth deneyinde en derin tiled yolun da
kalite ve hesap farkı olması bununla uyumludur; farkın tamamını tek bir
mekanizmaya bağlamak için ayrıca kontrollü replay gerekir.

## Mimari destek hesabı

İncelenen dondurulmuş shared-exit kodunda her trunk block, stride 1'de bir
3×3 depthwise convolution içeriyor; diğer trunk işlemleri pointwise.
Dört bloklu ortak stem'den sonra, depth k için k−4 conditional blok kalır.
İç tile için bu yolun yapısal bağlam yarıçapı k−4 feature hücresidir.
Bir feature hücresi sekiz görüntü pikseline karşılık gelir.

| Toplam derinlik | Conditional blok | Feature halo yarıçapı | Görüntü karşılığı | 32×32 core çevresindeki giriş alanı / core alanı |
|---|---:|---:|---:|---:|
| 6 | 2 | 2 | 16 piksel | 1,266× |
| 8 | 4 | 4 | 32 piksel | 1,562× |
| 10 | 6 | 6 | 48 piksel | 1,891× |
| 12 | 8 | 8 | 64 piksel | 2,250× |

Son sütun yalnız `(32 + 2h)^2 / 32^2` geometrisidir. Katmanlar arasında
halo daraltılması, değişen aktif batch, ortak stem/head ve entropy bunun
içinde değildir; decoder MAC veya wall-time çarpanı olarak kullanılamaz.
Bu hesap, ağırlıkların fiilî receptive field'ini değil mimarinin mümkün
etki desteğini verir.

Yeterli giriş halo'su tek başına global görüntü kenarlarında eşdeğerlik
kanıtı değildir. Katman bazındaki padding, residual yolların crop hizası
ve bias etkileri de aynı olmalıdır. [Kim ve arkadaşları, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_Block-based_Learned_Image_Compression_without_Blocking_Artifacts_CVPR_2026_paper.pdf),
§4–5, bu nedenle overlap hesabını boundary handling ile birlikte ele alır.
Mevcut replicate-halo helper'ını açmak, o tam protokolü uygulamakla aynı şey
olarak sunulmamalıdır.

## En küçük karşılaştırma

Önce aynı e15 ağırlıklarıyla her uniform reachable depth için tam görüntü
özelliklerini üret. Bunu yeterli halo ve katman başına doğru sınır işlemiyle
oluşturulan tiled özelliklerle, adapter öncesinde ve head sonrasında
karşılaştır. Maksimum mutlak tensor farkını, cropped RGB/444 hatasını ve
geçerli piksel desteğini birlikte kaydet. Bu, exact-context kontrolüdür;
mevcut erken çıkış yoluna otomatik olarak bir hız avantajı kazandırmaz.

Ardından aynı sabit mixed map üzerinde üç yol ölç: mevcut zero-halo+repair,
zero-halo fakat repair identity, ve context-preserving conditional yol.
İlk aşama yalnız sabit-weight hassasiyetidir. Tasarım tercihi için bu
kolları aynı mixed-map eğitim bütçesiyle yeniden optimize etmek gerekir.

Her koşulda map, derinlik histogramı, QP, padding/crop, latent ve checkpoint
kimliği sabit tutulmalı. Sınır bandı/interior farkı ile tam görüntü kalite
farkı ayrı raporlanmalı. Runtime kıyası; halo taşıma, grouping, scatter/gather
ve ortak head dahil, aynı cihazda gerçek çalıştırmayla yapılmalı. Sonuç
olumsuz çıkarsa repair'i büyütmek yerine paylaşılmış stem derinliği, daha
büyük core veya daha az routing granularity daha iyi aday olabilir.
