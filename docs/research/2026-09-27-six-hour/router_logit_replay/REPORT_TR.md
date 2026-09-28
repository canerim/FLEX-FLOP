# Router checkpoint tekrar üretimi

53 CTC ilk frame, QP32, frozen e15 + ce_soft_6k. CPU FP32 yeniden üretilen log olasılıkları ile tarihsel GPU arşivi karşılaştırıldı. İki cross-fit kontrol aynı kaldı; source hata etiketleri karara girmedi.

En büyük mutlak log-olasılık farkı: 0.022657394.

| Kontrol | Değişen görüntü | Değişen tile | Ortalama tasarruf değişimi (pp) |
|---|---:|---:|---:|
| mean | 0/53 | 0/1765 | +0.00000000 |
| q90 | 0/53 | 0/1765 | +0.00000000 |

Router Conv2d+Linear maliyeti: [289.71484375] MAC/padded pixel. Pooling, LayerNorm, aktivasyon, argmax, bellek/dispatch ve kaynak encode maliyeti buna dahil değil. Geçersiz latent/bit branch'leri hâlâ saklanan parametrelerdir; yürütülen convolution yalnız stem projection'dır.
Bu sayılar duvar saati kazancı değildir. Map eşitliği yalnız ölçülen QP/kontrol/cohort içindir; bütün beta veya platformlarda eşitlik iddia edilmez.
