# Shared early-exit: bağımsız CPU bitstream decode kontrolü

Önceden seçilen BasketballPass ve BQMall ilk frame/QP32 üzerinde dört sabit exit haritası: arşivlenmiş mean/Q90 router, tümü D12, döngüsel D6/D8/D10/D12. Kaynak görüntü kalite ölçümüne göre seçilmedi.

8/8 çıktı, latent, entropy sembolü ve index izi encoder ile bit düzeyinde eşleşti. Ayrı decoder süreci yalnız stream/output yollarını kabul etti; kaynak encoder ve hyper-encoder çağrıları hata verecek şekilde kapatıldı. Aynı frame’de dört haritanın inner entropy byte dizisi birebir aynı.

| Frame | Profil | Payload byte | Map byte | Araştırma header byte | Toplam byte |
|---|---|---:|---:|---:|---:|
| BasketballPass_416x240_50.yuv | mean | 2198 | 1 | 176 | 2375 |
| BasketballPass_416x240_50.yuv | q90 | 2198 | 1 | 176 | 2375 |
| BasketballPass_416x240_50.yuv | deepest | 2198 | 1 | 176 | 2375 |
| BasketballPass_416x240_50.yuv | cyclic | 2198 | 1 | 176 | 2375 |
| BQMall_832x480_60.yuv | mean | 10954 | 2 | 176 | 11132 |
| BQMall_832x480_60.yuv | q90 | 10954 | 2 | 176 | 11132 |
| BQMall_832x480_60.yuv | deepest | 10954 | 2 | 176 | 11132 |
| BQMall_832x480_60.yuv | cyclic | 10954 | 2 | 176 | 11132 |

Hatalı stream reddi: 7 kontrol geçti.

FUFEXIT1 + FUFREF1 CPU FP32 araştırma formatı; map açıkça iletiliyor. Header maliyeti minimum üretim formatı değildir. Bu preflight autonomous routing, tüm53 frame, native CUDA wire formatı, cross-device exactness veya runtime üstünlüğü kanıtı değildir.
