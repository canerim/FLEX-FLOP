# Halo, padding ve depth tasarrufunun paydası

Aşağıdaki ilk tablo yalnız FUFREF1 CPU image-pad64 protokolüne aittir. Native CUDA image-pad16 + latent-pad4 uygular; aynı maliyet olduğu varsayılamaz.
2×2 gridde her256 core yalnız bir iç sınır yönünde context alır. CPU-pad64 için halo32:288→320 padding; halo64:320→320. İkisinde de kodlanan alan full512'nin1,5625 katı.
Halo64 pencere başlangıcını64-pixel hyperlatent gridine de hizalar. Bu eşit PSNR, payload veya latency anlamına gelmez.

| Model | Halo | Neural decoder / full D12 (%) | Encoder+reconstruction / full D12 (%) |
|---|---:|---:|---:|
| D2 | 0 | 51.79 | 64.38 |
| D2 | 32 | 80.92 | 100.60 |
| D2 | 64 | 80.92 | 100.60 |
| D4 | 0 | 61.43 | 71.51 |
| D4 | 32 | 95.99 | 111.73 |
| D4 | 64 | 95.99 | 111.73 |
| D6 | 0 | 71.08 | 78.63 |
| D6 | 32 | 111.05 | 122.86 |
| D6 | 64 | 111.05 | 122.86 |
| D8 | 0 | 80.72 | 85.75 |
| D8 | 32 | 126.12 | 133.99 |
| D8 | 64 | 126.12 | 133.99 |
| D10 | 0 | 90.36 | 92.88 |
| D10 | 32 | 141.18 | 145.12 |
| D10 | 64 | 141.18 | 145.12 |
| D12 | 0 | 100.00 | 100.00 |
| D12 | 32 | 156.25 | 156.25 |
| D12 | 64 | 156.25 | 156.25 |

## Native boyut düzeniyle ayrı Conv2d hesabı

288 görüntü native yolda288 kalır; y18→20 yalnız hyperanalysis için pad edilir. Hyperprior/fusion ve diğer ağların alanları aynı oranda büyümediğinden tek image-area çarpanı kullanılamaz. Aşağıdaki değerler katman bazında yeniden izlendi.

| Model | Halo | Neural decoder / full D12 (%) | Encoder+reconstruction / full D12 (%) |
|---|---:|---:|---:|
| D2 | 0 | 51.79 | 64.38 |
| D2 | 32 | 67.82 | 83.22 |
| D2 | 64 | 80.92 | 100.60 |
| D4 | 0 | 61.43 | 71.51 |
| D4 | 32 | 80.02 | 92.23 |
| D4 | 64 | 95.99 | 111.73 |
| D6 | 0 | 71.08 | 78.63 |
| D6 | 32 | 92.22 | 101.25 |
| D6 | 64 | 111.05 | 122.86 |
| D8 | 0 | 80.72 | 85.75 |
| D8 | 32 | 104.43 | 110.26 |
| D8 | 64 | 126.12 | 133.99 |
| D10 | 0 | 90.36 | 92.88 |
| D10 | 32 | 116.63 | 119.28 |
| D10 | 64 | 141.18 | 145.12 |
| D12 | 0 | 100.00 | 100.00 |
| D12 | 32 | 128.83 | 128.29 |
| D12 | 64 | 156.25 | 156.25 |

Bu tablo kalite eşleştirmesi değil, aynı source alanını kodlayan mimarilerin Conv2d hesabıdır. D8/D10 eğitilmiş sonuç değildir. Entropy coder, çağrı sayısı, bellek ve MLP maliyetleri yok.

Encoder latentinin doğrudan yapısal source aralığı136 piksel: [16i−64,16i+71]. Kernel incelemesi,192×192 gerçekCPU gradient probe'unda bbox[32,167] ile doğrulandı. Halo32 tek başına full-frame analiz eşdeğerliğini garanti etmez. Tüm codec receptive field veya gerekli minimum halo bundan ibaret değildir.
