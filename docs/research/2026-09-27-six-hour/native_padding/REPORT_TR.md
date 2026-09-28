# Padding nerede uygulanıyor? CPU politika karşılaştırması

Aynı16 görüntü,5 QP,D2/D6 epoch20 ve releasedD12. Yalnız halo32/context288 tekrar kodlandı. FUFREF1 görüntüyü320'ye pad eder; FUFREF2 görüntüyü288 tutup y18'i hyperanalysis için20'ye pad eder. Native GPU eşdeğerliği iddiası yok.

| Model | Payload bpp | Ortak n | Native-shaped − image-pad64 RGB PSNR (dB) |
|---|---:|---:|---:|
| D2 | 0.1 | 10/16 | +0.31613 |
| D2 | 0.2 | 16/16 | +0.27906 |
| D2 | 0.4 | 15/16 | +0.25937 |
| D6 | 0.1 | 10/16 | +0.30378 |
| D6 | 0.2 | 16/16 | +0.26522 |
| D6 | 0.4 | 15/16 | +0.24922 |
| D12 | 0.1 | 12/16 | +0.27135 |
| D12 | 0.2 | 16/16 | +0.24702 |
| D12 | 0.4 | 15/16 | +0.22995 |

Bütün beş varyantın ortak desteği kullanılır: full,halo0,halo32-pad64,halo64,halo32-native-shape. Farklırate/derinliklerde cohort değişebilir. Header ve payload ayrı; same-QP değişimler matched-rate tablosunun yerine geçmez.
Bu karşılaştırma padding yerini değiştirir; encoder sınır desteği ve hyperprior context'i birlikte etkilenir. Tek bir mekanizmanın nedensel katkısı ayrılamaz.
