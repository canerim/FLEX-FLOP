# Ucuz kaynak özelliği ile ek depth faydası arasındaki ilişki

Bu bir association teşhisidir; router eğitilmedi ve held-out seçim başarısı ölçülmedi. Bütün önceden belirlenmiş feature/depth/rate kombinasyonları gösteriliyor.

| Model farkı | Gerçek payload bpp | Feature | n | Spearman rho |
|---|---:|---|---:|---:|
| D4−D2 | 0.1 | rgb_std | 95 | +0.274 |
| D4−D2 | 0.1 | rgb_gradient_energy | 95 | +0.300 |
| D4−D2 | 0.1 | rgb_laplacian_energy | 95 | +0.197 |
| D6−D2 | 0.1 | rgb_std | 95 | +0.247 |
| D6−D2 | 0.1 | rgb_gradient_energy | 95 | +0.109 |
| D6−D2 | 0.1 | rgb_laplacian_energy | 95 | -0.004 |
| D4−D2 | 0.2 | rgb_std | 99 | +0.265 |
| D4−D2 | 0.2 | rgb_gradient_energy | 99 | +0.270 |
| D4−D2 | 0.2 | rgb_laplacian_energy | 99 | +0.168 |
| D6−D2 | 0.2 | rgb_std | 99 | +0.271 |
| D6−D2 | 0.2 | rgb_gradient_energy | 99 | +0.111 |
| D6−D2 | 0.2 | rgb_laplacian_energy | 99 | +0.011 |
| D4−D2 | 0.4 | rgb_std | 93 | +0.298 |
| D4−D2 | 0.4 | rgb_gradient_energy | 93 | +0.238 |
| D4−D2 | 0.4 | rgb_laplacian_energy | 93 | +0.142 |
| D6−D2 | 0.4 | rgb_std | 93 | +0.349 |
| D6−D2 | 0.4 | rgb_gradient_energy | 93 | +0.093 |
| D6−D2 | 0.4 | rgb_laplacian_energy | 93 | -0.010 |

Yüksek korelasyon tek başına kullanışlı router kanıtı değildir; uzman seçiminin hatası, gerçek rate–quality ve feature/selector maliyeti ayrı ölçülmeli. Düşük korelasyon ise bu basit özelliğin tek başına yetersiz olabileceğine işaret eder; öğrenilmiş ucuz temsilin başarısını dışlamaz.
Ortak rate desteği dışındaki görüntüler ekstrapole edilmedi. Farklı rate noktalarındaki kohortlar farklı olabilir. Sonuçlar epoch20 ara checkpoint'lerine koşulludur.
