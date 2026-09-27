# Derinlik hesabında payda farkı

Aşağıdaki değerler512×512 girişte Conv2d MAC izidir. Entropy recovery ağları decoder toplamına dahildir; gerçek entropy coder, elementwise işlem ve bellek maliyeti değildir. Hız sonucu olarak okunamaz.

| Model | Synthesis GMac | Neural decoder GMac | Encoder+reconstruction GMac | Synthesis azalma % | Neural decoder azalma % |
|---|---:|---:|---:|---:|---:|
| D2 | 14.495 | 45.574 | 76.686 | 74.53 | 48.21 |
| D4 | 22.979 | 54.058 | 85.170 | 59.63 | 38.57 |
| D6 | 31.463 | 62.542 | 93.654 | 44.72 | 28.92 |
| D8 | 39.947 | 71.026 | 102.138 | 29.81 | 19.28 |
| D10 | 48.431 | 79.510 | 110.622 | 14.91 | 9.64 |
| D12 | 56.915 | 87.994 | 119.106 | 0.00 | 0.00 |

D8/D10 dahil bütün mimariler sayıldı; bu iki modelin eğitilmiş kalite sonucu yok. Meta-tensor izinin D2/64×64 eşdeğeri gerçek CPU yürütmeyle birebir karşılaştırıldı.
Encoder reconstruction ve entropy coding örtüşebildiğinden, encoder MAC azaltımı da wall-time oranına çevrilemez. MAC değerlerinin toplamsal olması, sürelerin toplamsal olduğu anlamına gelmez.
