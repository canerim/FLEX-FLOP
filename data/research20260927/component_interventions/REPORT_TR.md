# Sabit checkpoint üzerinde adapter ve repair müdahalesi

53 CTC ilk frame, QP32, aynı Q90 router haritaları. 2×2 identity-substitution kontrolü; yeniden eğitim yapılmadı. Orijinal çıktı ve cache edilen repair/head yolu yeniden doğrulandı.

| Kapatılan | RGB PSNR kaybı (dB) | Sınır MSE değişimi (%) | Interior MSE değişimi (%) |
|---|---:|---:|---:|
| repair_identity | +0.00208 | +0.192 | +0.039 |
| adapters_identity | +1.23091 | +37.919 | +43.737 |
| both_identity | +1.25384 | +39.436 | +45.224 |

Pozitif PSNR kaybı, identity müdahalesinin mevcut checkpoint kalitesini düşürdüğünü belirtir. Bir bileşen olmadan yeniden optimize edilmiş ağın kalitesini göstermez. Aynı harita korunur; çıkarılan modüllerin MAC maliyeti burada hızlanma olarak sunulmaz.
