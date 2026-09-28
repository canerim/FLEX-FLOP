# Aynı support üzerinde released D12 referansı

53 CTC ilk frame, QP32. Kaynak önce shared deneydeki gibi256 katına replicate-pad edilir, metrik geçerli kaynak alanına crop edilir. Stock DMCI released ağırlıkları, CPU FP32; native CUDA/bitstream çıktısı değil.

Released RGB PSNR ortalaması: 34.71763dB; e15 full-frame: 34.70855dB.
e15 full-frame'in released'e göre RGB kaybı: +0.00908dB; CI [0.007851270902897627, 0.010334925902186749].

| Kontrol | Kural | Released RGB kaybı (dB) | 0.1 üzeri /53 |
|---|---|---:|---:|
| mean | router | +0.09587 | 20 |
| mean | dither | +0.09722 | 20 |
| mean | uniform | +0.08661 | 17 |
| q90 | router | +0.06592 | 7 |
| q90 | dither | +0.06544 | 5 |
| q90 | uniform | +0.06069 | 4 |

Referans değişimi aynı çıktıların raporunu değiştirir; router ve dither arasındaki paired farkı değiştirmez. Nominal bütçe padded444 ve farklı kalibrasyon hedefiyle seçildiği için bu sayılar ayrı bir RGB garanti testi değildir. Her görüntünün ağırlığı aynı; en kolay görüntüler seçilmedi.
