# DIV2K100: gerçek payload ile epoch20 derinlik değerlendirmesi

D2/D4/D6: 20/105 epoch. D12: released model, farklı eğitim geçmişi. CPU FP32 araştırma formatı; GPU hız veya released-format uyumluluğu iddiası yok.
100 merkez512 crop × dört model × beş QP = 2000 ayrı encode/decode vakası. Her reconstruction bağımsız süreçte birebir doğrulandı.

| Payload bpp hedefi | Ortak görüntü sayısı | D4−D2 dB | D6−D2 dB | Released D12−D2 dB |
|---:|---:|---:|---:|---:|
| 0.1 | 95/100 | +0.1195 [+0.0998, +0.1391] | +0.1839 [+0.1580, +0.2122] | +0.6360 [+0.5496, +0.7316] |
| 0.2 | 99/100 | +0.1207 [+0.1018, +0.1420] | +0.1834 [+0.1572, +0.2115] | +0.6986 [+0.5989, +0.8112] |
| 0.4 | 89/100 | +0.1084 [+0.0912, +0.1272] | +0.1748 [+0.1504, +0.2011] | +0.7688 [+0.6660, +0.8843] |

Aralıklar %95 paired görüntü bootstrap aralığıdır; checkpoint ve ortak bitrate desteğine koşulludur. Seed belirsizliği değildir. Her rate noktasında kapsanan görüntüler farklı olabilir; eksik görüntüler için ekstrapolasyon yapılmadı.

| Model | Released D12'ye göre BD-rate (%) | Geçerli görüntü sayısı |
|---|---:|---:|
| D2 | +21.559 | 100/100 |
| D4 | +17.844 | 100/100 |
| D6 | +15.689 | 100/100 |

BD-rate her görüntüde dört modelin ortak PSNR aralığında hesaplanır; görüntü sonuçları eşit ağırlıkla ortalanır. Released karşılaştırması saf derinlik etkisini ayırmaz.

Research container başlığı 88 byte/crop = 0,00268555 bpp; bunun 64 byte'ı kimlik/bütünlük hash'idir. Bu, minimum production header veya router side-bit maliyeti değildir.
Entropy tahmini, rANS payload ve container sonuçları `analysis.json` içinde ayrı bulunur. CPU ve önceki GPU ölçümleri birleştirilmedi; ilk dört crop'un backend farkları ayrıca kaydedildi.

Dört crop'luk monitor ile ek96 validation görüntüsünün farkları her eş-rate çiftinde korunur. Veri, checkpoint seçmek için kullanılmadı.
