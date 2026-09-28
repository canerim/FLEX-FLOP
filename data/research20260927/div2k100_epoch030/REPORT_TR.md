# DIV2K100: gerçek payload ile epoch30 derinlik değerlendirmesi

D2/D4/D6: 30/105 epoch. D12: released model, farklı eğitim geçmişi. CPU FP32 araştırma formatı; GPU hız veya released-format uyumluluğu iddiası yok.
100 merkez512 crop × dört model × beş QP = 2000 ayrı encode/decode vakası. Her reconstruction bağımsız süreçte birebir doğrulandı.

| Payload bpp hedefi | Ortak görüntü sayısı | D4−D2 dB | D6−D2 dB | Released D12−D2 dB |
|---:|---:|---:|---:|---:|
| 0.1 | 95/100 | +0.1409 [+0.1218, +0.1619] | +0.1122 [+0.0769, +0.1446] | +0.6310 [+0.5454, +0.7245] |
| 0.2 | 99/100 | +0.1117 [+0.0911, +0.1322] | +0.1071 [+0.0790, +0.1366] | +0.6727 [+0.5799, +0.7793] |
| 0.4 | 89/100 | +0.1017 [+0.0854, +0.1191] | +0.0788 [+0.0531, +0.1063] | +0.7322 [+0.6357, +0.8413] |

Aralıklar %95 paired görüntü bootstrap aralığıdır; checkpoint ve ortak bitrate desteğine koşulludur. Seed belirsizliği değildir. Her rate noktasında kapsanan görüntüler farklı olabilir; eksik görüntüler için ekstrapolasyon yapılmadı.

| Model | Released D12'ye göre BD-rate (%) | Geçerli görüntü sayısı |
|---|---:|---:|
| D2 | +20.958 | 100/100 |
| D4 | +16.522 | 99/100 |
| D6 | +17.869 | 100/100 |

BD-rate her görüntüde dört modelin ortak PSNR aralığında hesaplanır; görüntü sonuçları eşit ağırlıkla ortalanır. Released karşılaştırması saf derinlik etkisini ayırmaz.

Research container başlığı 88 byte/crop = 0,00268555 bpp; bunun 64 byte'ı kimlik/bütünlük hash'idir. Bu, minimum production header veya router side-bit maliyeti değildir.
Entropy tahmini, rANS payload ve container sonuçları `analysis.json` içinde ayrı bulunur. CPU ve önceki GPU ölçümleri birleştirilmedi; ilk dört crop'un backend farkları ayrıca kaydedildi.

Dört crop'luk monitor ile ek96 validation görüntüsünün farkları her eş-rate çiftinde korunur. Veri, checkpoint seçmek için kullanılmadı.
