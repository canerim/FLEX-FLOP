# Aynı haritada region birleştirme maliyeti

256 core ve32 halo ile D2/D6 yüzde50/yüzde50. Dört288×288 pencere yerine iki512×288 bölge; source pikselinin atanmış derinliği değişmez. Image16/latent4 geometri.

| Maliyet | Dört patch | İki bölge | Değişim |
|---|---:|---:|---:|
| synthesis_macs | 29.0825 G | 25.8511 G | -11.1111% |
| entropy_neural_macs | 41.3319 G | 35.8049 G | -13.3722% |
| neural_decoder_macs | 70.4144 G | 61.6560 G | -12.4383% |
| encoder_with_reconstruction_macs | 109.8532 G | 96.6836 G | -11.9884% |

Yatay/dikey ve iki fazda sonuç aynı. Dikdörtgen Conv2d boyutları ayrı izlendi; D2/288×512 ve512×288 gerçekCPU trace'leri meta trace ile birebir eşleşti. Bu kalite veya runtime ölçümü değildir; bitstream kodlama, MLP, aktarım ve gruplama maliyetleri dahil değildir.
