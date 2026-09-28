# Sabit derinlikte patchleme: kalite, gerçek byte ve sınır hatası

Önceden belirlenen16 DIV2K merkez512 crop; D2/D6 epoch20, released D12; beş QP. CPU FP32 araştırma formatı. Router deneyi veya GPU hız ölçümü değil.

| Model | Payload bpp | Ortak görüntü | Halo0−full dB | Halo32−full dB | Halo64−full dB |
|---|---:|---:|---:|---:|---:|
| D2 | 0.1 | 10/16 | -0.0895 | -1.3259 | -1.5752 |
| D2 | 0.2 | 16/16 | -0.0606 | -1.3085 | -1.6701 |
| D2 | 0.4 | 15/16 | -0.0492 | -1.3705 | -1.8711 |
| D6 | 0.1 | 10/16 | -0.0899 | -1.2970 | -1.5484 |
| D6 | 0.2 | 16/16 | -0.0611 | -1.2914 | -1.6666 |
| D6 | 0.4 | 15/16 | -0.0551 | -1.3602 | -1.8710 |
| D12 | 0.1 | 12/16 | -0.0939 | -1.2715 | -1.5388 |
| D12 | 0.2 | 16/16 | -0.0791 | -1.3079 | -1.6969 |
| D12 | 0.4 | 15/16 | -0.0679 | -1.4040 | -1.9483 |

Ortak destek her modelde full/halo0/halo32/halo64 eğrilerinin kesişimidir. Derinlikler arasında kapsanan görüntüler farklı olabilir; bu tablo tek başına derinlik sıralaması için kullanılmaz.
Paired görüntü bootstrap aralıkları ve lineer–PCHIP duyarlılığı analysis.json içindedir. Veriler küçük, önceden belirlenmiş bir ara kontroldür.

32-pixel halo ile dört288×288 pencere, codec padding sonrası dört320×320 alan kodlar. Toplam kodlanan alan full512'nin1,5625 katıdır. Bu geometrik oran hız oranı değildir.
64-pixel halo doğrudan320×320 pencere kullanır: 32-halo ile aynı kodlanan alan, daha fazla gerçek context. Bu varyant hiçbir patch sonucu görülmeden eklendi.
Dört bağımsız stream toplam352 research-header byte taşır; full-frame88 byte. Extra264 byte =0,00805664bpp. Payload ve container ayrı raporlanır.
Seam ölçümleri aynı QP'de full-frame hatası çıkarılarak yapılır; kalite ve bitrate birlikte değişebilir. Aynı-bitrate global PSNR karşılaştırması yukarıdadır.
