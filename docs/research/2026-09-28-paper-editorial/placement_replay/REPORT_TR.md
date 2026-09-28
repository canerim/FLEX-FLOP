# Sabit derinlik dağılımında gerçek yerleştirme kontrolü

53 ilk frame × 2 sabit Q90 politika × (orijinal + 3 permütasyon) = 424 harita vakası. Tüm vakalar tamamlandı; orijinal haritalar önceki CPU replay çıktısını yeniden üretti. Tile derinlik sayıları ve geçerli piksel alanı tabakalarının derinlik dağılımları korundu.

Birincil ölçüm her görüntü için 10 log10(üç permütasyonun ortalama RGB MSE’si / orijinal RGB MSE). Pozitif değer orijinal yerleştirmenin daha iyi olduğunu belirtir. Ardından 53 görüntünün eşit ağırlıklı ortalaması alınır.

| Politika / fark | RGB yerleştirme kazancı (dB) | %95 sequence bootstrap aralığı |
|---|---:|---:|
| router | +0.013925 | [+0.009490, +0.019303] |
| dither | +0.000358 | [-0.000314, +0.001050] |
| router_minus_dither | +0.013567 | [+0.009204, +0.018969] |

Aralıklar yalnız bu üç permütasyona koşulludur; permütasyon örnekleme hatasını, yeniden eğitim belirsizliğini veya bağımsız test genellemesini kapsamaz. Değişmeyen haritalar ve uniform örnekler çıkarılmadı. 51 içerik grubuyla duyarlılık analizi JSON içinde yer alır. Önceki tile-error-table permütasyon tanısıyla protokol ve metrik farklıdır; sayılar doğrudan birleştirilmemelidir. Bu deney bitstream veya hız ölçümü değildir.
