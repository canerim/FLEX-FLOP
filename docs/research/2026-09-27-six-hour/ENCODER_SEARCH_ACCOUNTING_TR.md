# Encoder'da best-match ile MLP seçiminin karşılaştırma sözleşmesi

Ana hedef, altı farklı DCVC-UF derinliğinden birini patch kodlanmadan önce
seçmek. Encoder ve entropy ağırlıkları bağımsız öğrenildiğinden, bir
modelin encoder çıktısı diğer modele ücretsiz taşınamaz.

## Üç farklı best-match uygulaması

**Gerçek payload kullanan kaynak bilgili arama.** Her expert encode edilir,
gerçek rANS byte sayısı ve reconstruction hatası ölçülür. Seçilen stream
saklanmışsa yeniden encode etmeye gerek yoktur. Fakat diğer beş expert'in
analysis, prior, synthesis ve entropy coding maliyeti zaten ödenmiştir.
Bu, gerçek `(D,R)` oracle etiketine yakın bir ölçüm kontrolüdür; düşük
maliyetli encoder sayılmaz.

**Entropy tahminiyle neural arama.** Her expert'in neural forward'ı aday
distortion ve entropy tahmini üretir. Yalnız seçilen expert gerçek entropy
coding'den geçirilir. Latent/parametre cache'i yoksa seçilen analysis ve
prior tekrar çalışır; cache varsa cache taşıma ve bellek maliyeti ölçülür.
Seçim gerçek byte yerine tahmine dayandığı için ilk uygulamayla aynı
optimizasyon hedefi değildir.

**Encoder öncesi MLP.** Ucuz kaynak özellikleri çıkarılır, bir expert
seçilir ve yalnız o codec kodlanır. Encoder'da reconstruction hesaplanması
yalnız codec'in zorunlu yolu veya ayrıca bildirilen kalite doğrulaması
gerektiriyorsa dahil edilir. Araştırma referans codec'imiz doğrulama için
reconstruction üretiyor; onun CPU encode süresini bu optimize edilmiş
uygulamanın süresi diye kullanamayız.

Bu maliyet ayrımı literatürde de önemlidir: spatial competition yönteminin
mode search aşaması bütün aday codec'lerin analysis, entropy ve synthesis
yollarını değerlendirir. Aynı yöntemin nihai encoding'i seçilen codec ile
yapılır; decoder yalnız seçilmiş modeli çalıştırır.
[Spatial competition, §2.2](https://arxiv.org/html/2605.13243v1).

## Ölçülecek süreler

| Yol | Encoder ölçümü | Decoder ölçümü | Rate ölçümü |
|---|---|---|---|
| Sabit expert | Analysis + prior + gerçek entropy coding + I/O | Entropy recovery + synthesis + I/O | Payload ve gerekli header |
| MLP | Feature + selector + grouping + seçilen encoderlar + entropy + I/O | Mode parse + grouping + seçilen decoderlar + assembly | Payload + mode map + header |
| Best-match, gerçek byte | Tüm aday encode/decode/metric işleri + seçim + cache/assembly | MLP ile aynı seçilmiş-codec yolu | Seçilen gerçek payload + signalling |
| Best-match, tahmini rate | Tüm aday neural forward'ları + seçim + seçilen gerçek coding | Seçilmiş-codec yolu | Son gerçek payload; seçim tahmini kullandı diye etiketle |

Birden fazla patch aynı expert'te gruplanırsa süre `sum_k tau_k(n_k)` ile
ölçülmeli. `mean(tau_k(1)) × N` batch doluluğunu ve kernel sayısını kaçırır.
Model ağırlıklarının GPU'da hazır olduğu resident-bank ölçümü ile expert
yükleme/taşıma içeren cold-bank ölçümü ayrı tutulmalı.

## Net kazancın kabul ölçütü

MLP'nin hedefi oracle ile aynı etiketi yüksek doğrulukla tahmin etmek
değil; **aynı gerçekleşen rate–quality düzeyinde toplam süreyi azaltmak**.
Etiket doğruluğuna ek olarak seçim regret'i, sabit expert'e göre net süre,
blind karışıma göre ek kazanç, tail latency, toplam model belleği ve
yanlış seçimde kalite kaybı dağılımı raporlanmalı.

Eğer D2 ve D6 ortalama PSNR'de yakınsa, bu MLP'nin yararsız olduğunu da
yararlı olduğunu da tek başına göstermez. Gerekli kanıt patch'ler arasında
derinliğin marjinal kalite katkısının değişmesi, bu değişimin ucuz
özelliklerden tahmin edilebilmesi ve gruplanmış yürütmede seçimin kendi
maliyetini karşılamasıdır. Şu anki dört-crop/100-crop depth değerlendirmeleri
ilk soruya destek sağlar; son iki sorunun deneyleri henüz yapılmadı.
