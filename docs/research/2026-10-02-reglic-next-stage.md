# RegLIC: eğitim bitişi ve kanıt önceliği

**Durum zamanı:** 2 Ekim 2026, yaklaşık 13:48 Europe/Berlin. Bu plan, çalışan eğitimlerin tarifini veya GPU tahsisini değiştirmez.

## Kalan süre

Her epoch 24.049 batch. D6 512×512 fazında; D8/D10/D12 ilk 90 epoch'un 256×256 fazında. Tamamlanan D2/D4 ve çalışan D6 loglarında 512 fazı, güncel 256 fazına göre yaklaşık 3,3–3,7 kat daha yavaş. Son pencere hızı ve bu aralık kullanılarak, değerlendirme/kesinti/GPU çekişmesi hariç:

| Koşu | 2 Ekim durum | Saf eğitim kalan | İyimser bitiş (Berlin) |
|---|---|---:|---|
| D6 | epoch 99 | ~14 saat | 3 Ekim sabah erken |
| D8 | epoch 19 | ~4,5–4,9 gün | 7 Ekim sabah |
| D10 | epoch 14 | ~4,9–5,2 gün | 7 Ekim öğleden sonra |
| D12 scratch | epoch 9 | ~5,2–5,5 gün | 7 Ekim gece / 8 Ekim başı |

Bu aralıklar bir vaat değil; `status.json` ve 256→512 geçişinde yeniden hesaplanmalı. D8/D10/D12'nin Kodak ve runtime otomatik değerlendirmesi eğitime ek süre ister. Resmî 105 epoch tarifini süre için kısaltmıyoruz.

## Eğitim tamamlanınca sıralama

1. **Bütünlük:** 105 epoch / 2.525.145 batch, finite ağırlık ve optimizer, doğru decoder blokları, manifest ve kaynak hash'leri, 24 Kodak görüntüsü × 5 QP, aynı değerlendirme kodu. D12 scratch ile released D12'nin eş oranlı RGB/YUV 6:1:1 kıyası yeniden üretilebilirlik kontrolü; eşit checkpoint beklenmez.
2. **Sabit-expert Pareto eğrisi:** D2/4/6/8/10/12 tam görüntü RD ve aynı koşullarda encoder, decoder, entropy ve toplam wall-clock. Tahmini entropy bpp ile gerçek bitstream baytlarını ayrı göster. Kodak yanında önceden ayrılmış başka test görüntüleri kullan; test setinde politika ayarlama.
3. **Patch bedeli:** Router eklemeden, full-frame D12 ile uniform patch-D12'yi aynı kayıpsız kodlama protokolünde kıyasla. Bitstream header/mode bitleri, entropy reset, sınır PSNR ve görüntü kalitesini ölç. Bu maliyet bilinmeden altı-model spatial kazancın yorumu geçersiz.
4. **Router:** Bağımsız codec bankasında seçim RGB patch/ucuz özellikler ve QP'den *encoding öncesi* yapılır; model latentleri ortak değildir. Altı codec'i çalıştıran oracle yalnız eğitim etiketi ve üst sınırdır; inference encoder maliyetine yazılmaz. Ayrı eğitim/kalibrasyon/test görüntüleri; görüntüye ait patch'ler split'ler arasında karışmaz. MLP, expert başına distortion, gerçek byte ve measured latency/regret tahmin eder; QP ve runtime bütçesi girdidir. Model ID ve gruplama maliyeti bitstream'e eklenir.
5. **Karşılaştırmalar:** Her sabit expert; en iyi global expert; basit varyans/kenar kuralı; görüntü-düzeyi seçim; dither/random; aynı expert histogramıyla spatial shuffle; tam oracle. 2/3/6 expert, patch boyutu, halo, router özelliği ve spatial-coherence cezası ablasyonları. Hepsi aynı bitrate/kalite/runtime bütçesinde, aynı patch pipeline'ında.

## Eğitimle paralel, ana GPU işlerini etkilemeden

- Codec byte muhasebesi ve latency ölçüm protokolünü kod/CPU testleriyle bitir: yan bilgi, entropy stream, model ID, padding, aktarım, microbatch/stitching ve bellek dahil. Büyük GPU inference koşuları eğitim slotlarına bindirilmez.
- Router eğitim/kalibrasyon/test split manifestini ve source-informed oracle etiket formatını dondur. Mevcut D2/D4 ve released D12 ile yalnız CPU düzeyinde veri şeması, interpolation ve leakage testleri hazırlanabilir; altı expert oracle etiketi nihai checkpoint'leri bekler.
- Ölçüm plotlarını ve makale iddia tablosunu hazırlayıp her cümleyi `measured`, `diagnostic` veya `prospective` olarak işaretle. Scratch D12 finali ve gerçek bitstream olmadan “released parity” veya altı-model routing zaferi yazılmaz.
- [Spatial Competition](https://arxiv.org/abs/2605.13243) zaten codec başına region seçiyor; bu nedenle katkı “ilk spatial codec routing” olamaz. [MixCompress](https://arxiv.org/abs/2607.14334) değişken derinlik uzmanları kullanıyor; bizim kanıtlanması gereken ayrım içerik-bağımlı spatial seçim ve ölçülmüş codec zaman/byte bedeli.

## Yayın için karar eşiği

En güçlü tez, *aynı teslim edilen kalite ve gerçek bit hızında* sabit/dither/image-level kontrollerinden üstün, anlamlı **uçtan uca decode hızı** ve yönetilebilir encode/router bedeli. Eğer MLP'nin bu kontroller üzerindeki kazanımı belirsiz kalırsa, sonucu dürüstçe shared-exit ve depth–runtime frontier üzerine kur; altı bağımsız codec bankasını doğrulanmamış ana katkı diye sunma. Şu anki `1088×1920` modül medyanları (D2 45,3+23,1 ms; D4 45,3+35,8 ms; released D12 45,4+86,5 ms) yalnız neural encoder+decoder için yaklaşık 1,93×/1,63× potansiyel gösterir, gerçek codec speedup değil.
