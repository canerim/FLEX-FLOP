# FLEX-UF — Karar Günlüğü (Decision Log)

Bu dosya projenin **her adımını, neden yapıldığıyla birlikte** kaydeder.
Kural: bir satır "ne yaptım", altında "**Neden:**" ve gerekiyorsa "**Kanıt:**".
Hiçbir adım gerekçesiz girilmez. Ölçülmemiş sayı yazılmaz — ölçülene `[ölçüldü]`,
tahmine `[tahmin]`, kaynaktan alınana `[kaynak: ...]` etiketi konur.

Proje: DCVC-UF üzerine içerik-uyarlamalı **early exit** (erken çıkış) decoder.
Hedef: **çok küçük PSNR kaybı, çok büyük FLOP kazancı.**
Uzak depo: `git@github.com:canerim/FLEX-FLOP.git`
Yerel klasör: `~/FLEX-UF` (yeni isim — eski çalışma `~/Flex-Flop`'ta duruyor, ona dokunulmadı)

---

## 0. Başlangıç durumu tespiti (2026-08-15)

### 0.1 GPU envanteri

`nvidia-smi` ile 8× NVIDIA RTX A6000 (her biri 49140 MiB) bulundu. Anlık durum:

| GPU | util | bellek | durum |
|----:|-----:|-------:|-------|
| 0 | %88 | 23.4 / 49 GB | başkası kullanıyor |
| 1 | %99 | 5.1 / 49 GB | başkası kullanıyor |
| 2 | %100 | 44.2 / 49 GB | başkası kullanıyor |
| 3 | %0 | 44.2 / 49 GB | bellek dolu, iş yok — **dokunma** |
| **4** | **%0** | **12 MiB** | **BOŞ** |
| 5 | %100 | 44.2 / 49 GB | başkası kullanıyor |
| **6** | **%0** | **12 MiB** | **BOŞ** |
| **7** | **%0** | **12 MiB** | **BOŞ** |

**Karar: sadece GPU 4, 6, 7 kullanılacak.**
**Neden:** 0/1/2/5 aktif iş çalıştırıyor; onlara girmek başkasının eğitimini OOM'la
düşürür. GPU 3 boş görünüyor ama 44 GB bellek tutuluyor — sahipsiz görünen bir
process ya da çökmüş bir job olabilir; boş sanıp üzerine iş atmak riskli.
Her komutta `CUDA_VISIBLE_DEVICES=4,6,7` ile sınırlanacak.

### 0.2 Erişim / ortam

| şey | durum | not |
|---|---|---|
| GitHub SSH | ✅ çalışıyor | yeni ed25519 anahtar üretildi, kullanıcı GitHub'a ekledi. `Hi canerim!` ile doğrulandı |
| Python | 3.10.12 (sistem) | conda yok |
| PyTorch | ❌ **kurulu değildi** | kuruldu → §1.1 |
| nvcc / CUDA toolkit | ❌ yok | derleyici gerektiren UF inference eklentileri için sorun → §2 |
| NVIDIA driver | 535.309.01 | **CUDA 12.2 tavan** |
| Disk | `/` 1.6 TB boş, `/data10` 4.3 TB boş | dataset için `/data10` uygun |

**Kritik kısıt: driver 535 → CUDA 13 wheel'leri ÇALIŞMAZ.**
**Neden:** DCVC-UF README'si `pip install torch --index-url .../cu130` diyor ve
CUDA 13 runtime'ı ≥ r580 driver ister. Bizde 535 var. CUDA 12.x wheel'leri ise
minor-version uyumluluğu sayesinde 535'te sorunsuz koşar.
**Sonuç:** README'nin `cu130` satırı bu makinede **birebir uygulanamaz**;
`cu124` kullanılacak. Bu, recipe'den bilinçli ve zorunlu tek sapma —
eğitim tamamen saf PyTorch olduğu için sonuçları etkilemez (CUDA eklentileri
sadece gerçek bitstream encode/decode için gerekir, eğitim için değil).

### 0.3 Dataset — ❗ BULUNAMADI

UF recipe'sinin istediği veriler (`training.md`):
- **Open Images** train subset 0, 1, 2 → intra (görüntü) modeli için
- **Vimeo-90k** septuplet + original videos → video modeli için

Aranan yerler ve sonuç:

| yer | sonuç |
|---|---|
| `/data10/shareddata/` | sadece `ScanNetv2`, `view_of_delft` — **OpenImages yok** |
| `/data10/staff/`, `/data10/student/` | dataset klasörleri var ama hepsi başka projelere ait, OpenImages/Vimeo yok |
| `/mnt`, `/srv`, `/data`, `/shared` | mevcut değil |
| `~` (ev dizini) | sadece `Flex-Flop` (413 MB) ve `DCVC` (21 MB) |
| tüm dosya sistemi (depth-6 tarama) | `*openimage*`, `*vimeo*`, `*septuplet*`, `*uvg*`, `*kodak*` → **0 eşleşme** |
| diğer kullanıcıların ev dizinleri | `drwxr-x---` — **okuma izni yok** (`id`: sadece `can_karsal`, `docker` gruplarındayım) |

**Sonuç: bu makinede erişebildiğim hiçbir yerde eğitim verisi yok.**
**Neden önemli:** "0'dan eğit" talebi veri olmadan başlayamaz. Veri ya
indirilecek (Open Images subset 0-2 yüzlerce GB) ya da başka kullanıcının
dizinindeyse bana o dizin açılacak. Bu, insan kararı gerektiren bir engel — §5'e işlendi.

### 0.4 Checkpoint — indirilemedi

DCVC-UF README'si 4 checkpoint veriyor (1 image + 3 video: HT-L, HT-S, LD),
OneDrive linki üzerinden. `curl` ile denendi → **HTTP 403**.
**Neden:** OneDrive paylaşım linki JS tabanlı bir sayfa; başsız indirme için
tarayıcı etkileşimi ya da özel API çağrısı gerekiyor. Otomatik indirme
yolları ayrıca araştırılıyor (§5).

---

## 1. Yapılan hazırlıklar

### 1.1 Depolar çekildi

```
~/Flex-Flop   ← git@github.com:circuitmaster/Flex-Flop.git   (referans, dokunulmayacak)
~/DCVC        ← git@github.com:microsoft/DCVC.git            (UF kaynağı, dokunulmayacak)
~/FLEX-UF     ← yeni proje (bu repo → canerim/FLEX-FLOP)
```

**Neden `~/DCVC` tek clone:** `microsoft/DCVC` bir monorepo. Deponun **kökü**
DCVC-UF'in kendisi (CVPR2026); eski DCVC (NeurIPS 2021) ise
`DCVC-family/DCVC/` alt klasöründe. Yani "DCVC-UF'i çek" ve "DCVC'yi de çek"
tek clone ile karşılandı, ayrı depo yok.
**Kanıt:** `~/DCVC/README.md` başlığı `# DCVC-UF: Ultra-Fast Neural Video Compression`;
`~/DCVC/DCVC-family/` içinde `DCVC, DCVC-TCM, DCVC-HEM, DCVC-DC, DCVC-FM, DCVC-RT, EVC`.

**Neden `~/Flex-Flop`'a dokunulmuyor:** o bitmiş bir çalışma (63 sonuç CSV'si,
31 figür, 5 branch). Yeni iş ayrı klasörde, ayrı repoda — eski sonuçlar
kazara bozulmasın.

### 1.2 `training.md` güncelliği kontrol edildi

Kullanıcı "training modülünü güncelle, yenilemişlerdir" dedi.
`git fetch origin` → **"Already up to date"**. Depodaki `training.md`
(378 satır, son commit 2026-07-23) kullanıcının elindeki `training-2.md` ile
**birebir aynı**.
**Neden yine de kontrol edildi:** güncel olduğunu varsaymak yerine ölçmek gerekiyordu;
eski bir recipe'ye göre kod yazmak tüm eğitimi boşa çıkarırdı.

### 1.3 Ortam kuruldu

`~/FLEX-UF/.venv` içine PyTorch **cu124** kuruldu.
**Neden cu124, cu130 değil:** §0.2 — driver 535 CUDA 13'ü çalıştıramaz.
**Neden sistem Python'a değil venv'e:** makine paylaşımlı; sistem paketlerini
değiştirmek diğer kullanıcıların işlerini bozabilir.

---

## 2. Mimari inceleme — UF decoder, FLEX'in hedeflediği yapıya çok yakın

`~/DCVC/src/models/image_model.py:21-47` (IntraDecoder):

```
dec_1 = [ ResidualBlockUpsample(256 → 384),      ← açılış bloğu, hep çalışır
          DepthConvBlock(384→384) × 12 ]          ← 12 blokluk gövde (trunk)
        × quant_step
dec_2 = DepthConvBlock(384 → 192)                 ← paylaşılan head
PixelShuffle(8)                                   ← 0 MAC, saf reshape
```

FLEX'in DCVC-RT üzerinde kullandığı yapı (`docs/FORMAL_PAPER.md` §2):
**upsample + 12 DC blok + head + PixelShuffle(8)** — tek fark kanal genişliği
(RT'de C=368, UF'te C=384).

**Neden bu önemli:** FLEX'in çok-çıkışlı merdiveni (12 bloğu K gruba bölüp her
grup sonuna sıfır-init adapter koymak) UF'in intra decoder'ına **neredeyse
bire bir** oturuyor. Warm-start anahtar eşlemesi de aynı kalıpta
(`dec_1.0 → upsample`, `dec_1.j → groups[g][i]`, `dec_2 → head`).
Yani port riski düşük; sıfırdan mimari icat etmeye gerek yok.

Video tarafı (`src/models/video_model_ht.py`) farklı ve daha zengin:
`g_ch_d = 512`, hts'te 7 blok / htl'de 11 blok, ayrıca chunk tabanlı kodlama
(`g_frame_delay` kare tek latent'e giriyor, kare-özel decoder'lar paralel
çözüyor). Bu, early exit için **ekstra bir eksen** açıyor: bir chunk içindeki
farklı kareler farklı derinlikte çıkabilir. Detaylı analiz devam ediyor.

---

*(devam edecek — sonraki bölümler: tasarım kararı, 3 GPU deney planı, kod)*

---

## 3. Referans makaleler ve kayıp fonksiyonu

### 3.1 ClassSR (CVPR 2021) — temel felsefe

Görüntüyü alt-görüntülere böl → her birinin zorluğunu bir "Class-Module" ile
skorla → zorluğa uygun kapasitedeki dala yönlendir.

FLEX-UF'te dallar ayrı ağlar değil, **tek decoder'ın önekleri**. Bu ClassSR'a göre
net kazanç: 3 ayrı SR ağı yerine tek gövde, ekstra parametre yok, ve en derin dal
değiştirilmemiş codec'in ta kendisi.

Kullandığımız denklemler:
- **Eq (1)** `y = Σ_i f_i(x)·P_i(x)` — yumuşak karışım, router'a gradyan aksın diye
- **Eq (2)** `L = w1·L1 + w2·L_c + w3·L_a`, w1=2000, w2=1, w3=6 (makalenin kendi değerleri)
- **Eq (3) Class-Loss** `L_c = -Σ_{i<j}|P_i-P_j|` — dağılımı one-hot'a iter
- **Eq (4) Average-Loss** `L_a = Σ_i|Σ_j P_i(x_j) - B/M|` — tüm çıkışları kullanımda tutar

**Neden üçü birden gerekli:** Eq(1) tek başına türevlenebilir yapar ama çıkarımda
argmax çalışır → eğitilen fonksiyon ile koşan fonksiyon farklı olur. Eq(3) bu
açığı kapatır. Eq(4) olmazsa router her patch'i en derin çıkışa yollar (distorsiyon
için optimal, kazanç sıfır) — dejenere çözümü engelleyen tek şey odur.

**Bizim eklediğimiz:** ClassSR'ın 3 sabit dalı var, bizim sürekli bir maliyet
yelpazemiz. Bu yüzden `L_comp = β·Σ_i P_i·C_i` eklendi; β süpürülerek frontier
çizilir.

### 3.2 İkinci makale — tespit edildi

Kullanıcının tarif ettiği "farklı exit'leri nasıl train edeceğine dair sistem,
her exit için Eq 6-7" araması sonucu:

**Scardapane, Scarpiniti, Baccarelli, Uncini — "Why should we add early exits to
neural networks?", Cognitive Computation 12 (2020), arXiv:2004.12814.**

Bölüm 5 tam olarak "çok-çıkışlı ağlar nasıl eğitilir" taksonomisi, ve:
- **Eq (6)** `f* = arg min { L + Σ_{i=1}^{L-1} α_i·L_i }` — joint training
- **Eq (7)** `L_i = Σ_n ℓ(y_n, c_i(x_n))` — her exit'in yardımcı (companion) kaybı

Elenen adaylar ve nedeni: MSDNet (numaralı denklemi yok), Shallow-Deep (numaralı
denklemi yok), BranchyNet (Eq 6-7 yok), "Improved Techniques" ICCV19 (Eq 7 var
ama baseline olarak). Scardapane tek tam eşleşme.

### 3.3 Eq (6)-(7)'nin sıkıştırmaya uyarlanması — ve bir tuzak

`ℓ` yerine Microsoft'un kendi RD kaybı konur (`common.py:166-171`):
`ℓ = λ·mse + bpp`.

**Tuzak:** bpp her exit'te **aynı** — entropy decode bir kez, full-frame, decoder
çağrılmadan önce çalışır. Eq(6)'yı düz yazarsak aynı rate terimi K kez toplanır,
K farklı distorsiyon terimi toplanır → rate'in efektif ağırlığı K katına çıkar ve
model Microsoft'un λ'larının tarif ettiğinden **başka bir RD noktasına** oturur.

**Çözüm:** Eq(6) toplam ağırlığa bölünerek normalize edilir:
```
L = [ L_{K-1} + Σ α_k L_k ] / [ 1 + Σ α_k ]  =  λ·(exit'ler üzerinden ağırlıklı ortalama mse) + bpp
```
Böylece RD çalışma noktası **birebir Microsoft'unki kalır**; tek yaptığımız
distorsiyon denetimini merdivene yaymak. `flexuf/losses.py`.

---

## 4. Ölçülen mimari gerçekler

### 4.1 MAC denetimi — UF decoder'ı FLEX'in hedefiyle neredeyse aynı

`scripts/mac_audit.py`, 1920×1088, her `nn.Conv2d`'ye forward hook, MAC'ler
gerçek çıktı şekillerinden:

| parça | GMAC | pay | FLEX'in DCVC-RT ölçümü |
|---|---:|---:|---:|
| açılış upsample | 37.01 | **%8.16** | %8.1 |
| 12 bloklu trunk | 405.64 | **%89.44** | %89.4 |
| head | 10.89 | **%2.40** | %2.5 |
| **toplam** | **453.54** | %100 | — |
| pointwise 1×1 | 452.02 | **%99.66** | %99.7 |

**Neden bu önemli:** UF'in MAC profili DCVC-RT'ninkiyle neredeyse birebir aynı
çıktı. Yani FLEX'in tüm mimari kararları (K=6, j-split, zero-init adapter,
full-frame head) UF'e taşınabilir; sıfırdan mimari icat etmeye gerek yok.

Erken çıkış tavanı (upsample+head hep çalışır = MAC'in %10.56'sı):

| g blok sonrası çık | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---:|---:|---:|---:|---:|---:|
| kazanç | %74.5 | %59.6 | %44.7 | %29.8 | %14.9 | %0 |

### 4.2 Adapter neden 1×1 conv — sayılarla

Bir DepthConvBlock `8C² + 9C` MAC/px. Bunun:
- FFN kısmı `6C²` = **%74.8**
- tek uzamsal operatör (3×3 depthwise) `9C` = **%0.3**

Yani erken çıkmak neredeyse **saf pointwise kanal-karıştırma kapasitesi**
kaybettiriyor. Doğru telafi de pointwise olmalı → 1×1 conv.

**İkinci ve kritik avantaj:** 1×1'in alıcı alanı **yok**. Dolayısıyla patch sınır
cezasına **sıfır** katkı yapar. İçinde 3×3 olan bir adapter, tam da erken çıkan
(yani en kırılgan) patch'lere ek dikiş hasarı verirdi.

Maliyet: `C²` MAC/px = bir bloğun 1/8'i. Adapter'lar toplam **739,200 parametre**
= modelin %1.72'si [ölçüldü].

### 4.3 ❗ Halo yerleşimi — talebi doğrudan uygulamak projeyi bozuyordu

İstek: "HALO'yu latent HALO yap ve 1px'den fazla yap, yoksa router tarafında
sıkıntı çıkar."

Latent halo=2 (feature 4px) alıp **per-patch trunk'a** uygulayınca ölçülen net
kazanç (`flexuf/cost.py`, j=2, 128px tile):

| exit | halo@head | halo@taper | halo@trunk |
|---:|---:|---:|---:|
| 2 | %43.8 | %26.8 | %24.0 |
| 3 | %28.9 | %6.1 | **%-9.5** |
| 4 | %14.0 | %-8.6 | **%-43.1** |
| 5 | %0.0 | %-22.1 | **%-74.5** |

64px tile'da daha da kötü: **%-178.9**.

**Neden:** halo'lu bir tile, tuttuğu pikselden `((F+2h)/F)²` kat fazla piksel
hesaplar. F=16, h=4 → **2.25×**. Bu çarpan ağın pahalı kısmına (trunk %89.44)
binince kazanç negatife dönüyor — decode tam decode'dan pahalı hale geliyor.
FLEX bundan kaçınmıştı çünkü halo'yu sadece head'e (%2.40) uygulamıştı: 1.5625×
× %2.40 = decode'un +%1.35'i.

**Karar — talebin gerekçesi korunarak, mekanizması değiştirildi:**

Halo'nun iki ayrı işi var ve bunlar ayrıştırılabilir:
1. **Rekonstrüksiyon kalitesi için** → head'de halo. Ucuz, FLEX'te kanıtlı
   (1px feature halo dikişin %99.1'ini siliyor).
2. **Router'ın temiz istatistik görmesi için** → *senin asıl gerekçen buydu.*
   Router **latent'i** okur, decode etmez. Router'ın maliyeti decode'un
   **%0.009'u**. Yani router'a halo vermek **bedava**.

Bu yüzden: router girdisine **cömert halo (varsayılan 4 latent px, decode
halo'sunun 2 katı)**, head'e yapılandırılmış halo, per-patch trunk'a halo yok.
Böylece "router sıkıntı çıkarmasın" hedefi tam karşılanıyor, FLOP kazancı da
duruyor. `halo@taper` yine de bir deney kolu olarak kodda duruyor.

---

## 5. ❗ AÇIK ENGEL — eğitim verisi

Talimat: "dataset /mnt/data_local/datasets altında, shared'lardan bulursun,
kendin 0'dan indirme."

Yapılan arama ve sonuç:

| arandı | sonuç |
|---|---|
| `/mnt/data_local/datasets` | **yok** — `/mnt` tamamen boş, fstab'da yok, autofs yok |
| `data_local` adında herhangi bir yol (tüm fs) | 0 eşleşme |
| `/data10` (tüm alt dizinler okunabilir) | OpenImages yok; `description.json` hiç yok |
| `/data10/shareddata` | sadece ScanNetv2 (786G), view_of_delft (27G) |
| diğer `/home/*` | hiçbiri okunabilir değil (`drwxr-x---`) |
| tüm fs'de `open_images`/`openimages`/`data_local` string'i geçen kod/config | 0 eşleşme |
| docker volume'ları | sadece toolchain imajları, dataset yok |

**Sonuç: erişebildiğim hiçbir yerde OpenImages yok.** Kullanıcı onayı gerekiyor.

İndirme talimat gereği **durduruldu** (`pkill`, doğrulandı). Yarım kalan
`/data10/shareddata/openimages/train_0.tar.gz` = 41 GB (train_0'ın %87'si)
**silinmedi**, çünkü silmek geri alınamaz ve gerekebilir — kullanıcı kararına
bırakıldı.

**GPU disiplini doğrulandı:** `nvidia-smi` compute-apps listesinde bana ait
**hiçbir process yok**; dolu GPU'lar root/cankan/an_li'ye ait. Sadece GPU 4'te
saniyelik smoke test'ler koştu.

---

## 6. Mimari karar — nereye, ne koyuyoruz

### 6.1 Merdiven yerleşimi

`IntraDecoder.dec_1[1..12]` (12 DepthConvBlock, hepsi 384→384) K=6 gruba
bölünür, b = 12/6 = 2 blok/grup. Her grubun sonuna sıfır-init adapter.
`dec_1[0]` (upsample) ve `dec_2` (head) hep çalışır.

**Neden K=6:** FLEX bu ekseni süpürmüş ve serbest bir knob **olmadığını**
bulmuş — K=3 ve K=12 çöküyor, router patch'lerin ≥%97'sini tek-iki çıkışa
yığıyor. K=6 onun çalışma noktası, UF'te de trunk aynı 12 blok.

### 6.2 ClassSR yapısı

```
ortak gövde (grup 0..j-1, full-frame)     ← ClassSR'ın paylaşılan girişi
        ↓
patch'lere böl (128×128 RGB)              ← ClassSR'ın alt-görüntüleri
        ↓
per-tile merdiven (grup j..K-1)           ← ClassSR'ın farklı kapasiteli dalları
        ↓
1×1 adapter → paylaşılan head → RGB
```

**ClassSR'a göre kazancımız:** onun 3 dalı 3 ayrı ağ (3× parametre). Bizim
dallarımız tek decoder'ın önekleri — ek parametre yok, ve **en derin dal
değiştirilmemiş codec'in kendisi** (bit-exact doğrulandı).

### 6.3 Patch 128×128 seçimi

| p (latent) | RGB tile | 256-crop'ta tile | 512-crop'ta tile | 1080p'de tile |
|---:|---:|---:|---:|---:|
| 4 | 64px | 16 | 64 | ~480 |
| **8** | **128px** | **4** | **16** | **~120** |

**Neden 128:**
1. Boundary Law: ceza dikiş yoğunluğuyla, yani `1/kenar` ile ölçekleniyor.
   FLEX ölçümü: 128px'te **0.856 dB**, 64px'te **2.484 dB** — 128 aynı fikir
   için üçte bir distorsiyon ödüyor.
2. Microsoft 256 (0-89. epoch) ve 512 (90+) crop'ta eğitiyor. 128 ikisini de
   tam bölüyor, artık kalmıyor.
3. Halo amortismanı: overhead `((F+2h)/F)²`. F=16'da h=4 → 1.56×; F=8'de 2.25×.
   Büyük tile halo'yu ucuzlatıyor.
4. 1080p'de ~120 tile — yönlendirme çözünürlüğü hâlâ fazlasıyla yeterli.

E3 deneyi 64'e geri sürüyor ki bu takas **varsayılmasın, ölçülsün**.

---

## 7. Üç deney — her biri E1'den TEK değişkende ayrılıyor

| | GPU | j | tile | halo | adapter | ne test ediyor |
|---|---:|---:|---|---|---|---|
| **E1** | 4 | 2 | 128px | 2 lat | 1×1 | ana konfigürasyon, en geniş yönlendirilebilir aralık |
| **E2** | 6 | 4 | 128px | 2 lat | 1×1 | **split derinliği ekseni** |
| **E3** | 7 | 2 | 64px | 2 lat | 1×1 | **tile boyutu ekseni** (Boundary Law testi) |

**Neden tam olarak bu üçü:** her biri E1'den tek değişkende farklı, yani her
karşılaştırma temiz bir ablation — iki değişken aynı anda değişseydi sonucu
hangisinin ürettiğini söyleyemezdik.

**Hipotezler (önceden yazıldı, sonuca göre değiştirilmeyecek):**
- E1: 8 blok per-tile koşuyor; 1×1 adapter'lar dikişi yeterince emerse
  ölçülen tavan %58.7 kazanç. Risk: dikiş hasarı adapter kapasitesini aşar.
- E2: sadece 4 blok per-tile → çok daha az sınır hasarı, ama tavan %28.9.
  Beklenti: daha iyi dB, daha az kazanç. j ekseninin şeklini verir.
- E3: Boundary Law 64px'in 128px'in ~2 katı dB'ye mal olacağını söylüyor,
  karşılığında 4× yönlendirme çözünürlüğü. Beklenti: 128 frontier'da kazanır.

**Neden 3 bağımsız tek-GPU koşusu, DDP değil:** tek konfigürasyonun 3× hızlı
koşması tek soru cevaplar; 3 koşu üç soru cevaplar ve bu üç eksen UF üzerinde
hiç ölçülmemiş. Her koşu `CUDA_VISIBLE_DEVICES` ile sabitlendi.

**GPU disiplini:** sadece 4, 6, 7. GPU 3 %0 utilization gösteriyor ama 44 GB'ı
`an_li`'ye ait — boş *görünen* GPU boş GPU değil.

---

## 8. Recipe sadakati — satır satır

| bileşen | Microsoft `train_image.py` | FLEX-UF | aynı mı |
|---|---|---|---|
| lr programı | satır 22-32, 105 epoch | birebir kopya | ✅ |
| epoch dağılımı | 45+25+20+5+4+4+2+1 | aynı | ✅ |
| lr | 2e-4→5e-5→1e-5→2e-4→5e-5→1e-5→1e-6 | aynı | ✅ |
| crop | 256 (0-89), 512 (90+) | aynı | ✅ |
| optimizer | `AdamW(lr=1e-4)` | aynı | ✅ |
| batch | 16 | aynı | ✅ |
| grad clip | `clip_grad_norm_(0.1)`, NaN'da batch atla | aynı | ✅ |
| λ | 10→2048, 64 QP'ye log-aralıklı | aynı | ✅ |
| QP örnekleme | örnek başına uniform(0,63) | aynı | ✅ |
| veri | Open Images | Open Images subset 0 | ⚠️ subset 0,1,2 yerine 0 |
| kayıp | `λ·mse + bpp` | Eq(6-7), ağırlık-normalize | ⚠️ kasıtlı |
| PyTorch | cu130 | cu124 | ⚠️ zorunlu (driver 535) |

Üç sapmanın hepsi gerekçeli ve §0.2 / §3.3 / §5'te kayıtlı.

---

## 9. Sıfır-tolerans kontrolleri — hepsi geçti

| kontrol | sonuç | neden kritik |
|---|---|---|
| warm-start round-trip: en derin çıkış = stok UF | `max|Δ| = 0.0` | değilse merdiven UF'in yeniden ifadesi değil, "tam decode'a fark" metriği port hatasını ölçer |
| eğitilmemiş adapter'lar = identity (6 çıkışın hepsinde) | `max|Δ| = 0.0` | sıfır-init garantisi; warm-start gerçek bir başlangıç noktası mı |
| j=K hibrit yol = tam decode | `max|Δ| = 0.0` | patchify/canvas/unpatchify/head kablolaması doğru mu |
| patchify → unpatchify | `max|Δ| = 0.0` | ölçtüğümüz dikişler reshape hatası değil |

Ölçülen model boyutu: toplam **42,918,528** parametre, adapter'lar **739,200**
(%1.72).

---

## 10. ❗ Kuru koşuda yakalanan yapısal sorun — merdiven ters dönüyordu

Gerçek eğitim scriptini 48 sentetik görüntüyle 2 epoch koşturdum (`--aux_schedule
constant`, α=1). Kayıp düzgün düştü (12584 → 143), bpp düştü (15.4 → 5.2),
PSNR yükseldi. Ama:

```
psnr_per_exit: [8.095, 8.186, 8.121, 7.925, 7.512, 6.637]
                 ↑ en sığ                        ↑ en derin
spread_dB: -1.458   (negatif!)
```

**En derin çıkış en kötü.** Merdiven ters.

**Neden oluyor:** Eq(6)'da α=1 alınca tüm çıkışlar eşit ağırlıklı, yani hedef
fonksiyon "tüm çıkışların ortalama MSE'si". Bu, en derin çıkışın **en iyi olması
için hiçbir baskı üretmiyor**. Rastgele init'te her ek blok sinyali bozuyor, ve
eşit ağırlık bu sıralamayı düzeltmeye zorlamıyor.

**Neden FLEX'te bu sorun yoktu:** orada backbone donuk ve warm-start'lıydı;
en derin çıkış **tanımı gereği** gerçek DCVC-RT decoder'ıydı (bit-exact). Sıfırdan
eğitimde böyle bir garanti yok. Bu, "0'dan eğit" talebinin getirdiği yeni risk.

**Neden kritik:** en derin çıkış bizim **kalite çıpamız**. Tüm frontier
"tam decode'a göre kaç dB kaybettik" olarak ölçülüyor. Çıpa en iyi değilse
"kayıp" negatif çıkar ve ölçüm anlamsızlaşır.

**Çözüm — makalenin kendi yapısında zaten var:**
Eq (6) `L + Σᵢ αᵢLᵢ` — L (son çıkış) ağırlığı **1**, yardımcılar αᵢ. Yani
makale zaten son çıkışa ayrıcalık veriyor; α=1 alarak bunu ben düzlemiştim.

`aux_schedule=warmup` ile α, 10 epoch boyunca 0 → 1 rampalanıyor:
- epoch 0: ağırlıklar `[0,0,0,0,0,1]` → **sadece tam derinlik eğitilir**,
  model önce iyi bir codec olur, çıpa kurulur
- epoch 1-9: α = 0.1 … 0.9 → sığ çıkışlar kademeli yukarı çekilir
- epoch 10+: α = 1 → tam ortak eğitim, ama derin çıkışın önden aldığı fark korunur

**Karar: üç deneyin üçü de `--aux_schedule warmup` ile yeniden başlatıldı.**
Koşular 2 dakikalıktı, maliyeti yok; aynı hatayı 40 epoch sonra fark etmek
günlere mal olurdu.

**Not:** Bu 48 sentetik görüntüde 2 epoch'luk bir gözlem — tek başına kanıt
değil. Ama risk yapısal (eşit ağırlık ⇒ sıralama garantisi yok), düzeltme ucuz,
ve `spread_dB` her koşuda loglanıyor. `scripts/status.sh` bunu **COLLAPSE RISK**
olarak işaretliyor: spread > 0.5 dB → OK, < 0.15 dB → alarm.

---

## 11. Eğitim başladı (2026-08-15 16:40)

Veri: `/data10/shareddata/openimages/dcvc_train` — Open Images train subset 0,
kısa kenarı ≥512 filtresinden geçen **154,723 görüntü**.

**Neden min-side 512 filtresi:** `ImageFolder.__getitem__` crop'tan küçük
görüntüleri **sıfırla padliyor** (`image_dataset.py:33-48`). Sıfır padding gerçek
görüntü değil; codec'e siyah kenar rekonstrüksiyonu öğretir. Recipe 90. epoch'tan
sonra 512×512 crop'a geçtiği için, kısa kenarı 512'nin altındaki her görüntü o
aşamada padding üretirdi.

| koşu | GPU | j | tile | halo | adapter | α programı |
|---|---:|---:|---|---|---|---|
| e1_j2_p128 | 4 | 2 | 128px | 2 lat | 1×1 | warmup |
| e2_j4_p128 | 6 | 4 | 128px | 2 lat | 1×1 | warmup |
| e3_j2_p64 | 7 | 2 | 64px | 2 lat | 1×1 | warmup |

Her koşu ~12.6 GB GPU belleği kullanıyor (48 GB kartlarda rahat).
GPU 0/1/2/3/5'e dokunulmadı.

---

## 12. Boru hatlarını çalıştırınca çıkan üç sorun

Router ve değerlendirme hatlarını gerçek veriyle koşturmak — güvenmek yerine —
üç şey ortaya çıkardı. Üçü de eğitim koşarken, sonuç üretmeden önce düzeltildi.

### 12.1 Router kaybının ölçeği tutmuyordu

ClassSR'ın `w1:w2:w3 = 2000:1:6` ağırlıkları **kendi** görüntü kaybına göre
ayarlanmış — mertebesi 0.02-0.05 olan bir L1 normu. Bizimki YCbCr-0.5 üzerinde
MSE, bambaşka bir ölçek. Eğitilmemiş bir checkpoint'te ölçtüm:

```
w1·L_image  = 2000 × 4.9  ≈ 9800
w3·L_a      =    6 × 0.15 ≈    0.9
β ·L_comp   =    1 × 0.62 ≈    0.62
```

**Dört mertebe fark.** Yani Class-Loss ve Average-Loss hiçbir şey yapmıyor,
router en düşük MSE'li çıkışa çöküyor. Log bunu doğruladı:
`exit_share_hard: [32, 0, 0, 0, 0, 0]` — 32 patch'in hepsi tek çıkışta.

**Neden yakalamak önemliydi:** yakınsamış bir modelde bu dengesizlik kendiliğinden
kayboluyor (MSE ~1e-3 olunca `w1·L_image ≈ 2`, diğerleriyle kıyaslanabilir).
Yani sorun **sadece ara checkpoint'lerde** görünür — ve biz router'ları tam olarak
ara checkpoint'lerde eğiteceğiz. Sadece yakınsamada doğru olan bir ağırlıklandırma
tuzaktır.

**Çözüm:** `L_image` en derin çıkışın MSE'sine bölünüyor → boyutsuz bir **oran**
oluyor. 1.0 = "tam decode kadar iyi", 1.2 = "tam decode'dan %20 fazla hata".
Checkpoint'ler, QP'ler ve üç deney arasında kıyaslanabilir. `w_image = 50` ile
ClassSR'ın *göreli* dengesi korunuyor (görüntü terimi diğerlerinin ~50-100 katı —
onun kasıtlı tasarımı).

Düzeltme sonrası ölçüm: `w_image·l_image = 16.7` vs `β·l_comp = 15.7` — frontier
için gereken takas tam da bu.

### 12.2 Değerlendirmedeki bit-exact kontrolü sahteydi

`evaluate.py`'deki kontrol, merdiven anahtarlarını (`upsample.*`, `groups.*`,
`head.*`) stok `IntraDecoder`'a (`dec_1.*`, `dec_2.*`) `strict=False` ile
yüklüyordu. İki isim uzayı **tamamen ayrık** — yani hiçbir tensör eşleşmiyor,
stok model rastgele init'te kalıyor, ve kontrol eğitilmiş decoder'ı gürültüyle
karşılaştırıyordu.

**Neden en kötü tür hata:** ya devasa bir fark basıp paniğe yol açardı, ya da —
daha kötüsü — biri toleransı gevşeterek "düzeltir" ve kontrol hiçbir şey
karşılaştırmadan sonsuza kadar geçerdi. FLEX'in kuralı: *başarısız olamayan bir
kontrol, kontrol değildir.*

**Çözüm:** `remap_ladder_to_stock()` ters eşlemesi yazıldı; kontrol eşleme
eksikse `RuntimeError` atıyor; ve 143 tensörün tamamı üzerinde
ladder→stock→ladder **bijeksiyon testi** eklendi (`max|Δ| = 0.0`).

### 12.3 Autopilot

Kullanıcı saatlerce başında olmayacağı için üç işi döngüye alan bir süreç:
1. **Watchdog** — ölen koşuyu yeniden başlatır. `train_flexuf_image.py`
   `status_latest.pth.tar`'dan devam ettiği için maliyet en fazla o anki epoch.
2. **Sağlık** — 10 dakikada bir `autopilot.log`'a satır yazar, kimse bakmazken
   ne olduğunun kaydı kalsın diye.
3. **Frontier** — her yeni `ckpt_epo*.pth.tar` için otomatik ölçüm yapar.
   **Neden:** fikrin çalışıp çalışmadığını öğrenmek için 7 gün beklemek 7 günü
   çöpe atmak olurdu. Sweep ~15 dk, checkpoint aralığı ~5.7 saat — çekişme
   gerçek ama küçük, erken cevap çok daha değerli.

---

## 13. Bekleyen / kullanıcı kararına bırakılan

| konu | durum | gerekçe |
|---|---|---|
| `git push` | **engellendi** | otomatik mod dışa açık işlemi bloke etti. 4 commit hazır, `origin` = `canerim/FLEX-FLOP` |
| 46 GB `train_0.tar.gz` | **saklandı, silinmedi** | silmek geri alınamaz ve yeniden indirme gerektirir — kullanıcı büyük indirmeyi açıkça yasakladı. 4.2 TB boş diskte %1, riski sıfır. Çıkarılmış veri (`dcvc_train/`) eğitimde kullanılıyor |
| Open Images subset 1, 2 | **alınmadı** | recipe 0,1,2 diyor; elimizde 0 var (154,723 kullanılabilir görüntü). `/mnt/data_local/datasets` bu makinede yok — doğru makine öğrenilince oradan alınabilir |

---

## 14. β aralığı eğitim aşamasıyla ölçeklenmeli — ilk gerçek checkpoint'te bulundu

Zinciri e3'ün `ckpt_epo0.pth.tar`'ında doğrularken router **tamamen çöktü**:
`exit_share_hard: [0, 0, 0, 0, 0, 256]`, kazanç %0.

Terimler:
```
w_image·l_image = 50 × 1.226 = 61.3
β      ·l_comp  = 25 × 0.989 = 24.7      ← kalite 2.5× ağır basıyor
w_avg  ·l_a     =  6 × 1.569 =  9.4      ← itiyor ama yetmiyor
```

**Bu bir hata değil.** Epoch 0'da çıkışlar arasında 8.5 dB fark var; sığ çıkışın
MSE'si en derinin ~7 katı. Sığ yönlendirmek gerçekten o kadar kötü ve router
doğru karar veriyor.

**Sorun benim β aralığımdaydı.** β'nın karşılaştırıldığı büyüklük
`w_image × (sığ çıkış ne kadar kötü)` ve bu ikinci çarpan **eğitim boyunca
değişiyor**:
- Eğitim başı: fark büyük (Δoran ~3-7) → anlamlı takas için β **yüzlerde**
- Eğitim sonu: çıkışlar yakınsıyor, Δoran küçülüyor → ilginç bölge **küçük β**'da

Sabit `0/25/100` aralığı erken checkpoint'lerde her noktada aynı önemsiz sonucu
(%0 kazanç) verirdi — yani hiçbir şey ölçmezdi.

**Düzeltme:** dört mertebeye yayılan log-aralıklı süpürme: `0, 25, 100, 400,
1600, 6400`. Her iki rejimi de kapsıyor. Eski aralıkla başlamış olan süpürme
öldürüldü (GPU'yu boşa yakmasın), marker atılmadığı için autopilot yeni
aralıkla tekrarlayacak.

---

## 15. Teşhis eksenini yanlış kurmuşum — ve altından ilk gerçek sonuç çıktı

`status.sh` daralan spread'i "COLLAPSE RISK" diye işaretliyordu. Epoch 1'de
e3'te olan şey:

```
ep0 step9600  spread +7.601   deepest 25.84   shallowest 18.24
ep1 step0     spread +7.059   deepest 25.16   shallowest 18.10
ep1 step200   spread +1.368   deepest 27.47   shallowest 26.10   ← +8 dB, 200 adımda
ep1 step400   spread +1.333   deepest 27.59   shallowest 26.26
```

Daralma **çöküş değil**: en derin çıkış düşmedi, **sığ çıkış fırladı**.

**Neden:** epoch 0 boyunca warmup α=0 olduğu için sığ çıkışların adapter'ları
hiç gradyan almadı — sıfır-init'te, yani identity'de durdular. Epoch 1'de α=0.1
olunca eğitilmeye başladılar, ve gövde zaten iyi olduğu için çok hızlı yakaladılar.

**Teşhisin hatası:** spread tek başına yanlış eksen. Daralan bir aralık iki zıt
şey anlamına gelebilir:
- tüm çıkışlar aynı **vasat** kaliteye yakınsıyor → FLEX'in sıfırdan eğitim
  çöküşü, yönlendirilecek bir şey yok, çıpa da bozuk;
- sığ çıkışlar **iyi** bir en-derin çıkışı yakalıyor → projenin tam istediği şey.

İkisini ayıran tek şey en derin çıkışın **mutlak** kalitesi. Teşhis artık her iki
eksene birden bakıyor.

### İlk gerçek sinyal (epoch 1 — erken, kesin değil)

| koşu | sığ | derin | fark | exit 0'ın kazancı |
|---|---:|---:|---:|---:|
| e1 (j=2, 128px) | 24.82 | 25.69 | **0.87 dB** | %58.7 |
| e2 (j=4, 128px) | 24.58 | 25.46 | **0.88 dB** | %28.9 |
| e3 (j=2, 64px) | 25.93 | 26.48 | **0.55 dB** | %58.7 |

**Ne söylüyor:** bir patch 12 bloktan sadece 2'sinden sonra çıkıp ~0.9 dB
kaybediyor, ve bu decoder hesabının %58.7'sini kurtarıyor.

**Ne söylemiyor — abartmamak için:**
1. 105 epoch'un **1'indeyiz**. Mutlak PSNR (~25-26 dB) hâlâ düşük. Model
   olgunlaştıkça en derin çıkışın önde açması beklenir (fazla kapasite = daha
   yüksek tavan), yani fark **büyüyebilir**.
2. Bu sayılar **tam-kare, tek-derinlik** ölçümleri. Patch'e bölmenin dikiş
   cezası bunlara **dahil değil**. Gerçek frontier, dikişli yönlendirilmiş
   decode ile ölçülecek — `evaluate.py`'nin işi bu.
3. Yönlendirme henüz devrede değil. Eğer tüm çıkışlar eşit iyi kalırsa router'ın
   yapacak bir şeyi olmaz — o zaman katkı "yönlendirme" değil "eğitilmiş
   adapter'lı erken çıkış" olur. Bu da güçlü bir sonuç, ama farklı bir iddia.

---

## 14. İlk ölçülen sonuçlar (epoch 0 checkpoint)

### 14.1 Saf patch'e-bölme maliyeti izole edildi — 0.143 dB

Frontier sweep'in **β=0** kolu beklendiği gibi her patch'i en derin çıkışa
yolladı (%0 tasarruf). Ama PSNR kaybı sıfır değil: **+0.143 dB**.

Bu fark tasarruftan bağımsız — tamamen **j=2 split'in dikiş cezası**. Yani
patch'lere bölmenin, hiçbir şey kazanmadan, ödediğimiz sabit bedeli.

| | 64px tile'da ceza |
|---|---:|
| FLEX ölçümü, iyileştirmesiz (halo yok, per-patch head) | **2.484 dB** |
| FLEX, 1px feature halo + full-frame head | ~0.054 dB |
| **FLEX-UF, 2 latent px halo + full-frame head** | **0.143 dB** |

Dikişin **%94'ü** siliniyor. §4.3'teki karar — halo'yu trunk'a değil router'a
(bedava) ve head'e (ucuz) koymak — burada karşılığını veriyor: FLOP kazancı
duruyor *ve* dikiş neredeyse yok.

### 14.2 Referans eğrisi (e3, j=2, 64px, ckpt_epo0)

Kontrol: en derin çıkış vs stok UF `max|Δ| = 0.0` ✅ — sayılar güvenilir.

| çıkış | tasarruf | PSNR | tam decode'a fark |
|---:|---:|---:|---:|
| 5 | %0.0 | 28.43 | — |
| 4 | %14.0 | 24.35 | −4.09 |
| 3 | %28.9 | 21.81 | −6.62 |
| 2 | %43.8 | 20.18 | −8.25 |
| 1 | %58.7 | 18.98 | −9.45 |
| 0 | %73.6 | 18.07 | −10.36 |

Epoch 0'da merdiven çok ayrışık (10.4 dB aralık) — çünkü α=0 idi, sığ çıkışlar
hiç eğitilmemişti. Epoch 1'de α=0.1 ile aralık 0.7 dB'ye indi. Yani sonraki
checkpoint'lerde erken çıkışlar **çok daha ucuza** gelecek; bu eğri iyileşecek.

### 14.3 Baseline eklendi — çünkü frontier tek başına yanıltıcı olabilir

**Sorun:** frontier "bizim en derin çıkışımıza göre kaç dB kaybettik" ölçüyor.
Ortak eğitim (Eq 6, α>0) en derin çıkışın kalitesini düşürüyorsa, frontier harika
görünür ama mutlak kalite kötü olur. Klasik tuzak.

**Çözüm:** `--num_exits 1 --split_depth 1` ile aynı kod yolundan bir baseline.
K=1'de model tam olarak stok `IntraDecoder`, kayıp tam olarak `λ·mse + bpp`.
Ölçülen parametre sayısı **42,179,328 / 0 adapter** — en başta ölçtüğüm stok
`DMCI` sayısıyla birebir aynı. Yani tek değişken izole: çok-çıkışlı hedef.

(Microsoft'un kendi `train_image.py`'ı Python 3.12 f-string sözdizimi kullandığı
için 3.10'da koşmuyor; kendi scriptim K=1 ile aynı işi görüyor ve daha temiz bir
ablation veriyor.)

**Epoch 0 karşılaştırması** (eşleşmiş adımlar):

| adım | baseline K=1 | e1 | e2 | e3 |
|---:|---:|---:|---:|---:|
| 200 | 16.35 | 16.29 | 14.88 | 16.49 |
| 400 | 16.72 | 18.19 | 19.82 | 18.01 |
| 600 | 19.42 | 19.06 | 19.31 | 18.95 |
| 800 | 20.98 | 19.50 | 22.55 | 19.59 |

Gürültü içinde aynı — ve bu **beklenen**: epoch 0'da α=0, yani çok-çıkışlı
koşular da sadece en derin çıkışı eğitiyor. Eşleşmeleri warmup'ın doğru
çalıştığının kanıtı, henüz ortak eğitimin maliyeti hakkında bir şey söylemiyor.
Asıl test epoch 1-5.

### 14.4 Teşhis düzeltildi — spread tek başına yanlış gösterge

`status.sh` başlangıçta daralan spread'i "COLLAPSE RISK" diye işaretliyordu.
Bu yanlış: daralan aralık **iki zıt** şey olabilir.
- tüm çıkışlar aynı **vasat** kaliteye çöküyor → FLEX'in sıfırdan eğitim
  başarısızlığı, yönlendirilecek bir şey yok, çıpa da kötü
- sığ çıkışlar **iyi** bir en-derin-çıkışa yetişiyor → projenin tam istediği
  sonuç, çünkü o zaman bir patch 12 bloktan 4'ünü çalıştırıp neredeyse hiç dB
  kaybetmiyor

Ayıran tek şey **çıpanın mutlak kalitesi**. Teşhis artık ikisine birden bakıyor
ve baseline'la kıyaslıyor.

---

## 15. Oracle teşhisi — "router bozuk" sanılan şeyin gerçek sebebi

### 15.1 Belirti

Frontier sweep her β'da 6144 patch'in **tamamını tek çıkışa** yolladı; β sadece
hangi çıkış olduğunu değiştirdi. Sabit bir fonksiyon — yönlendirme değil.
ClassSR'ın tüm önermesi patch'lerin farklı zorlukta olması; router ayrım
yapmıyorsa düz bir sığ decoder da aynı işi görürdü.

### 15.2 Doğru teşhis: oracle'a bak, router'ı suçlama

`scripts/oracle_diagnostic.py` üç soruyu sırayla soruyor:
1. **Oracle değişiyor mu?** (patch başına, tam decode'a τ dB içinde kalan en ucuz
   çıkış) — sabitse içerikte uyarlanacak bir şey yok, hiçbir router yardım edemez.
2. **Headroom ne kadar değerli?** oracle vs **aynı kalitedeki** en iyi tek derinlik.
3. **Sinyaller görebiliyor mu?** her sinyalin oracle seçimiyle korelasyonu.

**ckpt_epo0 sonucu:**
```
çıkış 4 (en sığ alternatif):  +5.198 dB ± 2.014
oracle, τ ≤ 1.0 dB:            patch'lerin %100'ü çıkış 5
headroom:                       +0.0 puan
```
Hiçbir dB bütçesi 5.2 dB'yi karşılamıyor → **oracle'ın kendisi sabit**. Router
suçsuz. Sinyaller de suçsuz — gerçek varyans taşıyorlar (s1 std=243, s4 std=18);
sabit olan **hedef**.

**Sebep beklenen ve geçici:** epoch 0'da warmup α'yı 0'da tutuyor, sığ çıkışlar
hiç gradyan almıyor, adapter'ları sıfır-init'te duruyor.

**Maliyeti:** o checkpoint'te frontier ölçmek ~1 saat GPU'ya mal oldu ve
sıfırlardan oluşan bir tablo üretti. Artık sweep bu 1 dakikalık teşhise bağlı.

### 15.3 ❗ Karşılaştırma hatam — ve düzeltmesi

İlk "HEADROOM" tablosu 0.5 dB'de routing'i uniform'dan **kötü** (−2.5pp)
gösteriyordu. Bu **matematiksel olarak imkânsız**: oracle en kötü ihtimalle
uniform'u taklit edebilir.

**Hata elma-armut karşılaştırmasıydı:**
- uniform tarafı: "**ortalama** dB ≤ τ" kısıtı
- oracle tarafı: "**her patch** ≤ τ" kısıtı

Oracle daha zor bir problem çözüyordu, o yüzden daha az tasarruf ediyor
görünüyordu.

**Düzeltme:** τ süpürülerek oracle'ın (**ulaşılan** ortalama dB, tasarruf) eğrisi
çiziliyor; uniform eğrisi K ayrık noktadan (mean_db[k], 1−cost[k]) lineer
interpolasyonla **aynı ulaşılan dB'de** okunuyor. Aradaki fark routing'in
gerçekten ne kazandırdığı.

**Neden bu proje için en kritik ölçüm:** "routing düz sığ bir decoder'ı yeniyor
mu" sorusunun tek dürüst cevabı bu. Yenmiyorsa tüm mekanizma gereksiz karmaşa.

### 15.4 İlk olumlu sonuç (e1, epoch 1, j=2, 128px)

Patch başına dB cezası:

| çıkış | tasarruf | ceza |
|---:|---:|---:|
| 0 | %73.6 | +0.705 ± 0.402 |
| 1 | %58.7 | +0.435 ± 0.327 |
| 2 | %43.8 | +0.313 ± 0.256 |
| 3 | %28.9 | +0.167 ± 0.137 |
| 4 | %14.0 | +0.085 ± 0.065 |

**Eşit kalitede routing vs uniform:**

| ulaşılan dB | oracle | uniform | kazanç |
|---:|---:|---:|---:|
| 0.039 | %18.7 | %6.2 | **+12.6 puan** |
| 0.121 | %33.4 | %19.8 | **+13.6 puan** |
| 0.195 | %42.6 | %31.1 | **+11.5 puan** |
| 0.342 | %55.5 | %46.2 | **+9.3 puan** |
| 0.582 | %69.1 | %66.6 | +2.4 puan |
| 0.701 | %73.3 | %73.2 | +0.1 puan |

Patch başına yönlendirme aynı kalitede hiçbir tek derinliğin ulaşamadığı
tasarrufu veriyor, 0.12 dB civarında **+13.6 puan** tepe yapıyor. Ve bu 105
epoch'un **birincisinde**.

### 15.5 ❗ Açık sorun: sinyaller oracle'ı göremiyor

τ=0.5'te sinyal–oracle korelasyonları:

| sinyal | r |
|---|---:|
| s1 rate-surrogate | +0.095 |
| s2 sparsity | −0.084 |
| s3 gradient | +0.006 |
| s4 spatial-var | +0.080 |

Hepsi ~0.1'in altında. **Headroom gerçek ama router onu göremiyor.** Oracle bir
üst sınır; router bu sinyallerle o sınıra yaklaşamaz.

Bu sıradaki iş. Muhtemel yönler (henüz denenmedi, ölçülecek):
- sinyaller latent'ten hesaplanıyor ama oracle **rekonstrüksiyon** hatasına
  bakıyor; aradaki bağ dolaylı olabilir
- öğrenilmiş bir sinyal çıkarıcı (birkaç bin parametrelik küçük conv) latent'ten
  doğrudan zorluk kestirebilir — router bütçesi decode'un %0.009'u, yani yer var
- oracle'ı doğrudan taklit eden bir denetimli kayıp (FLEX'in yaptığı), ClassSR'ın
  dolaylı yumuşak-karışım gradyanı yerine

---

## 16. Değerlendirme eğitim verisi üzerindeydi — held-out ayrım kuruldu

`evaluate.py` `description.json`'ın ilk N görüntüsünü alıyordu; o liste eğitim
listesiydi. Yani her rapor edilecek PSNR modelin zaten oturttuğu veriden
ölçülecekti. Bir codec için bu bir sınıflandırıcı kadar yıkıcı değil
(ezberlenecek etiket yok), ama **"X dB'de %Y hesap tasarrufu" bir genelleme
iddiası** ve eğitim verisinden ölçülen bir sayı onu taşıyamaz. FLEX de tam bu
yüzden sabit 192 kareyi ayırıp asla listeler arası kıyas yapmamıştı.

`prepare_openimages.py` artık deterministik bir dilim ayırıyor (sıralı listenin
her 400.'sü, 512 görüntü) ve **eğitim listesinden çıkarıyor**. Deterministik
olması şart: üç deney ve baseline aynı seti görsün, ve subset 1-2 eklenince set
değişmesin diye. Rastgele bölme bu kıyaslanabilirliği bozardı.
Doğrulandı: eğitim 269,478 / held-out 512 / **çakışma 0**.

### 16.1 Yan bulgu: ilk dataset listesi eksikmiş

Yeniden tarama **273,228** dosya buldu, ilk tarama **156,541**. İlk
`description.json`, extraction tamamlanmadan yazılmış. Yani üç koşu da mevcut
verinin **%57'siyle** eğitiliyormuş.

Diskteki gerçek durum doğrulandı (273,228 jpg, 82 GB açılmış, 45.9 GiB arşivden).
Koşular durdurulup checkpoint'lerinden yeniden başlatıldı; artık 269,478 görüntü
görüyorlar. Maliyet: kısmi epoch kaybı (~3 saat / 7 gün). Daha fazla veriyle ve
temiz bir genelleme iddiasıyla devam etmek buna değer.

---

## 17. ❗ Asıl soru soruldu: yönlendirilecek bir şey var mı?

Frontier süpürmesi **her patch'i aynı çıkışa yollayan** router'lar üretiyordu —
β değişince sadece hangi çıkış olduğu değişiyordu. Bu yönlendirme değil, **sabit
fonksiyon**. Ve projenin dayandığı önermeyi geçersiz kılar: ClassSR çalışıyor
çünkü alt-görüntüler zorlukta farklılaşıyor.

`scripts/oracle_diagnostic.py` bunu üç soruyla sınıyor: (1) oracle'ın kendisi
değişiyor mu, (2) headroom kaç puan, (3) sinyaller oracle'ı görebiliyor mu.

**e3'ün epoch-0 checkpoint'inde sonuç (2048 patch, qp63):**

| exit | patch başına ceza | kazanç |
|---:|---:|---:|
| 0 | **+12.088 dB** ± 2.10 | %73.6 |
| 2 | +10.081 dB ± 1.95 | %43.8 |
| 4 | +5.799 dB ± 1.82 | %14.0 |
| 5 | 0.000 | %0 |

Oracle her τ'da **sabit**: patch'lerin %100'ü en derin çıkışta. Headroom her
τ'da **+0.0pp**. Sinyaller oracle ile korelasyonsuz — çünkü korelasyon
kurulacak bir varyasyon yok.

**Verdict: bu checkpoint'te headroom yok** — ve teşhis nedenini de söylüyor:
epoch 0'da α=0 olduğu için sığ çıkışlar hiç eğitilmedi, adapter'ları sıfır-init'te
durdu, özellikleri head için hiç şekillenmedi. α rampalandıktan sonra tekrar
bakılmalı. Eğitim logları bunu destekliyor: epoch 1-2'de sığ çıkış 0.8 dB'ye
kadar yaklaştı.

**Autopilot'a kapı eklendi:** önce ucuz teşhis (tek geçiş), sadece
"headroom exists" çıkarsa pahalı süpürme (7 router × 800 adım). Oracle sabitken
süpürme yedi kez aynı önemsiz noktayı rapor ederdi — GPU'yu boşa yakar.

---

## 18. Tek çıkışlı baseline eklendi

`runs/baseline_singleexit` — K=1, adapter yok, **42,179,328 parametre** = birebir
stok DMCI, aynı recipe, aynı veri, aynı tohum düzeni.

**Neden gerekli:** "sığ çıkış tam decode'un 0.9 dB'si içinde" demek yetmiyor;
o *tam decode* düz UF kadar iyi mi? Çok çıkışlı eğitim çıpaya bir bedel
ödetiyorsa, tüm frontier düşmüş bir referansa göre çizilmiş olur. Baseline bu
soruyu cevaplayan tek şey.

---

## 16. ❗ Kendi hatam: çalışan bir bash script'ini düzenlemek

### 16.1 Ne yaptım

`add_subsets.sh` çalışırken (subset 2'yi indirip extract ederken) dosyayı
düzenledim — baseline'ın restart sonrası yeniden başlatılmasını eklemek için.

### 16.2 Neden bozdu

**bash script'leri artımlı okur, bayt konumuyla.** Yorumlayıcı dosyada bir
offset tutar ve komut çalıştıkça ilerler. Dosyayı yerinde düzenlemek, o offset'i
yeni dosyada **bambaşka bir yere** denk getirir; bash oradan devam eder.

Sonuç: script restart bölümüne erken sıçradı. `description.json`'ı **kısmi**
veri setiyle (subset 0+1, 269,478) yeniden kurdu ve dört koşuyu da restart etti,
oysa train_2 hâlâ extract oluyordu.

### 16.3 İkinci hasar: yarış durumu

`add_subsets.sh` restart yaparken tüm koşuları `pkill` ediyor. Autopilot'un
watchdog'u tam o boşlukta liveness kontrolü yaptı, koşuları ölü gördü ve
**kendi kopyalarını** başlattı. Sonuç: baseline ve e3 için **ikişer ana
process**, aynı `--save_dir`'e, aynı `status_latest.pth.tar`'a yazıyor.

Bozuk bir checkpoint kendini duyurmaz. Bu sessizce ilerleyip günler sonra
anlamsız sonuçlar olarak ortaya çıkabilirdi.

### 16.4 Maliyet

- Dört koşunun yarım kalan epoch'ları (~1 saat, 4 GPU)
- Baseline **sıfırdan** başladı: epoch 0'ı hiç bitirmemişti, dolayısıyla
  `status_latest.pth.tar`'ı yoktu — devam edecek bir şey yok. ~1.5 saat kayıp.
- Kaybedilmeyen: kod, veri, tamamlanmış checkpoint'ler.

### 16.5 Düzeltmeler

1. **Kopyalar öldürüldü.** Her koşu için tek ana process doğrulandı — parent'ı
   başka bir eğitim process'i olmayanları sayarak (dataloader worker'ları aynı
   komut satırını paylaştığı için naif `pgrep -c` yanıltıyor).

2. **`flock` koruması.** Autopilot artık yeniden başlatma bölümünü özel bir
   kilitle sarıyor, ve kilidi aldıktan **sonra** bir kez daha canlılık kontrolü
   yapıyor. İki başlatıcı arasındaki pencere kapandı.

3. **`main () { ... }; main "$@"` kalıbı.** Autopilot'un gövdesi artık bir
   fonksiyonda. Bash tek bir komut çalıştırmadan önce dosyanın **tamamını**
   ayrıştırmak zorunda, dolayısıyla bir düzenleme asla yarı yolda etkili olamaz.
   *(Gövde bilinçli olarak girintilenmedi: heredoc sonlandırıcıları sütun 0'da
   olmak zorunda — ilk denemede tam bu yüzden script bozuldu.)*

4. **`resume_autopilot.sh`.** Çalışan `add_subsets.sh` düzenlenemeyeceği için —
   sorunu yaratan şey tam olarak buydu — autopilot o bir tur boyunca bekletildi.
   Desen **sabitlenmiş** (`^bash /path/add_subsets.sh`): sabitlenmemiş
   `pgrep -f "bash .*add_subsets.sh"` bu durumu incelemek için kullanılan
   kabuk komutlarını da eşleştirip sonsuza kadar beklerdi.

### 16.6 Sonuç — nihai durum doğru

`add_subsets.sh` fazladan bir restart turuna rağmen doğru bitirdi:

```
description.json  379,614 eğitim görüntüsü   (subset 0+1+2 — recipe'nin tam spec'i)
description_val.json  512 ayrık doğrulama
diskte              384,795 jpg
train_*.tar.gz      doğrulanmış extraction sonrası silindi
dört koşu da        son epoch'larından devam etti
```

**Alınan ders:** çalışan bir script asla yerinde düzenlenmez. Düzenlenecekse ya
önce durdurulur, ya yeni bir dosyaya yazılıp atomik olarak `mv` edilir, ya da
gövdesi baştan bir fonksiyona sarılır.

---

## 17. Router'ın körlüğü çözüldü — sinyaller gövdeden okunuyor

### 17.1 Sorun (§15.5'ten)

Headroom gerçekti (eşit kalitede +13.6 puan) ama router'ın dört elle yapılmış
sinyali oracle ile |r| < 0.1 korele idi. Girdileri göremediği bir şeyi router
sömüremez.

### 17.2 Aday arama — `scripts/signal_search.py`

Elle sinyal uydurmak yerine **codec'in kendi makinesinden** okumayı denedim.
Ölçüm (e1, epoch 1, τ=0.3 dB, 384 tile, oracle seçimine karşı):

| sinyal | r | Spearman | maliyet |
|---|---:|---:|---|
| **stem_max** | **+0.440** | +0.412 | bedava |
| scales_max | +0.336 | +0.333 | bedava |
| stem_energy | +0.319 | +0.314 | bedava |
| stem_std | +0.317 | +0.316 | bedava |
| y_energy | +0.310 | +0.300 | bedava |
| s1 rate-surrogate (eski) | +0.206 | +0.238 | — |
| s3 gradient (eski) | +0.111 | +0.107 | — |
| s2 sparsity (eski) | −0.103 | +0.101 | — |

### 17.3 Neden gövde sinyalleri daha iyi — ve neden bedava

**Daha iyi, çünkü yapısal:** oracle "decoder bu tile'a ne yapıyor" sorusudur.
Gövde çıktısı decoder'ın **kendi ara durumu**; ham latent ise ondan bir dönüşüm
uzakta. Bir soruyu, cevabına bir adım daha yakın bir yerden sormak.

**Bedava, çünkü zaten hesaplanıyor:** j-split altında açılış upsample'ı ve grup
0..j−1 **her tile için tam kare** çalışıyor — çıkış nerede olursa olsun.
Yönlendirme kararının verilmesi gereken anda gövde zaten bellekte. `scales_hat`
de entropi decode'unun çıktısı, o da her koşulda hesaplanıyor. Router yolu
%0.009'da kalıyor.

### 17.4 Sonuç — router ilk kez yönlendiriyor

200 adımlık router, e1'in epoch-1 checkpoint'i, β=25:

```
önce (latent sinyalleri):  exit_share [0, 0, 0, 0, 0, 6144]   ← sabit fonksiyon
sonra (gövde sinyalleri):  exit_share [13, 16, 9, 2, 12, 12]  ← altı çıkış da kullanımda
                           tasarruf %36.28   kayıp 0.1625 dB
```

Aynı kalitede uniform decoder ~%20 veriyor (§15.4 tablosu). Router artık
headroom'un anlamlı bir kısmını gerçekten alıyor.

**Uyarı:** 200 adımlık bir router ve hareket halindeki bir checkpoint — ön
sonuç. Ama mekanizma artık çalışıyor; önceki durumda hiçbir β değeri
yönlendirme üretmiyordu.

---

## 18. İlk gerçek frontier noktası — router oracle'a ulaştı

Aynı ağırlıklar (`probe_ep2`, e1'in epoch-2 anlık kopyası), aynı 512 görüntülük
**ayrık** doğrulama seti, qp=63:

| | tasarruf | kayıp |
|---|---:|---:|
| **Router (ölçülen, β=0)** | **%42.65** | **0.1894 dB** |
| Oracle (üst sınır, τ=0.30) | %42.5 | 0.196 dB |
| En iyi tek derinlik (uniform) | %31.9 | 0.196 dB |

`exit_share = [312, 622, 171, 110, 1, 320]` — altı çıkışın hepsi kullanımda.
Kontrol: en derin çıkış vs stok UF `max|Δ| = 0.0`.

**Router oracle'a ulaşmış durumda** — mevcut headroom'un neredeyse tamamını
alıyor ve aynı kalitede hiçbir tek derinliğin veremediği **+10.7 puan** fazla
tasarruf sağlıyor.

### 18.1 β=0'da neden tasarruf var?

Karmaşıklık terimi kapalıyken (β=0) tasarruf **ClassSR'ın Eq (4)
Average-Loss'undan** geliyor: `w_avg=6` ile dengeli dağılım baskısı, tek başına
router'ı tüm çıkışları kullanmaya zorluyor. Yani ClassSR'ın "dejenere çözümü
engelle" terimi burada aynı zamanda **tasarrufun kaynağı**.

β büyüdükçe eğrinin daha derin tarafına gidilecek; süpürme devam ediyor.

### 18.2 Sayının güvenilirliği

| önlem | durum |
|---|---|
| ayrık doğrulama seti (512 görüntü, eğitimden çıkarılmış) | ✅ |
| bit-exact çıpa kontrolü (`max|Δ| = 0.0`) | ✅ |
| paylaşılan gövde amortize edilmiş | ✅ |
| adapter maliyeti ücretlendirilmiş | ✅ |
| halo maliyeti ücretlendirilmiş | ✅ |
| eşleşmiş kalitede uniform karşılaştırması | ✅ |

### 18.3 Sınırlar — açıkça

- **105 epoch'un 2.'si.** Model olgunlaşmadı; sayılar değişecek.
- **Tek QP (63).** Çok-oranlı süpürme yapılmadı.
- **600 adımlık router.** Daha uzun eğitim daha iyi olabilir.
- **Tek deney (e1).** e2/e3 ile karşılaştırma (j ve tile boyutu eksenleri) henüz yok.
- **Baseline karşılaştırması eksik.** Ortak eğitimin çıpaya maliyeti hâlâ ölçülmedi
  (baseline epoch 0'da). Bu olmadan "0.19 dB kayıp" mutlak değil, göreli bir sayı.

---

## 19. ❗ Muhasebe hatası — raporladığım tasarruf sayıları şişikti

### 19.1 İmkânsızlığı kovalamak

Frontier probe'da router 0.278 dB'de %56.36 verdi. Ama Lagrange-optimal üst
sınır aynı kalitede daha azını gösteriyordu. **Router üst sınırı aşamaz** —
demek ki bir taraf yanlıştı.

İki hata birden çıktı.

### 19.2 Hata 1: yanlış üst sınır

İlk "oracle"ım τ-eşikliydi: *her patch τ dB içinde kalsın*. Bu bir **kısıt
sağlayıcı**, (ortalama dB, tasarruf) düzleminin Pareto sınırı değil — bir
patch'in biraz daha ileri gitmesi başka yerde çok hesap kazandıracaksa bile
reddediyor.

**Gerçek sınır:** her patch için bağımsız olarak `mse_k + λ·cost_k`'yı
minimize eden k'yı seç, λ'yı süpür. Patch'ler bağımsız ve maliyet toplanabilir
olduğu için bu süpürme **tam Pareto sınırını** çiziyor. Eklendi.

### 19.3 Hata 2 — asıl olan: maliyet modeli decoder'la uyuşmuyordu

`forward()` içinde `exit_map.clamp(min=j)` var: j=2'de bir patch **en erken
grup 2'yi çalıştırdıktan sonra** çıkabiliyor.

Ama maliyet modeli `k_eff = max(k, j-1)` kullanıyordu — çıkış 0 veya 1 atanan
patch'i "split'te çıkmış" sayıp **grup j'nin maliyetini hiç yazmıyordu**.

Router β=100'de tüm patch'lere çıkış 0 verdi. Fatura: %58.70 tasarruf.
Gerçek: %43.79. **Hiç çalıştırılmayan bir decode için indirim yazılmış.**

Düzeltme: maliyet modeli decoder'ın clamp'ini birebir uyguluyor
(`k_run = max(k, j)`).

### 19.4 Düzeltilmiş frontier (e1, epoch 2)

| β | raporlanmış | **doğru** | kayıp |
|---:|---:|---:|---:|
| 0 | %42.65 | **%33.58** | 0.1894 dB |
| 10 | %45.84 | **%37.55** | 0.2240 dB |
| 30 | %56.36 | **%42.24** | 0.2777 dB |
| 100 | %58.70 | **%43.79** | 0.2893 dB |

j=2'de gerçek tavan **%43.79**.

**Ders:** maliyet modeli ile çalıştırılan kod ayrı yerlerde yaşıyorsa
ayrışırlar, ve ayrıştıklarında hata **her zaman** iyimser yönde olur — çünkü
iyimser sayı sorgulanmaz. Kontroller mimari eşdeğerliği doğruluyordu
(`max|Δ|=0`) ama *maliyet* eşdeğerliğini doğrulayan bir kontrol yoktu.

### 19.5 Ortaya çıkan tasarım kusuru: ulaşılamaz çıkışlar

j=2'de çıkış 0, 1, 2 **özdeş** — üçü de grup 2'ye kadar çalışıyor. Router'ın 6
çıktısı vardı ama 3'ü aynı şeyi ifade ediyordu, ve ClassSR'ın Average-Loss'u
(üniform dağılıma iten terim) kütlenin üçte birini **var olmayan ayrımlara**
harcıyordu.

Düzeltme: `min_exit=j` ile split'in altındaki logit'ler maskelendi. Denge terimi
artık gerçekten farklı olan çıkışlar üzerinde çalışıyor.

---

## 20. ❗❗ İki hata daha — ve routing'in şu an kaybettiği bulgusu

### 20.1 Kontrolün kendisi eksikti

§19'daki dersi kalıcılaştırmak için `tests/test_cost_matches_reality.py` yazıldı:
gerçek decode'u koştur, hook'la gerçek MAC say, maliyet modelinin tahminiyle
karşılaştır. Anında ateşledi ve **çok daha büyük** bir hata buldu.

### 20.2 Hata: trunk halo'su koda hiç uygulanmamıştı — ama koddaydı

`forward()` içinde `patchify_with_halo(feat, Fp, halo)` vardı, yani per-tile
trunk blokları 16px yerine **24px** tile'larda çalışıyordu: (24/16)² = **2.25×**.

Ölçüm: `j=2, hepsi en derin çıkışta → tam decode'un 1.745 katı`.
Yani hiçbir şey kazanmadan %74 fazla ödüyorduk.

Bu tam olarak §4.3'te ölçüp "asla yapmayacağız" dediğim şey. Kararı doğru
vermiştim — halo router'a (bedava) ve head'e (ucuz), trunk'a değil — ama
**decoder'a uygulamamıştım**. Maliyet modeli `halo_scope="head"` varsayıyordu,
kod `trunk` yapıyordu. İki taraf sessizce ayrışmıştı.

**Düzeltme:** `trunk_halo` ayrı bir knob oldu, varsayılan **0**. Router halo'su
(`latent_halo=2`) dokunulmadı — o gerçekten bedava. Dikiş full-frame head ile
hallediliyor.

Düzeltme sonrası kontrol: maliyet modeli gerçekle **%0.13 içinde** uyuşuyor,
10 senaryonun hepsinde. `j=2 all deepest: ölçülen 1.000, model 1.000`.

### 20.3 ❗ Asıl bulgu: routing şu an uniform'u YENMİYOR

Trunk halo'su kaldırılınca PSNR gerçek değerine oturdu. Aynı decode yolundan,
aynı metrikle, 96 ayrık doğrulama görüntüsünde (e1, epoch 2):

| konfigürasyon | tasarruf | kayıp |
|---|---:|---:|
| **uniform k=2** | **%43.79** | **0.1787 dB** |
| uniform k=3 | %28.88 | 0.2137 dB |
| uniform k=4 | %13.98 | 0.1006 dB |
| uniform k=5 | %0.00 | 0.0432 dB |
| routed β=0 | %32.77 | 0.2809 dB |
| routed β=30 | %39.06 | 0.3257 dB |

**Düz sığ decoder her iki eksende de kazanıyor.** Router'ın kattığı değer negatif.

**Neden önceki analiz tersini söylüyordu:** oracle teşhisi `forward_all_exits`
kullanıyordu — tam kare decode, **dikiş yok**. Gerçek yol per-tile çalışıyor ve
dikiş cezası ödüyor. O ceza hangi çıkışı seçtiğinden **bağımsız**, yani tüm
çıkışlara aynı sabit maliyeti ekliyor. Çıkışlar birbirine 0.3 dB içinde sıkışmış
durumdayken (epoch 2, spread +0.3..+0.7) yönlendirilecek fark kalmıyor ve en
ucuzunu her yerde almak daha iyi oluyor.

Bu tam olarak oracle teşhisine koyduğum kontrolün amacıydı:
*"routing uniform'u yenmiyorsa dürüst sonuç, düz daha sığ bir decoder yeterdi."*
Düzgün ölçünce yenmiyor.

**Sınır:** epoch 2/105. Çıkışlar ayrıştıkça değişebilir — ama şu anki veriyle
routing'in değer kattığı iddia edilemez.

### 20.4 Şu ana kadar üç kez düzeltilen manşet sayı

| aşama | iddia | neden yanlıştı |
|---|---|---|
| ilk | %42.65 @ 0.189 dB | maliyet modeli grup j'yi faturalamıyordu |
| clamp düzeltmesi | %33.58 @ 0.189 dB | PSNR trunk halo'suyla ölçülmüştü |
| **halo düzeltmesi** | **%32.77 @ 0.281 dB** | — |
| **doğru kıyas** | **uniform k=2 daha iyi: %43.79 @ 0.179 dB** | — |

Üçünde de hata **iyimser** yöndeydi. İyimser sayı sorgulanmıyor; kontrol
yazılana kadar hiçbiri yakalanmadı.

---

## 21. Ana DCVC-UF'e göre kıyas — engelli

Kullanıcı haklı olarak "kıyas ana DCVC-UF'e göre olmalı" dedi. Bunun için
Microsoft'un yayınladığı checkpoint gerekiyor.

**İndirilemiyor.** Üç yol denendi, hepsi başarısız:
- doğrudan `curl` → HTTP 403
- `api.onedrive.com/v1.0/shares/u!<b64>` → `unauthenticated`
- `graph.microsoft.com/v1.0/shares` → `InvalidAuthenticationToken`

Link `migratedtospo=true` ile SharePoint Online'a taşınmış; anonim API erişimi
kapalı. Tarayıcı etkileşimi zorunlu.

**Kullanıcının yapması gereken:** OneDrive linkinden `cvpr2026_image.pth.tar`
indirip `~/DCVC/checkpoints/` içine koymak.

**Geldiğinde açılan iki şey:**
1. **Warm-start** — `warmstart.py` hazır ve test edilmiş (143 tensörlük eşleme,
   bit-exact doğrulandı, ters eşleme bijeksiyon testi geçiyor). FLEX'in
   yaklaşımı buydu ve sıfırdan eğitimin çöktüğü yerde çalışmıştı.
2. **Mutlak kıyas** — "DCVC-UF'e göre %X FLOP, Y dB" denebilir.

Warm-start §20.3'teki sorunu da çözebilir: yakınsamış bir modelde çıkışlar arası
fark gerçek olur, yönlendirilecek bir şey doğar.

---

## 22. Warm-start geldi — ve monotonluk sorunu çözüldü

### 22.1 Transfer ve kontroller

```
398 tensör, 42,179,328 parametre  (ilk gün ölçtüğüm stok DMCI sayısıyla birebir)
stok DMCI'ye temiz yükleniyor      (eksik 0, fazla 0)
transfer: 398 tensör, 10 adapter tensörü sıfır-init'te (tasarım gereği)

CONTROL en derin çıkış vs yayınlanmış UF : max|diff| = 0.0
CONTROL her çıkış vs kesilmiş UF          : max|diff| = 0.0
```

### 22.2 Merdiven artık monoton — §20.3'ün cevabı

Sıfırdan eğitim (e1, epoch 2) vs warm-start, aynı ölçüm, aynı yol:

| çıkış | tasarruf | **sıfırdan** | **warm-start** |
|---:|---:|---:|---:|
| 2 | %43.79 | 0.1787 dB | 6.2736 dB |
| 3 | %28.88 | **0.2137 dB ✗** | 4.9893 dB ✓ |
| 4 | %13.98 | 0.1006 dB | 2.5705 dB ✓ |
| 5 | %0 | 0.0432 dB | 0.7076 dB ✓ |

Sıfırdan eğitimde çıkış 3, çıkış 2'den **15 puan fazla hesap harcayıp daha kötü**
sonuç veriyordu — routing'in sömüreceği takas yoktu. Warm-start'ta her adımda
daha derin kesinlikle daha iyi. **Sıralama artık öğrenilen bir şey değil,
inşaen garanti.**

### 22.3 Kayıpların büyük olması beklenen

Warm-start sütunundaki değerler (çıkış 2'de 6.27 dB) büyük çünkü **adapter'lar
henüz eğitilmedi** — sıfır-init'te, yani identity. Merdiven şu an "stok UF'in
k. derinlikte kesilmiş hali", telafisi olmayan. Stage A tam olarak bunu
kapatmak için koşuyor; FLEX de bu noktadan başlayıp +0.51..+0.90 dB kazanmıştı.

### 22.4 Dikiş cezası modelin keskinliğiyle büyüyor

En derin çıkışta (hiç erken çıkış yok, sadece patch'e bölme):

| model | tam decode PSNR | dikiş cezası |
|---|---:|---:|
| sıfırdan, epoch 2 | ~31.3 dB | 0.043 dB |
| **yayınlanmış** | **35.64 dB** | **0.708 dB** |

Keskin bir model dikişten daha çok zarar görüyor — mutlak MSE farkı büyüyor.
Bu, patch boyutu (E3 ekseni) ve j (E2 ekseni) kararlarını yakınsamış modelde
yeniden ölçmeyi gerektiriyor; sıfırdan modelde alınan ölçümler taşınmaz.

### 22.5 Boru hattı doğrulaması — referans figürle örtüşme

`rd_curve.py`, yayınlanmış ağırlıklar, dense decoder, 48 ayrık görüntü:

| QP | bpp | PSNR (6:1:1) |
|---:|---:|---:|
| 0 | 0.196 | 30.35 |
| 32 | 0.325 | 36.35 |
| **63** | **0.829** | **41.86** |

Kullanıcının referans figüründe QP63 ≈ **0.83 bpp / ~41.7 dB**. Encode, entropi
modeli, decode ve 6:1:1 ağırlıklandırma birlikte doğrulanmış oldu.

Düşük QP'de ayrışma test setinden: referans **CTC Image**, bizimki Open Images'ın
ayrık dilimi.

**Kaydedilen iki sınır:** oran burada entropi modelinin **tahmini**, gerçek
aritmetik-kodlanmış uzunluk değil (rANS derlendi ama `rd_curve.py`'ye
bağlanmadı). Ve karşılaştırmamızda oran **iki eğride de aynı** — aynı encoder,
aynı bitstream — dolayısıyla tahmini/gerçek ayrımı eğriler arasındaki dikey
farkı etkilemiyor, ikisini birlikte yatay kaydırıyor.

---

## 23. Adapter'ların gerekliliği ölçüldü

Warm-start merdiveni monoton (§22.2) ama **adapter'lar sıfır-init'te**, yani
erken çıkışlar telafisiz. Bu haldeyken β=30 ile router eğitildi:

```
step 0    exit_share [0,0,81,0,33,14]   tasarruf %31.32  kayıp 1.92 dB
step 100  exit_share [0,0, 0,0, 0,128]  tasarruf %0      kayıp 0.00 dB
step 200  exit_share [0,0, 0,0, 0,128]  tasarruf %0      kayıp 0.00 dB
```

Router her patch'i en derin çıkışa yolluyor — ve **doğru yapıyor**:

çıkış 5 → çıkış 4 geçişi 2.57 dB maliyetli. Normalize görüntü teriminde
`ratio = 10^0.257 = 1.81`, yani `w_image × (1.81−1) = 50 × 0.81 = 40.5` ceza.
Karşılığında `β × Δcost = 30 × 0.14 = 4.2` kazanç. 40.5 ≫ 4.2.

**Sonuç:** telafisiz erken çıkışlar 2.5–6 dB pahalı, hiçbir makul β bunu
karşılamaz. Adapter'lar süs değil, mekanizmanın çalışması için zorunlu —
FLEX'in +0.51..+0.90 dB'lik adapter kazancı tam bu boşluğu kapatıyor.

**§20.3'ün açık sorusu** ("monoton merdivende routing uniform'u yener mi")
eğitilmiş adapter'ları bekliyor. Stage A bitince ölçülecek.

**Yan not — tekrarlayan hata:** `pkill -f "rt_ws"` deseni kendi kabuk komutumu
da eşleştirip onu öldürdü, bu yüzden bu kayıt ilk seferde yazılamadı. Aynı
sınıf hata `resume_autopilot.sh`'te de çıkmıştı ve orada desen sabitlenerek
(`^bash /path/...`) çözülmüştü. `pkill -f` her zaman sabitlenmeli.

---

## 24. Dikiş azaltma — üç mekanizma, hepsi ölçüldü

Hata haritaları (§ `results/samples`) en derin çıkışta hatanın **sadece tile
sınırlarından oluşan bir ızgara** olduğunu gösterdi. qp63'te en sığ çıkışın
toplam kaybının %80'i buradan geliyor. Yani baskın terim derinlik değil, artefakt.

### 24.1 Patch'li yoldan eğitim — çıkarımda bedava

**Sorun:** eğitim `forward_all_exits` çağırıyordu, o da **tam kare** decode ediyor.
Hiçbir tile sınırı eksik komşuya karşı konvolve edilmiyordu. Adapter'lar dikişsiz
bir sinyale uyduruluyor, sonra çıkarımda dikişli bir dünyaya sokuluyordu.

**Çözüm:** FLEX'in Stage A hedefi — adım başına **tek** decode, patch başına
**rastgele** çıkış:
```
ch_p ~ U{j..K-1}, her adımda yeniden çekilir
L = || decode(ŷ, ch) − x ||²
```
Neden rastgele: deployment kareyi hiçbir zaman tek derinlikte decode etmiyor,
**karışık derinlikli** bir kare üretiyor — çıkış 2'deki patch, çıkış 5'tekinin
yanında duruyor ve head o süreksizliği de birleştiriyor. Her çıkışı ayrı ayrı
uniform decode etmek hiç oluşmayan bir durumu eğitiyordu.

Ölçülen maliyet: tam kare 1.0×, her çıkış ayrı patch'li **9.0×**, rastgele
derinlik tek decode **2.12×**. (İlk implementasyonum görüntü başına döngü
yapıyordu ve 4.9× idi; decoder zaten batch'i işliyor, tek çağrıya inince 2.12×.)

### 24.2 Replicate dolgu — %0 maliyet

DepthConvBlock'taki tek uzamsal operatör 3×3 depthwise, ve j-split altında tile
sınırında ötesi yok. Stok dolgu **sıfır** — komşunun gerçek değeri için kötü bir
tahmin. Kenar değerini kopyalamak bedava ve gerçeğe çok daha yakın.

Ölçüm (referans her zaman **stok UF tam kare decode**):

| qp | zeros | replicate | kazanç |
|---:|---:|---:|---:|
| 0 | 0.1306 | 0.1333 | −0.003 dB |
| 32 | 0.3521 | 0.2960 | +0.056 dB |
| 63 | 0.8727 | **0.6315** | **+0.241 dB** |

Yüksek oranda dikişin %28'i siliniyor, düşük oranda gürültü içinde. Sadece
per-tile bölümde uygulanıp geri alınıyor, yani `forward_full` bit-exact kalıyor.

**❗ Bu ölçümde hata yaptım ve düzelttim.** İlk versiyonu replicate'i **tüm**
bloklara uygulamıştı — referans tam kare decode de kaymıştı. İki tarafı birden
kaydırınca aradaki fark küçülüyor ve iyileşme şişiyor: "+0.794 dB" çıkmıştı,
gerçeği +0.241. **Referans her zaman stok UF olmalı.** Bu, §19'daki maliyet
modeli hatasıyla aynı sınıf: iyimser bir sayı ve onu kontrol etmeye zorlayan
hiçbir şey yok.

### 24.3 Dikiş onarım modülü — %0.95 maliyet

```
Repair(f) = f + PW_{C→C}( WSiLU( DW3×3(f) ) ),   PW sıfır-init
```

`unpatchify`'dan sonra, head'den önce, **tam karede**.

**Neden orada:** dikiş uzamsal bir artefakt ve düzeltmek için tile sınırının
**karşısını görebilen** bir 3×3 gerekiyor. Per-tile trunk'ın içinde hiçbir kernel
bunu yapamaz — halo'nun satın alacağı şey buydu ve pahalı kısımda 2.25× maliyetle
kazancı tersine çeviriyordu. Canvas birleştikten sonra tek geçiş aynı erişimi
sabit ~%1'e veriyor.

| varyant | maliyet |
|---|---:|
| sadece depthwise | %0.022 |
| depthwise + 1×1 | **%0.951** |
| (kıyas: bir per-tile trunk bloğu) | %7.45 |

Sıfır-init, yani eğitilene kadar tam identity — hiçbir şeyi kötüleştiremez.

**Maliyet modeline eklendi.** Kontrol +%1.1 ölçmüştü, model %0 diyordu ve %3
toleransta geçiyordu. Sistematik eksik faturalama §19'daki şişik sayıların tam
sebebiydi; şimdi faturalanıyor ve kontrol %0.25'e kadar uyuşuyor.

### 24.4 Halo'nun yeni rolü — decode'da yok

| yer | halo | gerekçe |
|---|---|---|
| per-tile trunk | **0** | 2.25× maliyet, kazancı tersine çeviriyor (§20.2) |
| head | yok | zaten tam karede çalışıyor |
| adapter'lar | yok | 1×1, alıcı alanı yok |
| **router girdisi** | **4 latent px** | maliyeti %0.009 — bedava, ve tile istatistiklerini sınır artefaktından temiz tutuyor |

Yani halo artık bir kalite parametresi değil; tek işlevi router'ın temiz veri
görmesi. Dikiş, yukarıdaki üç mekanizmayla ele alınıyor.

---

## 25. Çıpa ölçümü — ortak eğitim gerçekten maliyetli

§14.3'te "frontier tek başına yanıltıcı olabilir" diye baseline eklemiştim.
Baseline epoch 0'ı bitirince eşleşmiş karşılaştırma yapıldı: aynı recipe, aynı
veri, aynı epoch — tek fark çok-çıkışlı hedef.

| QP | baseline (K=1) | e1 (K=6) |
|---:|---|---|
| 0 | 0.228 bpp, **30.47 dB** | 0.290 bpp, 29.77 dB |
| 32 | 0.379 bpp, **35.73 dB** | 0.475 bpp, 34.62 dB |
| 63 | 0.719 bpp, **38.39 dB** | 0.689 bpp, 36.67 dB |

Eşit bpp'de (0.689): baseline ~38.16, e1 36.67 → **1.49 dB fark**.
qp0'da e1 hem daha fazla bit harcıyor hem daha az kalite veriyor.

**Sonuç:** Eq(6) tüm çıkışları eşit ağırlıkla optimize edince en derin çıkışın
kalitesi sığ çıkışlar uğruna feda ediliyor. Ve o çıkış kalite çıpamız — tüm
frontier ona göre ölçülüyor.

Bu, sıfırdan-eğitim kolunun neden gerçek UF ile kıyaslanabilir sonuç üretemediğinin
ikinci sebebi. Birincisi encoder'ın da eğitilmesiydi (latent kayması, §20.5);
ikincisi bu.

**Warm-start yolunda ikisi de yapısal olarak imkânsız:** encoder donuk (latent
birebir aynı) ve backbone donuk (çıpa bit-exact yayınlanmış UF). Eğitilen tek şey
adapter'lar ve dikiş onarımı — ikisi de sıfır-init, yani en kötü ihtimalle
hiçbir şey değiştirmiyorlar.

---

## 26. İlk tam RD eğrisi — gerçek DCVC-UF referansına karşı

`results/rd_deliverable.png`, `results/rd_warmstart.json`

Warm-start Stage A epoch 0'ın adapter'ları, β=30 router, 48 ayrık görüntü.
Encoder donmuş → **latent, bitstream ve bpp gerçek DCVC-UF ile birebir aynı**,
dolayısıyla eğriler arasındaki dikey fark tamamen decoder'a ait.

| qp | bpp | dense | routed | ΔPSNR | tasarruf |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.194 | 30.37 | 29.81 | −0.554 | **%24.9** |
| 8 | 0.214 | 31.75 | 31.27 | −0.476 | %20.8 |
| 16 | 0.234 | 33.53 | 32.77 | −0.764 | %15.7 |
| 24 | 0.278 | 34.61 | 34.15 | −0.457 | %9.2 |
| 32 | 0.340 | 36.05 | 35.78 | **−0.270** | %4.1 |
| 40 | 0.397 | 37.77 | 37.15 | −0.611 | %1.2 |
| 48 | 0.489 | 39.30 | 38.79 | −0.504 | −%0.1 |
| 56 | 0.630 | 40.79 | 40.32 | −0.470 | −%0.7 |
| 63 | 0.752 | 42.20 | 41.60 | −0.601 | **−%0.8** |

### 26.1 Router orana göre kendi kendine uyarlanıyor

QP0'da %24.9, QP63'te sıfır. Kimse söylemedi — düşük oranda latent seyrek,
erken çıkış ucuz; yüksek oranda yoğun, pahalı. Router bunu gövde sinyallerinden
buluyor. §17'deki sinyal değişikliğinin karşılığı bu.

### 26.2 Ama hedef tutmuyor, ve yüksek oranda tasarruf negatif

- dB kaybı 0.27–0.76, **0.1 dB hedefinin 3–8 katı**
- qp48 ve üstünde tasarruf **negatif**: router her patch'i en derin çıkışa
  yolluyor ama dikiş onarımı %0.95 faturalanıyor → net maliyet tam decode'dan
  yüksek

İkincisi doğrudan açıklanabilir: **dikiş onarımı hiç eğitilmedi.** Stage A
3 epoch'un 1'ini bitirdi; adapter'lar üçte bir eğitilmiş, onarım modülü sıfır-init
(identity). Yani şu an bedelini ödüyoruz, karşılığını almıyoruz. Yüksek orandaki
negatif tasarruf, düzelmesi gereken ilk şey.

### 26.3 Ne bekleniyor

| bekleyen | neden önemli |
|---|---|
| Stage A'nın kalan 2 epoch'u | adapter'lar + dikiş onarımı tam eğitilir |
| GPU 7'de encoder-frozen / decoder-trained | decoder'ın **tamamı** (14.97M param) çok-çıkışlı yapıya uyum sağlar; şu an sadece 739k adapter'la sınırlı |
| j=4 / 256px varyantı | ölçülen tek konfigürasyon ki dikiş tabanı 0.1 dB'nin altında (0.054) |

---

## 27. Stage A tamamlandı — adapter'lar 2.2 dB'ye kadar kazandırdı

3 epoch, patch'li yoldan, donmuş backbone. qp63, 32 ayrık görüntü:

| çıkış | tasarruf | eğitim öncesi | eğitim sonrası | kazanç |
|---:|---:|---:|---:|---:|
| 2 | %42.8 | 6.2736 dB | **4.0385** | +2.235 |
| 3 | %27.9 | 4.9893 | **3.2133** | +1.776 |
| 4 | %13.0 | 2.5705 | **1.7940** | +0.777 |
| 5 | −%0.95 | 0.7076 | **0.6214** | +0.086 |

Merdiven monoton kaldı. FLEX'in adapter kazancı +0.51..+0.90 dB idi; burada sığ
çıkışlarda 2.2 dB'ye çıkıyor — çünkü başlangıç noktası daha kötüydü (telafisiz
erken çıkışlar).

**Ama hedef hâlâ uzak.** En ucuz çıkış 4.04 dB, en pahalısı 0.62 dB. Bu, §20.2'de
ölçülen yapısal kısıtın doğrulanması: **j=2 / 128px'te dikiş tabanı 0.837 dB**,
yani hiçbir adapter eğitimi 0.1 dB'ye indiremez. Konfigürasyon değişmeden hedef
ulaşılamaz.

Hedefe ulaşabilecek tek ölçülen konfigürasyon **j=4 / 256px** (taban 0.054 dB,
tavan %14 tasarruf) — kuyrukta.

Not: bu checkpoint `seam_repair` eklenmeden önce başlatıldığı için o modül
identity, ama maliyet modelinde faturalanıyor — çıkış 5'teki −%0.95 tasarruf
bundan. Dürüst rakam: ödeyip karşılığını almıyoruz.

---

## 28. Encoder-frozen / decoder-trained koşusu başladı

Kullanıcının istediği yapılandırma, GPU 7'de:

```
DONMUŞ    27,947,520 param  — encoder + hyperprior + entropi (Microsoft'un)
EĞİTİLEN  14,971,008 param  — decoder'ın TAMAMI: gövde + head + adapter + onarım
```

Latent, bitstream ve bpp gerçek DCVC-UF ile birebir aynı → oran ekseni sabit,
ölçülen her fark decoder'a ait. Patch'li yoldan, replicate dolgu ve dikiş onarımı
aktif.

**Neden bu Stage A'dan daha umut verici:** Stage A sadece 739k adapter
eğitebiliyordu, yani decoder gövdesi çok-çıkışlı yapıya hiç uyum sağlayamıyordu —
sadece çıkış noktalarına yama atılıyordu. Burada 14.97M parametre serbest, gövde
kendini erken çıkışa uygun şekilde yeniden düzenleyebilir.

---

## 29. ❗ Kullanıcı grafikten bir ölçüm hatası yakaladı

**Gözlem:** "QP16'nın orta noktası farklı, ikisinde bpp aynı noktadaymış gibi
görünmüyor."

**Doğrulandı — 9 noktanın 9'unda da bpp farklıydı:**

| qp | dense bpp | routed bpp | fark |
|---:|---:|---:|---:|
| 16 | 0.23423 | 0.23882 | **+0.00459** |
| 40 | 0.39705 | 0.41073 | +0.01368 |
| 48 | 0.48870 | 0.50326 | +0.01456 |

İki eğri **aynı bitstream'i** okuyor; aralarında bpp farkı olması imkânsız.

**Sebep:** `ImageFolder.__getitem__` crop konumunu `random.randint`, yatay
çevirmeyi `random.choice` ile seçiyor. `sweep()` iki kez çağrılıyordu — bir
dense, bir routed — ve her çağrı veriyi baştan dolaşıp **farklı rastgele
crop'lar** üretiyordu. Dense eğrisi A görüntülerinde, routed eğrisi B
görüntülerinde ölçülmüştü. Eğriler arasındaki dikey fark decoder'a değil,
kısmen crop varyansına aitti.

**Düzeltme:** `sweep_both()` — her iki decoder **aynı batch içinde** koşuyor.
Aynı görüntü, aynı latent, aynı bitler, sadece synthesis farklı. bpp artık
inşaen özdeş (doğrulandı: 0/9 fark).

**Etkisi — gürültü iyimser yöndeydi:**

| | kirli | temiz |
|---|---:|---:|
| QP0 | −0.554 dB / %24.9 | −0.603 dB / %24.7 |
| QP32 | −0.270 dB / %4.1 | −0.339 dB / %3.7 |
| QP63 | −0.601 dB / −%0.8 | −0.625 dB / −%0.7 |

Beşinci kez: hata iyimser yöndeydi.

**Ders:** bu hatayı hiçbir kontrolüm yakalamamıştı — kontroller bit-exact
eşdeğerlik ve maliyet-gerçeklik uyumunu doğruluyor, ama *iki ölçümün aynı veri
üzerinde yapıldığını* doğrulayan bir şey yoktu. Grafikteki tutarsızlık insan
gözüne çarptı. Aynı bitstream'i paylaşması gereken iki eğrinin bpp'si artık
`rd_curve.py` içinde inşaen tek bir yerden geliyor, yani ayrışması mümkün değil.

---

## 30. ❗ Değerlendirme rastgele crop çekiyordu — mutlak sayılar ~1 dB oynuyordu

**Kullanıcı gözlemi:** "dense decoder QP0 için 30'un biraz altında olmalı, QP63
için 42'nin çok az altında; doğru ölçtüğünden emin misin?"

**Doğrulandı.** Aynı model, aynı formül, iki ayrı ölçüm:

| ölçüm | QP0 | QP63 |
|---|---:|---:|
| A | 29.54 | 41.47 |
| B (rd_curve) | 30.58 | 41.97 |

**~1 dB salınım** — ölçmeye çalıştığımız etkilerin çoğundan büyük.

**Sebep:** `ImageFolder.__getitem__` crop konumunu `random.randint`, yatay
çevirmeyi `random.choice` ile seçiyor. Eğitim için doğru, **ölçüm için yanlış**.
Her değerlendirme farklı pikselleri görüyordu.

**Düzeltme:** `DeterministicCrop` — merkez crop, çevirme yok. Doğrulandı: aynı
komut iki kez, PSNR birebir aynı (31.17 / 37.05 / 42.72).

### 30.1 Formül doğruydu

Şüphelendiğim şey kroma altörneklemeydi; kontrol ettim:

| qp | Y | U 4:4:4 | V 4:4:4 | 6:1:1 (4:4:4) | 6:1:1 (4:2:0) |
|---:|---:|---:|---:|---:|---:|
| 0 | 26.54 | 37.87 | 39.27 | 29.54 | 29.67 |
| 63 | 39.97 | 45.43 | 46.52 | 41.47 | 41.82 |

`rgb2ycbcr_np` üç düzlemi de [0,1]'e koyuyor, dataset −0.5 kaydırıyor, aralık
1.0 → `10·log10(1/mse)` doğru. 4:4:4 ile 4:2:0 arasındaki fark 0.13–0.35 dB;
kayda değer ama 1 dB'lik salınımı açıklamıyor. Sorun crop rastgeleliğiydi.

### 30.2 Test seti farkı — mutlak kıyas için kapatılması gereken açık

Kullanıcının beklediği değerler **CTC Image** setinden. Bizimki Open Images'ın
ayrık dilimi. Deterministik merkez crop'la bizim set: QP0 31.17, QP63 42.72.
Mutlak kıyas için CTC gerekiyor; şu an yok.

### 30.3 Ders

Bu, kullanıcının grafikten yakaladığı **ikinci** ölçüm hatası (birincisi §29:
iki eğrinin farklı crop'larda ölçülmesi). İkisi de aynı kökten: `ImageFolder`
eğitim için tasarlanmış ve rastgele; değerlendirme onu olduğu gibi kullanıyordu.
Kontroller bunu yakalayamazdı — hepsi tek bir ölçümün iç tutarlılığına bakıyor,
hiçbiri *iki ölçümün karşılaştırılabilir olduğunu* doğrulamıyordu.

---

## 31. Dikiş için paper aradım, buldum, uyguladım — ve ölçüm paperın kendi uyarısını doğruladı

**Aranan:** tiled CNN inference'ta tile sınırındaki padding'in yarattığı hatayı
çözen güncel bir yöntem. Bulunan: Kaseva et al., *"Per-channel autoregressive
linear prediction padding in tiled CNN processing of 2D spatial data"*
(arXiv:2502.12300).

**Yöntem:** tile'ın kenarının ötesindeki bilinmeyen komşuyu sıfırla ya da kenarı
kopyalayarak uydurmak yerine, her kanal için tile'ın kendi içinden en küçük
kareler ile bir AR modeli fit edip padding'i o modelin beklenen değerinden
üretmek.

**Paperın kendi sonucu — atlamadım, aynen aktarıyorum:** hatayı sıfır/replicate'e
göre yalnızca *"slightly reduced"*, karşılığında *"moderate increase in time
cost"*, ve yazarlar çıktıyı birkaç piksel kırpmanın muhtemelen daha iyi olduğunu
söylüyor. Yani paperı bulmuş olmam uygulamak için yeterli sebep değildi. Ama
**padding parametresizdir** — eğitim gerektirmeden, mevcut warm-start
checkpoint'i üzerinde doğrudan ölçülebilir. Replicate'i ölçtüğüm gibi ölçtüm.

**Uygulama notu.** PyTorch Conv2d yalnızca zeros/reflect/replicate/circular
kabul eder, bu yüzden yeni şemalar `flexuf/backbone/padding.py` içinde
konvolüsyonu sarmalayarak kuruldu: elle pad, sonra `padding=0` ile konvolve.
Sadece 3x3 depthwise'lar sarmalandı, çünkü bir DepthConvBlock'ta uzamsal uzanımı
olan tek operatör odur (9C MAC/px, bloğun 8C²+9C'sinin %0.3'ü) — bir sınırın
padding ile karşılaşabileceği tek yer.

**Kontrol (önce bu):** `zeros` ile sarmalanmış decoder, sarmalanmamış decoder ile
`max|diff| = 0.0`. Bu geçmeseydi ölçtüğüm şey padding değil wrapper'ın hatası
olurdu.

**Ölçüm.** Saf dikiş cezası, dB — j=2, 128px tile, referans daima stok UF tam
kare decode, deterministik crop, 32 held-out görüntü. Küçük iyi.

| mod | qp0 | qp32 | qp63 | zeros'a göre |
|---|---|---|---|---|
| zeros | 0.1222 | 0.2855 | 0.8399 | — |
| replicate | 0.0883 | 0.1320 | 0.2093 | +0.034 / +0.153 / +0.631 |
| linear | 0.2294 | 0.3434 | 0.5223 | −0.107 / −0.058 / +0.318 |
| **arls** | **0.0710** | **0.1039** | **0.1625** | **+0.051 / +0.182 / +0.677** |

**Okunuşu.**

1. **arls her üç QP'de de kazandı** ve qp63'te dikişin %81'ini siliyor
   (replicate %75). Replicate'in üzerine ek kazanç +0.047 dB — paperın dediği
   gibi *slight*, ama tutarlı ve bedava. Parametre yok, eğitim yok.
2. **`linear` (birinci derece ekstrapolasyon) replicate'ten KÖTÜ.** Düşük QP'de
   zeros'tan bile kötü (0.229 vs 0.122). Yerel eğimi dışarı uzatmak, kenarda
   gradyan büyükse aşırı sapıyor. Bunu ölçmeseydim "lineer tahmin daha akıllı,
   elbette daha iyidir" diye varsayacaktım. Değil.
3. arls'in replicate'e üstünlüğü QP arttıkça büyüyor (+0.017 → +0.047), çünkü
   yüksek QP'de latent daha zengin ve kanal-içi korelasyon replicate'in örtük
   varsaydığı "korelasyon = 1"den daha bilgilendirici hale geliyor.

arls'in tam olarak bu davranışı göstermesi tesadüf değil: fit edilen katsayı
a = <x_t, x_{t+1}> / <x_{t+1}, x_{t+1}> olduğundan, kanal mükemmel korelasyonlu
ise replicate'e, korelasyonsuz ise zeros'a indirgeniyor. Yani arls iki ucuz
şemanın *ölçülmüş* interpolasyonu — bu yüzden ikisinden de kötü olamıyor, ve
tablo bunu doğruluyor.

**Karar:** `tile_pad_mode` varsayılanı `replicate` → **`arls`**. Zero-tolerance
kontrollerinin tamamı yeni varsayılanla da geçiyor.

---

## 32. e1 rafa kaldırıldı, yerine arls deneyi (kullanıcı talimatı)

**Neden e1, e2 değil.** Yeni deney j=2 / 128px / donuk encoder. e1 tam olarak
j2_p128'di, yani yeni koşu e1'in sorduğu soruyu daha sağlam cevaplıyor (deepest
exit bit-exact DCVC-UF, e1'de ise anchor 1.49 dB aşağıda). e2 (j4/128) hâlâ ayrı
bir konfigürasyon sorusu, o yüzden ona dokunmadım.

**Geri alınabilir:** `runs/e1_j2_p128/` içinde `ckpt_epo5.pth.tar` ve tam log
duruyor; `--pretrain` ile kaldığı yerden devam ettirilebilir. İptali ancak bunu
doğruladıktan sonra yaptım.

**Yeni koşu — `runs/wdec_j2_p128_arls`, GPU4.** Kullanıcının hedefi "%30-40
tasarruf, −0.1 dB, dikiş hatasını dramatik düşür" olduğu için bu koşu bütün
dikiş önlemlerini üst üste bindiriyor:

| önlem | ne yapıyor | maliyet |
|---|---|---|
| `--tile_pad arls` | sınırdaki komşuyu AR(1) ile tahmin | 0 |
| `--seam_repair full` | birleştirilmiş canvas üzerinde eğitilebilir tam-kare onarım | %0.95 |
| `--train_patched` | decoder'ı *dağıtılan* patch'li yol üzerinden eğit, dikişi telafi etmeyi öğrensin | 0 |
| `full_frame_head` | son upsample dikişi görerek çalışsın | 0 |
| `latent_halo 2` | router'a sınır bağlamı | 0 (yalnız router) |

`--freeze_encoder`: encoder, hyperprior ve entropy modeli donuk → latent ve
bitstream stok UF ile birebir aynı, bpp karşılaştırması yapı gereği geçerli.
16 epoch ≈ 2.3 gün.

**Beklenti, ölçülmüş sayılardan:** arls ile saf dikiş tabanı qp32'de 0.104 dB.
j=2'nin tavanı %74.5. Hedef %30-40 tasarruf bu tavanın yarısından azı, yani
adapter'ların telafi etmesi gereken şey dikiş değil derinlik kaybı — ve Stage A
adapter'larının daha önce +2.235 dB'ye kadar getirdiği ölçülmüştü. Hedef
ulaşılabilir görünüyor; koşu bunu doğrulayacak ya da çürütecek.

---

## 33. arls gerçek CTC'de de kazanıyor — ve dikiş orada crop'takinden daha kötü

İlk ablasyon 512x512 OpenImages crop'unda, 16 tile ile yapılmıştı. Dağıtım
koşulu bu değil: 1080p kare, tile katına hizalandığında 15x9 = 135 tile. Piksel
başına sınır oranı aynı ama içerik aynı değil — video kareleri, OpenImages
crop'unun taşımadığı geniş düz alanlar ve uzun düz kenarlar içeriyor. Crop
sonucunun aktarılacağını varsaymak, daha önce replicate hatasını üreten türden
bir varsayımdı; o yüzden ölçtüm.

Saf dikiş cezası, dB — j=2, 128px tile, CTC native (UVG 4 + HEVC_E 3, 18 kare):

| mod | qp0 | qp32 | qp63 |
|---|---|---|---|
| zeros | 0.2271 | 0.4993 | 1.1670 |
| replicate | 0.1339 | 0.1830 | 0.2125 |
| linear | 0.3841 | 0.4741 | 0.5021 |
| **arls** | **0.1125** | **0.1533** | **0.1785** |

**zeros gerçek içerikte çok daha kötü** (qp63'te 1.167 vs crop'ta 0.840): düz
alanlarda sıfır dolgu felaket, çünkü uydurduğu komşu gerçek komşudan maksimum
uzakta. replicate ve arls ise neredeyse aynı kalıyor — ikisi de içerikten
türediği için içerik değiştiğinde birlikte uyum sağlıyorlar. `linear` yine kötü,
bu sefer qp0'da zeros'un 1.7 katı. Sıralama iki farklı veri kümesinde aynı çıktı.

---

## 34. GridSeamRepair — modüle dikişin NEREDE olduğunu söylemek

**Tespit ettiğim kusur.** Mevcut `SeamRepair` tüm canvas üzerinde öteleme-değişmez
bir 3x3. Yani hangi pikselin tile sınırında olduğunu yalnızca içerikten çıkarmak
zorunda, ve aynı düzeltmeyi pikselin %77'sini oluşturan temiz iç bölgeye de
uyguluyor — orada herhangi bir düzeltme zarardan başka bir şey değil. Modüle
elimizdekinden daha zor bir problem çözdürüyorduk.

**Oysa ızgara bilinmiyor değil.** `unpatchify` tile'ları orijinden başlayan
düzenli bir P x P kafese diziyor; konum hem eğitimde hem çıkarımda tam olarak
biliniyor. Düzeltmeyi tile İÇİNDEKİ konuma göre indekslenen öğrenilmiş bir kapı
ile geçirdim:

    Repair(f) = f + G[i mod P, j mod P] · PW( WSiLU( DW3x3(f) ) )

G, 384 kanalın tamamında paylaşılan P x P = **256 skaler**. Decoder
parametrelerinin %0.0007'si, ve bir broadcast çarpımı dışında MAC maliyeti yok —
cost modelinde `full` ile aynı %0.951 olarak faturalandırıldı, bedavaymış gibi
davranılmadı.

**Başlangıç bilgiyi atmıyor, taşıyor.** G, d = tile kenarına feature-piksel
uzaklığı olmak üzere exp(−d/τ) ile başlatılıyor: ölçülen değerler kenarda 1.000,
bir içeride 0.607, merkezde 0.030. Yani adım 0'da kapı zaten dikişin üzerinde
yoğunlaşmış; eğitim onu keşfetmek yerine rafine ediyor. `pw` hâlâ sıfır
başlatmalı, dolayısıyla modül adım 0'da tam olarak birim — **`grid` ile de
deepest exit == stok UF, max|diff| = 0.0**. G'nin başlatması ilk çıktıyı değil
ilk gradyanı şekillendiriyor.

**Koşu.** `runs/wdec_j2_p128_arls_grid`, GPU4, `--tile_pad arls --seam_repair
grid`, donuk encoder, 16 epoch. Bir önceki arls koşusunu (2 saatlik) durdurdum:
ızgara kapısı yapısal bir iyileştirme ve 2 saatten fazla değer.

**Kontrol grubu var:** GPU7'deki `wdec_j2_p128` aynı konfigürasyonu
replicate + düz `full` onarım ile koşuyor. arls'in payı zaten sıfır-başlatmada
ayrı ayrı ölçüldüğü için (bölüm 31 ve 33), iki koşunun farkı ızgara kapısına
atfedilebilir.

---

## 35. Bir kill sarmalayıcıyı vurdu, trainer'ı değil — ve iki koşu aynı GPU'yu paylaştı

`nohup ... &` sonrası `$!` 1822822 verdi, ama gerçek python trainer 1822825'ti.
`kill 1822822` sarmalayıcıyı öldürdü, eğitim devam etti. Yerine yeni koşuyu
GPU4'e başlattığımda, eski koşu hâlâ o kartı tutuyordu: iki eğitim aynı GPU'da,
ikisi de yarı hızda.

**Nasıl yakalandı:** monitör. Yeni koşuyu başlattıktan sonra hem `ARLS-j2/128
s6000` (öldürdüğümü sandığım) hem `wdec_j2_p s200` satırları göründü, ve ARLS'in
adım sayısı kill'den SONRA artmıştı. Log tabanlı bir monitör bunu göremezdi —
ölü bir koşunun son satırı da aynen orada durur. Canlılığı `/proc`'tan okuma
kararı (bölüm 34 öncesi) bu yüzden işe yaradı.

**Yanlış ders:** "`$!` yerine şunu kullan". Doğru ders: **öldürdüğünü, kartın
boşaldığını görerek doğrula.** `nvidia-smi -i 4 --query-compute-apps` iki pid
gösteriyordu; tek bakışta belli.

**Kalıcı önlem — ve önlemin kendisi iki kez yanlıştı.** İlk denemede GPU çakışma
alarmı ekledim; ilk tetiklemede iki hata birden verdi:

1. `*** DEAD: WD-j4/256, GRID-j2/128 ***` — ikisi de gayet iyi çalışıyordu.
   Sebep: kendi shell'lerimi eşlememek için filtreyi `/train_flexuf_image.py`
   diye baştaki eğik çizgiyle sabitlemiştim, bu da **göreli yolla** başlatılan
   her koşuyu görünmez yaptı. Temiz görünen bir filtre, iki yanlış alarm üretti.
2. `*** GPU PAYLASIMI: GPU6=BASE+e2 ***` — ama GPU başına iki koşu **kasıtlı**.
   Alarm, normal olan şeye alarm veriyordu.
3. Bonus: `ckpt_warm` diye olmayan bir koşu belirdi, çünkü `runs/` altındaki her
   yolu koşu adı sayıyordum — `--pretrain .../runs/warmstart/ckpt_warmstart.pth.tar`
   de oraya düşüyor.

Doğru tespit `scripts/heartbeat.py` içine, gerekçeleriyle yazıldı: prosesin
argv0'ı gerçekten python olacak ve argümanlarından biri `train_flexuf_image.py`
ile bitecek; koşunun kimliği **`--save_dir`** argümanı olacak; ve alarm GPU
paylaşımı değil **ZOMBIE = alive − expected** olacak — yani "öldü sandığım ama
kartı hâlâ tutan koşu", ki fiilen olan tam olarak buydu ve yanlış pozitifi yok.
Monitöre bağlamadan önce gerçeğe karşı çalıştırıp yedi koşunun yedisini de doğru
saydığını gördüm.

---

## 36. Denenmemiş en iyi konfigürasyon: j=2 + 256px tile

Deneyler j'yi ve tile boyutunu birlikte değiştirdiği için bir hücre boş kalmıştı:

|  | 128px tile | 256px tile |
|---|---|---|
| j=2 (tavan %74.5) | WD-j2/128, GRID-j2/128 | **boş** |
| j=4 (tavan %14) | — | WD-j4/256 |

Saf dikiş cezası, arls ile, CTC native (küçük iyi):

| konfig | qp0 | qp32 | qp63 | tavan |
|---|---|---|---|---|
| j2 / 128px | 0.1125 | 0.1533 | 0.1785 | %74.5 |
| **j2 / 256px** | **0.0518** | **0.0707** | **0.0879** | **%74.5** |
| j4 / 256px | ~0.054 | — | — | %14 |

Tile kenarı iki katına çıkınca dikiş yarıya indi — Boundary Law'un (ceza ∝
1/tile-kenarı) doğrudan doğrulaması, ve bu sefer varsayım değil ölçüm.

**j2/256 diğer ikisini de domine ediyor:** j4/256 ile aynı dikişe sahip ama
tavanı 5 katı; j2/128 ile aynı tavana sahip ama dikişi yarısı. Ve **her QP'de
0.1 dB'nin altında** — kullanıcının −0.1 dB hedefi, dikiş için, adapter'lar hiç
çalışmadan zaten sağlanmış durumda. Kalan bütçenin tamamı derinlik kaybına
harcanabilir, ki asıl mesele oydu.

Tek bedeli yönlendirme çözünürlüğü: 1080p'de 40 tile (128px'te 135). Router için
fazlasıyla yeterli.

**Yer açmak için e3 rafa kaldırıldı** (`runs/e3_j2_p64`, `status_latest.pth.tar`
ve `ckpt_epo0.pth.tar` duruyor, resume edilebilir). Gerekçe: 64px tile her
konfigürasyonun en kötü dikişine sahip (Boundary Law 128px'in iki katını
söylüyor), üstüne sıfırdan eğitim anchor'ı gerçek DCVC-UF'in 1.49 dB altında
tutuyor, ve 34 gün kalmıştı. Çalışanlar arasında kullanılabilir sonuç üretme
ihtimali en düşük olan oydu. Bunu kullanıcı e3 için ayrıca istemedi — e1/e2
arasından seçmemi istemişti — o yüzden açıkça yazıyorum: **bu benim kararım**,
ve tek komutla geri alınır.

**Ortaya çıkan tasarım temiz:**

| koşu | dolgu | onarım | ne yalıtıyor |
|---|---|---|---|
| WD-j2/128 | replicate | full | kontrol |
| GRID-j2/128 | arls | grid | dolgu + ızgara kapısı |
| GRID-j2/256 | arls | grid | tile boyutu (üsttekine karşı) |
| WD-j4/256 | replicate | full | j (tavan/dikiş dengesi) |

`--min_crop 512`: 256px tile 256px crop'a tek tile olarak sığar ve dikiş hiç
oluşmaz — j4/256 deneyinde bir kez düşülen tuzak. 512 crop 4 tile veriyor.
Bellek için batch 16 yerine 8.

---

## 37. Watchdog rafa kaldırılan deneyleri diriltmeye çalışıyordu

`scripts/autopilot.sh`'in watchdog'u "kasten durduruldu" ile "çöktü"yü ayırt
edemiyor. e1 ve e3'ü rafa kaldırdıktan sonra **17 ardışık döngü** boyunca onları
geri koymaya çalışmış:

    [09:26:29] e1_j2_p128: DIED — restarting on GPU 4
    [09:36:50] e3_j2_p64:  DIED — restarting on GPU 7
    ... (5 dakikada bir, 17 kez)

Hedef kartlar tam olarak yerlerine geçen koşuların kartları: e1 → GPU4 (GRID-j2/128),
e3 → GPU7 (GRID-j2/256). Yeniden başlatmalar bir sebeple tutmamış, ama her an
tutabilirdi — ve tuttuğunda sonuç bölüm 35'teki arızanın aynısı olurdu: iki koşu
tek kartta, ikisi de yarı hızda, dışarıdan hiçbir şey yanlış görünmeden.

**Emekliye ayrılmış bir deneyi dolu bir karta geri koyan watchdog, watchdog
olmamasından kötüdür.**

**Düzeltme.** İzleme listesi yalnızca bu script'in *doğru* yeniden başlatabildiği
koşulara indirildi: `e2_j4_p128` ve `baseline_singleexit`.

`wdec_*` / `*_arls_grid` koşuları da bilerek listeye alınmadı: `--pretrain`,
`--freeze_encoder`, `--train_patched`, `--tile_pad`, `--seam_repair` argümanlarına
ihtiyaç duyuyorlar ve script'in `ARGS` tablosunda bunların hiçbiri yok. Buradan
yeniden başlatmak, **aynı isim altında sessizce farklı bir deney** üretirdi. Onlar
`scripts/heartbeat.py`'nin DEAD alarmıyla izleniyor ve elle başlatılıyor.

Not: `declare -A GPU=(...)` `main()`'in dışında olduğu için çalışan örneğin
belleğinde zaten çözülmüştü — dosyayı düzenlemek yetmezdi, süreci yeniden
başlatmak gerekti.

---

## 38. DÜZELTME: arls bedava değil — kazancı ölçüp faturayı ölçmemiştim

Bölüm 31'de arls'i varsayılan yaptım ve "parametresiz, bedava" dedim. Paperın
kendi uyarısını (*"moderate increase in time cost"*) aktarmıştım ama **ölçmemiştim.**
Bu, erken tasarruf rakamlarını şişiren hatanın aynı sınıfı: kazanç ölçülüyor,
fatura ölçülmüyor.

**Gecikme, 1080p tam decode, deepest exit:**

| mod | j2/128px | j2/256px |
|---|---|---|
| zeros | 383.7 ms | 436.0 ms |
| replicate | 384.0 (+0.1%) | 430.4 (−1.3%) |
| linear | 409.5 (+6.7%) | 442.5 (+1.5%) |
| arls | **423.3 (+10.3%)** | **471.2 (+8.1%)** |

**Ayırt edici ölçüm — bu benim sarmalayıcımın mı, AR uydurmasının mı maliyeti:**

    native cudnn zeros dolgu    440.5 ms   (referans)
    sarmalayici + zeros         440.1 ms    -0.1%   <- sarmalayici bedava
    sarmalayici + replicate     429.1 ms    -2.6%
    sarmalayici + arls          487.8 ms   +10.7%   <- fatura gerçekten AR uydurması

MAC sayısı bunu **yakalayamazdı**: AR uydurması tile ve blok başına iki
indirgeme, MAC olarak ihmal edilebilir. Maliyet kernel launch ve bellek bandı.

**Değerlendirme.** arls, replicate'e göre qp63'te +0.034 dB alıyor, karşılığında
decode'un %10.7'sini harcıyor. Cost modelinde bir trunk bloğu %7.45 → %10.7 ≈
**1.34 blok**. Yönlendirme o bütçeyi 0.034 dB'den çok daha iyi harcar.

### Kurtarma denemesi 1 — sabit küçültme (başarısız)

arls'in katsayısı korelasyon 1'de replicate'e, 0'da zeros'a indirgeniyor. Sabit
bir a bunun çoğunu yakalarsa uydurma boşuna ödeniyor demekti. `shrink<a>` modu,
maliyeti tam replicate kadar. 10 CTC karesi, j=2/128px:

| mod | qp0 | qp32 | qp63 |
|---|---|---|---|
| replicate | 0.1288 | 0.1771 | 0.2160 |
| shrink0.95 | 0.1228 | 0.1774 | 0.2138 |
| shrink0.9 | 0.1249 | 0.1852 | 0.2185 |
| shrink0.8 | 0.1356 | 0.2009 | 0.2279 |
| arls | **0.1082** | **0.1473** | **0.1797** |

Sabit katsayı arls'in kazancının **hiçbirini** yakalamıyor; 0.95 replicate'e
oturuyor, altındaki her şey daha kötü. Kazanç katsayının **uyarlanabilirliğinde**.

### Kurtarma denemesi 2 — öğrenilen kanal-başı katsayı (çalışıyor ama yine ucuz değil)

arls kanal VE tile başına uyarlanıyor. `learned` modu kanal-başı yarısını
tutuyor: `pad = a_c · kenar`, a_c eğitilen 384 skaler, 1.0 ile başlatılıyor
(yani **tam olarak replicate**). Doğrulandı:

    learned(a=1) == replicate     max|diff| = 0.0
    pad_coef gradyani             384/384 kanala ulasiyor
    gecikme  replicate 372.7 | learned 397.7 (+6.7%) | arls 422.3 (+13.3%)

Ucuz değil. **Eager PyTorch'ta her özel dolgu %5-13 arası maliyetli**, çünkü
`F.pad`'in replicate'i füzyonlu tek kernel, diğer her şey fazladan bir bellek
geçişi. %6.7 ≈ 0.9 trunk bloğu; 0.034 dB için hâlâ kötü.

### Karar

`tile_pad_mode` varsayılanı **`replicate`**. `arls`, `learned`, `shrink*`
ölçülmüş maliyetleriyle birlikte kodda ve belgede kalıyor — bulgu korunuyor,
seçim açık. Bir üretim CUDA kernel'inde AR dolgusu konvolüsyona füzyonlanabilir
ve hikâye değişebilir; ama **ölçtüğümü raporluyorum, mümkün olabileceği düşünülen
şeyi değil.**

**İki GRID koşusu replicate ile yeniden başlatıldı** (`wdec_j2_p128_grid` GPU4,
`wdec_j2_p256_grid` GPU7). Yan fayda: deney tasarımı da temizlendi — GRID artık
(replicate + grid), WD-j2/128 ise (replicate + full), yani aralarında **tek
değişken** var: ızgara kapısı. arls+grid iken iki değişken vardı.

Ayrıca `wrap_tile_padding` artık zaten sarmalanmış bir konvolüsyonu sarmalamayı
reddediyor: iç içe geçtiğinde hata `conv2d`'nin içinden
`'PaddedDepthwise' object has no attribute 'weight'` diye çıkıyordu — sebebinden
çok uzakta. Ve `pad_coef`, warm-start'ın bilinen-yeni tensör beyaz listesine
eklendi; liste onu **eklendiği anda yakaladı**, zaten bunun içindi.

---

## 39. Manşet yüzde saatte de gerçek — ve j=2'nin tavanını yanlış söylemişim

arls'te yakaladığım hatanın (kazancı bir birimde ölç, faturayı başka birimde öde)
aynısını **manşetin kendisinde** aradım. Rapor ettiğimiz tasarruf `flexuf/cost.py`
üzerinden MAC sayıyor. MAC bir makale için doğru birim, bir vaat için yanlış
birim: MAC'lerinin %40'ını atlayıp yalnızca %15 hızlanan bir decoder, kullanıcının
hissedebileceği hiçbir şeyin %40'ını kazanmamıştır.

`scripts/saving_is_real.py`: 1080p bir kareyi her tile aynı çıkışta olacak şekilde
decode et, ölçülen süre oranını `exit_costs()` ile karşılaştır.

| çıkış | MAC | saat (128px) | saat (256px) |
|---|---|---|---|
| 2 | 43.4% | 43.6% | 42.5% |
| 3 | 28.6% | 28.8% | 28.3% |
| 4 | 13.8% | 14.9% | 14.3% |
| 5 | 0.0% | 0.5% | 0.5% |

**±1% içinde.** Manşet dürüst. (Bu, replicate'e dönmüş olmanın da bir sonucu:
arls ile bu tablo her satırda ~10 puan bozulurdu.)

### Ama tabloyu yaparken kendi hatamı buldum

Kullanıcıya "j=2'nin tavanı %74.5" dedim ve "%30-40 hedefi tavanın yarısından
azı" diye ekledim. **İkisi de yanlış.** %74.5, decode'un tile başına koşan payı;
ulaşılabilir tasarruf değil.

Doğru aritmetik: j'de gruplar 0..j−1 (blok 0..2j−1) her zaman koşar. En erken
izin verilen çıkış grup j, yani blok 0..2j+1 koşar ve 10−2j blok atlanır:

| j | atlanan blok | maks tasarruf |
|---|---|---|
| 1 | 8 | %59.6 |
| 2 | 6 | **%44.7** (onarım sonrası %43.4) |
| 3 | 4 | %29.8 |
| 4 | 2 | %14.9 |

j=4 için daha önce söylediğim %14 doğruymuş; j=2 için %74.5 değil **%43.4**.
Hem cost modeli hem saat bunu doğruluyor.

**Sonucu önemli:** kullanıcının %30-40 hedefi tavanın yarısı değil, **%70-92'si**.
Tile'ların çoğunun en sığ izinli çıkışa (grup 2) gitmesi ve derinlik kaybının
adapter'larca telafi edilmesi gerekiyor. Hâlâ mümkün — j2/256'da saf dikiş
qp32'de 0.071 dB olduğu için −0.1 dB bütçesinin çoğu derinliğe kalıyor, ve Stage A
adapter'ları daha önce +2.235 dB'ye kadar getirmişti — ama "rahat" değil, "tam
sınırda". İki gün sonra değil şimdi söylenmesi gereken bir şey.

---

## 40. Pohpohlayıcı noktayı seçmeyi engelleyen kodun kendisi onu seçiyordu

`scripts/finish_run.sh`'e CTC geçişi için router seçen bir parça yazmıştım ve
yanına şunu not düşmüştüm: *"pohpohlayıcı noktayı sessizce seçmek bir frontier'ı
yalana çevirmenin yoludur"*. Kod tam olarak onu yapıyordu.

Seçici `dpsnr` ve `router` sütunlarını okuyordu. `collect_frontier.py` ise
`beta / qp / saving_pct / psnr_loss_dB / exit_share / control_max_diff` yazıyor.
Yani:

- `dpsnr` yok → `.get(...,0.0)` → her satır için `abs(0.0) <= 0.1` **doğru**
  → dB filtresi hiçbir şeyi elemiyor,
- seçim "en yüksek tasarruf, maliyeti ne olursa olsun"a çöküyor,
- `router` yok → yol boş → CTC hiç koşmuyor, "router yok" diye kafa karıştırıcı
  bir mesaj basılıyor.

Sessizce yanlış sayı üretmedi (yol boş kaldığı için hiç koşmadı), ama üretmesi
an meselesiydi: `frontier.tsv` bir gün `router` sütunu kazansa, filtre hâlâ ölü
olduğu için bütçeyi aşan bir nokta seçilip CTC'ye gönderilirdi.

**Düzeltmeler:**

1. Sütun adları artık **iddia ediliyor**; eksikse `FRONTIER-SCHEMA-DEGISTI`
   basıp çıkıyor. Eksik anahtarın 0.0'a düşmesi bu hatanın mekanizmasıydı.
2. Doğru sütun `psnr_loss_dB`.
3. **Beta başına en KÖTÜ qp ile toplama.** tsv satırları (beta, qp) çifti; bir
   beta bir router demek ve o router bütçeyi tüm QP aralığında sağlamalı, en
   elverişli QP'sinde değil. Satır bazında filtrelemek, tek bir QP'de 0.1 dB'nin
   altına düşen bir router'ı "bütçe içinde" ilan ederdi.
4. Hiçbir beta bütçeye girmiyorsa, sessizce en iyisini seçmek yerine
   `NONE-IN-BUDGET` diye söylüyor.

Ders: bir tehlikeye karşı yorum yazmak o tehlikeyi önlemez. Sütun adları
doğrulanabilirdi ve doğrulanmamıştı.

---

## 40. Oracle: hedef ulaşılabilir, ama yönlendirmenin değeri QP ile artıyor

Router'ı beklemeden, bu decoder'la **herhangi bir** router'ın ulaşabileceği üst
sınırı ölçtüm (`scripts/oracle_diagnostic.py`, WD-j2/128 epoch-0 checkpoint'i).
−0.1 dB bütçesine denk gelen satırlar:

| QP | ulaşılan dB | oracle tasarruf | tek-derinlik | fark |
|---|---|---|---|---|
| 0 | 0.101 | **42.3%** | 42.2% | +0.2pp |
| 32 | 0.091 | **37.0%** | 30.2% | +6.8pp |
| 63 | 0.102 | **29.3%** | 15.6% | +13.7pp |

**Hedef (%30-40, −0.1 dB) qp0 ve qp32'de karşılanıyor, qp63'te kıl payı altında** —
ve bu yalnızca **bir epoch** eğitim almış checkpoint.

### Beklemediğim bulgu: qp0'da yönlendirme hiçbir şey katmıyor

0.1 dB bütçesinde qp0'da tek tip sığ decoder da %42.2 veriyor; oracle'ın %42.3'ü
yanında fark +0.2pp. Yani **düşük kalite hedefinde içeriğe uyarlanmaya gerek yok**,
uniformca sığ bir decoder aynı işi görüyor. Yönlendirme parasını yüksek kalitede
kazanıyor: qp63'te +13.7pp, qp32'de +6.8pp.

Bu, projenin hikâyesini değiştiriyor ve dürüst olan bu: FLEX-UF'in katkısı
"her yerde daha iyi" değil, "**yüksek kalitede, tek bir derinliğin veremediğini
verir**". qp0'da tavanın zaten tek derinlikle alınabildiğini söylememek, sonucu
olduğundan geniş göstermek olurdu.

(Daha sıkı bütçelerde qp0'da da fark var — tau=0.1'de oracle %36.0'a karşı
uniform %11.8, +24.2pp. Karşılaştırma **eşit ulaşılan dB'de** yapılmalı, tau'da
değil; tau bir eşik, sonuç değil.)

### Ve bu bir üst sınır, başarı değil

Oracle her tile'ın doğru çıkışını bilerek seçiyor. Gerçek router bunu sinyallerden
bulmak zorunda. Açık soru: bu %37'nin ne kadarı yakalanabiliyor. Dönen frontier
taraması tam olarak onu ölçüyor.

### Tanı scriptinin kendisi yanıltıcıydı

`oracle_diagnostic.py` sinyal korelasyonlarını **router'ın artık kullanmadığı**
eski elle yapılmış latent istatistikleri üzerinden basıyordu (en iyisi r = −0.202).
Router bölüm 20'den beri stem sinyallerini kullanıyor. Hiç basmamaktan kötüydü:
"sinyaller oracle'ı göremiyor" diyordu — kullanılmayan sinyaller hakkında — ve
araştırmayı çözülmüş bir probleme geri gönderirdi. Gerçek sinyalleri raporlayacak
şekilde düzeltildi, ve `STEM_NAMES` sinyalleri üreten fonksiyonun yanına kondu:
etiketlediği şeyden ayrı yaşayan bir isim listesi sessizce kayar, ve yanlış
etiketli bir korelasyon hiç korelasyon olmamasından kötüdür.

---

## 41. En derin çıkış gerçek DCVC-UF'ten kaymış — recipe'yi yanlış noktadan uyguluyordum

Kullanıcı doğru soruyu sordu: "kusursuz bir router ile gerçek DCVC-UF'e göre %37
tasarruf 0.1 dB ile, doğru mu?" Cevap **hayır**, ve sebebi ölçülebilir.

Oracle'ın %37'si (bölüm 40) **bizim kendi en derin çıkışımıza** karşı. O çıkış
warm-start'ta bit-exact stok UF'ti, ama eğitim onu taşıyabilir.
`scripts/anchor_drift.py` — donuk encoder olduğu için latent birebir aynı
(`max|diff| = 0.0` ile doğrulandı), tek fark sentez:

| qp | stok DCVC-UF | bizim en derin | kayma |
|---|---|---|---|
| 0 | 33.252 | 33.100 | **−0.153** |
| 32 | 38.981 | 38.781 | **−0.201** |
| 63 | 43.827 | 43.564 | **−0.263** |

**Tek epoch'ta.** Yani gerçek DCVC-UF'e karşı iddia %37 @ 0.1 dB değil,
%37 @ ~0.29 dB. Manşet dört kat yanlış olurdu.

### Sebep: recipe doğru, uygulandığı nokta yanlış

`get_training_strategy()` birebir kopyalanmıştı — ama **epoch 0'dan** okunuyordu:

    [0,   2e-4, 256] × 45   <- koşularımız burada
    ...
    [69,  1e-5, 256] × 20
    [90,  2e-4, 512] × 5 ... [103, 1e-6, 512]

2e-4 **sıfırdan eğitimin başlangıç** lr'si. Ama `--pretrain` bize bu tablonun
**105. epoch'unun çıktısını** veriyor. Yakınsamış bir modele başlangıç lr'si
uygulamak onu optimumundan tekmeler — kayma tam olarak bu. Microsoft'un recipe'sine
uymak, onu *doğru yerinden* okumak demek.

### Üstelik önlemi zaten yazmıştım ve kapalı bırakmıştım

`--anchor_weight`, en derin çıkışı yayınlanmış decoder'a MSE ile bağlıyor
(`loss += w · λ̄ · ||deep − released||²`). Varsayılanı 0.0 ve hiçbir koşuda
verilmemiş. Kullanılmayan yolda ölü bir hata da birikmişti: `load_flexuf_state`
import edilmemişti, ilk kez açtığımda `NameError` ile patladı. **Varsayılan olarak
kapalı bir güvenlik mekanizması, olmayan bir mekanizmadır.**

### Düzeltme — dört warm-start koşusu yeniden başlatıldı

| ayar | değer | neden |
|---|---|---|
| `--epoch_offset 75` | lr 1e-5 @ 256px | recipe'nin ince ayar rejimi; devam eden eğitimin doğru yeri |
| `--new_lr_scale 20` | adapter'lar 2e-4 | sıfır-başlatmalı modüller yakınsamış gövdeyi koruyan lr'de hiçbir şey öğrenemez; tek lr ikisine birden hizmet edemez |
| `--anchor_weight 1.0` | — | en derin çıkış yayınlanmış decoder'a bağlı |

Başlangıçta `anchor_mse = 0.0` — yani pin tam yerinde. Eski koşular
`*.drifted` olarak saklandı, silinmedi.

Ayrıca `scripts/launch_wdec.sh`: `setsid` ile başlatılıyor. `nohup` yalnızca
SIGHUP'ı engelliyor; süreç grubunu öldüren bir denetleyici shell dört koşuyu
birden götürdü, bir kez oldu.

### Yan cevap: kalite noktası başına ayrı eğitim gerekmiyor

DCVC-UF tek model, 64 QP seviyesi (`q_scale_enc/dec` qp ile indeksleniyor).
Eğitim her görüntüye rastgele qp örneklüyor, router da qp-koşullu
(`router.assign(sig, qp)`). qp0/32/63 tablolarının hepsi **aynı modelden**.

---

## 42. Router: vekil hedefi bırakıp doğrudan oracle'a olan mesafeyi minimize etmek

Kullanıcı "router'a yeni bir şey dene, loss'u değiştirebilirsin, oracle'a
yaklaşman lazım" dedi. Mevcut hedefin neden yaklaşamadığı yapısal.

**Sorun.** Oracle her tile için `k* = argmin_k(mse_k + λ·C_k)` seçiyor. Router'ın
işi bunu sinyallerden üretmek. ClassSR'ın Eq.(2)'si bu işi *ifade etmiyor*:
beklenen bozulmayı minimize ediyor, sonra eğitimdeki yumuşak karışım ile
çıkarımdaki argmax arasındaki uyumsuzluğu **onarmak için** Class-Loss ekliyor, ve
çökmeyi engellemek için Average-Loss ekliyor. Üç vekil, üç ağırlık, ve hiçbiri
karşısında ölçüldüğümüz büyüklük değil.

**Çözüm — beklenen pişmanlık.** Ölçüldüğümüz büyüklüğün adı var:

    L = Σᵢ Pᵢ · [ (mseᵢ + λ·Cᵢ) − min_k (mse_k + λ·C_k) ]

Her terim negatif olamaz; **tam olarak** P bütün kütlesini k*'a koyduğunda sıfır;
ve **değeri** router'ın oracle'a karşı ödediği fazla Lagrange maliyeti. Yani hem
doğru hedef hem doğru ilerleme ölçüsü: "0.004" demek "oracle'ın 0.004 üstünde"
demek, üstelik frontier'ın kendi biriminde. λ'yı süpürmek frontier'ı β'nın
yaptığı gibi tarıyor, ama her nokta artık üç vekilin dengesi değil, iyi tanımlı
bir problem.

**Class-Loss atıldı, çünkü yamaladığı uyumsuzluk kaynağında yok ediliyor.**
`hard=True` ile ileri geçiş Gumbel-Softmax straight-through örneği kullanıyor:
eğitim de çıkarım gibi karar veriyor — tek çıkış, seçilmiş — gradyan yine yumuşak
olasılıklardan akıyor. Bu, uzamsal-uyarlanabilir çıkarımda ayrık kapılar için
standart çözüm (Verelst & Tuytelaars, CVPR 2020, arXiv:1912.03203) ve oradaki
yapı — her uzamsal birim için hesaplama harcanıp harcanmayacağını seçen küçük bir
kapı — bizimkinin aynısı.

**Average-Loss korundu ama varsayılan KAPALI.** ClassSR'ın ihtiyacı var çünkü
onun dalları ayrı ağlar ve kullanılmayan dal hiç gradyan almıyor. Bizim
çıkışlarımız yapı gereği ağırlık paylaşıyor ve router eğitimi sırasında decoder
donuk, yani kullanılmayan bir çıkış hiçbir şeyi bozmuyor; kullanımı zorlamak,
hedefin "yanlış" dediği çıkışlara tile göndermek olurdu. Yine de `w_avg > 0` ile
açılabiliyor, çünkü bu gerekçe inanılmak yerine kontrol edilmeli.

**Kontroller (sentetik, kapalı formda doğrulanabilir):**

    kusursuz router  regret = 0.000e+00   oracle_agree = 1.000
    kotu router      regret = 2.087e-01   oracle_agree = 0.000
    gradyan logitlere ulasiyor |g|max = 1.638e-03

**İlk gerçek koşu** (warm-start checkpoint'i, 100 adım, yalnızca makinenin
çalıştığını göstermek için — adapter'lar hâlâ sıfır-başlatmalı, mutlak sayılar
anlamlı değil): `oracle_agree` 0.297 → 0.438 (rastgele 0.25 olurdu), regret
0.084 → 0.061.

**`oracle_agree` raporlanıyor ama optimize EDİLMİYOR**, ve bu bilinçli: iki
neredeyse eşit çıkış arasında bölünen bir router az pişmanlık öder ama seçimi
yanlış yapar, yani kayıp düşerken bu metrik takılabilir. İkisi birlikte
loglanıyor, hiçbirine tek başına güvenilmiyor.

---

## 43. Canvas-coupled tiling — dikişi onarmak yerine oluşmasını engellemek

Ve bu, bölüm 11'deki kendi kararımı çürütüyor.

**Çürütülen karar.** Trunk halo'yu "tam decode'un 1.745 katı" diye reddetmiştim.
O ölçüm halo'yu **bütün bloğa** uyguluyordu, yani halodan hiçbir fayda görmeyen
%99.666'ya da (P+2h)²/P² çarpanını ödüyordu. Yanlış şeyi halolamışım.

**Gözlem.** `DepthConvBlock` = 1×1 → WSiLU → **3×3 depthwise** → 1×1, sonra
1×1 → act → 1×1. C=384'te 1,035,648 MAC/px'in yalnızca **3,456'sı** depthwise:
**%0.334**. Diğer her operatör pointwise ve komşusunun ne tuttuğunu hiç
umursamıyor. Yani trunk'ın tam olarak %0.334'ü dikişin **tüm** sebebi.

| tile | tüm bloğa halo | sadece depthwise |
|---|---|---|
| 128px | +%26.6 blok | **+%0.066 decode** |
| 256px | +%12.9 blok | **+%0.032 decode** |

GridSeamRepair %0.951'e mal oluyordu — bu 14-30 kat ucuz.

**Uygulama** (`flexuf/backbone/coupling.py`): her tile'ın depthwise-öncesi
aktivasyonu tek bir paylaşılan canvas'ta yaşıyor; her depthwise o canvas üzerinde
çalışıyor, yani tile'ın kenarı komşusunun **gerçek** değerini okuyor. Çıkmış
tile'lar son aktivasyonlarını yerinde bırakıp bedavaya bağlam olmaya devam
ediyor.

**Ölçüm 1 — aynı derinlikte dikiş var olmayı bırakıyor:**

    128px, 16 tile, hepsi ayni derinlikte:  tam kare − parcali  max|diff| = 0.0
    256px,  4 tile, hepsi ayni derinlikte:  tam kare − parcali  max|diff| = 0.0

Yaklaşık değil, birebir — birleştirilmiş canvas üzerindeki depthwise, zaten
tam-kare decoder'ın çalıştıracağı depthwise'ın kendisi.

| konfig | qp0 | qp32 | qp63 |
|---|---|---|---|
| 128px replicate | 0.1462 | 0.2299 | 0.3663 |
| 128px **coupling** | **0.0000** | **0.0000** | **0.0000** |
| 256px replicate | 0.0767 | 0.1296 | 0.2320 |
| 256px **coupling** | **0.0000** | **0.0000** | **0.0000** |

**Ölçüm 2 — ama karışık derinlikte DAHA KÖTÜ, ve bunu saklamıyorum:**

| konfig | qp0 | qp32 | qp63 |
|---|---|---|---|
| 128px replicate | 1.5305 | 2.3521 | 3.8408 |
| 128px coupling | 1.6522 | 2.4669 | 3.9774 |
| 256px replicate | 1.4115 | 2.1815 | 3.6411 |
| 256px coupling | 1.4667 | 2.2337 | 3.7001 |

Sebep yapısal: derin bir blok, sığ komşusunun **altı blok önceki** özelliğini
okuyor ve o özellik onun için dağıtım dışı. Replicate ise tile'ın kendi kenarını
kopyaladığı için istatistik tutarlı kalıyor. (Bu satırlardaki mutlak değerler
derinlik kaybını da içeriyor; karşılaştırılabilir olan iki satır arasındaki
**fark**, çünkü çıkış haritası ve tohum aynı.)

Uyarı: ölçüm warm-start checkpoint'inde, yani sığ çıkışlar ham kesilmiş UF —
derinlikler arası uyumsuzluğun mümkün olan **en kötü** hali.

**Ölçüm 3 — saat maliyeti, MAC'ten büyük (arls dersi tekrar):**

    128px  replicate 228.8 ms  ->  coupling 241.2 ms  (+5.4%)
    256px  replicate 254.4 ms  ->  coupling 263.4 ms  (+3.5%)

MAC +%0.03, saat +%3.5. Fark eager PyTorch'un unpatch/patch yeniden şekillendirme
ve tam-canvas depthwise ek yükü. Yine de takas arls'inkinin tersi: %3.5 karşılığı
qp63'te 0.232 dB'nin **tamamı**, ve routing'in hiçbir derinlikte veremeyeceği bir
şey — tam kare kalitesi.

**Hipotez ve deney.** Karışık derinlikteki kayıp, çıkışların özellik
dağılımlarının uyuşmamasından geliyor. Ladder distillation (bölüm 42) tam olarak
onları birbirine benzetmek için yazılmıştı. `runs/coupled_j2_p256` ikisini
birlikte koşuyor: ikincinin birinciyi düzeltip düzeltmediği deneyin sorusu.

---

## 44. HEADS-ONLY — anchor'ı kayıpla değil, YAPIYLA garanti etmek

Bütün gün en derin çıkışı gerçek DCVC-UF'te tutmaya çalıştık: önce fark ettik
kaydığını, sonra `--anchor_weight` ile bir kayıp terimiyle bağladık. Ama kayıp
terimi bir *baskı*, garanti değil.

`runs/heads_only_j2_p256`: `--freeze_backbone`. Gövdenin tek bir tensörü
optimizer'da değil — **3,848,704 / 46,028,032 parametre (%8.36)** eğitiliyor, o
da yalnızca exit head'leri. En derin çıkış yayınlanmış decoder'ın ta kendisi
olmaya **yapısal olarak** mahkûm; kaymak için değişebilecek bir ağırlık yok.
`--anchor_weight` bu koşuda anlamsız olduğu için verilmedi.

Karşılığında head'ler büyüdü: her çıkışta `FFNAdapter`, yani atlanan blokların
gerçekten içerdiği expand/activate/contract dizisinin aynısı. Başka hiçbir şey
kapasite için yarışmadığından bunu karşılayabiliyor.

Bu koşu aynı zamanda bir üst-sınır ölçümü: **gövdeye hiç dokunmadan** ne kadar
gidilebiliyor. Eğer buradaki sonuç gövdeyi de eğiten koşulara yakınsa, gövdeyi
eğitmenin riski (kayma) getirisini karşılamıyor demektir.

---

## 45. Router kör — ve zengin temsil kurtarmıyor (hipotezim yanlıştı)

İlk regret sweep'i **sabit** router üretti: λ=0.0002'de bütün tile'lar en derin
çıkışa, λ=0.002'de hepsi en sığa. Arada karışım yok. Sebebini aradım.

**Sinyal korelasyonları** (warm-start ckpt, qp32, düzeltilmiş tanı scripti — daha
önce eski elle yapılmış sinyalleri raporluyordu):

    stem_max +0.034 | stem_energy +0.071 | stem_std +0.077
    scales_max -0.097 | scales_mean -0.117 | y_energy +0.047

Hiçbiri |r| = 0.12'yi geçmiyor. Oysa **oracle değişiyor** (std 0.348 çıkış),
yani içerikte ayrışacak şey var, kayıp sinyal tarafında. (Not: daha önce
"stem_max r=+0.440" demiştim; o eski bir ölçümdü, bu checkpoint'te geçerli değil.)

**Hipotezim:** darboğaz. Tile başına 6 skaler, 384 kanallı 32×32 bir tile'ı
tarif etmeye yetmiyor. Stem zaten hesaplanmış, havuzlamak bedava.

**Ölçüm çürüttü.** `scripts/signal_probe.py`, doğrusal probe, %70 fit / **%30
ayrılmış**:

| λ | oracle dağılımı | 6 skaler | 768 havuz | taban |
|---|---|---|---|---|
| 3e−4 | [0,1,33,40,320,246] | **0.719** | 0.609 | 0.500 |
| 1e−3 | [0,1,400,27,184,28] | 0.672 | 0.740 | 0.625 |
| 3e−3 | [0,1,609,0,30,0] | 0.938 | 0.969 | 0.952 |

768 boyutlu temsil ilgilendiğimiz noktada 6 skalerden **daha kötü**. Darboğaz
değilmiş.

**Ve script'in ilk hâli tam tersini söylüyordu.** Ayrılmış küme olmadan 768 havuz
her λ'da **1.000** veriyordu — 160 tile'a karşı 769 özellik, probe ezberliyordu.
Bu kadar temiz bir sayıya inanmadan önce şüphelenmek gerekiyordu; şüphelendim.

**İsabet teslimat değil, asıl soru yanlış tile'ın NEREYE gittiği.** Aynı ayrılmış
tile'larda gerçekleşen (tasarruf, dB):

| λ | oracle | probe (6 skaler) |
|---|---|---|
| 3e−4 | 10.9% / +0.431 dB | 8.6% / +0.360 dB |
| 1e−3 | 31.7% / +2.123 dB | 37.8% / **+3.379 dB** |
| 3e−3 | 41.9% / +3.728 dB | 43.8% / +4.395 dB |

Probe frontier'ın **üstünde değil, dışında**: daha çok tasarruf edip orantısız
daha çok kaybediyor. Yanlış seçtiği tile'lar komşu çıkışa değil uzağa gidiyor.

**Bu sayılara ne kadar anlam yüklenmeli.** Ölçüm warm-start checkpoint'inde,
sıfır-başlatmalı adapter'larla. dB kayıpları projenin 0.1 dB bütçesinin **4 ila
37 katı** — yani ilgilendiğimiz rejimin tamamen dışında. Bu, işlerin ne kadar
kötü olabileceğinin üst sınırı; tahmin değil. Eğitilmiş merdiven üzerinde
tekrarlanacak, ve sıfır-başlatmalı bir merdiverde ölçülen ayrışabilirliğin
eğitilmişinkiyle aynı olması için hiçbir sebep yok.

---

## 46. HEADS-ONLY sonucu: anchor yapıyla garantilendi, ama tavan çöktü

İlk gerçek checkpoint `runs/heads_only_j2_p256/ckpt_epo0.pth.tar` (gövde tamamen
donuk, yalnızca exit head'leri eğitildi, FFN adapter). Değerlendirme
checkpoint düşer düşmez kendiliğinden çalıştı.

**Anchor — bütün günün açık sorusu, kapandı:**

    qp    stok UF   bizim en derin   kayma
     0    33.260    33.260          +0.000
    32    38.984    38.984          +0.000
    63    43.828    43.828          +0.000

**Tam sıfır.** Gövdeyi dondurmanın vaadi buydu: en derin çıkış yayınlanmış
DCVC-UF'in *kendisi*, kaymak için optimizer'da tek bir tensör yok. Bütün gün bir
kayıp terimiyle *baskı* uyguluyorduk; bu **garanti**. (Karşılaştırma: aynı ölçüm
düzeltme öncesi tek epoch'ta −0.153/−0.201/−0.263 dB veriyordu.)

**Ama oracle tavanı, 0.1 dB bütçesinde:**

| qp | oracle tasarrufu | tek-derinlik |
|---|---|---|
| 0 | %11.5 | %4.7 |
| 32 | **%0.4** | −%0.7 |
| 63 | −%1.0 | −%1.0 |

Gövdesi de eğitilen epoch-0 checkpoint'inde aynı ölçüm qp32'de **%37.0**
vermişti. Burada %0.4.

**Sebep, ve beklenmesi gerekirdi:** gövde donuk olduğu için sığ çıkışlar
iyileşemiyor. Exit head'leri tek başına altı bloğun kaybını kapatamıyor — FFN
adapter 5C²/px iken atlanan altı blok 6×8C². Üç mertebe fark.

**Sonuç — bu bir üst sınır ölçümüydü ve cevabı olumsuz.** "Gövdeye hiç dokunmadan
ne kadar gidilebilir?" sorusunun cevabı: neredeyse hiç. Yani gövdeyi eğitmek
**zorunlu**, ve anchor yapıyla değil kayıp terimiyle korunmak zorunda. İyi haber:
o terim çalışıyor — diğer dört koşuda 53-55 dB sadakat ölçüldü.

Koşu kapatılmadı. Negatif sonuç da sonuçtur, ve bu tavanın nerede olduğunu tek
başına söyleyen ölçüm odur; ileride "gövdeyi eğitmesek olmaz mıydı" sorusu
sorulduğunda cevabı burada duruyor.

---

## 46. DÜZELTME: %37'lik oracle tavanı kaymış anchor'ın ürünüymüş

İlk eğitilmiş checkpoint (`heads_only_j2_p256/ckpt_epo0`) iki şeyi birden ölçtü.

**İyi olan — anchor artık tam olarak sıfır kaymış.** CTC, 10 kare, aynı encoder
aynı latent (`max|diff| = 0.0` ile doğrulandı):

| qp | stok UF | bizim en derin | kayma |
|---|---|---|---|
| 0 | 33.260 | 33.260 | **+0.000** |
| 32 | 38.984 | 38.984 | **+0.000** |
| 63 | 43.828 | 43.828 | **+0.000** |

Sabah aynı ölçüm −0.153 / −0.201 / −0.263 veriyordu. Fark `--freeze_backbone`:
gövdenin tek bir tensörü optimizer'da değil, yani sapma bir kayıp terimiyle
*bastırılmıyor*, **yapısal olarak imkânsız**.

**Kötü olan — temiz anchor'la oracle tavanı çöktü.** −0.1 dB bütçesinde:

| | kaymış anchor (sabah) | temiz anchor (şimdi) |
|---|---|---|
| qp0 | %42.3 @ 0.101 dB | **%11.5 @ 0.049 dB** |
| qp32 | %37.0 @ 0.091 dB | **%5.3 @ 0.058 dB** |
| qp63 | %29.3 @ 0.102 dB | **~%0** |

**Mekanizma.** Oracle'ın referansı kendi en derin çıkışımız. O çıkış 0.20 dB
*daha kötüyken*, sığ çıkışlarla arasındaki fark küçük görünüyordu — erken çıkış
ucuzmuş gibi. Anchor yerine oturunca erken çıkışın gerçek bedeli ortaya çıktı.
Yani sabahki tablo **kendi bozulmuş modelimize karşı** ölçülmüştü.

Kullanıcı tam bunu sormuştu ("gerçek UF'ye yakınsadığımız tek sonuç pre-trained
ile yaptığımız mıydı") ve ölçüm onu doğruladı.

**Karşılaştırmanın adil olmayan yanı, saklamadan.** Bu checkpoint HEADS-ONLY:
gövde tamamen donuk, sadece exit head'leri eğitiliyor — tasarım gereği **en
kısıtlı** konfigürasyon, çünkü gövde sığ çıkışlara hizmet edecek şekilde yeniden
düzenlenemiyor. Sabahki kol gövdeyi de eğitiyordu ve 128px tile'daydı. Gerçek
sayı ikisinin arasında; gövdeyi eğiten kolların checkpoint'leri ayıracak.

**Ayakta kalan.** Yönlendirmenin kendisi hâlâ kazandırıyor: 0.1 dB'de oracle
%11.5'e karşı tek tip sığ decoder %4.7 — **2.4 kat**. Daha geniş bütçelerde fark
qp32'de +12.4pp, qp63'te +15.0pp. ClassSR'ın dayandığı iddia veride mevcut.

**Ayakta kalmayan.** %30-40 hedefi bu konfigürasyonda ulaşılabilir değil. Gövdeyi
eğiten kollar farklı çıkmazsa hedefin kendisi yeniden konuşulmalı.

Ders, bugün üçüncü kez aynı: **bir referansa göre ölçülen her sayı, o referansın
doğruluğu kadar doğrudur.** arls'te birim, cost modelinde adapter, burada anchor.

---

## 47. 0.3 dB bütçesi hedefi kurtarıyor — ve tabloyu çıkarırken iki hata daha

Kullanıcı "0.1 dB zaten düşük, bir de 0.3 dB limiti koy" dedi. `scripts/budget_table.py`
τ'yu tarayıp **ulaşılan** kayıp bütçenin altında kalan en yüksek tasarrufu
raporluyor — "%22.2 @ 0.320 dB" bir "0.3 dB'de ne kadar?" sorusunun cevabı
değildir.

Oracle (kusursuz router), CTC, `heads_only_j2_p256/ckpt_epo0`:

| qp | ≤0.05 | ≤0.1 | ≤0.2 | **≤0.3** | ≤0.5 | ≤1.0 dB |
|---|---|---|---|---|---|---|
| 0 | 14.2% | 22.4% | 32.3% | **38.9%** | 42.4% | 42.5% |
| 16 | 9.8% | 17.7% | 28.5% | **35.8%** | 42.2% | 42.5% |
| 32 | 8.3% | 13.9% | 23.8% | **30.9%** | 39.8% | 42.3% |
| 48 | 6.6% | 10.7% | 19.0% | **25.4%** | 35.3% | 42.3% |
| 63 | 5.2% | 8.4% | 15.5% | **21.1%** | 29.8% | 41.5% |

0.1 → 0.3 dB geçişi tasarrufu **~1.7 kat** artırıyor ve %30-40 hedefini qp0-32
aralığında karşılıyor.

### İki hata, ikisi de tabloyu uydurmuş olurdu

**1. Bisection ayrık sıçramada yanlış tarafa düşüyordu.** Dört kullanılabilir
çıkış ve birkaç yüz tile ile ulaşılan dB sıçramalarla hareket ediyor; bisection
süreksizliğe yakınsayıp **yanlış tarafını** raporluyordu — "≤0.1 dB" sütunu
0.22 dB gösteriyordu. Izgara taraması + `ulaşılan <= bütçe` filtresi bunu
yapamaz, çünkü kısıt **raporlanan sayının kendisi** üzerinde kontrol ediliyor.

**2. Oracle eşiği yanlış paydayı kullanıyordu.** `M <= deep_mean * 10^(τ/10)`,
yani her tile kendi en derin hatasıyla değil **kare ortalamasıyla**
karşılaştırılıyordu. Sonuç: kolay tile'lar her eşikte geçiyor, zor tile'lar
hiçbirinde — tablo "%0, sonra uçurum, sonra %25" merdiveni veriyordu. Bu içeriğin
değil paydanın özelliğiydi. Doğrusu `M <= M[:, -1:] * 10^(τ/10)`.

Ayrıca ilk ızgara logaritmikti ve 0-1 dB aralığına yalnızca **altı** nokta
düşürüyordu, üstelik alt ucu **negatif τ** üretiyordu — sessizce "her tile en
derin çıkışa" demek, ve tabloyu bulguymuş gibi görünen sıfırlarla doldurmak.

### Görsel rapor

`scripts/plot_checkpoint.py` → `results/checkpoint_report.png`. Eğri ve çubukların
yanında **resimler** de var, çünkü tasarruf sayısı yönlendirmenin yapı bulup
bulmadığını gösteremez. Beauty / qp32: dense 38.32 → yönlendirilmiş 38.25 dB
(**−0.071**), tasarruf **%37.7**, ve çıkış haritası rastgele değil — yüz ve saç
derin çıkışlara, düz arka plan en sığa.

(Resimler ilk denemede macenta-yeşil çıktı: YCbCr düzlemlerini RGB sanıp
çiziyordum. Decoder'da bir hata gibi görünüyordu, oysa çizimde.)

### Doyma sorusu

HEADS-ONLY 53.651 / 759.216 adım = planın **%7.1'i**. Doymuyor: en sığ çıkış
23.42 → 23.93, en derin 27.89 → 28.69. **Ama spread de büyüyor** (+4.46 → +4.76),
yani en derin çıkış sığ olandan daha hızlı iyileşiyor — istediğimizin tersi.
`--freeze_backbone`'un beklenen kısıtı: gövde sığ çıkışlara hizmet edecek şekilde
yeniden düzenlenemiyor. DISTILL kolunda spread +1.2 ve ters yönde; karşılaştırma
o kolun checkpoint'inde kesinleşecek.

---

## 47. İlk dürüst tablo: hedefe ulaşılmıyor, en büyük tek engel anchor kayması

`runs/wdec_j2_p128_grid/ckpt_epo0.pth.tar` — gövdesi eğitilen, anchor kayıp
terimiyle tutulan ilk checkpoint. Değerlendirme kendiliğinden çalıştı.

**Anchor kayması — düzeltme işe yaradı ama yetmedi:**

| qp | düzeltme öncesi | şimdi | HEADS-ONLY (gövde donuk) |
|---|---|---|---|
| 0 | −0.153 | **−0.025** | 0.000 |
| 32 | −0.201 | **−0.051** | 0.000 |
| 63 | −0.263 | **−0.074** | 0.000 |

3.5 kat iyileşme. Ama qp63'te 0.074 dB, projenin **0.1 dB bütçesinin %74'ü** —
daha routing başlamadan harcanmış durumda.

**Oracle tavanı, ve gerçek DCVC-UF'e karşı dürüst hesap:**

| qp | oracle | bize göre dB | +kayma | gerçek UF'e göre |
|---|---|---|---|---|
| 0 | %27.3 | 0.082 | 0.025 | **0.107 dB** |
| 32 | %17.2 | 0.097 | 0.051 | **0.148 dB** |
| 63 | %6.6 | 0.059 | 0.074 | **0.133 dB** |

Hedef %30-40 @ 0.1 dB. Ulaşılan: %27 @ 0.107 (qp0), %17 (qp32), %7 (qp63).
**Ulaşılmıyor.**

**Ama gövdeyi eğitmenin değeri doğrulandı:** aynı ölçüm HEADS-ONLY'de (gövde
donuk) qp0'da %11.5 veriyordu, burada %27.3 — iki katından fazla. Bölüm 46'nın
"gövdeyi eğitmek zorunlu" tespiti sayıyla desteklendi.

**Ve yönlendirmenin kendi katkısı, eşit kalitede tek-derinlik decoder'a karşı:**
+16.1 / +13.4 / +5.6 puan (qp0/32/63). İçeriğe uyarlanma gerçekten kazandırıyor;
bu, projenin premisinin en temiz doğrulaması, çünkü karşılaştırma eşit kalitede
yapılıyor.

**Üç sınırlama, saklamadan:** bu tek epoch; 128px tile (dikişi 256px'in iki
katı); ve daha önce raporladığım %37'lik oracle farklı bir konfigürasyondan
(epoch_offset öncesi j2/128) geliyordu — ikisi karşılaştırılamaz, temiz olan bu
tablo.

**Sıradaki adım, ölçüme dayalı:** anchor kayması en somut engel ve
`--anchor_weight` doğrudan onu hedefliyor. Ama ağırlığı artırmak sığ çıkışların
serbestliğini kısar, yani kaymayı azaltırken tavanı da düşürebilir. Bu bir takas
ve **varsayılmayacak, ölçülecek**.

---

## 48. Gövdeyi eğitmek tavanı 1.6-2.2 kat yükseltiyor — ve anchor'ın bedeli ölçüldü

İkinci checkpoint (`wdec_j2_p128_grid/ckpt_epo0`) "donuk gövde mi kısıtlıyor,
yoksa hedef mi ulaşılmaz" sorusunu ayırdı. Aynı script, aynı CTC verisi, aynı
metrik — bu şart, çünkü ilk karşılaştırmam farkında olmadan OpenImages crop'u ile
CTC'yi yan yana koyuyordu.

**Oracle tavanı** (kusursuz router):

| qp | ≤0.1 dB GRID | ≤0.1 dB HEADS-ONLY | ≤0.3 dB GRID | ≤0.3 dB HEADS-ONLY |
|---|---|---|---|---|
| 0 | **28.2%** | 22.4% | **42.0%** | 38.9% |
| 16 | **26.2%** | 17.7% | **39.7%** | 35.8% |
| 32 | **22.8%** | 13.9% | **35.5%** | 30.9% |
| 48 | **20.1%** | 10.7% | **31.8%** | 25.4% |
| 63 | **18.1%** | 8.4% | **28.2%** | 21.1% |

Gövdeyi eğitmek her QP'de kazandırıyor, ve fark QP arttıkça büyüyor: qp32'de
1.6 kat, qp63'te **2.2 kat**. Donuk gövdenin beklenen kısıtı gerçekmiş — adapter
tek başına, gövdenin sığ çıkışlara hizmet edecek şekilde yeniden düzenlenmesinin
yerini tutmuyor.

**0.3 dB bütçesinde %28-42**, yani hedef tek epoch sonrası (planın %6'sı)
karşılanıyor.

### Bedeli: anchor kayması

| kol | qp0 | qp32 | qp63 |
|---|---|---|---|
| düzeltmesiz (bölüm 41) | −0.153 | −0.201 | −0.263 |
| **GRID** (`--anchor_weight 1.0`) | **−0.025** | **−0.051** | **−0.074** |
| **HEADS-ONLY** (`--freeze_backbone`) | **0.000** | **0.000** | **0.000** |

**Kayıp terimi kaymayı 3.5 kat azaltıyor ama sıfırlamıyor. Dondurmak
sıfırlıyor.** Bir kayıp terimi *baskıdır*, dondurmak *garantidir* — ve artık
ikisinin de sayısı var.

Sonucu okurken bu düzeltilmeli: GRID'in qp32'deki %22.8'i, gerçek DCVC-UF'ten
0.051 dB aşağıda bir referansa göre. Gerçek maliyet ~0.15 dB. Hâlâ HEADS-ONLY'yi
geçiyor, ama fark ham tablodaki kadar büyük değil.

**Sıradaki tasarım sorusu** ikisinin arasında: gövdeyi eğit ama anchor ağırlığını
yükselt. `--anchor_weight` süpürülmesi gereken bir parametre olarak ortaya çıktı;
şu ana kadar tek değerde (1.0) sabitti ve hiç sorgulanmadı.

---

## 48. Router v2: literatür üç kusur gösterdi, uyum 0.56 → 0.86

**v1 neden kördü.** Altı elle yapılmış sinyalden hiçbiri oracle seçimiyle |r|>0.12
korele değildi; 768 boyutlu havuzlanmış temsil daha da kötüydü (bölüm 45). İki
deneme de aynı soruyu soruyordu: *sabit* bir stem'in *hangi sabit fonksiyonu*
doğru çıkışı verir. Stem hiç o soruyu cevaplamak için kurulmamıştı.

**Literatürden üç düzeltme:**

1. **Entropy modelinin ölçekleri girdiye alındı.** En yüksek korelasyonlu iki
   sinyal `scales_mean` (−0.117) ve `scales_max` (−0.097) idi — decode sırasında
   hesaplanıp atılıyorlardı. Latent de öyle. Router'a, tile başına zorluk
   tahmininin ta kendisi olan büyüklük verilmemişti.
2. **Kapasite doğru yere taşındı.** MLP tile başına TEK vektöre uygulanıyor
   (1080p'de 40 tane), yani genişliği bedava; 1×1 ise piksel başına ve tek
   maliyetli parça. v1 ikisini de dar tutup ihtiyatın bedelini iki kez ödemişti.
   v2: stem 1×1 384→48 + latent/scales 1×1 512→32, sonra 256 genişlikte iki
   katmanlı MLP. **144K parametre, %0.162 MAC** (dikiş onarımının altıda biri).
3. **Hedef doğrudanlaştı.** Beklenen pişmanlık minimize edilecek doğru şey ama
   *uyum* istenen şey, ve regret ona ancak sonuçlar üzerinden ulaşıyor. Oracle
   etiketine çapraz entropi doğrudan sinyali veriyor — her tile **kararın
   maliyetiyle** ağırlıklı, ki kapasite önemli yerlere gitsin.

**Çökme kontrolü yardımcı kayıpla değil bias ile** (Loss-Free Balancing, 2024):
büyük yardımcı kayıplar hedefe girişim gradyanı sokuyor. Bias karardan ÖNCE
uygulanıyor ve **oracle'ın kendi dağılımına** doğru itiliyor, tekdüzeye değil —
MoE'de uzmanlar birbirinin yerine geçer, bizde geçmez; yüksek λ'da oracle
gerçekten herkesi tek çıkışa yolluyor ve orada yayılmayı zorlamak hata yaptırmak
olurdu.

**Sonuç, ayrılmış tile'larda** (her batch'in yarısı eğitilir, yarısı ölçülür —
144K parametreyle in-sample sayı istenen yere tırmanır ve hiçbir şey ifade etmez):

    v1 (regret, elle yapilmis sinyaller)   0.562
    v2 (CE + scales + kapasite + bias)     0.863

Ve çökme yok: dağılım baştan sona dört çıkışa yayılı. v1 λ≥3e−4'te
`[0,0,100,0,0,0]`'a çökmüştü.

### Öngörümü ölçüm çürüttü

"%86'nın bir kısmı kıl payı tile'larda kasıtlı kayıtsızlıktır, maliyet-ağırlıklı
uyum daha yüksek çıkar" dedim. **Çıkmadı.** CTC üzerinde:

| qp | uyum | maliyet-ağırlıklı | gerçekleşen regret | tasarruf | oracle |
|---|---|---|---|---|---|
| 0 | 0.743 | 0.721 | 0.00366 | %36.1 | %34.6 |
| 32 | 0.913 | 0.889 | 0.00184 | %40.8 | %41.3 |
| 63 | 0.913 | 0.890 | 0.00367 | %40.7 | %41.8 |

Maliyet-ağırlıklı uyum her QP'de ham uyumun **altında** — router yanlışlarını
ucuz yerlerde değil, **kararın önemli olduğu** yerlerde yapıyor. Tahminimin tersi.

**Ama üçüncü sütun asıl olan ve o iyi:** gerçekleşen regret 0.002-0.004, yani
oracle'ın ödediğinin %0.2-0.4 üstü. Tasarruf oracle'a neredeyse eşit (qp32:
%40.8 vs %41.3). Çelişki yok, açıklama var: router farklı seçtiğinde çoğunlukla
**komşu** çıkışı seçiyor ve komşular birbirine yakın.

**Dürüst sonuç:** kullanıcının istediği %95 uyuma ulaşılmadı (qp32/63'te %91.3,
qp0'da %74.3, ortalama %86.3). Ve ölçüm şunu da söylüyor: **uyum hedeflenmesi
gereken metrik olmayabilir** — router zaten oracle'ın tasarrufunun %99'unu,
maliyetinin %100.4'üne alıyor, ve kalan uyuşmazlığın bedeli 0.002. %95 uyum bunu
kayda değer iyileştirmeyebilir. Yine de istenen o, ve qp0 (%74) açık zayıf halka.

---

## 49. "%95 uyum" hedefi kendi kendini sabote edebilir — ölçüm tuzağı ve korumasi

Kullanıcı router'ın oracle ile **en az %95** uyuşmasını istedi. λ süpürmesi bunu
verdi. Ama vermesi bir şey ifade etmiyor, ve sebebini tabloyla göstermek gerekiyor.

**qp0, oracle'ın en çeşitli olduğu QP:**

| λ | uyum | tasarruf | oracle | oracle entropi (bit) |
|---|---|---|---|---|
| 1e−4 | 0.743 | %36.1 | %34.6 | **1.207** |
| 1.2e−4 | 0.801 | %37.9 | %36.5 | 1.053 |
| 1.5e−4 | 0.842 | %39.6 | %38.6 | 0.837 |
| 2e−4 | 0.873 | %41.4 | %40.9 | 0.547 |
| 3e−4 | **0.952** | %42.9 | %42.6 | **0.268** |

Uyum ile oracle entropisi **mükemmel ters orantılı**. λ=3e−4'te %95.2 elde ediliyor
— ama entropi 1.207'den 0.268'e düşmüş: oracle artık neredeyse karar vermiyor,
router da onunla uyuşmak için karar vermiyor. Tasarruf %42.9, j=2'nin mutlak
tavanı olan %43.4'ün bir tık altı, yani "neredeyse bütün tile'lar en sığ çıkışa".

**Sabit router, sabit oracle ile %100 uyuşur. Ortada yönlendirme yoktur.**

Bu v1'de de olmuştu: λ=3e−3'te `oracle_agree 1.000`, dağılım `[0,0,100,0,0,0]`.
Dağılıma bakmasaydım zafer ilan edecektim.

**Koruma.** `scripts/eval_router2.py` artık her satırda oracle'ın kendi çıkış
dağılımının entropisini basıyor ve 0.15 bitin altında
`<- ORACLE SABIT, uyum anlamsiz` diye işaretliyor. Bir metrik, kendisini
anlamsızlaştıran koşulu yanında taşımadan raporlanmamalı.

**Ve bu, önceki bir paradoksu çözüyor.** qp0'da uyumun en düşük (%74), qp32/63'te
en yüksek (%91.3) olması router'ın qp0'da kötü olması değil: qp0'da oracle
entropisi 1.207, qp32/63'te 0.37-0.48. Düşük uyum **zor problem** işareti.

**Dürüst sonuç:** anlamlı yönlendirmenin olduğu rejimde (entropi ≥ 1 bit) uyum
**%74-80**. Kullanıcının istediği %95'e ancak yönlendirmeyi kapatarak ulaşılıyor,
ve öyle raporlamak teknik olarak doğru pratik olarak yanıltıcı olurdu.

**Bir düzeltme daha:** "1.2e−4 ile 1.5e−4 arasında uyum düzleşiyor" demiştim; o
eğitim-içi ayrılmış ölçümdü (batch başına 48 tile, gürültülü). CTC üzerindeki
gerçek değerlendirme düzleşme göstermiyor, monoton artıyor. Eğitim-içi metriğe
dayanıp sonuç çıkarmak hataydı.

---

## 50. Uzun eğitim qp0'ı kıpırdatmadı — sınır optimizasyonda değil

Bölüm 49'da "%74 uyum kapasite sınırı mı, yarım kalmış eğitim mi" diye sordum.
CE 3000 adımda hâlâ düşüyordu, yani ikisi de mümkündü. Dört kat adımla ölçtüm.

**Eğitim gerçekten ilerledi:** CE 0.5 → **0.0935**, regret 0.0026 → 0.00111.

**Ama CTC'de qp0 kıpırdamadı:**

| | qp0 uyum | qp0 regret | qp32 uyum | qp32 regret |
|---|---|---|---|---|
| 3.000 adım | 0.743 | 0.00366 | 0.913 | 0.00184 |
| 12.000 adım | **0.741** | **0.00366** | 0.934 | 0.00106 |

qp0'da uyum 0.743 → 0.741, regret **birebir aynı**. Model eğitim verisinde dört
kat daha iyi uyuyor ve test performansı sabit — sınır optimizasyonda değil.

qp32'de küçük kazanç var (0.913 → 0.934) ama orada oracle entropisi zaten 0.477;
problem kolay olduğu için kolay kazanç.

**Sonuç: qp0 — oracle'ın 1.207 bit ile gerçekten dağıttığı, yönlendirmenin asıl
anlamlı olduğu yer — router'ın GÖRDÜĞÜ bilgiyle çözülemiyor.** Stem ve latent,
tile'ın hangi çıkışta ne kadar kaybedeceğini yeterince kestirmiyor.

Bu bilgili bir başarısızlık: "daha uzun eğit" yolu elendi, ve elenmesi iki mimari
yolu öne çıkardı —

1. **Stem'in kendisi yönlendirilebilir hale gelsin.** BEST koşusu tam olarak bunu
   deniyor: router'a sabit bir temsili okutmak yerine temsili router'ın işine göre
   şekillendirmek. Bu, bölüm 45'te "sabit bir stem'in hangi sabit fonksiyonu"
   sorusunun yanlış soru olduğu tespitinin doğal devamı.
2. **Router'a daha fazla derinlik göster** — ilk per-tile bloğun çıktısı da
   girdiye girsin. Maliyeti var, ölçülebilir.

Kullanıcının istediği %95'e ulaşılmadı ve artık **sebebi biliniyor**, tahmin
edilmiyor.

---

## 51. Çözüm mimaride değil protokolde: çıkış haritasını İLET, tahmin etme

Bölüm 50 "daha uzun eğit" yolunu eledi: CE 0.5 → 0.0935 inerken qp0 uyumu
0.743 → 0.741, regret beş ondalıkta aynı. Eğitim verisine dört kat iyi uyup test
davranışının hiç değişmemesi **eksik bilginin** imzasıdır.

**Eksiklik yapısal.** Oracle `argmin(mse_k + λ·C_k)` seçiyor ve `mse_k`
**kaynağa** karşı hata. Decoder kaynağı asla görmüyor. Router'dan girdisinde
olmayan bir şeyi çıkarması isteniyordu; hiçbir mimari bunu düzeltmez.

**Ama encoder kaynağı görüyor.** Ve video kodlamada mod kararları zaten iletilir:
HEVC ve VVC blok bölümlemesini, tahmin modunu, dönüşüm ağacını gönderir; decoder
tahmin etmez. Burada alışılmadık olan, decoder tarafında tahmin etmeye
çalışmaktı — konvansiyonel olan değil.

**Ölçüm** (`scripts/signalled_curve.py`, gerçek DCVC-UF referanslı, CTC, harita
maliyeti bpp'ye DAHİL):

| qp | tasarruf | dB | bpp artışı | harita |
|---|---|---|---|---|
| 0 | %24.2 | 0.0776 | 0.000139 | 229 bit |
| 16 | %21.2 | 0.0841 | 0.000141 | 232 bit |
| 32 | %15.7 | 0.0880 | 0.000130 | 217 bit |
| 48 | %11.0 | 0.0878 | 0.000120 | 198 bit |
| 63 | %8.6 | 0.0904 | 0.000105 | 173 bit |

Oracle üst sınırının **1-2 puan içinde** (26.0/19.4/16.5/13.4/9.6). Fark
haritanın faturasından ve 0.1 dB kısıtının o fatura dahil sağlanmasından geliyor.

**Harita entropi kodlu faturalanıyor**, 2-bit en kötü hâliyle değil: dağılım
tekdüze olmaktan uzak ve gerçek bir codec en kötü hâli göndermez. Kendi maliyetini
abartmak da eksik göstermek kadar yanlıştır.

**Router ile karşılaştırma, qp32:**

    tahmin eden router   %24.8 tasarruf,  0.367 dB   <- frontier'in DISINDA
    sinyalli sistem      %15.7 tasarruf,  0.088 dB   <- butcenin ICINDE

Router daha çok tasarruf ediyor gibi görünüp dört kat fazla kalite ödüyor.

**Kullanıcının %95 uyum hedefi bu tasarımda konu dışı kalıyor** — uyum tanım
gereği %100. Saatlerdir kovalanan metrik, doğru soruyu sorunca ortadan kalktı.

**İki uygulama hatası da not:** işi bir kez shell zaman aşımı öldürdü (setsid ile
çözüldü) ve tamponlanmış stdout bunu yirmi dakika gizledi (`python -u`). Ayrıca
λ süpürmesi her adımda `forward_all_exits`'i yeniden hesaplıyordu — tile hataları
λ'dan bağımsız, yani 25 katı boşa iş; önbelleğe alındı.

---

## 52. Neden qp0'da yüksek qp63'te düşük — ve frontier'ın kapalı formu

**Mekanizma, ölçülmüş.** Her çıkışın yayınlanmış DCVC-UF'e göre dB kaybı (bütün
tile'lar o çıkışta, CTC):

| qp | çıkış 2 | çıkış 3 | çıkış 4 | çıkış 5 | bpp |
|---|---|---|---|---|---|
| 0 | 0.345 | 0.152 | 0.063 | 0.023 | 0.194 |
| 16 | 0.435 | 0.218 | 0.082 | 0.033 | 0.216 |
| 32 | 0.615 | 0.332 | 0.110 | 0.047 | 0.257 |
| 48 | 0.864 | 0.482 | 0.139 | 0.057 | 0.328 |
| 63 | **1.157** | 0.631 | 0.158 | 0.060 | 0.456 |

Erken çıkmanın bedeli QP ile **3.4 kat** büyüyor. Sebep bpp sütununda: qp63'te
latent 2.4 kat daha fazla bilgi taşıyor ve derin bloklar tam olarak o bilgiyi
detaya çeviren şey. qp0'da latent kaba, fazladan blokların yapacak işi az.

### Frontier'ın kapalı formu — türetildi, sonra test edildi

Tile'ların p_k oranı çıkış k'ya atansın, Σp_k = 1. O zaman

    C(p) = Σ p_k C_k        (maliyet tile'lar üzerinde toplanır)
    D(p) = Σ p_k D_k        (MSE tile'lar üzerinde ortalamadır)

**İkisi de p'de doğrusal.** Dolayısıyla ulaşılabilir (maliyet, bozulma) kümesi,
K noktanın **tam olarak konveks zarfıdır**; verimli frontier onun sol-alt
zarfıdır. İki sonuç deney gerektirmeden çıkar:

1. **Bitişik köşeler arasında frontier (maliyet, MSE) düzleminde DÜZ bir
   doğrudur** — karışım, eğrilik değil. (tasarruf, dB) düzleminde eğri
   görünmesinin tek sebebi dB'nin MSE'nin logaritması olması.
2. **Lagrange çarpanı eğimin kendisidir:** optimumda aktif köşeler
   λ = −(D_k − D_k')/(C_k − C_k') sağlar. λ süpürmenin zarfı taraması bundandır.

Yani tekdüze-karışım frontier'ı QP başına **2K sayıyla tamamen belirlenir.**

### Test: tahmin vs ölçülen tile-başına oracle

| qp | tasarruf | tahmin dB | ölçülen dB | oracle kazancı |
|---|---|---|---|---|
| 0 | %20 | 0.1004 | 0.0793 | +0.021 |
| 32 | %20 | 0.2038 | 0.1222 | +0.082 |
| 63 | %20 | 0.3615 | 0.2638 | +0.098 |
| 63 | %30 | 0.6833 | 0.4214 | **+0.262** |

Kazanç her yerde pozitif ve **bu hata değil** — Jensen. Tahmin tile'ları sabit
oranlarda **körlemesine** karıştırıyor, gerçek oracle **tile başına** seçiyor.
Fark tam olarak içeriğe uyarlanmanın değeri, yani projenin premisinin ölçüsü. Ve
QP ile büyüyor: qp0'da +0.02, qp63'te +0.26 dB.

### İki kesit (`results/two_views.png`)

Tasarruf sabit → dB maliyeti:

| qp | %15 | %20 | %30 |
|---|---|---|---|
| 0 | 0.046 | 0.079 | 0.131 |
| 32 | 0.079 | 0.122 | 0.292 |
| 63 | 0.158 | 0.264 | 0.421 |

dB sabit → tasarruf:

| qp | 0.1 dB | 0.2 dB | 0.3 dB |
|---|---|---|---|
| 0 | %26.0 | %32.5 | %41.5 |
| 32 | %16.5 | %29.4 | %35.4 |
| 63 | %9.6 | %19.6 | %26.3 |

0.3 dB bütçesinde qp63'te bile **%26.3**, qp0'da **%41.5** — yani hedef bandı
0.1 dB'de değil ama 0.3 dB'de rahatça karşılanıyor.

---

## 53. 256px tavsiyem yanlıştı — ve sebebi tile boyutu değil

Bölüm 36'da "j2/256 diğer konfigürasyonları domine ediyor" demiştim, dikiş
ölçümüne dayanarak (256px dikişi yarıya indiriyor: qp63'te 0.0879 vs 0.1785).
Sinyalli sistem 256px checkpoint'inde ölçülünce **yarı yarıya kötü** çıktı.

İlk açıklamam: "büyük tile heterojenliği ortalıyor, uyarlanma kazancı düşüyor."
Makul geliyordu ve teorimle de uyumluydu (Δ heterojenlikten geliyor). **Ölçüm
çürüttü.**

### 2×2 yalıtım — aynı ağırlık, iki tile boyutu

Dikiş onarımı iki tarafta da kapalı, çünkü `GridSeamRepair`'in kapısı P×P ve
checkpoint'i tile boyutuna kilitliyor (bunu da bu ölçüm sırasında keşfettim;
"merdiven tile-boyutundan bağımsız" demiştim, ızgara kapısı için değilmiş).

Tasarruf, ≤0.1 dB, yayınlanmış DCVC-UF referanslı, qp0/32/63:

| | 128px tile | 256px tile |
|---|---|---|
| **128px ağırlık** | 24.4 / 15.8 / 8.7 | 24.0 / 14.3 / 7.5 |
| **256px ağırlık** | 17.2 / 10.0 / 6.3 | 11.6 / 9.9 / 5.1 |

- **tile etkisi** (128px ağırlıkla, A→B): −0.4 / −1.5 / −1.2 puan
- **ağırlık etkisi** (256px tile'da, B→C): −12.4 / −4.4 / −2.4 puan

Ağırlık etkisi qp0'da tile etkisinin **on katı**. Açıklamam yanlıştı.

(C kolu, daha önce CONTROL için ayrı ölçtüğüm 11.4/9.8/5.0 ile uyuşuyor —
bağımsız tutarlılık kontrolü.)

### Tanı: fark sığ çıkışlarda, ve anchor tersine dönüyor

Çıkış-başına dB kaybı, qp0, `scripts/why_qp.py` her iki checkpoint'te:

| checkpoint | çıkış 2 | çıkış 3 | çıkış 4 | çıkış 5 (anchor) |
|---|---|---|---|---|
| 128px eğitilen | **0.3447** | **0.1516** | **0.0634** | 0.0228 |
| 256px eğitilen | 0.5725 | 0.2953 | 0.0809 | **0.0165** |

256px koşusu **anchor'ı daha iyi korumuş** ama **sığ çıkışları belirgin kötü**.
Tasarruf tamamen sığ çıkışlardan geldiği için fark buradan geliyor.

### Çıkarılacak ders

Dikiş ölçümü doğruydu; 256px dikişi gerçekten yarıya indiriyor. Ama bu
checkpoint'te **bağlayıcı kısıt dikiş değil, sığ çıkış kalitesi**. Darboğaz
olmayan bir şeyi iyileştirip sonucu iyileşmiş saymak, bu projede ikinci kez
oldu (ilki: arls'in dB kazancını maliyetsiz sanmak). Bir iyileştirmenin
**hangi kısıtı** gevşettiğini sormadan konfigürasyon seçmemek gerekiyor.

Sunumun 12. slaydındaki "larger tiles cut the seam but also the adaptivity"
cümlesi de bu yüzden yanlış ve düzeltildi.

## 54. Hedef karşılandı — ve oraya giden yol on üç ölçüm hatasından geçti

BEST bir tam epoch sonunda 0.1 dB'de **%34.7 / 31.4 / 27.4 / 24.3 / 21.5**
veriyor (qp 0…63, 40 CTC sekansı, kare-başına dB, harita maliyeti bitrate'in
içinde). **%30 hedefi qp0 ve qp16'da tutuyor.** Integre BD-saving %30.8,
anchor kayması −0.003 / −0.016 / −0.033 dB.

Neden çalıştığı üç koşuyla ölçüldü — aynı K=6/j=2/256px, aynı warm start,
birer epoch, yalnızca eklentiler farklı:

| anchor_weight | 0.1 dB'de tasarruf |
|---|---|
| 0 (tarif olduğu gibi) | **hiçbir oranda ulaşılamaz** — kayma tek başına 0.146–0.217 dB |
| 1 | 14 / 12 / 9 / 7 / 5 % |
| 10 + ölçekli adaptör + distilasyon + ortak router | 35 / 31 / 27 / 24 / 22 % |

Anchor terimi bir iyileştirme değil, **ön koşul**: onsuz bütçe, tek bir tile
erken çıkmadan kaymaya harcanıyor.

### Ölçüm tarafında bulunanlar

Sayıların doğru olduğuna güvenmeden önce şunlar düzeltildi. Her biri rapor
edilen bir rakamı değiştirecek türdendi. (Sayıyı gün boyunca farklı yerlerde
farklı verdim — 13'ten 16'ya. Doğrusu aşağıdaki liste; sunum/figür tarafındaki
düzeltmeler ayrı ve buraya dahil değil.)

1. **λ ızgara okuması** — "bütçe altındaki en iyi örnek" kuralı taramanın
   nereye örnek koyduğuna bağlı. Gönderilen sistemi 3.1 puana kadar eksik
   gösteriyordu; bisection'a geçildi.
2. **dB konvansiyonu** — iki betik aynı ismi farklı şey için kullanıyordu
   (havuzlanmış vs kare-başına). Fark bütçenin ¼–⅓'ü ve aralarındaki 5.4
   puanın tamamını açıklıyor. Kodek konvansiyonu manşet oldu, ikisi de JSON'da.
3. **Kare sayısı** — biri 1 biri 2 kare kullanıyordu; olmayan bir düşük-bitrate
   gerilemesi rapor ettirmişti.
4. **BD aralığı, iki yönde** — eğri başına türetmek checkpoint'leri
   kıyaslanamaz yaptı; sabitlemek BEST'in ucundan taştı ve üç oranın
   ortalamasını beşle kıyaslattı. `common_interval.py` ikisini de çözüyor.
5. **`np.interp` kırpması** — sınırın dışındaki bütçe için taban değerini
   döndürüyordu.
6. **Korumanın kendi unpack hatası** — `(tasarruf, dB)` sırasını ters açtım,
   her oranı "ulaşılamaz" ilan ediyordu. Çıktıyı okuyarak yakalandı: "13.7 dB"
   hiçbir şeyin desibeli değil.
7. **Kuartik uydurma** — konveksliğin eğrinin %30'unda bozulduğunu söylüyordu;
   artefaktmış. Sekant testi (uydurmasız) her yerde %100 veriyor.
8. **K'ya sabit referans** — K=12 koşusu ölçülemiyordu.
9. **25 kat gereksiz hesap** — λ döngüsü kare döngüsünün dışındaydı.
10. **Alarm, iki kez** — önce saf gürültüde (|t|≈1.6), sonra küçük bir eğilimin
    üstündeki dalgalanmada. Artık epoch boyunca eğim uyduruyor.
11. **"Eğitim yok" etiketli figür** eğitilmiş checkpoint gösteriyordu — hem de
    başından beri. Warm start'ın gerçek sayıları 9 kat farklı (exit 2: 2.796 dB).
12. **`ckpt_step.pth.tar` üzerine yazılıyor** — iki dosya aynı yolu gösterip
    saatlerce farklı ağırlıkları ölçebiliyordu. En sinsi olanı: bütün ara
    kontroller geçiyordu, yalnızca iki bağımsız yolun çeliştirilmesiyle çıktı.
13. **Maliyeti eşit çıkışlar** — j=2 altında exit 0/1/2 aynı maliyette,
    histogram yalnızca birini sayıyordu (%29.5 yerine %33.8).

### Geri aldığım dört sonuç

- BEST128'in düşük-bitrate gerilemesi — kare uyuşmazlığıydı, gerileme yok.
- FINE12'nin anchor'ında kayma — gürültülü izden üç nokta seçmişim.
- BD tablosunda FINE12'nin BEST128'i geçmesi — 12 000 adımı 8 000'e karşı.
- Konvansiyon farkının oranla büyüyeceği tahminim — küçülüyor.

### Sıradaki tura not

Bağlayıcı kısıt yer değiştirdi: baz çizgide exit 2'ydi, BEST'te exit 3 (her
oranda tile'ların ~yarısı orada, en ucuz çıkışın payı %34'ten %2'ye çöküyor).
Ayrıca yardımcı kaybın üçte biri, j-bölünmesi yüzünden exit 2'den fazla tasarruf
edemeyecek çıkışlara gidiyor — ama tile'ların sekizde birinde o çıkışlar
gerçekten daha iyi olduğu için kaldırmak değil azaltmak sorusu. İkisi de deney
sorusu; koşan deneye dokunulmadı.

## 55. Daha fazla eğitim CONTROL'ü bozdu — merdiven en derin çıkışa doğru çöküyor

CONTROL ikinci epoch'unu bitirdi ve 0.1 dB'de tasarrufu **%14.1'den %4.7'ye**
düştü (qp0; her oranda düşüş). Sebebi aramak zorunda kalmadım çünkü epoch 0
checkpoint'i hâlâ diskteydi — çıkarım yerine **doğrudan ölçüm** yapabildim.

Elenenler: anchor kayması neredeyse sabit (−0.023/−0.035/−0.053, öncekiler
−0.015/−0.037/−0.058), taban bütçenin rahatça altında (0.014–0.058 dB), ölçüm
kurulumu aynı.

Çıkış başına dB, epoch 0 → epoch 1:

| qp | exit 2 | exit 3 | exit 4 | exit 5 (en derin) |
|---|---|---|---|---|
| 0 | +0.290 | +0.258 | +0.123 | +0.002 |
| 32 | +0.478 | +0.393 | +0.163 | −0.004 |
| 63 | +0.664 | +0.572 | +0.201 | −0.013 |

**En derin çıkış yerinde durdu, hatta yüksek oranlarda iyileşti; her sığ çıkış
kötüleşti; ve gerileme derinlikle monoton.** Ağ, karşılayamayacağı yardımcı
baskı taşımayan tek çıktıyı, karşılayabileceklerinin pahasına iyileştiriyor.

Merdiven distilasyonu ve ölçekli adaptörler tam olarak bunu önlemek için var ve
CONTROL'de ikisi de yok. İkisini de taşıyan BEST128 aynı aralıkta ters yöne
gitti: 8 000 → 16 000 adımda qp32–63'te +0.8 ile +1.2 puan.

### Pratik sonuç

*"CONTROL'e daha fazla epoch verirsek BEST'e yaklaşır"* **yanlış**. Daha fazla
eğitim onu uzaklaştırıyor. Eklentiler bir hızlandırıcı değil; merdiveni
eğitmenin bir merdivene yakınsamasını sağlayan şey.

Bu, 54. bölümdeki anchor merdivenini tamamlıyor: anchor'sız hedef **baştan**
erişilemez (VERBATIM), eklentisiz ise **zamanla** erişilemez hale geliyor
(CONTROL).

### Çekince

Tek koşu, tek epoch sınırı, ve CONTROL BEST'ten dört şeyde farklı. Çöküşün
eklentiler olmadan **gerçekleştiğini** kuruyor, hangi eklentinin onu önlediğini
değil. Ayrıca `--aux_weight`'in j-bölünmesi altında hâkim çıkışlara giden
üçte birlik payı (bkz. `flexuf/losses.py`) bu tabloda ayrıca incelenmedi.

### 55a. Önceden kaydedilmiş tahmin: VERBATIM daha sert çökmeli

VERBATIM de epoch 1'de (~2 saat sonra biter) ve **hiçbir eklentisi yok** —
anchor yok, distilasyon yok, ölçekli adaptör yok, seam repair yok. CONTROL en
azından anchor'ı 1.0'da ve grid seam repair taşıyor.

Çöküş açıklaması doğruysa VERBATIM'de aynı şekli **daha büyük** görmeliyiz:
en derin çıkış yerinde ya da iyileşiyor, sığ çıkışlar geriliyor, gerileme
derinlikle monoton, ve büyüklük CONTROL'ün qp63'teki +0.664 / +0.572 / +0.201
değerlerinden fazla.

Yanlışlanması: VERBATIM'in sığ çıkışları iyileşirse ya da gerileme monoton
değilse, "eklentisizlik merdiveni çökertir" açıklaması bu haliyle yetersizdir
ve CONTROL'deki düşüş başka bir sebebe bağlanmalıdır.

Tahmin ölçümden ÖNCE yazıldı. Sonradan kurulan açıklama sınama değildir.

### 55b. Mekanizma iddiam bir adım fazlaydı — spread çöküşü görmüyor

55'te "merdiven en derin çıkışa doğru çöküyor" dedim, ki bu yardımcı kaybın
derin çıkışı kayırdığını ima ediyor. Ölçüm bunu **desteklemiyor**.

CONTROL'ün epoch 1'inde eğitim-batch spread'i yükselmedi: −0.11 dB/epoch
(t = −0.6, anlamsız), anchor'ı da düz (t = −0.2). Yardımcı kayıp derin çıkışı
sığ olanların pahasına kayırıyor olsaydı spread yükselirdi.

Ama aynı epoch'ta çıkışlar **CTC karelerinde** ciddi biçimde bozuldu (qp63'te
exit 2: +0.664 dB). Yani bozulma eğitim dağılımında görünmüyor, değerlendirme
dağılımında görünüyor.

Eğitim 512px OpenImages kırpmaları, değerlendirme 1080p CTC kareleri. Bu tabloya
uyan açıklama **sığ çıkışların eğitim dağılımına aşırı uyum sağlaması** — derin
çıkışın onları ezmesi değil.

Ölçülen olgu değişmedi: CONTROL'ün çıkışları CTC'de kötüleşti ve gerileme
derinlikle monoton. Değişen, sebebine dair söylediğim şey. 55'teki "ağ,
karşılayamayacağı yardımcı baskı taşımayan tek çıktıyı iyileştiriyor" cümlesi
veriden bir adım öteye geçmişti.

### Yan sonuç: kurmayı düşündüğüm alarm ölü

Çöküşü oluşurken yakalamak için "spread yükselirken anchor sabit" alarmı
kuracaktım. Kalibre etmek için çöküşün gerçekleştiğini bildiğim epoch'a baktım
ve spread hiç kıpırdamamış. Alarm yanlış şeyi ölçecekti. Kurmadan önce kalibre
etmek, kurduktan sonra güvenmekten iyi.

Eğitim-içi sinyalle çöküşü yakalamak istiyorsak, eğitim dağılımında değil
**tutulan bir doğrulama kümesinde** çıkış-başına kalite ölçmek gerekir. Bu bir
deney kararı, koşan deneye yapılacak bir ekleme değil.

### 55c. Düzeltmemin düzeltmesi: bozulma tutulan veride de var, aşırı-uyum değil

55b'de "sığ çıkışlar eğitim dağılımına aşırı uyum sağlıyor" dedim, gerekçem
eğitim-batch spread'inin kıpırdamamasıydı. **Yanlış.**

Eğitim dağılımından hiç görülmemiş 512 görüntüde (description_val.json, eğitimle
sıfır örtüşme) CONTROL'ün epoch 0 → 1 değişimi:

| qp | küme | exit 2 | exit 3 | exit 4 | exit 5 |
|---|---|---|---|---|---|
| 0 | tutulan | +0.146 | +0.116 | +0.049 | −0.001 |
| 0 | CTC | +0.290 | +0.258 | +0.123 | +0.002 |
| 63 | tutulan | +0.513 | +0.544 | +0.215 | −0.016 |
| 63 | CTC | +0.664 | +0.572 | +0.201 | −0.013 |

Aynı şekil, aynı monotonluk, aynı işaret. Yani **ezberleme değil**: sığ çıkışlar
kendi eğitim dağılımının görülmemiş örneklerinde de geriliyor.

CTC'deki bozulma daha büyük (qp0'da iki kat), qp63'te ikisi eşitleniyor. Yani
fotoğraftan videoya transfer bir **bileşen**, ana etki değil.

Kalan soru: eğitim-batch spread'i bunu neden görmüyor? Muhtemel sebep o
istatistiğin 64 QP'nin rastgele karışımı üzerinden ortalanması — etki QP'ye göre
değişiyor (qp0'da 0.15, qp63'te 0.51) ve karışım onu seyreltiyor. Yani spread'in
kıpırdamaması mekanizmaya karşı kanıt değil, o istatistiğin duyarsızlığı.

### Ne kurulmuş oluyor

Eklentiler olmadan daha fazla eğitim sığ çıkışları **genel olarak** bozuyor.
Mekanizmaya dair 55'teki ifademi ("ağ derin çıkışı diğerlerinin pahasına
iyileştiriyor") hâlâ kanıtlamış değilim — bunun için tutulan kümede eğitim
boyunca çıkış-başına ölçüm gerekir, ki bu bir deney kararı.

İki kez düzelttim: önce olguyu mekanizmayla karıştırdım, sonra yanlış mekanizmayı
seçtim. Kalan ifade en dar olanı ve ölçümle birebir örtüşen: **bozulma gerçek,
derinlikle monoton, ve eğitim dağılımına özgü değil.**

### 55d. Eksik kontrol, ve ne zaman gelecek

Bulgu şu an tek taraflı: bozulmayı eklentileri **taşımayan** bir koşuda ölçtüm
(CONTROL, epoch 0 → 1, iki değerlendirme kümesinde). Aynı ölçümü eklentileri
**taşıyan** bir koşuda yapamadım.

Sebebi mekanik: BEST128 ve FINE12 ara checkpoint yazıyor ama `ckpt_step.pth.tar`
her seferinde üzerine yazılıyor, dolayısıyla 8 000 ve 16 000 adımlardaki halleri
artık diskte yok. BEST'in ise henüz tek epoch checkpoint'i var.

BEST epoch 1'i bitirdiğinde (~15 saat) `ckpt_epo0` korunmuş olarak duruyor ve
epoch 1 `status_latest`'ten terfi edecek — yani **aynı öncesi/sonrası** ölçümü
BEST'te yapılabilecek. Beklenen: bozulma yok ya da çok daha küçük, çünkü
sinyalli ölçümü zaten iyileşiyor (BEST128 8 000 → 16 000 arasında qp32–63'te
+0.8 ile +1.2 puan).

O ölçüm gelene kadar kurulan şey "eklentisiz koşu bozuluyor"dur, "eklentiler
bozulmayı önlüyor" değil. İkisi arasındaki fark, kontrolün varlığıdır.

Not: `ckpt_step`'in üzerine yazılması bugün ikinci kez maliyet çıkardı (ilki:
iki dosyanın aynı yolu gösterip farklı ağırlıkları ölçmesi). Ara checkpoint'leri
adıma göre adlandırmak bunu çözerdi ama koşan deneye dokunmak gerekir; sonraki
tura not.

## 56. Tahmin çürüdü: VERBATIM iyileşti, "eklentisizlik bozar" açıklaması düştü

55a'da VERBATIM'in CONTROL'den **daha sert** çökmesini bekliyordum, çünkü ondan
da çıplak: anchor yok, seam repair yok. Ölçüm tersini verdi.

Epoch 0 → epoch 1, çıkış başına dB (negatif = sürüme daha yakın):

| qp | koşu | exit 2 | exit 3 | exit 4 | exit 5 |
|---|---|---|---|---|---|
| 0 | VERBATIM | −0.025 | −0.011 | −0.023 | −0.026 |
| 0 | CONTROL | +0.290 | +0.258 | +0.123 | +0.002 |
| 63 | VERBATIM | −0.055 | −0.056 | −0.049 | −0.044 |
| 63 | CONTROL | +0.664 | +0.572 | +0.201 | −0.013 |

VERBATIM'in bütün çıkışları iyileşti, üstelik yaklaşık **tekdüze** — derinlikle
monoton hasar yok. Yazdığım yanlışlayıcı birebir gerçekleşti.

**"Eklentiler olmadan daha fazla eğitim merdiveni bozar" açıklamasını geri
çekiyorum.** Eklentisi daha az olan koşu bozulmadı.

### Geriye kalan: CONTROL'e özgü ne var

| | CONTROL | VERBATIM |
|---|---|---|
| schedule konumu | 75 → lr 1e−5, crop 256 | 90 → lr 2e−4, crop 512 |
| `--min_crop` | 512 (crop'u 512'ye zorluyor) | yok |
| `--new_lr_scale` | **20** | yok |
| `--anchor_weight` | 1.0 | yok |
| `--seam_repair` | grid | none |
| `--grad_accum` | 1 (efektif batch 8) | 2 (efektif batch 16) |

En güçlü aday `--new_lr_scale 20`: CONTROL'de adaptörler ve seam repair kapısı
temel lr'nin **yirmi katıyla** eğitiliyor. Epoch 1'de temel lr 1e−5, yani yeni
parametreler 2e−4 görüyor — VERBATIM'in bütün ağı için kullandığı lr kadar, ama
yalnızca yeni parametrelerde ve daha küçük batch ile.

Ama bu bir **hipotez**, ve bugün mekanizma hipotezlerinde üç kez yanıldım. Test
etmenin yolu tek bayrak değiştiren bir koşu; koşan deneye dokunmadan yapılamaz,
dolayısıyla sonraki tura.

### Ne kurulmuş durumda

- CONTROL'ün sığ çıkışları epoch 1'de bozuldu (iki değerlendirme kümesinde,
  ölçülmüş)
- VERBATIM'inkiler bozulmadı, iyileşti (ölçülmüş)
- Sebep "eklentisizlik" **değil** (çürütüldü)
- Sebep bilinmiyor; adaylar yukarıda

BEST'in epoch 1'i (~12 saat) üçüncü veri noktasını verecek: eklentilerin
tamamını ve `new_lr_scale 20`'yi birlikte taşıyor.

---

## 57 — `docs/` yazıldı; yazarken üç ölçüm hatası çıktı

Kullanıcı deney planı, model topolojisi, kalite/dB grafiği ve loss grafiği
istedi. `docs/` altında beş İngilizce belge ve dört figür üretildi
(`scripts/make_docs_figs.py`, `scripts/snapshot_runs.py`).

Bir sistemi baştan anlatmak, onu tarif eden sayıları tek tek doğrulamayı
gerektirdi. Üçü yanlış çıktı — hepsi de mevcut figürlerde ve anlatımda aylardır
duruyordu:

1. **Adaptör tipi.** Topoloji figürü her çıkışın altına "1×1 adapter" yazıyordu.
   BEST `--adapter_kind scaled` kullanıyor: çıkış 0–3'te **FFN** (739,200
   parametre), çıkış 4'te 1×1 (147,840), çıkış 5'te hiç yok. Yani beş
   adaptörün dördü yanlış etiketlenmişti, ve yanlış yönde — FFN, 1×1'in iki
   katı maliyetli (bir `DepthConvBlock`'un 0.249'u vs 0.125'i). Figür artık
   etiketi kurulmuş modelden okuyor, uydurmuyor.

2. **Sinyalleşme maliyeti.** Figür "1.2e-4 bpp" diyordu. Ölçülen `map_bits`
   94 bit/kare; 1920×1080'de bu **4.5e-5 bpp**. 1.2e-4, 1280×720'nin rakamı.
   İki buçuk kat abartılmış, ama her iki değer de bitstream'in yanında
   önemsiz olduğu için sonucu değiştirmiyor.

3. **argv ayrıştırıcısı tek tireli bayrakları yutuyordu.** `snapshot_runs.py`
   yalnızca `--` üzerinden bölünce VERBATIM'in `epoch_offset`'i "90 -e 15",
   `batch_size`'ı "8 -n 8 -e 16" olarak okundu. Figür var olmayan bir schedule
   pozisyonu yazdıracaktı. Düzeltince VERBATIM'in `--grad_accum 2` ile
   **efektif batch 16** kullandığı da ortaya çıktı — diğerlerinin iki katı, ve
   54/55 serisindeki CONTROL-VERBATIM farkları listesine giren yeni bir madde.

Ayrıca `run_tree` figüründe bir string-replace sessizce eşleşmemiş, kod
fallback değerle çalışmış ve **altı koşunun altısını da K=6 warm start'a
bağlamış** gibi görünen bir çizim üretmişti — FINE12 K=12'den geliyor, ve iki
ayrı remap tam olarak bu karışıklığı önlemek için var. Fallback makul bir
görüntü ürettiği için hata sessizdi; yalnızca çizilen PNG'ye bakınca görüldü.

### Düzeltme: `--new_lr_scale 20` hipotezi hakkında

54/55'te "CONTROL'ün bozulmasının en güçlü adayı `--new_lr_scale 20`" dendi.
Bu tur canlı argv'ler dondurulunca **BEST'in de `--new_lr_scale 20` kullandığı**
görüldü. Bu hipotezi çürütmüyor — BEST henüz epoch 1'i bitirmedi — ama BEST'in
epoch 1'inin ne test ettiğini değiştiriyor: tek bayrak değil, "20× lr + güçlü
anchor (10.0)" birleşimi. BEST bozulursa güçlü anchor kurtarmıyor demektir;
bozulmazsa 20× lr tek başına yetmiyor ve zayıf anchor (1.0) şüpheli hale gelir.
Hiçbiri tek bayraklı bir ablation'ın yerini tutmaz.

### Loss platosu sorusu

Kullanıcı "1 epoch sonunda plato oluşuyor mu" diye sordu. Cevap: **loss ~5,000
adımda düzleşiyor, ama loss yanlış soruyu ölçüyor.** `corr(loss, batch bpp) =
0.97` — loss `λ·MSE + bpp` ve rastgele kırpımın bitrate'i ardışık batch'ler
arasında iki kat oynuyor. Sabit CTC karelerindeki tasarruf ise hâlâ tırmanıyor
(BEST128 8k→20k: 12.07→13.38%; FINE12 12k→24k: 15.76→17.55%). Figürdeki panel c
bu yüzden eklendi: loss eğrisinin cevaplayamadığı soruyu cevaplıyor.

---

## 58 — Tasarrufun paydası yanlıştı: her manşet sayı 0.6–0.75 puan iyimser

`docs/` için "sabit tasarruf hedefi kaç dB'ye mal olur" tablosunu çıkarırken her
qp'de tavanın tam **%42.5** olduğunu gördüm. Bu mimari bir tavan (bütün tile'lar
exit 2'de), ama sayı tuhaftı: exit 2'nin maliyeti 0.5809, yani stok kod
çözücüye göre tasarruf 1 − 0.5809 = **%41.91** olmalı. %42.46 çıkması paydanın
1.0 değil **1.0095** olduğu anlamına geliyordu.

Öyleymiş. `paper_curve.py` ve `signalled_curve.py` ikisi de

    saving = 1 - cost[k].mean() / cost[-1]

hesaplıyor, ve `cost[-1]` **bizim merdivenimizin tam derinliği**: deepest exit
`seam_repair="grid"` ödüyor, stok kod çözücü ödemiyor. Yani raporlanan sayı
"erken çıkış, kendi tam-derinlik yolumuza göre ne kazandırıyor" sorusunu
cevaplıyordu; etrafındaki her cümle ise "yayınlanmış DCVC-UF'ye göre" diyordu.
İkisi aynı şey değil ve fark hep bizim lehimize.

Bağımsız doğrulama: `flexuf/cost.py::saving()` zaten payda 1.0 kullanıyor.
Eğrideki bir satırın histogramından kareyi `frame_relative_cost` ile baştan
fiyatladım — X = 0.651298 stok decode. Payda 1.0 → %34.87; payda 1.0095 →
%35.48; eğrinin yazdığı değer %35.48. Kod tabanında iki normalizör varmış ve
raporlama yolu okşayıcı olanı seçmiş.

Düzeltmeden sonra (`scripts/renormalise_saving.py`, saklı eğriler üzerinde tam
cebir, GPU gerekmiyor):

| qp | 0 | 16 | 32 | 48 | 63 |
|---|---|---|---|---|---|
| eski | 34.66% | 31.35% | 27.45% | 24.34% | 21.55% |
| yeni | 34.04% | 30.70% | 26.76% | 23.62% | 20.80% |

Ve tam da "hesaba katılmıştır" dediğim seam-repair vergisini paydadan çıkarıyordu.

### İki sonuç

- **Aynı koşunun checkpoint'leri arasındaki sıralama etkilenmiyor** — dönüşüm
  afin ve monoton.
- **Koşular arası sıralama etkileniyor.** VERBATIM `seam_repair="none"` ile
  çalışıyor, yani paydası zaten 1.0; sayıları hiç oynamıyor. Diğer beş koşu
  0.6–0.9 puan kaybediyor. Yani düzeltilmemiş karşılaştırma seam-repair'li
  koşuları kayırıyordu — **tam da 57'deki CONTROL vs VERBATIM sorusunda**.

`saving_pct` alanına dokunmadım (bütün saklı sonuçlar ve BD karşılaştırmaları
onu kullanıyor); iki üretici de yanına `saving_pct_vs_release` yazıyor artık.

## 59 — "byte-identical bitstream" iddiası fazla güçlüydü

Kullanıcı "router işini nasıl çözdün, encoder'a dokunmadın dimi" diye sordu.
Encoder'a dokunulmadı — altı koşu da `--freeze_encoder`, ve üç değerlendirme
scripti her çalışmada `max|enc diff| == 0.0` iddia ediyor, yani varsayılmıyor,
kanıtlanıyor.

Ama soruyu cevaplarken belgelerde iki ayrı iddiayı birbirine karıştırdığımı
gördüm. Encoder donuk olduğu için **kodlanmış yük** stok DCVC-UF ile bit-aynı.
Dosyanın kendisi ise gönderilen konfigürasyonda bit-aynı **değil**: signalled
sistemde encoder exit haritasını yolluyor, kare başına ~94 bit — bitrate'in
%0.008–0.020'si (qp0'da 94/461,493; qp63'te 90/1,133,348). Ölçülüyor ve
`bpp_added` olarak kaydediliyor, ama "byte-identical" demek yanlıştı.

Doğrusu iki konfigürasyon:

| | dosya | decoder maliyeti |
|---|---|---|
| decoder-tarafı router | byte-identical | +%0.044 |
| signalled harita (manşet sayılar) | yük aynı, +94 bit/kare | 0 |

Her iki durumda da stok kod çözücü yükü okuyabiliyor; FLEX-UF kod çözücüye stok
akış verilirse harita yok, kendi router'ına ya da tam derinliğe düşüyor.

---

## 60 — BEST'in router'ı çökmüş: sabit bir politika, qp63'te oracle ile uyum 0.000

Kullanıcı "bit rate hiç değişmesin de deneyelim" dedi. Sıfır ek bit demek,
kararı encoder'ın yollaması yerine kod çözücünün kendi router'ıyla vermesi
demek. Bu konfigürasyon hiç ölçülmemişti — `paper_curve` oracle'ı,
`signalled_curve` encoder'ı kullanıyor, router'la eğri çıkaran araç yoktu.
`scripts/router_curve.py` yazıldı.

İlk çalıştırmada qp0'da bütçeye hiç oturmadı. Teşhis:

| qp | router (240 tile) | oracle | uyum |
|---|---|---|---|
| 0 | 240 → exit 2 | çoğu exit 2 | 0.838 |
| 32 | 210 → exit 3 | 238 → exit 2 | 0.125 |
| 63 | 239 → exit 4 | 240 → exit 2 | **0.000** |

Ortalama güven 0.9646–0.9998. Router **içeriğe hiç bakmıyor**; qp'den ibaret bir
kural öğrenmiş ("qp yükseldikçe derin çıkış"), yani oracle'ın koşullu değil
marjinal dağılımını. qp0'daki 0.838 uyum bunun aksi delili değil: orada oracle da
neredeyse her şeyi exit 2'ye yolluyor, sabit bir politika onunla kendiliğinden
uyuşuyor.

Sonucu keskin: sabit politika bir eğri değil tek nokta, dolayısıyla bisection
edilecek bütçe yok. qp0'da ulaşılabilen iki tahsis var — %41.9 / 0.234 dB
(bütçenin iki katı) ve %−1.0 / 0.003 dB (hepsi en derin çıkışta; −%1.0 seam
repair vergisi). Arada hiçbir şey yok.

Yani **94 bit bir kolaylık değil, içerik uyarlamasının tamamını taşıyor**, ve
02'deki "encoder işbirliği yapmazsa decoder-tarafı router devreye girer" cümlesi
bu checkpoint için yanlıştı. Belgelerden çıkarıldı.

Bunun bilgi-kuramsal bir sınır olmadığını düşünmek için sebep var: aynı projede
donuk kod çözücüye karşı, oracle'ın kendi çıkış karışımına doğru kayıpsız
yük-dengelemeli eğitilen v2 başlığı eski bir checkpoint'te 0.86–0.93 uyum almıştı
(`results/router2_*.log`). Çöken şey **ortak eğitim** — model.py'nin kendi
docstring'inin uyardığı "hareketli hedefe karşı eğitilen router" riski.

BEST'in donuk kod çözücüsüne karşı v2 başlığı (144,024 param) λ=1.3e-5'te 3000
adım eğitiliyor. λ qp ile bir mertebe değişiyor (4.9e-5 → 4.1e-6), o yüzden
geometrik ortada eğitilip her qp'de beta eğimiyle bütçeye oturtulacak.

Ölçülecek: sıfır ek bitle 0.1 dB'de ne kadar tasarruf, ve signalled
konfigürasyonun ne kadar gerisinde. O fark **değişmeyen bitstream'in bedeli**.

---

## 61 — Tavan mimari ve `j`'ye bağlı; FINE12'ninki BEST'ten 7.5 puan yüksek

Trade-off'u ters yönden ("şu kadar tasarruf kaç dB'ye mal olur") çıkarırken her
qp'de aynı tavana çarpıldığı görüldü: **%41.9**. Bu, bütün karoların exit 2'de
olduğu tahsis — baskılanmamış en sığ çıkış. Her hızda aynı olması eğitimden
değil **split'ten** geliyor: gövde (4 blok, tam kare), head ve seam repair hiçbir
zaman atlanamıyor.

Tavanın merdivene bağımlılığı (saf aritmetik):

| K | j | gövde bloğu | exit j'de çalışan blok | tavan |
|---|---|---|---|---|
| 6 | 1 | 2 | 4 | %56.8 |
| **6** | **2** | **4** | **6** | **%41.9** ← BEST |
| 6 | 3 | 6 | 8 | %27.0 |
| 12 | 4 | 4 | 5 | **%49.4** ← FINE12 |
| 12 | 5 | 5 | 6 | %41.9 |

BEST ile FINE12 satırları iki kez okunmalı: **gövdeleri aynı (4 blok), FINE12'nin
tavanı 7.5 puan yüksek.** Paylaşılan hesapta hiçbir fark yok; fark tanelilikte.
K=12'de her blokta bir çıkış var, karo gövdeden bir blok sonra çıkabiliyor;
K=6'da çıkışlar iki blokta bir, ilk durak iki blok sonra, ve her karo
ihtiyacı olmayabilecek bir blok için ödüyor.

FINE12 bugün ölçülen tasarrufta geride (qp63'te %16.76'ya karşı %20.80) ama 1.38
epoch'a karşı 0.58 epoch görmüş ve 7.5 puan uzaktaki bir tavanı hedefliyor.
Bu, onu şimdi tercih etmek için değil, **dört epoch'a kadar koşturmadan sonuç
çıkarmamak için** bir sebep. Seçim kriteri değişmiyor.

---

## 62 — Sıfır ek bit çalışıyor: değişmeyen bitstream'in bedeli 3.3–6.5 puan

Kullanıcı "encoder'da arama yerine decoder'da tahmin deneyini de yapalım, ikisini
de tut, kâğıtta karşılaştıracağım" dedi. Yapıldı.

`StemRouterHeadV2` (144,024 param), BEST'in **donuk** kod çözücüsüne karşı
λ=1.3e-5'te 3000 adım eğitildi; tutulan veride oracle uyumu **0.848** (BEST'in
ortak eğitilmiş başlığı qp63'te 0.000 veriyordu). Aynı checkpoint, aynı 40 kare,
aynı 0.1 dB bütçe, ikisi de yayınlanmış kod çözücüye göre:

| qp | A: encoder araması, harita sinyalli | B: decoder tahmini, hiçbir şey yollanmıyor | fark |
|---|---|---|---|
| 0 | 34.04% | **30.07%** | −3.96 |
| 16 | 30.70% | 27.44% | −3.26 |
| 32 | 26.76% | 22.66% | −4.09 |
| 48 | 23.62% | 19.35% | −4.27 |
| 63 | 20.80% | 14.31% | −6.49 |

B, router'ın kendi hesabı (%0.163) düşülmüş hâli — `exit_costs()` router'ı
içermiyor, ve onu dışarıda bırakmak 58'deki payda hatasının aynısı olurdu.

**qp0'da B = %30.07: hedef, dosyaya tek bit eklemeden tutuluyor.**

Farkın üst sınır olduğunu söyleyen iki sebep:

1. Router tek λ'da eğitildi; 0.1 dB'ye oturan λ qp0'da 4.9e-5, qp63'te 4.1e-6.
   Değerlendirme eğimi bunu telafi etmek zorunda kalıyor ve β +12.2'den
   −30.7'ye savruluyor. Farkın hız ile büyümesi (3.3 → 6.5) bunun izi.
2. λ router'a girdi değil, qp girdi. λ'yı koşullamak ya da çalışma noktası başına
   bir router eğitmek bu kaybı kaldırırdı. Koşan deneyi değiştirmemek için
   yapılmadı.

Yani 6.49 puanın hepsi "kaynak kareyi görememenin bedeli" değil; bir kısmı
kapatılabilir bir eğitim eksiği.

### İki yol da korunuyor

- A: `scripts/signalled_curve.py` → `results/signalled_*.json` (19 dosya,
  davranışı değişmedi)
- B: `scripts/router_curve.py` → `results/router_*.json`

`watch_ckpts.sh`'a `3b` aşaması eklendi: bundan sonraki her checkpoint ikisini de
**eşleşmiş çift** olarak ölçüyor, böylece kâğıttaki karşılaştırma checkpoint'ler
arası değil aynı checkpoint üzerinde. Zincirdeki B, checkpoint'in kendi
başlığını kullanıyor (çöküşün sürüp sürmediğini izlemek için); donuk kod
çözücüye karşı yeniden eğitim ayrı ve kasıtlı bir deney, yarım saat sürdüğü için
zincirden çalıştırılmıyor.

---

## 63 — A-B farkının çoğu tahmin hatası değil, çalışma noktası uyumsuzluğu

62'de sıfır-bit konfigürasyonunun 0.1 dB'de 3.3–6.5 puan geride kaldığı ölçüldü
ve bunu "üst sınır" diye raporladım. Ayrıştırdım.

`router_curve.py --at_lam` eklendi: bisection ve eğim devre dışı, oracle ile
router aynı λ'da. Sonra router'ın düştüğü dB'de oracle cephesinin ne verdiğine
bakıldı — yani **eşit kalitede** karşılaştırma, tek fark kararın kendisi.

| qp | uyum | eşit kalitede tahmin kaybı | 0.1 dB'deki fark | \|β\| |
|---|---|---|---|---|
| 0 | 0.554 | **7.23** | 3.96 | 12.2 |
| 16 | 0.622 | 3.64 | 3.26 | 3.0 |
| 32 | 0.731 | 2.08 | 4.09 | 12.5 |
| 48 | 0.830 | 0.95 | 4.27 | 23.1 |
| 63 | 0.861 | **0.80** | **6.49** | 30.7 |

**İki bileşen zıt yönlerde gidiyor.** Tahmin kaybı hızla neredeyse on kat
düşüyor; bütçedeki fark yükseliyor. Dolayısıyla qp63'teki 6.49 puan router'ın
tahmin edememesi değil — orada en iyi tahmini yapıyor (%86 uyum, eşit kalitede
0.8 puan kayıp).

Bütçedeki fark |β| ile gidiyor: 3.0→3.26, 12.5→4.09, 23.1→4.27, 30.7→6.49.
Mekanizma: β büyüdükçe maliyet terimi logitleri bastırıyor, tahsis "herkes aynı
çıkışa" doğru dejenere oluyor, ve router'ın tek katkısı olan içerik sıralaması
tam da bu yüzden atılıyor.

Bu bir sınır değil, düzeltilebilir bir eksik gösteriyor: λ router'a girdi değil,
yalnızca qp girdi. λ'yı koşullamak ya da çalışma noktası başına bir router
eğitmek yüksek hızdaki farkın çoğunu geri kazandırmalı.

### Toplamsal değil

Bu iki sayı **bileşenlere ayrılıp toplanamaz** — farklı çalışma noktalarında
ölçülüyorlar. qp0'da tahmin kaybının (7.23) bütçedeki farktan (3.96) büyük
olması çelişki değil: router'ın eğitildiği nokta 0.0605 dB'de, cephenin 0.1
dB'den çok daha dik olduğu yerde. Toplamsal bir ayrıştırma gibi sunmak yanlış
olurdu.

### 63a — Ön kayıtlı tahmin (ölçümden ÖNCE yazıldı)

63'teki mekanizma iddiası şu: qp63'teki 6.49 puanlık fark tahmin hatası değil,
router'ı eğitildiği λ'dan (1.3e-5) bütçenin λ'sına (4.1e-6) sürüklemenin bedeli.

Bunu yanlışlayacak deney: λ=4.1e-6'da ikinci bir router eğit, aynı protokolle
ölç. İddia doğruysa qp63'te |β| küçülmeli ve fark belirgin biçimde daralmalı.

**Tahmin: qp63'te fark 6.49 puandan 3 puanın altına iner.**

Yanlışlanma koşulu: fark 5 puanın üstünde kalırsa mekanizma açıklaması yanlıştır
ve 63'teki yorum geri çekilecek. Bu projede mekanizma hipotezlerinde dört kez
yanıldım; tahmini önceden yazmanın sebebi bu.

### 63b — Tahminin keskinleştirilmesi (yine ölçümden ÖNCE)

Eğitim sırasında router'ın dağılımı en derin çıkışa yakınsıyor gibi görününce
"hedef dejenere, test geçersiz" diye düşündüm. **Yanlış.** curve_BEST'in
histogramları λ=4.4e-6'da gerçek yayılma gösteriyor (qp0'da en derin %62 ama
exit 3'te 291, exit 4'te 220 karo; qp63'te en derin yalnızca %12). CE'nin düşük
olmasından hedefin trivial olduğunu çıkarmak aceleciydi; 24 karoluk tek bir
batch'in argmax histogramına fazla anlam yükledim.

Ama veri tahmini keskinleştiriyor. λ=4.4e-6'nın düştüğü dB:

| qp | 0 | 16 | 32 | 48 | 63 |
|---|---|---|---|---|---|
| dB | 0.0005 | 0.0175 | 0.0414 | 0.0719 | **0.1055** |

Yani yeni router **qp63 için tam bütçede** eğitiliyor, qp0 için çok derinde —
ilk router'ın (λ=1.3e-5, qp32'de doğru) tam aynası. Mekanizma iddiası doğruysa:

- **qp63'te fark 3 puanın altına iner** (63a'daki asıl tahmin, değişmedi)
- **qp0'da fark kötüleşir** — 3.96'nın üstüne çıkar, çünkü bu sefer sürüklenen
  uç orası

İkincisi bir bonus kontrol: mekanizma "eğitim noktasından uzaklık" ise, iki
router'ın hataları qp ekseninde zıt yönlerde eğilmeli. İkisi de aynı yönde
çıkarsa açıklama yanlıştır.

### 63c — SONUÇ: tahmin doğrulandı, ama |beta| yeterli istatistik değil

λ=4.1e-6'da eğitilen ikinci router (tutulan-veri uyumu 0.923) ölçüldü.

| qp | A | B λ=1.3e-5 (fark) | B λ=4.1e-6 (fark) | en iyi |
|---|---|---|---|---|
| 0 | 34.04% | 30.07% (3.96) | 28.53% (**5.51**) | 3.96 |
| 16 | 30.70% | 27.44% (3.26) | 26.87% (3.82) | 3.26 |
| 32 | 26.76% | 22.66% (4.09) | 23.93% (2.82) | 2.82 |
| 48 | 23.62% | 19.35% (4.27) | 22.27% (1.35) | 1.35 |
| 63 | 20.80% | 14.31% (6.49) | **19.30% (1.50)** | 1.50 |

- 63a: qp63'te fark 3'ün altına iner → **6.49 → 1.50, doğrulandı**
- 63b: qp0'da fark kötüleşir → **3.96 → 5.51, doğrulandı**

|β| qp63'te 30.7 → 4.2. Mekanizma açıklaması testi geçti; 57'de dört kez
yanıldığım mekanizma hipotezlerinin aksine bu tutuyor.

**Ama iddiamı zayıflatmam gerekiyor.** 63'te farkın |β| ile "neredeyse birebir"
gittiğini yazmıştım. On noktayı birlikte alınca korelasyon yalnızca **0.663**,
ve ikinci router monotonluğu bozuyor: |β|=13.4'te fark 1.35, |β|=16.0'da 5.51.
Aynı |β|'da çok farklı farklar var. Yani |β| yeterli bir istatistik değil,
"eğitim noktasından uzaklık"ın kaba bir vekili. Verinin desteklediği ifade
şudur: **router'ı çalışma noktasının λ'sında eğitmek önemli, ve en çok
uyumsuzluğun büyük olduğu yerde önemli.** Tek değişkenli bir yasa gibi sunmak
fazla olurdu.

### Kâğıt için asıl sonuç

Hız başına uygun router seçilince değişmeyen bitstream'in bedeli **3.3–6.5 değil,
1.35–3.96 puan**. Beş hız için beş router 5 x 144,024 = 720,120 parametre, yani
45.4M'lik modelin %1.6'sı — dağıtım açısından ihmal edilebilir. Her hızın kendi
λ'sında eğitilmiş beş router muhtemelen qp0/16'da da daha iyisini verir; iki
router'lı zarf bunun alt sınırı.

---

## 64 — Düzeltme: FINE12'nin tavanı %49.4 değil %50.3

61'de tavan tablosunu BEST'in konfigürasyonunu alıp K ve j'yi değiştirerek
hesapladım. FINE12 ayrıca `latent_patch 8` ve `conv1x1` adaptör kullanıyor;
ikisi de halo ve adaptör kalemlerini değiştiriyor. Her koşunun **kendi**
meta.json'undan hesaplayınca:

| koşu | K | j | latent_patch | adaptör | tavan |
|---|---|---|---|---|---|
| BEST | 6 | 2 | 16 | scaled | %41.9 |
| BEST128 | 6 | 2 | 8 | scaled | %41.9 |
| FINE12 | 12 | 4 | 8 | conv1x1 | **%50.3** |

Fark 7.5 değil **8.4 puan**. 61'deki tablo hatalı; belgeler düzeltildi. Hata
sınıfı tanıdık: bir koşunun sayısını başka bir koşunun konfigürasyonuyla
üretmek (bkz. 54'teki referans karışıklığı).

## 65 — "İnce merdiven yüksek qp'yi düzeltir" tahmini şu an DESTEKLENMİYOR

61'de ince merdivenin yüksek hızda avantajlı olacağını savundum: qp63'te K=6'nın
basamakları bütçeyi kuşatıyor (%41.9 çok yıkıcı, %27.0 karşılanabilir) ve K=12
aradaki %34.5'i sunuyor. Mevcut veriyle test ettim — profilin **düzleşmesi**
gerekirdi:

| koşu | tavan | qp0 | qp63 | qp63/qp0 |
|---|---|---|---|---|
| BEST | %41.9 | 34.04 | 20.80 | **0.611** |
| FINE12 | %50.3 | 32.20 | 16.76 | **0.521** |
| BEST128 | %41.9 | 33.87 | 12.56 | 0.371 |

FINE12'nin profili daha düz değil, **daha dik**. Yani tahmin şu an desteklenmiyor.

Ama sonuç çıkarılamaz: FINE12 0.58 epoch görmüş, BEST 1.38. Eğitim süresi ile
karışık, ve az eğitilmiş sığ çıkışlar en çok yüksek qp'de zarar verir — yani
gözlenen diklik eksik eğitimin de imzası olabilir. Dört epoch'ta tekrar
bakılacak. Şimdilik 61'deki iddia "ölçülmemiş beklenti" olarak işaretlendi.

---

## 66 — Quantization: tahmin ÇÜRÜDÜ, mekanizma tam tersi

Plana yazarken şu tahmini kaydetmiştim: kuantizasyon gürültüsü ile erken çıkış
toplamsal-altıdır, çünkü sığ çıkışın ek bir bozulmayı soğuracak kapasitesi daha
azdır; somut olarak qp63 ve 8 bitte exit 2'nin bozulması en derin çıkışınkini
iki kattan fazla aşmalı.

**Ölçüm:** exit 2 = 0.1076 dB, en derin = 0.1113 dB, oran **0.97**. Çürüdü.

fp32'ye göre eklenen dB (ağırlık-yalnız, per-channel, kalibrasyonsuz PTQ,
32 tutulan OpenImages, `scripts/quant_sweep.py`):

| bit | qp | exit 0 | exit 2 | exit 5 | açıklık (e0−e5) |
|---|---|---|---|---|---|
| fp32 | 63 | — | — | — | 2.864 |
| 8 | 63 | 0.124 | 0.108 | 0.111 | 2.876 |
| 6 | 63 | 1.472 | 1.946 | 2.112 | 2.224 |
| 4 | 63 | 6.358 | 7.117 | 7.824 | 1.398 |

Mekanizma tahminimin tersi: kuantizasyon **derin** çıkışlara daha çok zarar
veriyor. En derin çıkış neredeyse kusursuz başlıyor (qp63'te 0.073 dB) ve
eklenen gürültü tabanını soğuracak payı yok; sığ çıkışın hatası zaten attığı
bloklardan geliyor, gürültü onun yanında küçük kalıyor.

### Merdiven için asıl sonuç

Bit azaldıkça **çıkışlar birbirine yakınsıyor**. qp63'te açıklık 2.864 → 2.224 →
1.398. Routing'in sömürdüğü şey tam olarak bu açıklık; o daralınca yönlendirecek
bir fark kalmıyor. 4 bitte altı çıkışın hepsi 1.4 dB içinde.

- **8 bit** neredeyse bedava (qp0'da ≤0.013 dB) ve açıklığı **koruyor** (2.876 vs
  2.864) — iki kaldıraç temiz biçimde birleşiyor. BOPs 0.062×.
- **6 bit ve altı** merdivenin kendisini aşındırıyor; kuantizasyon gürültüsü
  çıkışlar arası farkı bastırıyor.

Bir uyarı: yüksek hızda 8 bit bile tabanı büyütüyor (qp63'te en derin çıkış
0.073 → 0.184, 2.5 kat). Taban zaten bütçenin %30'unu yiyordu (63); 8 bit onu
daha da büyütür. Yani "8 bit bedava" ifadesi **çıkışlar arası açıklık** için
doğru, **taban** için değil.

Bu ölçüm OpenImages 512px üzerinde; CTC'deki taban rakamlarıyla (0.0296 dB)
doğrudan karşılaştırılamaz, yalnızca göreli ifadeler taşınır.

---

## 67 — Anchor'ın λ ölçeklemesi kodun kendi amacına aykırıydı; bayrak arkasına alındı

Yüksek qp'de neden az tasarruf olduğunu ayrıştırırken tabanın bütçenin %30'unu
yediği ölçüldü (qp63'te 0.0296 dB / 0.1 dB). Taban = en derin çıkışın yayınlanmış
kod çözücüden sapması, yani anchor teriminin engellemek için var olduğu şey.
Tabanı sıfırlamak qp63'te **3.28 puan** getirir (curve_BEST'ten, 0.1 dB ile
0.1+taban okunarak).

Anchor terimine bakınca:

    ld["loss"] += args.anchor_weight * lambdas.mean() * anchor_mse

Yorumu şöyle diyor: "reconstruction teriminin taşıdığı **aynı** λ ile ölçeklenir,
dolayısıyla QP başına yeniden ayarlanması gerekmez." `lambdas.mean()` bunu
yapmıyor — batch ortalaması. λ qp0'da 10, qp63'te 2048, ortalama ~393:

| qp | λ | anchor/RD oranı | drift |
|---|---|---|---|
| 0 | 10 | **39.3×** | 0.003 dB |
| 63 | 2048 | **0.19×** | 0.029 dB |

Anchor, driftin on kat büyük olduğu yerde beş kat zayıf. 207 kat ters, ve tam
olarak kodun kendi yorumunun engellemeyi vaat ettiği şey.

### Uygulanmadı, bayrak arkasına alındı

`--anchor_per_sample`, varsayılan kapalı. Sebep: altı koşu aynı dosyayı
kullanıyor ve biri çökme sonrası yeniden başlarsa değişikliği sessizce yutar —
ölçtükleri şey deney ortasında değişirdi. Yeni bir koşu **başlatılmadı** da:
yedinci iş altı koşunun hepsini yavaşlatırdı.

## 68 — `--help` zaten kırıkmış

Bayrağı doğrulamak için `--help` çalıştırdım ve çöktü. Benim düzenlemem
öncesinde de çöküyormuş: yardım metinlerindeki kaçırılmamış `%` işaretlerini
(`"+0.066% of the decode"` vb.) argparse format belirteci sanıyor ve
`TypeError: %o format` veriyor. Yedi satırda `%%` olarak kaçırıldı; davranış
değişmedi, yalnızca `--help` artık çalışıyor.

Küçük ama kayda değer: bir eğitim scriptinin `--help`'i çalışmıyorsa
bayraklarını okumanın tek yolu kaynağı okumak, ve bu oturumda argv'nin tek
kayıt yeri olmasının ne kadar pahalıya patladığını (57, 58) zaten gördük.

---

## 69 — MAC tasarrufu duvar saatine dönüşmüyor: 2–3 kat abartı

CVPR planındaki en yüksek riskli satır ölçüldü (`scripts/latency.py`,
1920×1088, 40 serpiştirilmiş yineleme, medyan):

| qp | stock | hepsi-derin | yönlendirilmiş | ek yük | **gerçekleşen** | öngörülen |
|---|---|---|---|---|---|---|
| 0 | 303.6 ms | 314.9 | 244.6 | %3.7 | **%19.4** | %34.9 |
| 32 | 295.7 | 314.7 | 261.2 | %6.4 | **%11.6** | %26.6 |
| 63 | 296.0 | 315.2 | 276.4 | %6.5 | **%6.6** | %21.5 |

Ayrışma: sabit **%3.7–6.5 ek yük** (bütün karolar en derin çıkışta, yani stock
ile birebir aynı aritmetik — karolama makinesinin bedeli), artı kalan **8–12
puan** düşen aritmetik yoğunluk (bir grup 1500 yerine 300 karo üzerinde koşunca
GPU doymuyor).

### Yöntem hatası, yakalandı ve düzeltildi

İlk sürüm 40 stock, sonra 40 deep, sonra 40 routed ölçtü. Kart %100 dolu ve
üzerinde VERBATIM eğitiliyor; bloklar arası yük kayması doğrudan orana yansırdı
— ki oran bu scriptin ürettiği tek şey. Serpiştirilmiş hâlde yeniden ölçüldü:
cevaplar bir puandan az oynadı, yani çekişme kayması açıklama değilmiş. Bulgu
gerçek.

### Mutlak sayılar alıntılanamaz

453 GMAC boş bir A6000'de ~12 ms sürmeli; 300 ms ölçtük. Kart dolu, fp32,
channels_last yok, torch.compile yok, CUDA graph yok. Bu koşullar dağıtımı
temsil etmiyor ve mutlak latency makaleye konamaz. Ölçülen şey **oran**.

### Makalenin çerçevesi değişti

Soyut "%27 daha hızlı" diyemez. "%27 daha az MAC, bu uygulamada %6–19 daha
hızlı" diyebilir — ve aradaki fark kendi başına bir sonuç: içerik-uyarlamalı
derinlik MAC cinsinden söylemesi kolay, GPU'da tahsil etmesi zor. Bunu erken
ölçmenin sebebi tam olarak buydu; soyut yazıldıktan sonra öğrenilseydi makale
yanlış iddiayla gidecekti.

---

## 70 — Gevşek bütçe daha iyi makale; ve "neden küçük decoder" sorusunun ilk cevabı

0.3 dB'de latency yeniden ölçüldü. Gerçekleşen tasarruf öngörülene çok daha
yakın: qp0'da oran 0.56 → **0.86**, qp63'te 0.31 → 0.58.

| bütçe | BD-Rate | MAC | duvar saati | hızlanma |
|---|---|---|---|---|
| 0.1 dB | %0.88 | %27.2 | %6.6–19.4 | 1.14× |
| 0.3 dB | **%2.51** | %39.8 | **%20.8–36.1** | **1.38×** |
| 0.5 dB | %3.36 | %41.7 | ölçülmedi | |

Sebep iki katlı ve aynı yöne çalışıyor: gevşek bütçede karolar en sığ çıkışa
yığılıyor, derin gruplar çok az karo üzerinde koşuyor (yani tasarruf "çok küçük
grup" yerine "işi tamamen atlamak"tan geliyor), ve sabit ek yük iki katı
tasarrufa yayılıyor.

### DCVC-UF'nin kendi tablosu "neden küçük decoder" sorusunu kısmen cevaplıyor

Makale iki model boyutu yayınlıyor, ve aralarındaki takas alanın hız için ne
ödediğinin bedava bir kalibrasyonu:

| | verilen BD-Rate | alınan hız | birim hız başına puan |
|---|---|---|---|
| DCVC-UF HT-L → HT-S | 10.6 | 1.66× | **16.1** |
| FLEX-UF @ 0.3 dB | 2.51 | 1.38× | **6.6** |

Uyarlanabilir derinlik, modeli küçültmekten **~2.4 kat daha ucuza** hız satın
alıyor.

İki uyarı: onların çifti tüm video, bizimki yalnızca intra — paydalar farklı,
yani bu bir kalibrasyon, kafa kafaya karşılaştırma değil. Ve HT-S, HT-L'den
yalnızca derinlikte değil genişlik ve blok sayısında da farklı. Kendi statik
baseline'ımız (plan §2) hâlâ gerekli. Ama hakemin sorusuna cevabın lehimize
olduğuna dair güçlü bir işaret, ve zaten basılı bir tablodan bedavaya geldi.

---

## 71 — Hızın fiyatı sabit: birim hızlanma başına ~6.5 BD-Rate puanı

Üç bütçede de latency ölçüldü:

| bütçe | BD-Rate | MAC | duvar saati | hızlanma | puan/birim |
|---|---|---|---|---|---|
| 0.1 dB | %0.88 | %27.6 | %12.6 | 1.14× | **6.2** |
| 0.3 dB | %2.51 | %39.5 | %27.6 | 1.38× | **6.6** |
| 0.5 dB | %3.36 | %41.7 | %34.0 | 1.51× | **6.5** |

**Fiyat bütün aralıkta sabit.** Bu, tek bir çalışma noktasından daha iyi bir
sonuç: yöntemin kiraz toplamakla suçlanabilecek bir tatlı noktası yok, bir
döviz kuru var. Uygulama hangi noktayı istiyorsa onu seçer.

DCVC-UF'nin kendi HT-L → HT-S takası 16.1 puan/birim. Yani uyarlanabilir
derinlik hızı **2.4 kat ucuza** alıyor, ve bunu tek bir elverişli noktada değil
sabit oranda yapıyor.

### Ek yük için nokta tahmini vermiyorum

Dokuz ölçümde %1.5–8.2, medyan %6.2. İki ~300 ms ölçümün küçük farkı ve kart
%100 dolu; nokta tahmini sahte hassasiyet olurdu. Aralık raporlanıyor.

### Belgelerdeki bütün MAC iddialarına uyarı eklendi

README, 03 ve 05'te başta duruyor: bunlar MAC, duvar saati değil, ve MAC
tasarrufunu hızlanma diye alıntılamayın. Altı belge boyunca bu ayrım yoktu.

---

## 72 — "Düşen aritmetik yoğunluk" GERİ ÇEKİLDİ; gerçek sebep grup arası muhasebe

69'da duvar saati farkını "sabit ek yük + düşen aritmetik yoğunluk" diye
ayırdım ve bunu belgelere de yazdım. İkinci yarısı **ölçülmeden ilan edilmişti**
ve yanlış çıktı.

`scripts/latency_profile.py`, qp32, 40 karo:

| aşama | 0.1 dB | 0.3 dB | MAC modeli |
|---|---|---|---|
| gövde (upsample + grup 0–1) | 106.86 ms (%53.6) | 108.57 (%65.4) | %38 |
| patchify | 0.19 (%0.1) | 0.19 (%0.1) | — |
| unpatchify | 0.19 (%0.1) | 0.19 (%0.1) | — |
| seam repair | 2.05 (%1.0) | 2.05 (%1.2) | %0.95 |
| head | 11.67 (%5.9) | 11.70 (%7.1) | **%2.4** |

Karo başına grup maliyeti (0.1 dB): grup 2 → 1.03, grup 3 → 0.93, grup 4 → 0.73,
grup 5 → 0.94 ms/karo. **Düz, hatta küçük kümelerde daha iyi** — muhtemelen L2'ye
sığdıkları için. Yani aritmetik yoğunluk düşmüyor; iddiam desteklenmiyor.

### Gerçek sebep, ölçülmüş

Aşamaların toplamı 199.34 ms, uçtan uca 261.24 ms. ~62 ms hiçbir aşamada yok.
Ayrı ölçüldü:

| | ms |
|---|---|
| grup döngüsü, yalnızca konvolüsyonlar | 77.50 |
| grup döngüsü, yalnızca muhasebe (konvolüsyonsuz) | **36.43** |
| ikisi birlikte | 113.34 |

**Muhasebe grup döngüsünün %32.1'i.** Her grup sınırında boolean maske
(`em[active] > g`), gather (`active[keep]`, `work[keep]`) ve canvas scatter var;
boolean indeksleme kaç eleman hayatta kalacağını bilmek zorunda olduğu için
cihaz→ana bilgisayar senkronizasyonu zorluyor. Dört grup, dört senkron.

### Bu iyi haber

Kurtarılabilir bir kayıp. Çözüm: karoları çıkış derinliğine göre **bir kez**
sırala, sonra her grupta boolean maske yerine bitişik dilim al. Senkronlar ve
gather'lar gider. Yani plandaki "kernel çağrı yükünü düşür" maddesi tahmin
değil, hedefi ölçülmüş bir mühendislik işi.

### Ve bir örüntü

Bu oturumda mekanizma iddialarında **dördüncü kez** yanıldım (57'deki üç, artı
bu). Örüntü net: sayıyı ölçüp mekanizmayı tahmin etmek. Ölçüm doğru çıkıyor,
açıklama çıkmıyor. Sayıyı raporlarken mekanizmayı ayrı ölçmeden yazmamalıyım.

---

## 73 — Muhasebe kaybı kurtarılabilir: karoları bir kez sırala, %23–37 geri gelsin

72'de grup döngüsünün %32'sinin muhasebe olduğu ölçüldü. Önerdiğim çözüm test
edildi (`scripts/sorted_exec.py`, ayrı bir kıyaslama — `decoder.py` altı canlı
koşu tarafından import ediliyor, düzenlenmiyor):

Karoları çıkış derinliğine göre **azalan** sırala. O zaman "grup g'de hâlâ aktif
olan karolar" bitişik bir **önek** olur; her grup gather değil dilim alır; dilim
sınırları tek bir kümülatif sayımdan gelir (grup başına senkron yerine döngü
başına bir tane); biten karolar sonda tek bir scatter ile ters permütasyondan
yazılır.

| bütçe | mevcut | sıralı | kazanç | çıktılar aynı mı |
|---|---|---|---|---|
| 0.1 dB | 114.44 ms | **87.64** | **%23.4** | evet, bit-aynı |
| 0.3 dB | 77.10 ms | **48.86** | **%36.6** | evet, bit-aynı |

Aritmetik birebir aynı — aynı karolar aynı gruplardan geçiyor. `torch.allclose`
atol=rtol=0 ile doğrulandı; hızlanma uydurma değil.

### Uçtan uca izdüşüm (ölçüm değil, izdüşüm)

qp32'de grup döngüsü kazancını ölçülmüş uçtan uca süreye uygulayınca:

| bütçe | şimdi | izdüşüm | MAC modeli |
|---|---|---|---|
| 0.1 dB | %11.6 | **%20.7** | %26.6 |
| 0.3 dB | %25.8 | **%35.4** | %40.5 |

gerçekleşen/öngörülen oranı 0.44 → 0.78 ve 0.64 → 0.87.

İzdüşüm, çünkü uçtan uca ölçmek `decoder.py`'ı düzenlemeyi gerektiriyor ve altı
koşu onu import ediyor; çökme sonrası yeniden başlayan bir koşu değişikliği
deney ortasında yutardı. Koşular bittiğinde ya da ayrı bir kopyada ölçülecek.

Bu, CVPR planındaki "kernel çağrı yükünü düşür" maddesini tahminden ölçülmüş
bir mühendislik katkısına çeviriyor.

### 73a — Kazanç bütün hızlarda tutuyor; değişimi açıklayamıyorum

| qp | 0.1 dB | 0.3 dB |
|---|---|---|
| 0 | %31.3 | %12.7 |
| 32 | %23.4 | %36.6 |
| 63 | %22.5 | %28.2 |

Hepsi bit-aynı çıktı veriyor. Ama %12.7–36.6 aralığı geniş ve **sebebini
bilmiyorum**. Bariz hipotez — kazanç, hâlâ karo geçiren grup sınırı sayısıyla
gider — bu altı noktada 0.32 korelasyon veriyor, yani desteklenmiyor. İddia
etmeden önce baktım; etseydim bu oturumda beşinci yanlış mekanizma olacaktı.

Kurulmuş olan: kazanç var, her yerde pozitif, ve aritmetiği değiştirmiyor.
Kurulmamış olan: neden 12.7 ile 36.6 arasında oynadığı. Boş bir kartta ve daha
çok noktada ölçülmeli.

### 73b — Sıralı yürütme kod çözücüye ALINMADI, bilerek

Kazanç ölçüldü (%12.7–36.6, bit-aynı), ama `decoder.py`'daki gerçek döngü
kıyaslamadan üç noktada farklı ve her biri permütasyon üzerinden yeniden
indekslenmek zorunda:

- `self._at_exit(work[leaving], g)` → bitişik dilim `work[hi:lo]`. Kolay kısım.
- `tile_gate[active[leaving]]` → straight-through router kapısı **orijinal karo
  kimliğiyle** indeksleniyor, sıralı konumla değil; `order[hi:lo]` gerekiyor.
  Bunu yanlış yapmak router'ın gradyanını sessizce bozar ama rekonstrüksiyon
  bit-aynı kalır — yani piksel eşitliği testi yakalamaz.
- `--tile_coupling` altında `cpl.active = active`. En temiz hamle sıralı yolu
  `not cfg.tile_coupling` ile sınırlamak.

Altı canlı koşu bu dosyayı import ediyor. İnce bir indeksleme hatası eğitimi
sessizce bozar ve günler sonra fark edilir. Uzun bir oturumun sonunda, ölçülmüş
bir kazancı almak için alınacak risk değil. Koşular bittiğinde, varsayılanı
kapalı bayrak arkasında, hem piksel hem gradyan testiyle alınacak.

---

## 74 — Zincire eklediğim aşama hiç çalışmayacaktı

62'de `watch_ckpts.sh`'a `3b` (router yolu) aşamasını ekledim ve "bundan sonraki
her checkpoint ikisini birden ölçecek" dedim. Yerinde doğrulamayı bugün yaptım:
**hiç çalışmamış, ve çalışmayacaktı da.**

Watcher'lar 22:05'te başlamış, dosyayı 00:24'te düzenledim. Bash `while true;
do ... done` gövdesini çalıştırmadan önce tamamen parse ediyor; çalışan süreç
diskteki yeni sürümü görmüyor. Altı watcher da eski gövdeyi koşturmaya devam
ediyordu.

Lock boşken yeniden başlatıldı (watcher'lar yalnızca yokluyor, eğitmiyor —
yeniden başlatmak hiçbir koşuyu etkilemiyor). Yeni süreçler 3665407–3665412.

Ders: uzun ömürlü bir bash döngüsüne yaptığın düzenleme, süreci yeniden
başlatmadan yürürlüğe girmez. Ve "ekledim, artık çalışıyor" demek, çalıştığını
görmekle aynı şey değil — bu oturumda `--help`'in kırık olması da aynı sınıftan
bir varsayımdı.

---

## 75 — 3b yerinde doğrulandı; ve anchor terimsiz taban bütçeyi tamamen yiyor

74'te watcher'ları yeniden başlattım. İlk gerçek testler geldi.

### 3b çalışıyor

`results/router_FINE12_0818_0458.json` — zincirin ürettiği ilk router sonucu.
Beş hızda da bütçeye ulaşıyor, sıfır ek bitle %9.2–14.4 (FINE12'nin kendi
ortak-eğitilmiş başlığıyla). Yani 62'de eklediğim, 74'te yürürlüğe soktuğum
aşama yerinde çalışıyor ve kâğıt için istediğiniz eşleşmiş çiftler birikmeye
başladı.

Yan bulgu: FINE12'nin β'sı −0.008…−0.021, yani router'ı **çökmemiş** — BEST'inki
qp63'te oracle uyumu 0.000 veriyordu (60). Demek ki çöküş ortak eğitimin
kaçınılmaz sonucu değil; K, j ya da adaptör tipi de rol oynuyor olabilir. Dört
epoch'ta bakılacak.

### Düzeltme: VERBATIM sürüklenmiyor, iyileşiyor

İlk okumamda "VERBATIM'in en derin çıkışı kötü sürüklenmiş" dedim. Veri bunu
desteklemiyor:

| VERBATIM (anchor YOK) | qp0 | qp32 | qp63 |
|---|---|---|---|
| epoch 0 | +0.1365 | +0.1561 | +0.2231 |
| epoch 2 | +0.0847 | +0.1241 | +0.1675 |

Taban **düşüyor**. Doğru ifade: anchor terimi olmadan en derin çıkış üç epoch
sonra bile yayınlanmış kod çözücüden 0.17 dB uzakta, ve bu 0.1 dB bütçesini
beş hızın dördünde ulaşılmaz kılıyor (qp16'dan itibaren "floor exceeds the
budget").

BEST (`--anchor_weight 10`) qp63'te +0.0296. **5.7 kat küçük.**

Bu, "taban qp63'te bütçenin %30'unu yiyor" bulgusunun (E1, 67) diğer ucu:
anchor'sız taban bütçenin **%167'si** oluyor. Anchor terimi gerçek iş yapıyor,
ve 67'deki λ ölçekleme düzeltmesinin neden önemli olduğunu da güçlendiriyor.

---

## 76 — Grid seam repair kendi testinin yanlış tarafında

Sunuma teknik seam slaytlarını yazarken piksel ölçeğinde bir figür ürettim
(`scripts/seam_patches.py`) ve grid repair'in beklenenden kötü davrandığını
gördüm. Tahmin etmek yerine ölçtüm — 8 dizi, qp63, aynı latent, aynı exit
haritası, repair AÇIK vs KAPALI:

| rejim | KAPALI | AÇIK | fark |
|---|---|---|---|
| bütün karolar tam derinlikte | 0.0672 | 0.0750 | **+0.0078 (kötü)** |
| 0.1 dB'de yönlendirilmiş | 0.1259 | 0.1239 | −0.0019 (iyi) |

Yani modül yönlendirilmiş rejimde **0.002 dB** kazandırıyor ve decode'un
**%0.95'ine** mal oluyor. Ayrıca tam derinlikte kötüleştirdiği için **tabanı
büyütüyor** — ki taban qp63'te bütçenin %30'unu zaten yiyor (67).

Kıyas kendi projemizden: `arls` %10.7 için 0.019 dB kazandırıyordu ve
reddedilmişti. Birim başına: arls 0.0018 dB/puan, grid repair 0.0020 dB/puan.
Neredeyse aynı. **Aynı kurala göre grid repair de reddedilmeliydi.**

Uyarı, slayta da yazıldı: modül ortak eğitildi, yani çıkarımda kapatmak onsuz
eğitmekle aynı şey değil. Temiz test 256px'te `seam_repair=none` ile bir koşu,
ve altı koşunun hiçbiri o değil. VERBATIM `none` kullanıyor ama başka altı
bayrakta da farklı.

Sıradaki deney listesine giriyor, ve E1 (anchor'ın λ ölçeklemesi) ile aynı yere
bakıyor: ikisi de tabanı küçültmeye çalışıyor, ve taban yüksek hızda bütçenin
en büyük tek kalemi.

## 77 — Sunum: 23 slayt, tamamı İngilizce, seam'e beş slayt

Kullanıcı ClassSR tarzı bir çıkış haritası, formüllerle teknik anlatım ve
git'e push istedi.

- `scripts/exit_map_figure.py` — Bosphorus qp32'de gerçek Lagrange tahsisi:
  tekne exit 4, su ve gökyüzü exit 2, %33.4 tasarruf 0.098 dB'de. Yazarken bir
  ölçüm hatası yakalandı: karo başına dB'yi kare ortalamasına bölüyordum, o
  yüzden kolay karolar −6 dB okuyordu ("yayınlanmış kod çözücüden iyi", ki
  hiçbir çıkış olamaz). Her karo kendi referansına bölünüyor artık.
- `scripts/seam_patches.py` — karo sınırında 192px pencere, üç dolgu modu,
  ×40 hata haritaları.
- Slayt 11–15: seam'in türevi (`1 − ((F−2b)/F)² ≈ 4b/F`, %75 kirlenme),
  dolgunun bir kestirimci olduğu, perimeter yasası, GridSeamRepair'in denklemi.

Ayrıca 10. slayttaki "seam var olmayı bırakıyor" cümlesi düzeltildi: o canvas
coupling'i anlatıyor ve `tile_coupling=False` altı koşunun altısında da. Ölçülmüş
bir seçenek, ama bu sonuçların kullandığı şey değil.

---

## 78 — Dört epoch'un doğru durak olup olmadığı ilk kez ölçüldü

Seçim noktası 4 epoch olarak takvim gerekçesiyle seçilmişti, hiçbir şeyin
yakınsadığı gözlendiği için değil. Artık koşu başına birkaç checkpoint var,
bakılabilir (`scripts/convergence.py`, GPU gerekmiyor — saklı ölçümler).

Beş hızın ortalaması, 0.1 dB'de, release paydasıyla:

| koşu | 47k'da | yarım-yarım eğim | zaman sabiti | 4 epoch öngörüsü | tavan |
|---|---|---|---|---|---|
| BEST128 | %27.1 | +0.180 → +0.151 | 106k adım | **%38.0** | %41.9 |
| FINE12 | %27.4 | +0.145 → +0.094 | 189k adım | **%39.6** | %50.3 |

**İkisi de yavaşlıyor** — ikinci yarının eğimi birincinin altında. Bu, üstel
uyumun anlamlı olmasının ön koşulu; yavaşlama olmasaydı "asimptot" diye
çizilen şey kılık değiştirmiş bir ekstrapolasyon olurdu.

Tavan **sabitlenerek** uyduruldu (`C = 1 − C_j`), çünkü merdiven onu aşamaz;
serbest C ile uydurmak maliyet modelinin zaten bildiği bir tavanı veriye
uydurmak olurdu.

### Sonuç: 4 epoch makul, ama sayı 4 kat ekstrapolasyon

İkisi de tavanının ~%80–90'ına varıyor. Yani 4 epoch keyfi bir sayı değilmiş.
Ama uyum 7 ve 5 noktaya iki parametre, ve verinin **4 katı** ötesine
uzatılıyor. Hata çubuğu yok. Rapor edilirken bu söylenecek.

### Ve şu an ki manşet muhtemelen kazanan değil

Belgelerdeki 34.0 / 20.8, BEST'in epoch 0'ından. BEST128 ve FINE12'nin 4
epoch öngörüleri %38–40 — yani nihai sayılar bugünkü manşetin belirgin
üstünde çıkacak. Manşeti değiştirmiyorum (koşular bitmeden kazanan ilan etmek
seçim kuralını bozar) ama beklenti bu.

CONTROL karşı örnek olarak duruyor: −0.073 puan/1k, üç checkpoint boyunca
monoton düşüş. Yavaşlamıyor, kötüleşiyor.

---

## 79 — Tavan bir MALİYET sınırı, ulaşılabilir bir nokta değil

Kullanıcı FINE12'nin %50.3'lük tavanına takıldı ve haklıydı: sayı doğru ama
"%50 tasarruf" diye okununca yanlış.

Aritmetik doğrulandı, kalem kalem:

```
FINE12, exit 4 (12 blokun 5'i)
  upsample            0.0816
  4 paylaşılan blok   0.2981
  1 karo bloğu        0.0745
  head                0.0240
  adaptör             0.0093
  seam repair         0.0095
  ------------------- 0.4971  ->  %50.3
```

**Ama o çıkışta kalite yok.** BEST'in exit 2'si 12 blokun 6'sını çalıştırıyor ve
qp63'te 0.559 dB kaybettiriyor — bütçenin beş katı. FINE12'nin exit 4'ü ondan da
sığ. Tavana ulaşmak *bütün* karoların orada olması demek, ve hiçbir dB bütçesi
buna izin vermiyor.

Tavanın söylediği şey: **merdiven en fazla oraya kadar gidebilir.** Eğitim seni
oraya doğru iter, asla geçiremez.

### Ve vaat henüz tahsil edilmemiş

| | tavan | 0.1 dB'de ölçülen ortalama |
|---|---|---|
| BEST | %41.9 | %27.18 |
| FINE12 | %50.3 | %26.93 |

8.4 puanlık tavan avantajı **sıfır ölçülen avantaja** dönüşmüş. 4 epoch
öngörüleri de %38.0 ve %39.6 — yani fark orada da 1.6 puan, tavan farkının
beşte biri.

Belgelerde tavanı hep "mimari sınır" diye yazmıştım ama maliyet-sınırı /
ulaşılabilir-nokta ayrımını öne çıkarmamışım. 03 ve 04 düzeltildi.

---

## 80 — "Router çöktü" fazla genel bir ifadeydi

Kullanıcı sordu: router çalışıyor mu, neden çöküş diyorsun. Haklı — iki ayrı
router'ı tek isimle anlatmışım.

Üç router, üç sonuç, hepsi ölçülmüş:

| router | sonuç |
|---|---|
| BEST, ortak eğitilmiş | qp'ye bağlı sabite çöktü, qp63'te uyum 0.000 |
| FINE12, ortak eğitilmiş | **çökmedi** — beş hızda da bütçeye ulaşıyor, β ≈ −0.01 |
| donuk kod çözücüye karşı yeniden eğitilmiş | uyum 0.848–0.923, beş hızda da bütçede |

Yani **routing çalışıyor.** Çöken şey BEST'in içindeki belirli bir router.
Ve ortak eğitim de otomatik olarak öldürücü değil — FINE12 aynı şeyi yapıyor
ve sağ kalıyor.

BEST'inki neden çöktü de FINE12'ninki çökmedi: **açık.** İkisi K, j, karo boyutu
ve adaptör tipinde farklı, yani karşılaştırma hiçbir şeyi izole etmiyor.

Sunumdaki "A claim overturned" slaytı ve docs/04 düzeltildi: artık üç sonucu
yan yana veriyor, "router çöker" değil.

## 81. The padding ablation was measured against a handicapped reference

`ctc_seam_ablation.py` installed the padding wrapper on `dec.groups[j:]` and
then called `forward_full` for the reference. `forward_full` runs those same
modules, so the reference full-frame decode was itself padded with the mode
under test -- at the FRAME border.

The released decoder was trained with zeros padding. Forcing replicate at the
image boundary is off-distribution and costs it real quality: on Johnny 720p at
qp 63 the reference fell 44.6611 -> 44.4273, i.e. **0.234 dB**. The seam is
reported as `reference - tiled`, so a degraded reference UNDERSTATES it -- and
only for the non-zeros modes, since for `zeros` the wrapper reproduces the
default. `zeros` was measured against a clean baseline and every alternative
against a handicapped one.

Corrected, 256 px, qp 63, replicate:

| | published | corrected |
|---|---|---|
| 10 sequences (the old test set) | 0.1070 | 0.2366 |
| 40 sequences (current) | — | **0.2161** |

Two independent code paths now agree to four decimals on the 3-sequence probe
(0.3162 from `seam_vs_qp.py`, 0.3162 from the fixed `ctc_seam_ablation.py`),
which is how the fix was verified rather than assumed.

**What survives.** replicate still removes 67% of the seam at qp 63 (0.6523 ->
0.2161 on 40 sequences) and still costs nothing, so the design decision stands.
What does not survive is the *size* of the published margin, and the `linear` /
`arls` rows are affected too -- both are being re-measured against the clean
reference before anything is claimed about their ranking.

**A design consequence, not yet implemented.** replicate helps at internal tile
borders and *hurts* at the frame border, where the release expects zeros. Tiles
on the image edge are identifiable at zero cost, so the right rule is replicate
inside, zeros on the frame boundary. That is a free improvement this ablation
was previously hiding.

## 82. Grid seam repair: the spatial measurement, and a ceiling

Error by distance from the nearest tile boundary, qp 63, routed at 0.1 dB, 6
sequences, repair OFF vs ON with the same exit map:

| band (px) | share | OFF | ON | change |
|---|---|---|---|---|
| 0-4 | 6.2% | 0.000047 | 0.000047 | **-0.27%** |
| 4-16 | 17.3% | 0.000044 | 0.000044 | +0.05% |
| 16-64 | 51.6% | 0.000043 | 0.000043 | +0.04% |
| 64-128 | 25.0% | 0.000041 | 0.000041 | +0.04% |

The seam is real and local -- the boundary band carries 15% more error than the
interior, falling monotonically with distance. But the gate leaks: the module
gains only in the 0-4 band and loses slightly across the other 94% of pixels.

The ceiling this implies settles the module. With a PERFECT gate -- zero
correction in the interior, the boundary gain unchanged -- the most it could
earn is

    0.062 x 0.000047 x 0.0027 / 4.30e-5 = 1.8e-4  ->  ~0.0008 dB

against 0.95% of decode. `arls` was rejected at 0.019 dB for 10.7%; this is 24x
worse per point. Tightening the gate cannot rescue it, because there is not
enough there to win. Decision: `seam_repair=none` plus `tile_coupling=True`
(0.032% of decode, removes the cause) on the next run.

The joint-training caveat still applies -- switching the module off at inference
is not the same as training without it -- but it now bounds a quantity that is
too small to matter either way.

## 83. The reported dB was a full-frame number. The tiling penalty was never in it.

`dec.forward_all_exits` runs the trunk FULL FRAME and taps each exit. That is
correct for training -- one trunk pass, K heads, gradient to every exit -- and it
is what every evaluation script in this repository used to build its per-tile,
per-exit distortion table. The reference is the release's full-frame decode. So
both sides of the ratio were full-frame, the seam cancelled, and the reported dB
described a decoder nobody ships.

Measured on RECIPE512, 8 CTC frames, every tile at the same exit:

| qp | exit | full-frame dB | deployed dB | missing |
|---|---|---|---|---|
| 0 | 5 (deepest) | 0.0040 | 0.0385 | **+0.0345** |
| 0 | 2 (shallowest) | 0.1845 | 0.1939 | +0.0094 |
| 32 | 5 | 0.0120 | 0.0506 | +0.0387 |
| 63 | 5 | 0.0252 | 0.0615 | +0.0362 |
| 63 | 2 | 0.3002 | 0.3080 | +0.0078 |

It is not a constant offset. It is four times larger at the deepest exit than at
the shallowest, because eight blocks have run per tile there against two -- so it
scales with exactly the quantity the Lagrangian is optimising. An operating point
labelled 0.1 dB was delivering roughly 0.12-0.13.

How it survived: three separate checks pointed at the reference and found it
correct. `ref.dec.forward_full` IS the right reference -- the release is
full-frame and the dB is quoted against it. The bug was one line further up, on
our own side, in a function whose name says "all exits" and not "full frame".
`model.forward_all_exits_patched` exists precisely to be the tiled counterpart
and its docstring spells the distinction out; no evaluation script used it.

Fix: `flexuf/eval.py`. `tiled_exit_mses` builds the table with one tiled decode
per exit, on the deployed path; `true_frame_mse` decodes the actual mixed map
once, because the table still measures each tile with its neighbours at the same
depth. The curve scripts now bisect on the cheap table and then correct against a
real decode, so the dB they print is what a decoder delivers. It also removes a
second, smaller error: the old table had distinct values for exits 0 and 1, which
`decoder.forward` clamps away, so an argmin could select an exit that does not
exist.

Cost: K tiled decodes per frame instead of one trunk pass -- minutes, not hours.

Everything measured with a real tiled decode was already right and does not
move: the seam tables (`ctc_seam_ablation.py`, `seam_vs_qp.py`), the floor, the
saturation points, the spatial repair measurement, the latency work.

## 84. Which scripts still measure on the full-frame path

`flexuf/eval.py` fixed the evaluation scripts whose numbers are reported. These
still call `dec.forward_all_exits` and are therefore still measuring a decoder
nobody ships. None of them produces a headline number; all are listed so the
next person does not have to grep for them.

    budget_table.py       signal_probe.py     plot_checkpoint.py
    make_slide_figs.py    oracle_diagnostic.py  per_qp_saving.py
    signal_search.py      theory_check.py     why_qp.py / why_qp_val.py
    train_router_head.py  ladder_health.py

`why_qp.py` is a special case and is fine: it reports BITRATE, which is a
function of the latent alone and identical on both paths.

`train_router_head.py` trains the V1 router head; V2 (`train_router2.py`) is
fixed and is what the reported configuration B uses.

Fixed and reported: `signalled_curve.py`, `paper_curve.py`, `router_curve.py`,
`routed_curve.py`, `eval_router2.py`, `db_convention.py`, `train_router2.py`,
`exit_map_figure.py`, `quant_sweep.py`, and the new `static_baseline.py` and
`saturation.py`.

## 85. The area law does not predict the seam. It grows as b squared.

The contamination fraction `1 - ((F-2b)/F)^2` has been quoted throughout this
project as the reason split depth matters. It is a correct statement about which
pixels are within reach of an invented value, and it is a bad predictor of the
penalty.

Sweeping the split depth j sweeps the per-tile block count b, untrained warm
start, every tile at full depth, 12 sequences:

| b | area fraction | qp 0 | qp 32 | qp 63 |
|---|---|---|---|---|
| 12 | 0.938 | 0.1748 | 0.3138 | 0.6057 |
| 10 | 0.859 | 0.0997 | 0.1490 | 0.2340 |
| 8 | 0.750 | 0.0649 | 0.1068 | 0.1944 |
| 6 | 0.609 | 0.0324 | 0.0527 | 0.0894 |
| 4 | 0.438 | 0.0088 | 0.0181 | 0.0382 |
| 2 | 0.234 | 0.0026 | 0.0056 | 0.0165 |
| 0 | 0.000 | **0.0000** | **0.0000** | **0.0000** |

The b = 0 row is the control and it is exactly zero at all three rates, which is
what licenses reading the rest as seam and nothing else.

Fitted with one free scale, the area fraction is wrong by 160-264% on average.
A power law is wrong by 12-22%:

    seam ~ b^alpha,  alpha = 2.38 / 2.22 / 1.93 at qp 0 / 32 / 63,
    r = 0.995 / 0.994 / 0.979 in log-log

The area fraction fails because it saturates. At b = 12 nearly every pixel is
already contaminated and the fraction can rise no further, while the penalty
still climbs by 3.5x from b = 8 to b = 12. What the fraction cannot express is
that a pixel reached by twelve convolutions is far more damaged than one reached
by two.

The exponent near two has a reading, and it is the one to state: contaminated
pixels grow like perimeter times depth, proportional to b, and the error
accumulated in each grows with how many convolutions reached it, again
proportional to b. Their product is b^2. The drift from 2.38 to 1.93 with rate is
not explained and is not claimed to be.

Corrected in paper/main.tex, paper/supplementary.tex, docs/07-seam.md and
scripts/build_pdf.py. The fraction is kept where it is used correctly -- as the
statement of which pixels are affected -- and is no longer used to predict how
much.

## 86. Canvas coupling is exact at uniform depth and destroys the routing

The paper's largest claimed remaining gain was Section 4.4: coupling costs
+0.032% of the decode, makes a tiled decode bit-identical to a full-frame one,
and closing the floor it removes is "worth roughly 3.5 points at the lowest rate
and 5 points at every higher one". Two of those three statements survive
measurement and the conclusion does not.

**Exactness: confirmed, after isolating it.** Measured on RECIPE512 as shipped,
a coupled tiled decode differs from a full-frame decode by 1.13e-2 -- not zero.
That is not coupling failing. GridSeamRepair runs on the stitched canvas and has
no counterpart in the full-frame path, so the two decodes differ by a MODULE.
With it off:

| configuration | max\|tiled - full\| at uniform depth |
|---|---|
| coupling ON, seam repair ON (as shipped) | 1.131e-02 |
| coupling ON, seam repair OFF | **0.000e+00** |
| coupling OFF, seam repair OFF (replicate) | 6.055e-02 |

**The floor: coupling removes most of it.** 16 CTC frames, every tile at full
depth, coupling switched on at inference only:

| qp | floor padded | floor coupled | removed |
|---|---|---|---|
| 0 | 0.0358 | 0.0031 | 91% |
| 32 | 0.0484 | 0.0144 | 70% |
| 63 | 0.0564 | 0.0300 | 47% |

**The routing: coupling destroys it.** Same model, same frames, bisected to the
same 0.1 dB budget:

| qp | saving, padded | saving, coupled |
|---|---|---|
| 0 | 33.44% | 31.94% |
| 32 | 25.94% | **4.15%** |
| 63 | 19.25% | **0.40%** |

So the projection of "+5 points" is wrong in sign at every rate above the lowest,
and catastrophically so at high rate.

The mechanism is in coupling's own docstring, which anticipated it and assumed it
was small: "what remains is only the boundary between tiles that chose DIFFERENT
depths". With replicate padding a tile is completely independent -- its output
does not depend on its neighbours at all -- and routing is therefore free to give
adjacent tiles any depths it likes. Coupling makes each tile depend on its
neighbours' features, and under routing those neighbours ran a different number
of blocks. A shallow tile reading a deep neighbour's activation is a
configuration the trained weights have never seen, and the resulting error is far
larger than the seam that was removed.

Coupling is exact when every tile is at the same depth. Routing is the
deliberate violation of that condition. The two are in tension by construction,
not by accident.

**Caveat, and it is the one that keeps this from being final.** This model was
trained with replicate padding, so switching to coupling at inference is a
distribution shift, and a run trained with coupling could reverse it. But the
tension above is structural rather than a training artefact, so the burden is now
on that run to show otherwise -- and until it exists the paper cannot claim the
gain. Section 4.4 and the limitations section are corrected accordingly.

## 87. Smaller tiles do not fix the low-resolution collapse

The per-class result said the method is governed by tile count: 40 tiles at
1080p gives 36% at q0, 8 tiles at 832x480 gives 21%, 2 tiles at 416x240 gives
11%. The obvious remedy is a smaller tile, and the paper said so.

Measured at q0, 0.1 dB, RECIPE512 (256 px) against BEST128 (128 px), which
multiplies the tile count by 3.4:

| class | tiles @256 | 256 px | tiles @128 | 128 px | delta |
|---|---|---|---|---|---|
| MCL-JCV 1080p | 40 | 36.30% | 135 | 34.33% | -1.97 |
| UVG 1080p | 40 | 33.14% | 135 | 32.94% | -0.20 |
| HEVC_B 1080p | 40 | 33.80% | 135 | 32.13% | -1.67 |
| HEVC_E 720p | 15 | 32.14% | 60 | 32.59% | +0.45 |
| HEVC_C 832x480 | 8 | 20.92% | 28 | 17.59% | -3.33 |
| HEVC_D 416x240 | 2 | 11.28% | 8 | **13.67%** | **+2.39** |

Smaller tiles help only where the count was 2. At 832x480 the count goes from 8
to 28 and the saving FALLS by 3.3 points: beyond a modest number of tiles the
extra seam costs more than the extra granularity is worth. That is consistent
with the b^2 law -- halving the tile side doubles the seam at fixed depth -- and
it means the granularity argument, which is correct as far as it goes, does not
license the remedy it suggests.

Confound, stated: BEST128 is a different training run at a different epoch, so
this is tile size and training together. The direction is consistent across six
classes and two rates, which is enough to withdraw the claim that a
resolution-adaptive tile size is an easy win, and not enough to quantify what a
matched-training comparison would give.

## 88. Every wall-clock number was timed on the wrong device

`torch.cuda.Event` is created on the process's **current** device, not on the
device the tensors live on. `torch.cuda.synchronize()` with no argument syncs
the current device. Both are silent when they are wrong: they return numbers.

`scripts/latency.py` defaults to `cuda:0` but was run with `--device cuda:2`
(the run notes say so). `scripts/encoder_cost.py` defaults to `cuda:7`.
`scripts/latency_profile.py` and `scripts/sorted_exec.py` had the same shape.
Every timing in this project was taken that way.

Re-measured with `torch.cuda.set_device(a.device)` first, RECIPE512:

| quantity | before | after |
|---|---|---|
| released decode, q0 | 274 ms | 111 ms |
| released decode, q32 | 401 ms | 111 ms |
| released decode, q63 | 512 ms | 111 ms |
| tiling overhead | 8.5% | 3.6% |
| routed, masked loop, q0 | 9.9% | 27.9% |
| routed, sorted loop, q0 | 28.5% | 29.1% |
| stem share of decode | 107 ms | 41 ms |
| encoder deployed table | 4.59x | 4.52x |

The tell was in the table the whole time. `forward_full` is a fixed synthesis
network; its arithmetic does not depend on the quality index, so it cannot take
1.9x longer at q63 than at q0. It does not: it takes 111 ms at all three rates.
Nobody looked, because the numbers were plausible in isolation and the ratios
were the thing being read.

Ratios of two long kernels mostly survive the bug -- which is why the encoder
cost barely moved and why this lived so long. Short kernels do not: the router
head measured 0.002 ms against a true 0.57 ms, a factor of ~300, and that is
what made it visible.

**What it costs the paper.** Finding (iv) claimed a 20% reduction in operations
made the decoder 10.9% *slower* until the loop was reordered. That did not
happen. The unsorted loop already realises 27.9% against a 35.3% prediction and
sorting adds 1.2 points. The claim is now the weaker true one: operations are an
optimistic bound, four fifths of the predicted saving arrives, and the optimism
grows with how much of the frame exits early (15.6% against 20.1% at q63).

**What was added so it cannot recur.** `tests/test_timing_device.py` asserts
that every script using `torch.cuda.Event`, or a bare `torch.cuda.synchronize()`,
also calls `torch.cuda.set_device`. It found two scripts beyond the two already
known. It also guards itself: if the pattern stops matching anything, a case
fails rather than the suite passing vacuously -- the failure mode of the two
entries already in the supplementary's "things that were not running".

## 89. The router's exit mask stopped being a mask

`StemRouterHeadV2` (and `StemRouterHead`) suppress exits below the split depth
with

```python
logits[:, : self.min_exit] = -1e4
```

That is a mask only while the head's own logits are much larger than -1e4. This
head's are not. A raw row, RECIPE512's router at q63:

```
[-10000.0, -10000.0, -10044.5, -10003.8, -10011.1, -10018.7]
```

The head's genuine outputs are the last four minus a common offset of about
-10000, and nothing in the training objective penalises a common offset --
cross-entropy and the regret term are both shift-invariant, so one drifted in.
The two "masked" entries are therefore the **largest** in the row. After
`log_softmax` they hold essentially all the mass, `argmax` picks one, and
`clamp(min=j)` turns it into exit `j`, the cheapest real exit.

So a large fraction of every configuration-B allocation has been going to the
cheapest rung for a reason unrelated to the tile.

**It is visible in the allocations.** q63, six sequences, 0.1 dB:

| decision rule | histogram | saving |
|---|---|---|
| `argmax` over all K, then clamp | {2: 49, 5: 191} | 7.98% |
| `argmax` over k >= j only | {3: 110, 4: 28, 5: 102} | 13.49% |

The broken form collapses onto a single confidence threshold and produces a
two-level allocation, which is a poor point on the RD curve however the
multiplier is bisected. It is not a bisection failure -- dB is monotone in beta
and the bisection lands correctly; the family it searches is simply degenerate.

**Fixed at the decision site**, not in the head. `flexuf/router/head.py` and
`head2.py` are imported by live training runs and a crash-restart would pick up
the edit mid-experiment. `scripts/router_curve.py` and `scripts/hybrid_curve.py`
now slice `lg[:, j:]` before the softmax, which is what a correct mask would
have produced. `tests/test_hybrid_map.py` asserts no selected exit is below the
split depth. The head's own mask should become scale-relative
(`logits.max() - 1e4`, or `-inf`) when nothing is training.

**It probably also explains the router's 0.718 agreement.** Training pushed the
real exits up against a suppression term sitting above them, so the head spent
capacity fighting its own mask. A retrain with a correct mask is the obvious
next experiment and has not been run.

**How it was found.** A blended rule at w=1 -- the same head, the same frames --
beat configuration B by seven points at q63. The first hypothesis was that a
single global beta over-tilts frames the head is confident about, so the
log-probabilities need per-frame normalisation. That was measured
(`results/router_RECIPE512_b01_pfs.json`) and is worth 0.25-0.47 points, not
seven. The hypothesis was wrong and the flag it added is kept only because it is
free. What actually differed between the two code paths was that the blend
sliced the logits and configuration B did not.

## 91. Configuration C at rho=1 does not quite reproduce configuration A

At rho=1 every tile is overridden by the encoder's exact choice, so configuration
C reduces to configuration A by construction. It was checked that way and the
check passed at 0.00 for weeks. On the re-measured data, on one checkpoint and
one cost model, it does not:

| qp | C at rho=1 | A | delivered dB, C | A | difference |
|---|---|---|---|---|---|
| 0 | 29.90 | 29.81 | 0.09983 | 0.09997 | +0.09 |
| 16 | 25.62 | 25.48 | 0.09997 | 0.09996 | +0.14 |
| 32 | 20.93 | 20.72 | 0.09999 | 0.09998 | +0.21 |
| 48 | 18.33 | 18.20 | 0.10000 | 0.10000 | +0.13 |
| 63 | 15.64 | 15.59 | 0.10000 | 0.10001 | +0.04 |

The obvious explanation is bisection noise, and it is wrong. The delivered
decibels agree to within 0.14 millibels, and around a 0.1 dB budget the saving
moves about 0.6 points per millibel, so noise of that size buys at most 0.1
points and at q32 it would have to buy 42 points per millibel. C is finding a
genuinely cheaper allocation at the same delivered quality.

That should not be possible. Both branches take the argmin of D + lambda c over
the same per-tile table; at rho=1 the router branch is empty and the two
searches differ only in how they bisect lambda. Either the two paths are not
computing the same objective, or one of them is charging differently.

Not yet diagnosed. Recorded rather than absorbed into a tolerance, because every
discrepancy tonight that looked like noise turned out to be a bug: the exit
clamp, the FFN adapter at 5C^2 billed as 2, the adapter picked by the exit the
map names rather than the one that runs, the mask that stopped masking, the
timing on the wrong device, and the result file chosen by a stale hand-ordered
list.

The check stays red until it is understood.

---

## 92. Training helps, and the paper is quoting a checkpoint three points behind

`ckpt_PAPER.pth.tar` is RECIPE512 at epoch 0. The run has since reached epoch 3,
and the per-checkpoint watcher measured epochs 1 and 2 on the way. Same 53
sequences, same 0.1 dB budget, same code path:

| epoch | q0 | q16 | q32 | q48 | q63 | mean |
|---|---|---|---|---|---|---|
| 0, the pinned checkpoint | 29.9 | 25.6 | 20.9 | 18.3 | 15.6 | **22.1** |
| 1 | 33.5 | 29.3 | 23.8 | 21.4 | 18.4 | **25.3** |
| 2 | 33.5 | 28.8 | 23.0 | 20.4 | 17.3 | **24.6** |

Epoch 1 is 3.2 points ahead of what the paper reports and epoch 2 is 2.5 ahead.
Two points is not enough to call epoch 1 a peak, so moving the headline there
would be picking the best of two rather than reporting the latest.

### The reading that was nearly published instead

The first pass through `results/signalled_*.json` looked like a collapse. BEST
went 28.9 to 19.8, FINE12 27.8 to 19.5, BEST128 27.6 to 19.2, all in the same
direction across four independent runs.

The test set had grown. The epoch-1 measurements are on 40 sequences and the
epoch-2 and epoch-3 ones on 53, because MCL-JCV and the smaller HEVC classes
landed in between. HEVC C is 8 tiles per frame at 832x480 and HEVC D is 2 at
416x240, and neither saves anything like a 1080p frame does. The whole apparent
drop was the test set, not the training. This is the same mistake the project
has already made twice in another form: comparing numbers measured over
different intervals, and comparing a table dB against a delivered dB.

### The comparison the ladder results cannot yet support

RECIPE512 and BEST are the same recipe. Same K, same split depth, same scaled
adapters, same grid seam repair, same 45,445,398 parameters, different runs.

| epoch | RECIPE512 | BEST | gap |
|---|---|---|---|
| 1 | 25.3 | 24.2 | 1.1 |
| 2 | 24.6 | 19.8 | 4.8 |

At epoch 2 two runs of one configuration are 4.8 points apart, which is as large
as the 256-against-128 tile difference the ladder table reports (256 px: 24.6;
128 px: 19.3 for BEST128 and 19.7 for FINE12). Until that variance is bounded by
repeated runs, no architectural comparison in this project is separable from
run-to-run noise, and the ladder table should say so rather than rank the rows.

BEST also falls from 24.2 at epoch 1 to 19.8 at epoch 2 while RECIPE512 does
not. One run of the pair degrades and the other does not. Not diagnosed.

### Anchor drift, which grows with training

The deepest exit is supposed to reproduce the released decoder. At the latest
RECIPE512 checkpoint it does not: -0.003 dB at q0, -0.015 at q32, -0.036 at q63.
At the high rate that is a third of the 0.1 dB budget, spent before any tile
exits early. The saving is measured against the released decoder, so drift
costs the method rather than flattering it, but it grows with training and at
some point the claim that the deepest exit is the released decoder stops being
true. Any move to a later checkpoint has to report this as its own row.

---

## 93. DECISIONS 91 was a units error in the checker, not a bug in configuration C

The open failure was "hybrid: rho=1 reproduces A", with C measuring 0.09 to 0.21
points above A at matched delivered decibels. It is closed, and the cause is not
what 91 assumed.

The checker compared the hybrid file's endpoint against a signalled file chosen
by `pick()`. Those two were not the same measurement: the hybrid run had been
made against one signalled run and the checker was reading another, and after
`signalled_RECIPE512_ctc53.json` was re-measured with four budgets the two lined
up. The endpoint now reproduces A to 0.0000, exactly, at every rate.

91 spent its length arguing that the discrepancy could not be bisection noise,
which was correct, and then concluded that one of the two code paths must be
charging differently, which was wrong. Both paths were right. What differed was
which file each side of the comparison was reading.

The lesson is the one this project keeps relearning in new costume: a
disagreement between two numbers is a disagreement between two measurements, and
the first question is whether they are measurements of the same thing. The
checker now goes through the same `sv()` the tables use, and where the hybrid
file carries no hook count it is compared against modelled endpoints rather than
measured ones, so the model's 0.008-per-decode offset cannot masquerade as a
failure of the interpolation.

check_paper is 85/85 for the first time.

---

## 94. Epoch 3 is snapshotted, and the earlier reading of the epoch series was wrong

`runs/RECIPE512/ckpt_eval.pth.tar` held epoch 3 and is overwritten by the
training loop, so it is now also `ckpt_PIN_e3.pth.tar`. That costs 182 MB and
buys the option of moving the paper onto it later, which the moving file would
otherwise have taken away without warning.

### The correction

DECISIONS 92 read the epoch series as peaking at epoch 1 and falling, and the
limitations section said so. Both were reading files that carry no hook count,
where `sv()` falls back to the arithmetic model, which over-reports by 0.4 to
0.8 points and not uniformly. Mixing the two definitions is the thing the whole
accounting pass existed to stop, and it had survived inside the one paragraph
about how much better the run gets.

On the hook count alone, same 53 frames, same 0.1 dB budget:

| epoch | mean saving | file |
|---|---|---|
| 0 | 21.5% | signalled_RECIPE512_ctc53.json, the pinned checkpoint |
| 1 | 22.8% | signalled_RECIPE512_e1.json |
| 2 | 24.0% | signalled_RECIPE512_0820_0140.json |
| 3 | 25.8% | signalled_RECIPE512_0820_2125.json |

Monotone, 4.3 points over four epochs, and epoch 4 was in progress at step
14,200 of 47,451 when this was written.

### What follows

Moving the paper to the latest checkpoint is no longer picking a peak, because
there is no peak; it is reporting the most recent measurement of a run that is
still improving. `scripts/repin.sh` does the whole chain in one command and
refuses to pin a moving file.

The decision is the author's. What is recorded here is that the argument against
moving, which I made in conversation, rested on a units error of mine.

## 95. The eval chain was on a card that stopped being ours (2026-08-21)

`watch_ckpts.sh` defaulted to `GPU=${2:-2}`. That was correct when GPU2 held
VERBATIM. VERBATIM ended, another user moved onto GPU2, and the default did not
notice: tonight's FINE12 evaluation started at 04:37 on a card `cankan` had been
running on for eight hours. Small (3.5 GB against 40 GB free) and short (~8 min),
but not ours to take, and the standing rule on this machine is explicit.

The fix is not a new number. A hardcoded index is a memory of who owned a card,
and cards change hands silently. `scripts/pick_gpu.sh` asks the driver instead:
it lists every GPU, reads the owner of each compute process on it, and returns
the one this account has to itself with the most free memory. If no card is free
of other users it prints nothing, so the caller gets an empty
`CUDA_VISIBLE_DEVICES` and fails loudly rather than quietly borrowing one.

`watch_ckpts.sh`, `pin_router.sh`, `run_beta_calibration.sh`, `run_signalled.sh`
and `run_signalled_256.sh` now go through it. The `launch_*.sh` scripts are left
exactly as they are: they record how the four running experiments were started,
and that record should not be edited after the fact.

The running eval was left to finish rather than killed. Killing it risked the
watcher marking the checkpoint evaluated with no measurement written, and the
next one lands on our own card regardless.

## 96. A check for the paragraph that says the same thing twice (2026-08-21)

Two paragraphs stated their point and then stated it again. Section 5.1 opened
"A tenth of a decibel is an engineering convention. It is an engineering
convention, chosen because...". Contribution (iii) gave the bit rule's result in
three sentences and then repeated all three in different words, with a sentence
fragment wedged between them. Every existing check passed both: they are not
long, not hedged, not bolded, and every number in them is right.

`prose_audit` now counts six-word phrases that occur twice inside one paragraph.
Six, not four: at four it fires nineteen times on this paper and all nineteen are
technical phrases the subject genuinely repeats, like "the calibrated bit rule".
At six a repeat inside one paragraph is an accident, and both real defects were
six-word repeats.

Two things the check needed before it was usable. The supplement builds numbers
with f-strings, so its paragraph source carries Python inside braces; left in,
that code counted as words and produced 43 false repeats. Braces now collapse to
a placeholder. And stripped mathematics leaves runs of single letters, so a
repeat must carry three distinct words of three letters or more.

Fatal in the paper, reported in the supplement. The supplement compares the same
quantity across rates, and "at q0 the smaller tile is ahead ... at q63 the
smaller tile is behind" is that sentence working, not failing.

## 97. Figure 18 was the words "[spread.png missing]" (2026-08-21)

For an unknown number of builds, Figure 18 of the paper was not a figure. It was
the italic string `[spread.png missing]`, set in caption type where the picture
should be, under a correct caption with correct numbering.

Every check passed. `check_paper` verified the claims the caption makes, and
they were right. `check_twins` confirmed main.tex and build_pdf.py agree on the
figure list, and they did. `check_tex` and `prose_audit` have no opinion about
images. The figure numbering was continuous in both files, because `_autonum`
increments whether or not a picture arrives.

The cause: `spread_figs.py` writes to `docs/figures` and `build_pdf.py` reads
`paper/figures`. Every other figure had been copied across at some point;
this one never was. It was found by counting the numbers in the built PDF's text
layer and noticing that 18 was absent between 17 and 19.

Three changes. `spread_figs.py` writes to both directories. `fig()` prints the
missing path to stderr and the build prints a summary line at the end, so the
placeholder is never silent again. And `check_paper` now checks that every
figure `build_pdf.py` names exists where `build_pdf.py` looks for it, which
makes a missing picture fail the same command that catches a stale number.

The general lesson is the one from DECISIONS 91 and 96: a check that reads the
source cannot see what the source failed to produce. Read the artefact.

## 98. A blank page in the middle of the paper (2026-08-21)

Page 5 of the built PDF was one word and a page number. Page 17 was three
fifths empty. Both are `figure_wide`, which has to force a page break because
reportlab has no float mechanism: it switches to a full-width page template,
emits the figure into the top band, and switches back. If the story happens to
be three lines into a fresh page when that runs, those three lines are the whole
page.

So the placement of the call in the story, not the figure, decides whether a
page is wasted. `scratchpad/sweep_wide.py` moves the call to each statement
boundary within ±9000 characters, builds, and counts text elements per page from
`pdftotext -bbox`. For `patchify.png`: the original anchor gives 21 pages with a
two-element page, and eighteen of the thirty-seven candidates give 20 pages with
no page thinner than the references tail. We took the good anchor nearest the
original so the figure still sits where the text discusses it.

Two things this cost, recorded because both were tempting and both were wrong.
Rebuilding the qualitative figure two-by-two at column width removes the page
break, and it also letterboxes a 16:9 frame beside square crops and runs into
the 1.25-inch cap that `figure()` puts on a column figure, which would leave the
crops under an inch. The picture is not the problem. Where the call sits is.

The measurement to keep: text elements per page. Fill measured as the lowest
line on the page reads 0.96 on every page of a paper with a blank one in it,
because the page number is on the blank page too.

## 99. Twenty pages, and none of them empty (2026-08-21)

The target was exactly twenty pages. Getting there meant fixing two things and
adding one.

Fixed: both `figure_wide` calls were forcing their page break three lines into a
fresh page, so page 5 was one word and page 17 was three fifths white. The sweep
in DECISIONS 98 moved each to the anchor nearest its original position that
wastes no page.

Added, from measurement rather than padding. A qualitative figure in Section 6:
the released decoder and ours from the same latent, the crop centred on the tile
that gave up the most. Two tables promoted from the supplement, generated from
the same dumps the supplement uses so the two documents cannot disagree: the
seven mechanisms that were built and dropped with the number that ended each,
and one frame per class showing the exit histogram collapsing to a single rung
at 416x240. Two subsections that were only in the supplement: what a set-level
budget hides -- 29 to 34 of the 53 sequences receive a decode worse than the
budget, because one multiplier is bisected for the set mean -- and what each
router input is worth, where the stem alone reaches the agreement that all five
inputs together reach.

And one new figure, `exit_vs_rate.png`: the allocation walking down the ladder
as the rate rises, 67% of tiles at the shallowest live exit at q0 against 22% at
q63. It is the paper's own explanation of why the saving falls with rate, drawn
rather than asserted, and panel b is the saturation of Section 5.4 in the same
units.

Final state: 20 pages, thinnest page 330 text elements and that is the tail of
the references, every other page above 780. Figures 1 to 35 and tables 1 to 24
with no gaps.

## 100. A caption is not a reference (2026-08-21)

Twenty-six of thirty-five figures and fifteen of twenty-four tables had no
sentence pointing at them. Every one had a correct caption, correct numbering
and a place in the flow, and nothing complained, because no check in this repo
knew the difference between a figure that exists and a figure the reader is sent
to. In a two-column paper that is not cosmetic: a float lands where it fits, and
without a reference the reader has no idea which paragraph it belongs to.

The reportlab build could not write one. `_autonum` numbers captions as they are
emitted, so a sentence written before a figure has no number to use. Now
`_figure_numbers()` and `_table_numbers()` read the emission order out of this
file's own source before anything is laid out -- the banner first, then
`content()` in source order, since there are no loops -- and prose writes
`[[fig:name]]` or `[[tab:name]]`, which `sub()` resolves. An unknown name raises
rather than printing a wrong number.

Every reference was placed on the sentence that already made the figure's point,
not appended as a new sentence. `check_paper` now reports how many figures and
tables the prose names, and the LaTeX side is checked by `check_tex`, which
already fails on a dangling `\ref`.

35/35 and 24/24, both documents, and the paper is still exactly 20 pages.

## 101. Column by column, not page by page (2026-08-21)

The author looked at the built PDF and listed six places with visible white:
page 15's right column "like a field", then pages 2, 7, 8, 16 and 17. Every
check here passed. The per-page fill measure read 0.96 for all twenty pages,
including the one that was blank, because the footer page number sits at the
bottom of every page and a naive "lowest text on the page" finds it.

Measured properly -- per column, footer excluded -- the list matched the author's
exactly: p15R 0.31, p2R 0.78, p8R 0.81, p17R 0.81, p16R 0.87, p7L 0.88.

The mechanism is `figure_wide`. reportlab has no float, so a full-width figure
switches page template, which forces a page break; wherever that break lands,
the rest of that page's columns are lost. `qualitative` was the worst case, and
it did not need to be full width: the fourth panel was the whole frame with a box
on it, which the caption can say in words. Three square crops tile a column
exactly. `patchify` stays wide and keeps its break, because panel d is a
left-to-right pipeline of four tensor shapes and a sentence, and at column width
matplotlib's tight bounding box expands the saved figure straight back to eight
inches to hold them.

For the rest, story order is the only lever, so `scratchpad/fill_columns.py`
finds the block that starts the column after the shortest one and tries it at
paragraph boundaries either side, keeping whatever leaves the least empty column.
One trap: the first version moved a figure among its neighbouring figures and
seven candidate layouts came out identical to two decimal places -- reordering two
figures that sit together changes nothing about how much text is above them.

Empty column space went 1.63 to 0.52 of a column and the worst column 0.31 to
0.76. `check_layout.py` now measures this from `check_paper`, so the next one is
caught by a command rather than by eye.

## 102. Two commits that never reached the PDF (2026-08-21)

An insertion swallowed the newline before the following call and `build_pdf.py`
stopped parsing. For two commits the build failed, and I read page counts and
column fills off the stale PDF and reported them.

`pdfinfo` on yesterday's PDF is indistinguishable from `pdfinfo` on today's. The
build prints `-> paper/FLEX-UF.pdf` when it writes one, and that line was simply
absent from output I was grepping for other things.

`check_paper` now compiles `build_pdf.py` and compares the PDF's mtime against
it. This is the third time tonight the same lesson has come around: DECISIONS 97
said a check that reads the source cannot see what the source failed to produce.
Here the check read the artefact and still could not tell, because the artefact
was last night's.

## 103. The heartbeat was wrapped in a loop it never left (2026-08-21)

`heartbeat_status.sh` loops internally and prints one line every 300 seconds. The
monitor I set up ran it inside another `while true` and piped it to `tail -1`.
`tail -1` prints nothing until its input closes, and the script never exits, so
the five-minute check produced no events at all after I restarted it. The monitor
was alive, the script was running, and nothing arrived.

The script is the event stream. The monitor now runs it directly.

## 104. The figures were a second, unchecked copy of the results (2026-08-21)

The author asked for the plots to be explained, the formulations set to
conference standard, and nothing overlapping or cut off. Auditing all sixty
figures for that turned up something larger: five of them printed numbers the
paper had corrected days earlier.

`rd_spread` said "0.1 dB, 24% saved" beside a table saying 21.5. `tradeoff` and
`budget_band` drew a 41.9% ceiling, which is the architectural ceiling as it
stood before the FFN accounting fixed it to 38.3. `raterank` and `router_ab`
drew configuration B from `router_..._b01_fixed.json`, an earlier experiment, at
27.2% at q0 where every table says 23.6 -- and with that file the trained head
sits ABOVE the calibrated bit rule at the low rates, so the figure contradicted
the paper's third claim on the page that states it.

One cause, three faces. Each figure script chose its own result file by a
hand-written first-that-exists list, its own saving definition, and its own
output directory. `make_paper_tables` had already fixed the first two for the
tables and nothing had carried the fix across. Twelve producers wrote
`docs/figures` only, so the copy the paper reads went stale in silence, and
eight more figures had not been regenerated since their own script changed.

What is in place now: one `sv()` and one `pick()` in `scripts/savings.py`, used
by the tables and the figures alike; every producer writes both directories; and
`check_figs_fresh` compares each of the sixty figures against the result files
its producer opens *and against the producer itself*, which is the case that
actually bit.

The rest of the audit, in one list. Text sitting on data in four figures, a
truncated axis label, two panel letters on top of the labels beside them, an
axis in two notations, six unkeyed visual elements -- shaded bands, endpoint
dots, dashed fits, tick marks -- and a Lorenz panel whose curves stopped at half
the axis because the sweep did.

And one claim that had quietly become false: configuration C's ends were said to
reproduce A and B "to the second decimal". They sit 0.43 to 0.78 points above
them, because the C sweep was never hook-counted while A and B were. The paper
says so now and `check_paper` holds both ends.

The lesson is DECISIONS 97 again in a larger form: a figure is a cached
artefact, and a cache with no invalidation is a second copy of the results that
nobody is checking.
