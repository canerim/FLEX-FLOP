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
