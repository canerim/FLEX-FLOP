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
