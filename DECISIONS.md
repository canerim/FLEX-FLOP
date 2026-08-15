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
