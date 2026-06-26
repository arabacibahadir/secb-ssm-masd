# SECB-SSM Subject-Independent LOSO Evaluation

Bu klasör, mevcut `attention_mamba/train.py` içindeki `MambaEEG` sınıfını kaynak
dosyayı değiştirmeden yükler ve strict leave-one-subject-out (LOSO) değerlendirmesi
yapar.

## Protokol

- 5 dış fold: S1-S5 sırayla tamamen test kümesinde tutulur.
- Validation, kalan dört katılımcının her birinden seçilen birer tam kayıttan oluşur.
- Aynı kaydın pencereleri train ve validation arasında bölünmez.
- Kanal ortalaması ve standart sapması yalnız training kayıtlarından hesaplanır.
- Pencere uzunluğu 256 örnek, stride 240 örnek ve örtüşme 16 örnektir.
- Proxy etiketleri: 0-10 dakika Focused, 10-20 dakika Unfocused, >20 dakika Drowsy.
- Full profil seed 42, 43, 44, 45 ve 46 ile toplam 25 eğitim çalıştırır.

## Katılımcı eşleme varsayımı

Kamuya açık `.mat` dosyalarında açık participant ID alanı bulunmamaktadır.
`subject_map.csv`, özgün çalışmada bildirilen katılımcı başına yedi deney ve
dosyaların kronolojik oluşturulma tarihleri kullanılarak şu eşlemeyi tanımlar:

- S1: kayıt 1-7
- S2: kayıt 8-14
- S3: kayıt 15-21
- S4: kayıt 22-28
- S5: kayıt 29-34

Son blokta yalnız altı kayıt bulunmaktadır. Bu eşleme bilimsel raporda açık bir
metadata varsayımı olarak belirtilmelidir.

## Kurulum

Çalışma klasörünün kökünde:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\loso_evaluation\requirements.txt
```

CUDA destekli PyTorch kurulumu gerekiyorsa uygun komutu PyTorch'un resmi kurulum
sayfasından seçin. Kod CUDA varsa otomatik kullanır; aksi halde CPU'da çalışır.

## Lisans ve veri erişimi

The source code in this repository is released under the MIT License.
The MASD EEG dataset is not redistributed in this repository and should be
obtained from the original Kaggle dataset page:
https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection.
Users are responsible for complying with the dataset license and terms of use.

## Önce smoke testi

Smoke profil tek fold (S1), tek seed (42), iki epoch ve split başına en fazla
64 pencere kullanır. Bu profil yalnız pipeline doğrulaması içindir; bilimsel
sonuç olarak raporlanmamalıdır.

```powershell
.\.venv\Scripts\python.exe -m loso_evaluation.run --profile smoke
```

## Tam LOSO çalışması

```powershell
.\.venv\Scripts\python.exe -m loso_evaluation.run --profile full
```

Kesilen çalışma aynı komutla yeniden başlatılabilir. `status: complete` içeren
koşular otomatik atlanır. Baştan çalıştırmak için `--force`, `.mat` önbelleğini
yeniden üretmek için `--rebuild-cache` kullanın.

Tek fold veya seed çalıştırma örneği:

```powershell
.\.venv\Scripts\python.exe -m loso_evaluation.run --profile full --subjects S3 --seeds 42
```

## Çıktılar

`results/<profile>/seed_<seed>/test_<subject>/` altında:

- `result.json`: validation/test metrikleri ve çalışma ayarları
- `split_manifest.json`: kayıt ayrımları ve train-only normalizasyon değerleri
- `history.json`: epoch geçmişi
- `predictions.npz`: sıkıştırılmış tahmin dizileri
- `predictions.csv.gz`: kayıt ve pencere kimlikli tahmin tablosu
- İsteğe bağlı `best_model.pt` (`--save-checkpoints`)

`results/<profile>/` altında:

- `summary.json`
- `summary.csv`
- `loso_results.md`
- `confusion_matrix.png`

Ana sonuç, her seed için beş held-out katılımcının tahminleri birleştirildikten
sonra hesaplanır; ardından beş seed üzerinden ortalama ve standart sapma raporlanır.

## Kod arşivine dahil edilmemesi gerekenler

Dergi veya ek materyal kod paketine çalışma sırasında üretilen `cache/`,
`results/`, `__pycache__/`, model checkpointleri ve veri dosyaları dahil
edilmemelidir. Kod, kamuya açık MASD `.mat` dosyalarının kullanıcı tarafından
yerel olarak indirilip `EEG Data/` klasörüne konmasını bekler.

## Makalede yorumlama

Bu çalışma yalnız SECB-SSM için LOSO sonucu üretir. EEGNet, TCN ve diğer baseline
sonuçları mevcut epoch-wise protokole aittir. Bu nedenle LOSO sonucuyla
baseline üstünlüğü iddia edilmemeli; baseline karşılaştırması açıkça
"within the original epoch-wise protocol" şeklinde sınırlandırılmalıdır.
