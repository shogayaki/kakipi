# Neurosurgery Image Annotation Guide

著作権切れの脳外科教科書から画像を取得し、物体検出モデル学習用のアノテーションデータを作成する手順。

## 画像ソース（パブリックドメイン）

以下の著作権切れ書籍から画像を取得します（すべて1928年以前に出版）。

| 書籍 | 著者 | 出版年 | Internet Archive ID |
|------|------|--------|---------------------|
| The Pituitary Body and Its Disorders | Harvey Cushing | 1912 | `pituitarybodyits00cushuoft` |
| Tumors of the Nervus Acusticus | Harvey Cushing | 1917 | `cu31924032537403` |
| Surgery of the Brain and Spinal Cord, Vol.1 | Fedor Krause | 1909 | `b21272116_001` |
| Surgery of the Brain and Spinal Cord, Vol.2 | Fedor Krause | 1912 | `surgeryofbrainsp02krauuoft` |
| Surgery of the Brain and Spinal Cord, Vol.3 | Fedor Krause | 1912 | `surgerybrainand01kraugoog` |

## ラベルクラス（14クラス）

| ID | クラス名 | 日本語 | カテゴリ |
|----|----------|--------|----------|
| 1 | `cerebrum` | 大脳 | 解剖学的構造 |
| 2 | `cerebellum` | 小脳 | 解剖学的構造 |
| 3 | `brainstem` | 脳幹 | 解剖学的構造 |
| 4 | `ventricle` | 脳室 | 解剖学的構造 |
| 5 | `dura_mater` | 硬膜 | 解剖学的構造 |
| 6 | `artery` | 動脈 | 解剖学的構造 |
| 7 | `vein` | 静脈 | 解剖学的構造 |
| 8 | `cranial_nerve` | 脳神経 | 解剖学的構造 |
| 9 | `spinal_cord` | 脊髄 | 解剖学的構造 |
| 10 | `skull` | 頭蓋骨 | 解剖学的構造 |
| 11 | `tumor` | 腫瘍 | 病変 |
| 12 | `hemorrhage` | 出血 | 病変 |
| 13 | `pituitary_gland` | 下垂体 | 解剖学的構造 |
| 14 | `surgical_instrument` | 手術器具 | 手術器具 |

## セットアップ手順

### 1. 画像のダウンロード

```bash
# 利用可能な書籍一覧を表示
python download_neurosurgery_images.py --list

# Cushing の下垂体の本からページ画像をダウンロード
python download_neurosurgery_images.py --book cushing_pituitary --pages 30-200

# Krause の脳外科教科書 Vol.1 をダウンロード
python download_neurosurgery_images.py --book krause_v1 --pages 50-300

# PDFとしてダウンロードする場合
python download_neurosurgery_images.py --book cushing_pituitary --pdf
```

### 2. PDF からの画像抽出（PDFダウンロードした場合）

```bash
pip install pdf2image
# Linux: apt-get install poppler-utils
# macOS: brew install poppler

python extract_pdf_pages.py \
    --input ./data/neurosurgery/raw/cushing_pituitary.pdf \
    --output-dir ./data/neurosurgery/raw \
    --pages 30-200 --dpi 200
```

### 3. 画像のリサイズ

```bash
python resize_images.py \
    --raw-dir ./data/neurosurgery/raw \
    --save-dir ./data/neurosurgery/images \
    --ext jpg \
    --target-size "(800, 600)"
```

### 4. Train/Test 分割

リサイズ済み画像を手動で分割します（目安: train 80%, test 20%）。

```bash
# 画像をtrain/testに移動
mv ./data/neurosurgery/images/[0-3]*.jpg ./data/neurosurgery/images/train/
mv ./data/neurosurgery/images/[4-9]*.jpg ./data/neurosurgery/images/test/
```

### 5. labelImg でアノテーション

```bash
pip install labelImg

# predefined_classes.txt を使ってlabelImgを起動
labelImg ./data/neurosurgery/images/train \
    ./data/neurosurgery/predefined_classes.txt
```

**アノテーションのコツ:**
- `w`: バウンディングボックスを描画
- `d`: 次の画像
- `a`: 前の画像
- `Ctrl+S`: 保存
- Pascal VOC 形式（XML）で保存する
- 1画像あたり複数のラベルを付けてOK
- 構造が不明瞭な場合は `difficult` フラグを使用

### 6. XML → CSV 変換

```bash
# Train データ
python xml_to_csv.py \
    -i ./data/neurosurgery/images/train \
    -o ./data/neurosurgery/annotations/train_labels.csv \
    -l ./data/neurosurgery/annotations/

# Test データ
python xml_to_csv.py \
    -i ./data/neurosurgery/images/test \
    -o ./data/neurosurgery/annotations/test_labels.csv
```

### 7. TFRecord 生成

```bash
python generate_tfrecord.py \
    --csv_input=./data/neurosurgery/annotations/train_labels.csv \
    --output_path=./data/neurosurgery/annotations/train.record \
    --label_map=./data/neurosurgery/label_map.pbtxt \
    --img_path=./data/neurosurgery/images/train

python generate_tfrecord.py \
    --csv_input=./data/neurosurgery/annotations/test_labels.csv \
    --output_path=./data/neurosurgery/annotations/test.record \
    --label_map=./data/neurosurgery/label_map.pbtxt \
    --img_path=./data/neurosurgery/images/test
```

## ディレクトリ構成

```
data/neurosurgery/
├── raw/                    # ダウンロードした生画像（.gitignore済み）
├── images/
│   ├── train/              # リサイズ済み学習画像 + XMLアノテーション
│   └── test/               # リサイズ済みテスト画像 + XMLアノテーション
├── annotations/            # CSV, TFRecord, label_map の出力先
│   └── .gitkeep
├── label_map.pbtxt         # 14クラスのラベル定義
└── predefined_classes.txt  # labelImg用の事前定義クラスリスト
```

## アノテーション品質ガイドライン

1. **バウンディングボックスは対象を密にカバー**: 余白を最小限に
2. **重複する構造にはそれぞれラベルを付与**: 例: 頭蓋骨の中に大脳、その中に腫瘍
3. **テキストのみのページはスキップ**: 解剖図・手術写真・病理写真のみをアノテーション
4. **難しい例には `difficult` フラグ**: 構造が不明瞭、部分的にしか見えない場合
5. **最低20枚以上をアノテーション**: 各クラスが最低5回以上出現するように
