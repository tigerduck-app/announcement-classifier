# 公告分類模型

為台科大 Moodle 課程公告標上 20 類標籤中的一個或多個（多標籤分類，不生成文字）。模型是 `intfloat/multilingual-e5-base` 的微調版，CPU 上單則約 35 ms。

[English](README.en.md)

## 標籤

定義與判斷規則見 [`labeling_guide.md`](labeling_guide.md)。

- `exam_info` 考試資訊：某次考試的日期、時間、地點、範圍、題型或規則
- `exam_change` 考試異動：已公布的考試改期、改地點、改範圍、取消，或澄清如期舉行
- `exam_seating` 考試座位：考試座位表，或依學號分配考場
- `exam_review` 考卷發放：發還考卷、看考卷、核對考卷分數的時間地點
- `exam_prep` 考題資料：考古題、模擬考、考前複習課、考題與小考的解答或講解
- `grade_release` 成績公布：任何成績已公布、已更正，或請學生核對
- `assignment_new` 新作業：發布作業、專題、報告等及其要求與截止時間
- `assignment_change` 作業時間異動：截止時間更動、開放補交或遲交、繳交區重開
- `assignment_note` 作業補充：對已發布作業的後續說明、更正、提示
- `class_cancel` 停課：某一次課不上
- `class_change` 上課異動：上課時間、教室、方式改變，補課、調課
- `class_prep` 上課準備：下次上課要帶的物品、先讀或先安裝的東西、教科書訂購
- `attendance` 出缺勤：點名方式、簽到、缺課紀錄、請假規定
- `materials` 教材：投影片、講義、錄影、習題與習題解答、閱讀資料
- `grading_policy` 評分方式：配分比重、評分標準、加分規則及其變更
- `course_info` 課程資訊：聯絡方式、Office hour、課程群組、大綱、進度表
- `grouping` 分組：分組表單、名單、期限、換組
- `admin` 行政事項：報名、加退選、停修、證明文件等行政程序
- `survey` 問卷：教學意見調查、課程問卷、回饋表單
- `event` 活動：演講、講座、研討會、競賽、展覽等活動

## 成績

在 300 則人工標註的 gold 公告上，以 2-fold cross-fitting 估計（閾值在另一半挑選）：

| micro F1 | macro F1 | 完全命中 |
|---|---|---|
| 0.870 | 0.843 | 0.703 |

分類閾值為全域單一值 0.35（`threshold.json`）。逐類：

| 標籤 | 筆數 | precision | recall | F1 |
|---|---|---|---|---|
| `exam_info` | 68 | 0.914 | 0.941 | 0.928 |
| `exam_change` | 13 | 0.909 | 0.769 | 0.833 |
| `exam_seating` | 7 | 1.000 | 1.000 | 1.000 |
| `exam_review` | 16 | 0.941 | 1.000 | 0.970 |
| `exam_prep` | 14 | 0.857 | 0.429 | 0.571 |
| `grade_release` | 35 | 0.889 | 0.914 | 0.901 |
| `assignment_new` | 41 | 0.872 | 0.829 | 0.850 |
| `assignment_change` | 7 | 0.714 | 0.714 | 0.714 |
| `assignment_note` | 14 | 1.000 | 0.571 | 0.727 |
| `class_cancel` | 38 | 0.925 | 0.974 | 0.949 |
| `class_change` | 36 | 0.875 | 0.778 | 0.824 |
| `class_prep` | 14 | 0.923 | 0.857 | 0.889 |
| `attendance` | 17 | 0.789 | 0.882 | 0.833 |
| `materials` | 63 | 0.862 | 0.889 | 0.875 |
| `grading_policy` | 13 | 0.750 | 0.692 | 0.720 |
| `course_info` | 19 | 1.000 | 0.895 | 0.944 |
| `grouping` | 14 | 0.778 | 1.000 | 0.875 |
| `admin` | 10 | 0.875 | 0.700 | 0.778 |
| `survey` | 6 | 0.750 | 1.000 | 0.857 |
| `event` | 13 | 0.786 | 0.846 | 0.815 |

## HTTP API

`scripts/serve.py` 提供標準函式庫的 HTTP 服務，一次只算一個請求，不記錄請求內容，沒有驗證，請只開在受信任的網路。

```
GET  /health       -> {"status": "ok", "labels": [...], "threshold": 0.35, "release": "model-2026-11-15"}
POST /v1/classify  {"posts": [{"subject": "...", "message": "<Moodle HTML>"}, ...]}
                or {"texts": ["<title>\n\n<Markdown content>", ...]}     (at most 64 items)
  -> {"results": [{"labels": ["exam_info"], "scores": {"exam_info": 0.97, ...}}, ...]}
```

`posts` 會先依訓練資料的方式把 Moodle HTML 轉成文字（`scripts/moodle_text.py`），回應另附該 `text`。模型讀取前會遮掉網址、email 與學號。

## 部署

需要 Python 3.13（`install.sh` 會用 uv 建立，沒有時 uv 會自動下載）、`uv` 與已登入的 `gh`。

Release 附上 `model.safetensors`、`config.json`、`tokenizer.json`、`tokenizer_config.json`、`threshold.json`、`eval.json` 與 `SHA256SUMS`。

```bash
deploy/install.sh model-2026-11-15   # 下載、驗證 SHA256SUMS、建立 venv、執行 test_serve.py
.venv/bin/python scripts/serve.py model --host 127.0.0.1 --port 40002 --threads 2
```

`deploy/ai.tigerduck.classifier.plist` 是 macOS launchd 的服務定義，路徑依部署環境調整。服務沒有驗證，請讓防火牆擋住區域網路。

## 重訓

```bash
.venv/bin/python scripts/train.py <announcements.jsonl> <out dir> --labels reviewed
.venv/bin/python scripts/evaluate.py <out dir>
.venv/bin/python scripts/compare_models.py <model A> <model B>
```

`train.py` 依課號（去掉學期）切出 10% 驗證集，訓練 4 epoch，保留驗證 loss 最低的一輪。`evaluate.py` 在 gold 上選閾值並寫出 `eval.json`。

## 訓練資料格式

訓練資料不公開。腳本預期的檔案：

- `<announcements.jsonl>`：每行 `{"id": "...", "text": "<標題>\n\n<內文>"}`，內文為 Markdown 純文字。
- `labels/<name>.jsonl`：每行 `{"id": "...", "labels": ["exam_info", ...]}`，標籤只能用上面 20 個。
- `gold/gold_items.jsonl`（`id`、`text`）與 `gold/gold_labels.jsonl`（`id`、`labels`）：人工標註的評估集，訓練時排除。

## 限制

- 只針對台科大 Moodle 的繁體中文（含部分英文）課程公告。其他學校或文體沒有量過。
- gold 只有 300 則，少數類別只有 6 到 13 則，逐類數字誤差大。`exam_prep`、`assignment_change`、`grading_policy`、`assignment_note` 較弱。
- 訓練資料中的人名沒有完全遮罩。
- 測試用的公告在 `tests/fixtures/synthetic_posts.jsonl`，是虛構的。

## 授權

[AGPL-3.0](LICENSE)。基礎模型 [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) 為 MIT 授權，感謝原作者。
