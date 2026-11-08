# ate-dehazing-reference

除霧論文（Upadhyay & Sarawadekar, IEEE TVLSI 2026）的 Python golden model，用來和 RTL、ISR 逐值比對。

> **注意**
> - 數值規則都是暫定值，尚未經組內確認。
> - 不是論文復現；本 repo 沒有 RTL。
> - 已在 Mac（Apple Silicon）的乾淨環境照本教學建立環境，並跑過兩個 selfcheck、`demo` 與 `compare`（2026-11-08）；Windows、Linux 還沒試過。遇到問題請告訴李艾軒。

## 1. 快速開始

先安裝 uv 和 git（第 2 節），再依序執行。以下指令三個系統通用：

```
git clone https://github.com/ahhhh71828/ate-dehazing-reference.git
cd ate-dehazing-reference
uv sync --locked
uv run --locked python -B ate_golden.py selfcheck
uv run --locked python -B ate_pte_lut.py selfcheck
```

- `git clone` 會在目前所在的資料夾裡建立 `ate-dehazing-reference`。
- 兩個 selfcheck 的最後一行分別是 `結果：489/489 PASS` 和 `結果：64/64 PASS`，看到就代表環境正常。

## 2. 安裝 uv 與 git

### 2.1 uv

uv 會自動準備正確版本的 Python 與套件，不需要另外安裝 Python。

**Windows（PowerShell）**

```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS、Linux**

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安裝後執行 `uv --version`，應該顯示版本號。

指令來源：uv 官方安裝說明 https://docs.astral.sh/uv/getting-started/installation/

### 2.2 git

**Windows**

到 https://git-scm.com/install/windows 下載 Git for Windows 並安裝。也可以在 PowerShell 執行：

```
winget install --id Git.Git -e --source winget
```

**macOS、Linux**

執行 `git --version`。沒有顯示版本號的話，依 https://git-scm.com/install/ 的說明安裝。

## 3. 建立環境

進入 repo 資料夾（有 `pyproject.toml` 的那一層）再建立環境。用第 1 節的 `git clone` 取得的話，以下指令三個系統通用：

```
cd ate-dehazing-reference
uv sync --locked
```

下載 ZIP 的話，改成 `cd` 到解壓後的資料夾，名稱可能是 `ate-dehazing-reference-main`。

應該看到：產生 `.venv` 資料夾，沒有錯誤訊息。第一次執行會下載套件，電腦沒有 Python 3.12 時也會一併下載，需要網路。

## 4. 不需要影像的檢查

在 repo 資料夾執行，三個系統通用。`-B` 讓 Python 不產生 `__pycache__` 資料夾。

| 指令 | 最後一行應該是 |
|---|---|
| `uv run --locked python -B ate_golden.py selfcheck` | `結果：489/489 PASS` |
| `uv run --locked python -B ate_pte_lut.py selfcheck` | `結果：64/64 PASS` |
| `uv run --locked python -B ate_golden.py mutation`（選用，較久） | `結果：79/79 個 mutation 在預期項目失敗` |
| `uv run --locked python -B ate_pte_lut.py mutation`（選用） | `結果：16/16 個 mutation 在預期項目失敗；對照組 2/2 依預期不算抓到` |
| `uv run --locked python -B ate_pte_lut.py measure`（選用，約需 0.5 GB 記憶體） | 倒數第二行是 `判定：推導與手算都符合` |

## 5. 使用自己的影像

路徑有空格時要加雙引號。下面以浮點版除霧為例：

**Windows（PowerShell）**

```
uv run --locked python -B ate_golden.py demo "C:\Users\你的名稱\Pictures\hazy.jpg"
```

**macOS、Linux**

```
uv run --locked python -B ate_golden.py demo ~/Pictures/hazy.jpg
```

應該看到：印出統計值、A_F 與 A_N、t 的範圍，最後一行是 `未指定 --out-dir：不寫檔`。加上 `--out-dir out` 會在 `out` 資料夾寫出除霧結果與對照圖。

其他影像命令的路徑寫法相同，下表以 `影像.jpg` 代表：

| 用途 | 指令 | 結果 |
|---|---|---|
| 定點版（目前全部定點的設定） | `uv run --locked python -B ate_golden.py static 影像.jpg --inv-frac 20 --pala-fixed --pte-frac 10 --sd lut` | 印出排程表與各定點步驟造成的差異，不寫檔 |
| PTE 查表版和定點版的比較 | `uv run --locked python -B ate_pte_lut.py compare 影像.jpg` | 倒數第二行是 `判定：guard 都成立`，不寫檔 |

`demo` 用影像本身的平均灰階當 T_h，`static` 第一次統計用預設的 T_h = 128，兩者的輸出不能直接比較。

## 6. 選用：資料集

只有要做畫質評估時才需要。資料集的授權還沒查清楚：請各自下載，不要再傳給別人，也不要放進 repo。

### 6.1 O-HAZE（45 組室外有霧／無霧照片）

- 官方頁：https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/
- 官方頁上的下載連結目前失效（2026-11-01 測試為 404）。可改用同一目錄下的這個網址，它不是官方頁列出的連結：https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/O-HAZE.zip
- 解壓後的結構：`# O-HAZY NTIRE 2018\hazy\`、`# O-HAZY NTIRE 2018\GT\`，各有 45 張。

檢查配對：會讀取並解碼每張影像，檢查配對與影像格式，不執行除霧，也不寫檔。指令三個系統通用，下例是 Windows 路徑，請換成你的解壓位置：

```
uv run --locked python -B ate_quality.py pairs "D:\data\# O-HAZY NTIRE 2018"
```

應該看到：`配對狀態：ok 45；missing 0`。

## 7. 常見問題

| 狀況 | 處理方式 |
|---|---|
| 安裝後輸入 `uv` 或 `git` 顯示找不到指令 | 關掉終端機重新開啟，再試一次 |
| Windows 不允許執行 uv 的安裝指令 | 確認有照第 2.1 節完整輸入（含 `-ExecutionPolicy ByPass`）；或改用 `winget install --id=astral-sh.uv -e` |
| `uv sync` 顯示 lock 需要更新 | `pyproject.toml` 或 `uv.lock` 被改過。不要執行 `uv lock`，重新下載 repo |
| selfcheck 的結果不是上面的數字 | 確認指令前面有 `uv run --locked`，並確認 repo 是最新版本 |
| 顯示 `無法讀取：<路徑>` | 路徑打錯，或格式不支援（例如 iPhone 的 HEIC，請先轉成 JPG 或 PNG） |
| 影像很大時跑很久或記憶體不足 | 先用 1920×1080 以下的影像 |
| Linux 上 `import cv2` 出現 `libGL.so.1` 錯誤 | 系統缺少圖形函式庫，需要另外安裝（一般經驗，本專案沒有在 Linux 測試過） |

## 8. 回報問題

請附上：

- 作業系統（Windows、macOS 或 Linux）
- `uv --version` 的輸出
- repo 的版本：用 git 取得的話附上 `git rev-parse --short HEAD` 的輸出，下載 ZIP 的話附上下載日期
- 完整指令，以及輸出的最後 20 行

## 9. 檔案說明

| 檔案 | 用途 |
|---|---|
| `ate_golden.py` | golden model：統計路徑、CPU 參數更新、PALA、PTE、SD、影片與靜態排程，以及各種定點設定 |
| `ate_pte_lut.py` | PTE 查表版的候選（B = 8、10），尚未選定 |
| `ate_quality.py` | 畫質評估工具：O-HAZE 的配對檢查與 PSNR／SSIM（選用，需要第 6.1 節的資料集） |
| `pyproject.toml`、`uv.lock`、`.python-version` | 環境設定，請不要修改 |

演算法、符號與數值格式的說明，見組內的《Golden Model 技術說明》。
