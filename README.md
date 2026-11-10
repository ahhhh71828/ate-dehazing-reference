# ate-dehazing-reference

除霧論文（Upadhyay & Sarawadekar, IEEE TVLSI 2026）的 Python 軟體 reference v0，以及 PTE 查表的候選版本。之後要用來和 RTL、ISR 逐值比對；目前還沒有測試向量的匯出（第 1 節）。

> **注意**
> - 數值規則都是暫定值，尚未經組內確認；本版實際使用的規則列在第 7 節。
> - 不是論文復現；本 repo 沒有 RTL。
> - 已在 Mac（Apple Silicon）的乾淨環境照本教學建立環境，並跑過兩個 selfcheck、`demo` 與 `compare`（2026-11-08）；Windows、Linux 還沒試過。遇到問題請告訴李艾軒。

## 1. 本版內容

- 演算法檔在首次交付（2026-11-08）時就已經在本 repo，之後沒有修改：
  - `ate_golden.py`：統計路徑、CPU 參數更新、PALA、PTE、SD、影片與靜態排程，以及定點選項。
  - `ate_pte_lut.py`：PTE 查表的候選版本。
- 之後新增的是評估工具與真實影像的驗證，不是重新修改演算法：
  - 在 O-HAZE 的 45 組照片上，比較 FX 組合（第 6.2 節）和把 PTE 換成查表（B = 8、10）之後的輸出與畫質。45 組都已完成。
  - 驗證結果：45 組的 t 差都沒有超過上限。上限是事先定好的門檻，不是量到的值：B = 8 時 3、B = 10 時 1（t 以 1/1024 為單位）。
  - 評估工具和結果沒有放進本 repo，這一版也不附評估數字。
- B = 8、10 仍然是候選，還沒有選定。
- 現在可以使用：建立環境、selfcheck、單張影像的浮點與定點計算、PTE 查表候選和 FX 的比較、O-HAZE 的配對檢查。
- 後續工作，本版沒有：
  - 匯出測試向量（中間值與輸出的格式還沒決定）。
  - 和 RTL 逐值對齊。

## 2. 快速開始

先安裝 uv 和 git（第 3 節），再依序執行。以下指令三個系統通用：

```
git clone https://github.com/ahhhh71828/ate-dehazing-reference.git
cd ate-dehazing-reference
uv sync --locked
uv run --locked python -B ate_golden.py selfcheck
uv run --locked python -B ate_pte_lut.py selfcheck
```

- `git clone` 會在目前所在的資料夾裡建立 `ate-dehazing-reference`。
- 兩個 selfcheck 的最後一行分別是 `結果：489/489 PASS` 和 `結果：64/64 PASS`，看到就代表環境正常。

## 3. 安裝 uv 與 git

### 3.1 uv

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

### 3.2 git

**Windows**

到 https://git-scm.com/install/windows 下載 Git for Windows 並安裝。也可以在 PowerShell 執行：

```
winget install --id Git.Git -e --source winget
```

**macOS、Linux**

執行 `git --version`。沒有顯示版本號的話，依 https://git-scm.com/install/ 的說明安裝。

## 4. 建立環境

進入 repo 資料夾（有 `pyproject.toml` 的那一層）再建立環境。用第 2 節的 `git clone` 取得的話，以下指令三個系統通用：

```
cd ate-dehazing-reference
uv sync --locked
```

下載 ZIP 的話，改成 `cd` 到解壓後的資料夾，名稱可能是 `ate-dehazing-reference-main`。

應該看到：產生 `.venv` 資料夾，沒有錯誤訊息。第一次執行會下載套件，電腦沒有 Python 3.12 時也會一併下載，需要網路。

## 5. 不需要影像的檢查

在 repo 資料夾執行，三個系統通用。`-B` 讓 Python 不產生 `__pycache__` 資料夾。

| 指令 | 最後一行應該是 |
|---|---|
| `uv run --locked python -B ate_golden.py selfcheck` | `結果：489/489 PASS` |
| `uv run --locked python -B ate_pte_lut.py selfcheck` | `結果：64/64 PASS` |
| `uv run --locked python -B ate_golden.py mutation`（選用，較久） | `結果：79/79 個 mutation 在預期項目失敗` |
| `uv run --locked python -B ate_pte_lut.py mutation`（選用） | `結果：16/16 個 mutation 在預期項目失敗；對照組 2/2 依預期不算抓到` |
| `uv run --locked python -B ate_pte_lut.py measure`（選用，約需 0.5 GB 記憶體） | 倒數第二行是 `判定：推導與手算都符合` |

## 6. 使用自己的影像

路徑有空格時要加雙引號。本節的命令都只印到螢幕；只有 `demo` 加上 `--out-dir` 時會寫檔。

本版有三種計算方式：

- 浮點路徑：程式預設，不加任何定點選項。
- FX 組合：要自己加四個定點選項，不是程式預設。
- PTE 查表候選：用另一個檔 `ate_pte_lut.py`。

### 6.1 浮點路徑（程式預設）

**Windows（PowerShell）**

```
uv run --locked python -B ate_golden.py demo "C:\Users\你的名稱\Pictures\hazy.jpg"
```

**macOS、Linux**

```
uv run --locked python -B ate_golden.py demo ~/Pictures/hazy.jpg
```

應該看到：印出統計值、A_F 與 A_N、t 的範圍，最後一行是 `未指定 --out-dir：不寫檔`。加上 `--out-dir out` 會在 `out` 資料夾寫出除霧結果與對照圖。

其他命令的路徑寫法相同，下表以 `影像.jpg` 代表：

| 命令 | 做什麼 |
|---|---|
| `uv run --locked python -B ate_golden.py demo 影像.jpg` | 單張展示：T_h 用這張影像自己的平均灰階；某一群為空時用另一群的 A；不做 step 4 |
| `uv run --locked python -B ate_golden.py static 影像.jpg` | 靜態排程：第一次讀取用第一張的預設參數（第 7.1 節 L5）統計，CPU 更新後，第二次讀取再除霧 |
| `uv run --locked python -B ate_golden.py video 影像1.jpg 影像2.jpg` | 影片排程：第一張用預設參數（第 7.1 節 L5），之後每張用上一張更新後的參數除霧；`--repeat K` 讓每張重複 K 個 frame |

`demo` 和 `static` 用的 T_h 不同，兩者的輸出不能直接比較。

### 6.2 FX 組合（定點；要加選項）

```
uv run --locked python -B ate_golden.py static 影像.jpg --inv-frac 20 --pala-fixed --pte-frac 10 --sd lut
```

四個選項都加上，才是 FX 組合；`video` 也可以加。印出排程表，以及各定點步驟和浮點相比造成的差異。

| 選項 | 意思 | 不加時 |
|---|---|---|
| `--inv-frac 20` | 1/I_max 改成 floor(2^20 / I_max)；也可以選 16 | 浮點 1/I_max |
| `--pala-fixed` | 定點 PALA；需要 `--inv-frac` | 浮點 PALA |
| `--pte-frac 10` | PTE 理想定點 reference，t 的小數位數 T = 10；也可以選 16；需要 `--pala-fixed` | 浮點 PTE |
| `--sd lut` | SD 用直接 LUT（只能配 `--pte-frac 10`）；也可以選 `ideal`（理想整數 SD） | 浮點 SD |

各選項的數值格式見第 7.2 節。

### 6.3 PTE 查表候選

```
uv run --locked python -B ate_pte_lut.py compare 影像.jpg
```

- 固定用 FX 組合與靜態排程，沒有其他選項。
- 把 PTE 分別換成 B = 8、10 的查表，和 FX 的理想定點 PTE 比較 t 與 8-bit 輸出。
- 倒數第二行是 `判定：guard 都成立`，不寫檔。guard 是 t 差的上限：B = 8 時 3、B = 10 時 1（t 以 1/1024 為單位）。
- 查表的參數：ratio 表 K_r = 23、I/A 表 L = 17、B = 8 或 10；只接受 T = 10。
- 查表版只在 `ate_pte_lut.py`；`ate_golden.py` 沒有查表模式。

## 7. 本版的暫定設定（對照目前的程式）

下表是程式實際使用的規則。「可切換」欄只列程式已經有的選項；寫「無」的表示目前只有一種做法。編號對應組內《Golden Model 對接事項》第 5.3 節的決定清單：L 是統計路徑，W 是主路徑。

### 7.1 統計路徑與 CPU 更新

| 編號 | 目前的做法 | 可切換 |
|---|---|---|
| L1 並列 | 同一群的 dark 最大值並列時，取 raster 順序第一個 | `--tie last`（`video`、`static`） |
| L2 下一張的 T_h | floor(gray_sum / 3N)，gray_sum 是降採樣後三色原值的總和 | `--th-round ceil`（`video`、`static`） |
| L3 降採樣 | 列、行都取 0、3、6…（每個 3×3 的左上角） | 無 |
| L4 step 4 | A_F、A_N 分開判斷；三色各自和上一個實際使用的值比，任一色差 > 3 時，三色都取 floor((本張 + 上一個) / 2) | 無 |
| L5 第一張的參數 | T_h = 128、A_F = A_N = (255, 255, 255)；第一次更新不做 step 4 | 無 |
| L6 空群組 | 統計結果把該群標成無效；排程中沿用上一個值，第一次更新改用另一群本張的值 | 無 |

- L6 的 register map 有效位元（例如 bit 24）是介面提案，程式沒有實作；golden 沒有暫存器模型。
- 其他已經實作、沒有選項的規則：
  - dark channel 先取 RGB 最小值，再取 5×5 範圍的最小值，影像外補 255。
  - dark ≥ T_h 歸遠群，否則歸近群。
  - ω = 0.9375、γ = 2。

### 7.2 主路徑

| 編號 | 浮點路徑（預設） | FX 組合 | 可切換 |
|---|---|---|---|
| W1 灰階 | 三色和（0–765）；I_max = A_F 的三色和 | 同左 | 無 |
| W2 1/I_max | 浮點 | floor(2^20 / I_max)；I_max = 0 時是 0 | `--inv-frac 16` |
| W3 η 與 A | 浮點 | η 1.0 = 256（取 floor）；A 保留 8 個小數 bit | 無 |
| W4 t | 截到 [1/1023, 1] | 1.0 = 2^10；下限 1/1024 | `--pte-frac 16` |
| W5 PTE 的除法 | 浮點除法 | 精確整數除法（理想 reference） | 查表版只在 `ate_pte_lut.py`（第 6.3 節） |
| W6 SD | 浮點；四捨五入（.5 進位）後截到 0–255 | 1/t 直接 LUT（K = 10、T = 10）；四捨五入後截到 0–255 | `--sd ideal` |
| W7 零分母 | A^c = 0 的 channel 不參加 min（三色都是 0 時取 0）；純黑 pixel 的 k = 1 | 同左 | 無 |

- W2 的暫存器欄位寬度是介面提案，程式沒有實作。
- η 超過 1 時的 saturation 規則還沒決定，程式沒有實作；遇到時直接報錯。

### 7.3 還沒決定的事

以下由組內決定，決定後程式與比較都要跟著更新：

- B 用 8 還是 10。
- 硬體的 t 小數位數 T。FX 組合用 10，W4 本身還沒定。
- 第 7.1、7.2 節的所有規則。L1、L2 若改變，FX 與查表版的比較要重做。
- 測試向量的格式，以及先對影片還是靜態排程。

## 8. 選用：資料集

只有要做畫質評估時才需要。資料集的授權還沒查清楚：請各自下載，不要再傳給別人，也不要放進 repo。

### 8.1 O-HAZE（45 組室外有霧／無霧照片）

- 官方頁：https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/
- 官方頁上的下載連結目前失效（2026-11-01 測試為 404）。可改用同一目錄下的這個網址，它不是官方頁列出的連結：https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/O-HAZE.zip
- 解壓後的結構：`# O-HAZY NTIRE 2018\hazy\`、`# O-HAZY NTIRE 2018\GT\`，各有 45 張。

檢查配對：會讀取並解碼每張影像，檢查配對與影像格式，不執行除霧，也不寫檔。指令三個系統通用，下例是 Windows 路徑，請換成你的解壓位置：

```
uv run --locked python -B ate_quality.py pairs "D:\data\# O-HAZY NTIRE 2018"
```

應該看到：`配對狀態：ok 45；missing 0`。

PTE 查表在這 45 組上的比較已經完成（第 1 節）；評估工具不在本 repo，不需要自己重跑。

## 9. 常見問題

| 狀況 | 處理方式 |
|---|---|
| 安裝後輸入 `uv` 或 `git` 顯示找不到指令 | 關掉終端機重新開啟，再試一次 |
| Windows 不允許執行 uv 的安裝指令 | 確認有照第 3.1 節完整輸入（含 `-ExecutionPolicy ByPass`）；或改用 `winget install --id=astral-sh.uv -e` |
| `uv sync` 顯示 lock 需要更新 | `pyproject.toml` 或 `uv.lock` 被改過。不要執行 `uv lock`，重新下載 repo |
| selfcheck 的結果不是上面的數字 | 確認指令前面有 `uv run --locked`，並確認 repo 是最新版本 |
| 顯示 `無法讀取：<路徑>` | 路徑打錯，或格式不支援（例如 iPhone 的 HEIC，請先轉成 JPG 或 PNG） |
| 影像很大時跑很久或記憶體不足 | 先用 1920×1080 以下的影像 |
| Linux 上 `import cv2` 出現 `libGL.so.1` 錯誤 | 系統缺少圖形函式庫，需要另外安裝（一般經驗，本專案沒有在 Linux 測試過） |

## 10. 回報問題

請附上：

- 作業系統（Windows、macOS 或 Linux）
- `uv --version` 的輸出
- repo 的版本：用 git 取得的話附上 `git rev-parse --short HEAD` 的輸出，下載 ZIP 的話附上下載日期
- 完整指令，以及輸出的最後 20 行

## 11. 檔案說明

| 檔案 | 用途 |
|---|---|
| `ate_golden.py` | 軟體 reference：統計路徑、CPU 參數更新、PALA、PTE、SD、影片與靜態排程；預設是浮點，定點要加選項（第 6.2 節） |
| `ate_pte_lut.py` | PTE 查表的候選版本（B = 8、10），尚未選定；已在 O-HAZE 45 組上和 FX 比較過 |
| `ate_quality.py` | 畫質評估工具：O-HAZE 的配對檢查與 PSNR／SSIM（選用，需要第 8.1 節的資料集） |
| `pyproject.toml`、`uv.lock`、`.python-version` | 環境設定，請不要修改 |

演算法、符號與數值格式的說明，見組內的《Golden Model 技術說明》。
