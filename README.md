# ate-dehazing-reference

除霧論文（Upadhyay & Sarawadekar, IEEE TVLSI 2026）的 Python 軟體 reference v0（golden model），附 PTE 查表的候選版本與一包預先算好的測試向量。

現在可以做的事：

- 負責 EPU 主路徑（PALA、PTE、SD）的組員，以及負責統計路徑、CPU 更新與 ISR 的組員：用 `results/vectors-v0/` 的向量，和自己的 RTL 或 ISR 逐值比對。
- 第一步：照第 2 節的最短路線，確認環境與向量檔都正常，再接到自己的 testbench。

> **注意**
> - 數值規則都是暫定值，尚未經組內確認；本版實際使用的規則列在第 7 節。
> - 不是論文復現；本 repo 沒有 RTL。
> - 已在 Mac（Apple Silicon）的乾淨環境照本教學建立環境，並跑過兩個 selfcheck、`demo` 與 `compare`（2026-11-08）。`ate_vectors.py` 與第 8 節的向量只在作者的開發環境跑過；Windows、Linux 都還沒試過。遇到問題請告訴負責 golden model 的組員。

## 1. 本版內容

- 演算法檔在首次交付（2026-11-08）時就已經在本 repo，之後沒有修改：
  - `ate_golden.py`：統計路徑、CPU 參數更新、PALA、PTE、SD、影片與靜態排程，以及定點選項。
  - `ate_pte_lut.py`：PTE 查表的候選版本。
- 之後新增的是評估工具與真實影像的驗證，不是重新修改演算法：
  - 在 O-HAZE 的 45 組照片上，比較 FX 組合（第 6.2 節）和把 PTE 換成查表（B = 8、10）之後的輸出與畫質。45 組都已完成。
  - 驗證結果：45 組的 t 差都沒有超過上限。上限是事先定好的門檻，不是量到的值：B = 8 時 3、B = 10 時 1（t 以 1/1024 為單位）。
  - 評估工具和結果沒有放進本 repo，這一版也不附評估數字。
- 2026-11-10 加上軟體 v0 的測試向量：`ate_vectors.py` 與 `results/vectors-v0/` 的 9 個檔（第 8 節）。它只呼叫既有的計算函式，不改演算法。
- B = 8、10 仍然是候選，還沒有選定。
- 現在可以使用：建立環境、selfcheck、單張影像的浮點與定點計算、PTE 查表候選和 FX 的比較、O-HAZE 的配對檢查、軟體 v0 的測試向量。
- 後續工作，本版沒有：
  - 測試向量的格式經組員確認；需要時再調整。
  - DRAM 的 24-bit packed 格式。
  - 和 RTL 逐值對齊。

## 2. 快速開始：最短使用路線

要用 golden 驗證自己的實作，走這 5 步就夠；第 5、6、9 節（golden 自檢、用影像展示、資料集）都是選讀。

1. 安裝 uv 與 git。Windows 和 macOS／Linux 的指令不同，見第 3 節。
2. 取得 repo 並建立環境。以下指令三個系統相同，Windows 在 PowerShell 執行：

   ```
   git clone https://github.com/ahhhh71828/ate-dehazing-reference.git
   cd ate-dehazing-reference
   uv sync --locked
   ```

3. 確認向量檔和程式一致：

   ```
   uv run --locked python -B ate_vectors.py verify
   ```

   最後一行應該是 `結果：32/32 PASS`。不是的話先看第 10 節，或告訴負責 golden model 的組員。
4. 依你負責的單元選檔，並選對版本（PTE 看 `src`、SD 看 `mode`）：第 8.2 節。
5. 接到自己的 testbench：讀檔範例在第 8.4 節（注意檔案路徑），比對流程在第 8.5 節。

- `git clone` 會在目前所在的資料夾裡建立 `ate-dehazing-reference`；下載 ZIP 的話見第 4 節。
- 想另外確認 golden 本身的自檢，可以跑第 5 節的兩個 selfcheck；這不影響上面的路線。

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

這幾個檢查是選用的：確認 golden 本身的自檢案例在你的電腦上符合預期，不需要任何影像。最短路線（第 2 節）只需要第 8.6 節的 `verify`；要跑這裡的檢查時，先跑前兩個，mutation、measure 再選用。

在 repo 資料夾執行，三個系統通用。`-B` 讓 Python 不產生 `__pycache__` 資料夾。

| 指令 | 這是什麼 | 最後一行應該是 |
|---|---|---|
| `uv run --locked python -B ate_golden.py selfcheck` | golden 的自我測試：用事先手算好的案例和另一種寫法的參考算法，逐值核對 golden 的結果。全部通過代表這些自檢案例在你的電腦上符合預期（不代表重現了論文作者的結果） | `結果：489/489 PASS` |
| `uv run --locked python -B ate_pte_lut.py selfcheck` | 同上，對象是 PTE 查表候選 | `結果：64/64 PASS` |
| `uv run --locked python -B ate_golden.py mutation`（選用，較久） | 檢查 selfcheck 本身有沒有用：自動把演算法故意改錯一處（例如 1/I_max 的 floor 改成四捨五入），確認 selfcheck 會失敗。79/79 代表這 79 種改錯都抓得到 | `結果：79/79 個 mutation 在預期項目失敗` |
| `uv run --locked python -B ate_pte_lut.py mutation`（選用） | 同上，對象是 PTE 查表候選。「對照組」是只會讓程式當掉的改錯，用來確認程式當掉不會被誤算成抓到錯誤 | `結果：16/16 個 mutation 在預期項目失敗；對照組 2/2 依預期不算抓到` |
| `uv run --locked python -B ate_pte_lut.py measure`（選用，約需 0.5 GB 記憶體） | 量查表的誤差：窮舉所有表項、100 萬個隨機 pixel 的 t 差，以及各張表的大小，確認都在設計時推導的上限內。給要決定 B 或設計查表硬體的人 | 倒數第二行是 `判定：推導與手算都符合` |

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
- 測試向量的格式（第 8 節是軟體這邊先定的版本），以及先對影片還是靜態排程。

## 8. 用測試向量驗證 RTL／ISR（軟體 v0）

`results/vectors-v0/` 的 9 個文字檔，是本版軟體 reference 在小型合成影像（最大 21×21）上的預期答案。用法：從檔案取出輸入送進你的電路或 ISR，再把結果和同一行的預期值逐欄比對；全部相同，代表你的實作在這些案例上和 golden 一致。

- 格式是軟體這邊先定的，還沒經組員確認；B、T 也還沒選定（第 7.3 節）。
- 本包的 baseline 設定：1/I_max 的 F = 20、t 的 T = 10、並列取第一個（first）、下一張 T_h 取 floor、SD 直接 LUT 的 K = 10。
- 同一個 pixel 有幾個版本的答案，各自標示，不能混用：
  - PTE：精確整數除法（EX，`src` = 0）、查表 B = 8（L8，`src` = 1）、查表 B = 10（L10，`src` = 2）。三版用同一份 RGB 與 A_q。
  - SD：理想整數（IDEAL，`mode` = 0）、直接 LUT（LUT，`mode` = 1）。
  - 只有「PTE 精確版＋SD 直接 LUT」（`src` = 0、`mode` = 1）等於第 6.2 節 FX 組合的輸出；其他組合不是 FX 的輸出。
- RGB 分欄、一個 pixel 一行，不是 DRAM 的 24-bit packed 格式。

目前的驗證狀態，四件事要分開看：

| 項目 | 證明什麼 | 狀態 |
|---|---|---|
| 軟體自檢（第 5 節的 selfcheck） | golden 符合自己寫下的規則與手算案例 | 作者環境通過 |
| 向量檢查（第 8.6 節的 `verify`） | 向量檔和程式重新算的結果逐 byte 相同，並通過手算、獨立參考算法與接線檢查 | 作者環境通過 |
| SystemVerilog 讀檔（第 8.4 節） | testbench 能正確讀出向量檔的每一欄 | 2026-11-10 在 lab89 用 VCS 跑過；只讀檔，沒有接電路 |
| RTL／ISR 比對 | 你的實作在這些案例上和 golden 一致 | 還沒做，要等各自的 testbench |

### 8.1 檔案

| 檔案 | 內容 |
|---|---|
| `frames.hex` | 合成輸入影像 |
| `stats_pixels.hex` | 統計路徑：每個降採樣點的 RGB、最小值、dark 與分群 |
| `stats.hex` | 每次統計的摘要：gray_sum、遠群與近群的代表點 |
| `cpu.hex` | CPU 更新：上一組參數、本張統計值、step 4、下一組參數 |
| `schedule.hex` | 排程：每張 frame 統計時、除霧時與下一張的參數（靜態與影片） |
| `pala.hex` | PALA：1/I_max、eta_q、A_q |
| `pte.hex` | PTE：EX、B = 8、B = 10 |
| `sd.hex` | SD：理想整數版與直接 LUT |
| `lut.hex` | 查表內容：PTE 的 ratio 表與 I/A 表（B = 8、10）、SD 的 1/t 表 |

讀法：

- `#` 開頭的是檔頭：產生它的程式版本（SHA-256）、設定、代碼表、案例，以及每一欄的位數、有無正負號、範圍、無效值寫法與 scale。欄位順序在 `# columns` 那一行。
- 其餘每一行是一筆資料：大寫 hex，欄位間一個空白，每一欄的位數固定；無效值寫成全 F。
- `vec_id` 連接 pala、pte、sd；`stats_id`、`upd_id` 連接 stats、stats_pixels、cpu、schedule；`frame_id`、`row`、`col` 對到 frames.hex。

### 8.2 依你負責的部分找檔案

| 你要驗證 | 檔案 | 用哪些行 | 送進電路的欄位 | 要比對的欄位 |
|---|---|---|---|---|
| PALA | `pala.hex` | 全部 | `R`、`G`、`B`；`AF_R`–`AF_B`（A_F）、`AN_R`–`AN_B`（A_N）；`inv_q` | `eta_q`、`A_q_R`–`A_q_B`；中間值 `S`、`delta`、`product` |
| PTE（精確整數除法） | `pte.hex` | `src` = 0 | `R`、`G`、`B`；`A_q_R`–`A_q_B` | `r_q`、`r2_q`、`k_q`、`m_c_R`–`m_c_B`（未飽和，可能大於 131071）、`m_q`、`p`、`u`、`t_q` |
| PTE（查表 B = 8 或 10） | `pte.hex` | `src` = 1（B = 8）或 2（B = 10） | 同上 | 同上，但 `m_c_R`–`m_c_B` 是查表後已飽和到 131071 的值；另有查表的 `n_R`–`n_B`、`j_R`–`j_B` |
| SD（理想整數） | `sd.hex` | `mode` = 0 | `R`、`G`、`B`；`A_q_R`–`A_q_B`；`t_q` | `num_R`–`num_B`、`den`、`rnd_R`–`rnd_B`、`out_R`–`out_B` |
| SD（直接 LUT） | `sd.hex` | `mode` = 1 | 同上 | `inv_t_q`、`num_R`–`num_B`、`rnd_R`–`rnd_B`、`out_R`–`out_B` |
| 統計路徑 | `stats_pixels.hex`、`stats.hex` | 用 `stats_id` 對應 | `frames.hex` 中 `frame_id` 那張影像；`stats.hex` 的 `t_h_prev` | 每個降採樣點的 `dark`、`far`；每次統計的 `N`、`gray_sum`、`gray_mean`、`far_*`、`near_*` |
| CPU 更新（ISR） | `cpu.hex` | 全部 | `prev_*`（上一組參數）、`gray_sum`、`N`、`far_valid`、`far_raw_*`、`near_valid`、`near_raw_*` | `next_t_h`、`next_AF_*`、`next_AN_*`、`next_I_max`、`next_inv_q`；中間值 `far_act`、`far_diff`、`far_trig`、`far_used_*`（近群同理） |
| 整條排程 | `schedule.hex` | 用 `case_id`、`seq` | `frames.hex` 的影像，依 `seq` 的順序；第一張用預設參數（T_h = 128、A = 255） | 每張的 `stat_*`（統計時）、`use_*`（除霧時）、`next_*`（下一張）；輸出在 `sd.hex` 從 `vec0` 起的 `nvec` 個 `vec_id`，`src` = 0、`mode` = 1 的那幾行就是 FX 組合的輸出 |

- 各單元可以分開驗證：每一行都附上該單元需要的全部輸入。例如只測 SD 時，`t_q` 就在 `sd.hex` 同一行，不必先做 PTE。
- `sd.hex` 的 `src` 標示 `t_q` 從哪裡來：0、1、2 分別是 pte.hex 的精確版、B = 8、B = 10，3 是直接給定的邊界案例。比較 SD 的兩種做法時，只比同一個 `src`。

### 8.3 讀一行的例子

`pte.hex` 的第一行資料是《技術說明》第 10.1 節的 pixel（案例 01）：

```
0000 01 FF FF 000 000 0 3C 5A 78 9A60 9C40 9E20 5555 1C71 0E38F FF FF FF FFF FFF FFF 0000637F 00009374 0000C246 0637F 052EA1DE1F 14B 2B5
```

| 欄位 | hex | 十進位 | 意思 |
|---|---|---|---|
| `vec_id` | `0000` | 0 | 第 0 個向量；pala.hex、sd.hex 的同一個 pixel 也是 0 |
| `case_id` | `01` | 1 | 案例 01，說明在檔頭的 `# case 01` |
| `seq`、`frame_id` | `FF`、`FF` | — | 全 F：無效。這是單元案例，不屬於排程 |
| `src` | `0` | 0 | 精確整數除法 |
| `R`、`G`、`B` | `3C 5A 78` | 60、90、120 | 輸入 pixel |
| `A_q_R` | `9A60` | 39520 | A = 39520 × 2^-8 = 154.375 |
| `k_q` | `0E38F` | 58255 | k = 58255 × 2^-16 ≈ 0.889 |
| `n_R`、`j_R` | `FF`、`FFF` | — | 全 F：精確版不查表 |
| `m_q` | `0637F` | 25471 | m = 25471 × 2^-16 ≈ 0.389 |
| `t_q` | `2B5` | 693 | t = 693 × 2^-10 ≈ 0.677 |

每一欄的位數、範圍、有無正負號、無效值寫法與 scale，都在檔頭的 `# field` 行。例如：

```
# field t_q digits=3 bits=12 signed=0 min=1 max=1024 invalid=- scale=2^-10 : …
```

意思是：3 個 hex 字元、無正負號、合法值 1–1024、沒有無效值、實際的 t = 欄位值 × 2^-10。

hex 的欄寬是檔案儲存用的位數（4 bit 的倍數），不是硬體 port 的位寬：

- `t_q` 存成 3 個 hex（12 bit），合法值 1–1024 只需要 11 bit。
- `k_q` 存成 5 個 hex（20 bit），最大 65536 需要 17 bit。
- `eta_q` 存成 3 個 hex（12 bit），最大 256 需要 9 bit。

比對時比數值。合法值放不進你的 port 時，代表位寬要和組內確認，不能直接截斷。

### 8.4 testbench 讀檔範例

下面是讀 `pte.hex` 的最小 SystemVerilog 範例：跳過 `#` 開頭的行，再依 `# columns` 的順序用 `$sscanf` 拆欄。**它只示範讀檔，沒有接任何電路（DUT）。**

檔案路徑：`$fopen` 的相對路徑是相對於模擬器的執行目錄（例如執行 `./simv` 時所在的資料夾），不是 `.sv` 檔所在的位置。

| 你在哪裡執行模擬器 | `$fopen` 的檔名要寫 |
|---|---|
| `pte.hex` 所在的資料夾（範例的寫法） | `"pte.hex"` |
| repo 的最上層 | `"results/vectors-v0/pte.hex"` |
| 其他位置 | 從執行目錄到 `pte.hex` 的相對路徑，或絕對路徑 |

驗證紀錄：2026-11-10 在 lab89 用 VCS 2025.06 編譯並執行下面這段原文，當時 `pte.hex` 和 `simv` 在同一個資料夾：讀到 7356 行，第一行的值和上一節相同；少一欄的檔案會停在 `$fatal`。改成其他路徑字串的寫法沒有另外跑過。

```systemverilog
module tb_read_pte;
  int     fd, n, rows;
  string  line;
  int     vec_id, case_id, seq, frame_id, row, col, src;
  int     r, g, b, aq_r, aq_g, aq_b, r_q, r2_q, k_q;
  int     n_r, n_g, n_b, j_r, j_g, j_b;
  longint mc_r, mc_g, mc_b, p;
  int     m_q, u, t_q;

  initial begin
    fd = $fopen("pte.hex", "r");
    if (fd == 0) $fatal(1, "cannot open pte.hex");
    rows = 0;
    while ($fgets(line, fd)) begin
      if (line.len() == 0 || line[0] == "#") continue;   // skip header lines
      n = $sscanf(line, "%h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h %h",
                  vec_id, case_id, seq, frame_id, row, col, src, r, g, b, aq_r, aq_g, aq_b,
                  r_q, r2_q, k_q, n_r, n_g, n_b, j_r, j_g, j_b, mc_r, mc_g, mc_b, m_q, p, u, t_q);
      if (n != 29) $fatal(1, "data row %0d: parsed %0d fields", rows, n);
      // Drive r, g, b, aq_r, aq_g, aq_b into your PTE here, then compare its outputs
      // with r_q ... t_q of this row (choose rows by src: 0 = exact, 1 = B=8, 2 = B=10).
      if (rows == 0)
        $display("first row: vec_id=%0d case_id=%0h src=%0d RGB=%0d,%0d,%0d A_q=%0d,%0d,%0d k_q=%0d m_q=%0d t_q=%0d",
                 vec_id, case_id, src, r, g, b, aq_r, aq_g, aq_b, k_q, m_q, t_q);
      rows++;
    end
    $fclose(fd);
    $display("data rows: %0d", rows);
    $finish;
  end
endmodule
```

改成讀其他檔時要注意：

- 變數的個數與順序照該檔的 `# columns` 行，`$sscanf` 的回傳值要等於欄數。
- 超過 31 bit 的欄位（`m_c_*`、`p`）用 `longint`，不要用 `int`。
- 無效值是全 F，讀進來會是最大值（例如 `FF` 是 255），要先看檔頭說明再比對。
- `sd.hex` 的 `num_*`（10 位）和 `rnd_*`（5 位）可能是負數，用二補數表示：讀進同樣寬度的 `logic signed [39:0]`、`logic signed [19:0]`。這個讀法也在 lab89 用 VCS 驗證過，D3 那行讀到 `rnd_R` = −100。

### 8.5 接到你自己的驗證

第 8.4 節的範例只負責讀檔。接到你的電路或 ISR 時，照下面的順序：

1. 先跑 `verify`（第 8.6 節），確認拿到的檔案和程式一致。
2. 選對行：依第 8.2 節選檔，再用 `src`、`mode` 篩出你要對的版本。例如做 PTE 查表 B = 8 的人只取 `src` = 1 的行。
3. 每讀一行，把這一行的輸入和預期答案一起存起來，並記下 `vec_id`（統計用 `stats_id`、CPU 用 `upd_id`），之後的輸出才對得回是哪一行。
4. 把輸入送進你的實作，用你自己的 port、clock 與 handshake。
5. 等你的實作表示這一筆輸出有效（例如你自己設計的 valid 訊號，或你確定的固定延遲之後），再取出同一筆的預期答案來比對。
6. 只比對你的實作實際有輸出、而且檔案有對應欄位的值。能觀察到的中間值（例如內部的 `k_q`、`m_q`）可以幫忙找出錯在哪一步。

有 pipeline 時要注意：第 N 筆輸入的結果要過幾個 clock 才出來，這段時間你可能已經送進第 N+1、N+2 筆。不能讀了下一行就拿當下的輸出和這一行比。做法是把預期答案照送入的順序排隊（先進先出），輸出有效時才取出最前面那一筆來比。

不一致時：

1. 找出第一個不同的步驟：依序比對中間值，第一個不同的欄位就是出問題的地方。
   - PALA：`product` → `eta_q` → `A_q_*`。
   - PTE：`r_q` → `k_q` → `m_c_*` → `m_q` → `u` → `t_q`。
   - SD：`inv_t_q` → `num_*` → `rnd_*` → `out_*`。
2. 確認行選對了：PTE 的 `src`、SD 的 `mode` 與 `src`。
3. 確認設定相同：F = 20、T = 10、並列取第一個、T_h 取 floor、SD LUT K = 10。
4. 還是找不到原因，把檔名、`vec_id`、欄位名稱、你的值和預期值傳給負責 golden model 的組員。

### 8.6 檢查與重新產生

在 repo 資料夾執行，三個系統通用：

| 指令 | 應該看到 |
|---|---|
| `uv run --locked python -B ate_vectors.py verify` | 最後一行是 `結果：32/32 PASS`：重新計算的結果和 repo 裡的 9 個檔逐 byte 相同，而且通過全部檢查；不寫檔 |
| `uv run --locked python -B ate_vectors.py selfcheck` | 最後一行是 `結果：32/32 PASS`；只在記憶體計算，不寫檔 |

- `uv run --locked python -B ate_vectors.py write` 會覆寫這 9 個檔，只有程式改了才需要。
- 2026-11-10 只在作者的開發環境跑過；乾淨環境還沒試。

### 8.7 限制

- 只有本包的固定案例。要用自己的輸入（例如特定的 pixel 或 A_q）算預期值，目前程式不支援，請找負責 golden model 的組員。
- 欄位或位寬不適合你的 testbench 時，告訴負責 golden model 的組員再調整格式。
- 這些是軟體 reference 的答案，不代表 RTL 已驗證，也不代表 B、T 已選定。
- 47 個案例是 golden 的比對資料，不等於課程要求的 EPU direct test。

## 9. 選用：資料集

只有要做畫質評估時才需要。資料集的授權還沒查清楚：請各自下載，不要再傳給別人，也不要放進 repo。

### 9.1 O-HAZE（45 組室外有霧／無霧照片）

- 官方頁：https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/
- 官方頁上的下載連結目前失效（2026-11-01 測試為 404）。可改用同一目錄下的這個網址，它不是官方頁列出的連結：https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/O-HAZE.zip
- 解壓後的結構：`# O-HAZY NTIRE 2018\hazy\`、`# O-HAZY NTIRE 2018\GT\`，各有 45 張。

檢查配對：會讀取並解碼每張影像，檢查配對與影像格式，不執行除霧，也不寫檔。指令三個系統通用，下例是 Windows 路徑，請換成你的解壓位置：

```
uv run --locked python -B ate_quality.py pairs "D:\data\# O-HAZY NTIRE 2018"
```

應該看到：`配對狀態：ok 45；missing 0`。

PTE 查表在這 45 組上的比較已經完成（第 1 節）；評估工具不在本 repo，不需要自己重跑。

## 10. 常見問題

| 狀況 | 處理方式 |
|---|---|
| 安裝後輸入 `uv` 或 `git` 顯示找不到指令 | 關掉終端機重新開啟，再試一次 |
| Windows 不允許執行 uv 的安裝指令 | 確認有照第 3.1 節完整輸入（含 `-ExecutionPolicy ByPass`）；或改用 `winget install --id=astral-sh.uv -e` |
| `uv sync` 顯示 lock 需要更新 | `pyproject.toml` 或 `uv.lock` 被改過。不要執行 `uv lock`，重新下載 repo |
| selfcheck 的結果不是上面的數字 | 確認指令前面有 `uv run --locked`，並確認 repo 是最新版本 |
| 顯示 `無法讀取：<路徑>` | 路徑打錯，或格式不支援（例如 iPhone 的 HEIC，請先轉成 JPG 或 PNG） |
| 影像很大時跑很久或記憶體不足 | 先用 1920×1080 以下的影像 |
| Linux 上 `import cv2` 出現 `libGL.so.1` 錯誤 | 系統缺少圖形函式庫，需要另外安裝（一般經驗，本專案沒有在 Linux 測試過） |

## 11. 回報問題

請附上：

- 作業系統（Windows、macOS 或 Linux）
- `uv --version` 的輸出
- repo 的版本：用 git 取得的話附上 `git rev-parse --short HEAD` 的輸出，下載 ZIP 的話附上下載日期
- 完整指令，以及輸出的最後 20 行

## 12. 檔案說明

| 檔案 | 用途 |
|---|---|
| `ate_golden.py` | 軟體 reference：統計路徑、CPU 參數更新、PALA、PTE、SD、影片與靜態排程；預設是浮點，定點要加選項（第 6.2 節） |
| `ate_pte_lut.py` | PTE 查表的候選版本（B = 8、10），尚未選定；已在 O-HAZE 45 組上和 FX 比較過 |
| `ate_vectors.py` | 產生與檢查第 8 節的測試向量；只呼叫既有的計算函式 |
| `results/vectors-v0/` | 軟體 v0 的測試向量（第 8 節），9 個文字檔 |
| `ate_quality.py` | 畫質評估工具：O-HAZE 的配對檢查與 PSNR／SSIM（選用，需要第 9.1 節的資料集） |
| `pyproject.toml`、`uv.lock`、`.python-version` | 環境設定，請不要修改 |

演算法、符號與數值格式的說明，見組內的《Golden Model 技術說明》。
