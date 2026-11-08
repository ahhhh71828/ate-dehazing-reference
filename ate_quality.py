"""AVSD Final E 靜態畫質評估第一輪（O-HAZE 靜態 PSNR／SSIM）：metric、O-HAZE runner 與 selfcheck。

依 E Notes 4.11 的量測契約撰寫。內容：
- PSNR（自寫）與 SSIM（scikit-image 0.26.0，參數全部明給）。
- 逐張平均、共用成功集合 S、ΔPSNR／ΔSSIM、D 與非有限值處理；NA、原因代碼與 formatter。
- O-HAZE 的實際檔名規則（2026-11-01 盤點）、配對核心 pair_files、讀檔與格式 gate（MPO 只用主圖，做法 A）。
- FL／FX 的 golden 執行（主路徑可按列分塊）、公平性比較、每對一列、合計與報表（含 4.11.4 的設定標記）。
- 資源 gate：校準值、預算、餘裕都要由命令列明給；本檔沒有預設預算（12 GiB、15 分鐘、1.5 倍都只是提案）。
- 結果檔（E Notes 4.11.17）：ohaze --record 每對完成後寫入一行（完整精度）並讀回驗證；ohaze-complete 在外層迴圈的
  45 對都通過事後檢查後追加完成紀錄；ohaze-report 驗證完成紀錄、ID 01–45、run、檔名與資料目錄後彙整。
- 子命令：selfcheck（Q1–Q34）、mutation（QM1–QM83 與對照組 QC1–QC6）、pairs、ohaze、calib、ohaze-complete、ohaze-report。

命令存在不等於已獲准執行：
- pairs：只讀資料、列清單與格式 gate，不呼叫 golden、不算畫質、不寫檔。2026-11-01 使用者准許對實際 O-HAZE 執行。
- ohaze（計分）與 calib（RSS 校準用的合成圖執行）：都要使用者另外批准才能執行（E Notes 4.11.12）。
  ohaze --record、ohaze-complete 會寫結果檔；用真實資料執行同樣要另外批准。ohaze-report 只讀結果檔與資料目錄的檔名。
  ohaze 缺少資源條件時拒絕開始；預設第一對完成後停止。

ate_golden.py 不改。本檔從它 import run_static、stats_pass、cpu_update、apply_pass、make_active、ACTIVE0_DEFAULT、
STEP、SD_LUT_K，以及底線開頭的 _check_frame、_Checker、_diff_stats、_run_selfcheck_source；golden 改到這些時，
要重跑本檔的 selfcheck。

用法（在 dehaze-e/ 執行；Python 一律 -B；讀資料時加 -I，程式路徑明確指定）：
    .venv/bin/python -B ate_quality.py selfcheck
    .venv/bin/python -B ate_quality.py mutation
    .venv/bin/python -I -B /Users/ahh/Aporia/AVSD-2026-Fall/Final-Project/dehaze-e/ate_quality.py pairs \\
        "/Users/ahh/Aporia/AVSD-2026-Fall/Final-Project/datasets/O-HAZE/# O-HAZY NTIRE 2018"
"""

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import shlex
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.metrics import structural_similarity

# 用 -I 執行時，Python 不會把本檔所在的目錄放進 sys.path；只加入本檔自己的目錄（可信的程式路徑），不加入資料目錄或 cwd
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
if _HERE and _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ate_golden as golden  # noqa: E402

PEAK = 255               # 8-bit 的最大值，PSNR 與 SSIM 的 data_range 固定用它，不從影像內容推估
COLS = ("H", "FL", "FX")  # 都和 GT 比較：H 是原始 hazy，FL 是浮點輸出，FX 是定點候選輸出（4.11.3）
WORSE = -0.5             # 另外報告 ΔPSNR < −0.5 dB 的對數；只是參考，不判 PASS／FAIL
USED_SAME = ("t_h", "a_far", "a_near", "i_max", "tie", "th_round", "measured")   # FL、FX 必須相同的 used 欄位
MUTANT_TIMEOUT = 120     # 每個 mutant 的 selfcheck 秒數上限；超時算例外，不算抓到
EXC_MARK = "［呼叫丟出例外，不是正常 FAIL］"   # selfcheck 的 check_cli 在 cli 呼叫、前置步驟或條件本身丟例外時加在 FAIL 標籤上；mutation 不把它算成抓到
GIB = 1 << 30
OHAZE_DIRS = {"hazy": "hazy", "GT": "gt"}   # 所在目錄 → 角色；檔名裡的角色要和所在目錄一致（4.11.6）
# 下載紀錄（README「外部來源」、run.log 第 10 節）。報表照印；本檔不重新驗證 zip。
OHAZE_SOURCE = {"url": "https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/O-HAZE.zip",
                "zip_sha256": "538421bb80798da2298a9b4f0478ca59ff37e5a305ba57a8a9a8e03ec477be09",
                "etag": '"222c2c93-5a0d2b46d9b6a"', "date": "2026-11-01"}
PAPER_TABLE1 = "論文 Table I（γ = 2）的 O-Haze：PSNR 15.92、SSIM 0.45"   # 4.11.9：只並列，設定不同
OHAZE_TITLE = "O-HAZE 靜態 PSNR／SSIM（O-HAZE 單獨報告，不和其他資料集合併平均）"
# 結果檔（E Notes 4.11.17）：資料行一行一對；45 對都通過外層迴圈的檢查後，最後追加一行完成紀錄
REC_SCHEMA = "ohaze-record-1"
DONE_SCHEMA = "ohaze-complete-1"
EXPECTED_IDS = tuple(range(1, 46))   # 本輪固定 01–45（O-HAZE 釋出版的 45 對）；不從資料目錄推定
RESOURCE_METRIC = "peak memory footprint (/usr/bin/time -l)"


# ---------- NA 與報表的數字格式（4.11.7） ----------

NA_TEXT = {"nonfinite": "非有限值", "empty": "N_ok = 0", "contains_na": "S 含 NA 的 Δ", "no_finite": "沒有有限值"}


@dataclass(frozen=True)
class NA:
    """不存在的統計量。Δ 計算與彙整的 helper 回傳 float 或 NA，不回傳 float NaN。"""
    reason: str

    def __post_init__(self):
        if self.reason not in NA_TEXT:
            raise ValueError(f"未定義的 NA 原因代碼：{self.reason!r}")


def fmt(v, digits):
    """報表的數字格式。只把 NA 印成文字、+inf 印成 +inf；其他 float 照格式印。
    float NaN 照 Python 的格式印成 nan，不改成 NA：報表出現 nan 就是 helper 的 bug（Q17 檢查）。"""
    if isinstance(v, NA):
        return f"NA（{NA_TEXT[v.reason]}）"
    if v == math.inf:
        return "+inf"
    return f"{v:.{digits}f}"


# ---------- PSNR、SSIM（4.11.7、4.11.8） ----------

def _check_pair(x, y):
    """兩邊都要是 uint8、H×W×3、H、W > 0 的 RGB 陣列，而且形狀完全相同；否則丟 ValueError。"""
    golden._check_frame(x)
    golden._check_frame(y)
    if x.shape != y.shape:
        raise ValueError(f"兩張影像的形狀不同：{x.shape} vs {y.shape}")


def mse(x, y):
    """RGB 合併的 MSE：所有 pixel 與三個 channel 一起平均。先轉 int64 再相減，避免 uint8 wraparound。"""
    _check_pair(x, y)
    d = x.astype(np.int64) - y.astype(np.int64)
    return float(np.sum(d * d)) / d.size


def psnr_from_mse(m):
    """MSE = 0 時回傳 +inf（明確分支，不靠除以 0）；否則 10·log10(255² / MSE)。"""
    if m == 0:
        return math.inf
    return 10 * math.log10(PEAK ** 2 / m)


def psnr(x, y):
    """一張影像一個 PSNR（RGB 合併的 MSE），不是三個 channel 各算再平均。"""
    return psnr_from_mse(mse(x, y))


def ssim(x, y):
    """scikit-image 0.26.0 的 SSIM，參數全部明給：逐 channel 算 2-D SSIM，再取三個值的算術平均。
    Gaussian 權重 σ = 1.5（視窗 11）、母體共變異數、K1 0.01、K2 0.03、data_range 255；
    濾波用 reflect 補邊，之後每邊裁掉 5 pixel 再平均。傳 uint8，不先轉 float。
    這組設定不等於論文的 MATLAB 評估設定，也不等於 Wang 的 ssim.m（不轉灰階、不降採樣）。"""
    _check_pair(x, y)
    return float(structural_similarity(x, y, channel_axis=2, data_range=PEAK, gaussian_weights=True, sigma=1.5,
                                       use_sample_covariance=False, K1=0.01, K2=0.03))


# ---------- 平均、Δ、D（4.11.3、4.11.7） ----------

def column_mean(values):
    """一欄的資料集平均：每對權重相同的算術平均。空集合回傳 NA("empty")；有 +inf 時平均就是 +inf，不丟掉。"""
    vals = list(values)
    if not vals:
        return NA("empty")
    return math.fsum(vals) / len(vals)


def finite_mean(values):
    """只當診斷：有限值的平均與個數 k（分母明寫，不用它重算 D）。回傳 (平均或 NA, k)。"""
    vals = list(values)
    if not vals:
        return NA("empty"), 0
    finite = [v for v in vals if math.isfinite(v)]
    return (math.fsum(finite) / len(finite) if finite else NA("no_finite")), len(finite)


def mean_psnr(rows, col):
    """資料集 PSNR：先逐張算 PSNR，再對成功的配對取算術平均，每對權重相同；不是 pooled MSE。"""
    return column_mean([r[col]["psnr"] for r in rows])


def delta(fx, fl):
    """ΔPSNR_i 或 ΔSSIM_i = FX − FL。任一個不是有限值時回傳 NA("nonfinite")，不算 inf − inf。"""
    if not (math.isfinite(fx) and math.isfinite(fl)):
        return NA("nonfinite")
    return fx - fl


def _delta_list(deltas):
    """Δ 彙整的共同規則：空集合 → NA("empty")；S 裡有 NA 的 Δ → NA("contains_na")（不改用子集合）；否則回傳 list。"""
    ds = list(deltas)
    if not ds:
        return NA("empty")
    if any(isinstance(d, NA) for d in ds):
        return NA("contains_na")
    return ds


def mean_delta(deltas):
    """帶正負號的 mean Δ。"""
    ds = _delta_list(deltas)
    return ds if isinstance(ds, NA) else math.fsum(ds) / len(ds)


def d_value(deltas):
    """D = |mean ΔPSNR|（4.11.3 的提案公式）。是 |mean Δ|，不是 mean |Δ|。草案門檻 0.5 dB，本檔不判 PASS／FAIL。"""
    m = mean_delta(deltas)
    return m if isinstance(m, NA) else abs(m)


def delta_extremes(ids, deltas):
    """ΔPSNR 的最小值與 ID、|Δ| 的最大值、Δ < −0.5 的對數、最小的三對。NA 規則和 D 相同。"""
    ds = _delta_list(deltas)
    if isinstance(ds, NA):
        return ds
    order = sorted(zip(ds, ids))
    return {"min": order[0], "absmax": max(abs(v) for v in ds), "n_worse": sum(v < WORSE for v in ds),
            "worst3": order[:3]}


def min_finite(values):
    """有限值的最小值（FX 對 FL 的 PSNR 用）。空集合 → NA("empty")；一個有限值都沒有 → NA("no_finite")。"""
    vals = list(values)
    if not vals:
        return NA("empty")
    finite = [v for v in vals if math.isfinite(v)]
    return min(finite) if finite else NA("no_finite")


# ---------- 讀檔與配對（4.11.6） ----------

def decode_rgb(raw):
    """hazy 與 GT 共用的解碼，做法和 golden 的 _read_rgb 相同：cv2.imdecode（IMREAD_COLOR）後 BGR → RGB。"""
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("無法解碼")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def pair_files(names, parse):
    """配對核心，只處理字串。parse(name) 回傳 (場景編號 int, "hazy" 或 "gt")；非影像檔回傳 None（忽略但計數）。

    O-HAZE 的實際檔名規則要下載後才確定，所以本檔不內建 parse。依場景編號配對，不依 list index；
    同一編號、同一角色有兩個以上檔案時丟 ValueError，停止整次執行（duplicate，不自行挑選）。
    回傳 {"pairs": [(編號, hazy 檔名, GT 檔名)]（依編號排序）, "missing": [編號], "ignored": [檔名]}。
    """
    table, ignored = {}, []
    for name in sorted(names):
        key = parse(name)
        if key is None:
            ignored.append(name)
            continue
        sid, role = key
        if type(sid) is not int or role not in ("hazy", "gt"):
            raise ValueError(f"parse 的結果不合法：{name!r} → {key!r}")
        slot = table.setdefault(sid, {})
        if role in slot:
            raise ValueError(f"duplicate：編號 {sid} 的 {role} 有 {slot[role]!r} 與 {name!r}")
        slot[role] = name
    ids = sorted(table)
    pairs = [(i, table[i]["hazy"], table[i]["gt"]) for i in ids if len(table[i]) == 2]
    return {"pairs": pairs, "missing": [i for i in ids if len(table[i]) != 2], "ignored": ignored}


class DirSource:
    """實際的資料目錄（只讀）。root 是 zip 解壓出的上層目錄（datasets/O-HAZE/# O-HAZY NTIRE 2018），下面是
    hazy/、GT/。列出所有檔案（含隱藏檔）；遇到 symlink 就停止，不跟隨。"""

    def __init__(self, root):
        if not os.path.isdir(root):
            raise ValueError(f"資料目錄不存在：{root!r}")
        self.root = os.path.abspath(root)

    def names(self):
        out = []
        for d, ds, fs in os.walk(self.root):
            for n in ds + fs:
                if os.path.islink(os.path.join(d, n)):
                    raise ValueError(f"資料目錄含 symlink，不跟隨：{os.path.relpath(os.path.join(d, n), self.root)!r}")
            out += [os.path.relpath(os.path.join(d, f), self.root).replace(os.sep, "/") for f in fs]
        return sorted(out)

    def open(self, name):
        return open(os.path.join(self.root, name), "rb")

    def read(self, name):
        with self.open(name) as fh:
            return fh.read()


def parse_ohaze(name):
    """O-HAZE 的實際檔名規則（2026-11-01 盤點，E Notes 4.11.6），交給 pair_files 用。
    name 是相對於解壓上層目錄的路徑，例如 hazy/01_outdoor_hazy.jpg、GT/03_outdoor_GT.JPG。
    - 檔名不是 NN_outdoor_hazy／NN_outdoor_GT 加 .jpg 或 .JPG → None；由 list_ohaze 再分成非影像與
      「檔名無法解析的影像」，不靜默忽略。
    - 檔名符合，但不在 hazy/、GT/ 的第一層，或檔名的角色和所在目錄不一致 → ValueError，停止整次執行。"""
    parts = name.split("/")
    m = re.fullmatch(r"(\d{2})_outdoor_(hazy|GT)\.(jpg|JPG)", parts[-1])
    if m is None:
        return None
    role = "gt" if m.group(2) == "GT" else "hazy"
    if len(parts) != 2 or OHAZE_DIRS.get(parts[0]) != role:
        raise ValueError(f"檔名和所在目錄不一致：{name!r}（{m.group(2)} 的檔案要在 {m.group(2)}/ 的第一層）")
    return int(m.group(1)), role


def _is_image(source, name):
    """pair_files 忽略的檔案是不是影像：Pillow 認得就算影像（只讀檔頭）。Pillow 認不得才算非影像；
    其他例外保守地算影像，讓它被列出來。"""
    try:
        with source.open(name) as fh, Image.open(fh):
            return True
    except UnidentifiedImageError:
        return False
    except Exception:
        return True


def list_ohaze(source):
    """列出資料目錄並配對（沿用 pair_files，不另寫配對核心）。duplicate、檔名和目錄不一致都丟 ValueError。
    pair_files 忽略的檔案分成 other（非影像，只計數並列名）與 unparsed（檔名無法解析的影像，ohaze 會拒絕開始）。"""
    names = source.names()
    res = pair_files(names, parse_ohaze)
    other, unparsed = [], []
    for n in res["ignored"]:
        (unparsed if _is_image(source, n) else other).append(n)
    missing_files = {}
    for n in names:
        k = parse_ohaze(n)
        if k is not None and k[0] in res["missing"]:
            missing_files.setdefault(k[0], []).append(n)
    return {"n_files": len(names), "pairs": res["pairs"], "missing": res["missing"], "missing_files": missing_files,
            "other": other, "unparsed": unparsed}


FORMAT_TEXT = {"unreadable": "無法讀取", "dtype": "不是 8-bit", "channels": "不是 3 channel",
               "orientation": "EXIF orientation 存在且不是 1", "decoded_size": "解碼後的尺寸和主圖 metadata 不同"}


def inspect_image(raw, decode=decode_rgb):
    """一個檔案的讀取與格式 gate（4.11.6「讀檔」、MPO 做法 A）。只用現成 API，不自製 JPEG／MPO／EXIF parser，
    也不旋轉、resize、crop 或轉換格式。回傳 (rgb 或 None, info)；info["bad"] 是空的才算通過。
    - 原始檔：bytes 與 SHA-256（整個檔案的位元組）。
    - Pillow（Image.open 時在 frame 0；MPO 的 frame 0 是主圖）：format、frame 數、H、W、EXIF orientation。
    - OpenCV IMREAD_UNCHANGED：原始的 dtype 與 channel 數，要是 uint8、3 channel。
    - decode（預設 decode_rgb）：shape 要等於 frame 0 的 (H, W, 3)。decoded RGB 的 SHA-256 是解回陣列的 C order 位元組，
      和原始檔的 SHA-256 分開。
    MPO 只用主圖，縮圖不參與。2026-11-01 Codex 對 4 個 MPO 做過一次「完整檔 vs 主圖區段」的解碼比對；這裡不重複做。"""
    info = {"bytes": len(raw), "raw_sha": hashlib.sha256(raw).hexdigest(), "bad": []}
    try:
        with Image.open(io.BytesIO(raw)) as im:
            info.update(format=im.format, frames=getattr(im, "n_frames", 1), h=im.size[1], w=im.size[0],
                        orientation=im.getexif().get(0x0112))
    except Exception as e:
        info["bad"].append("unreadable")
        info["msg"] = f"Pillow：{type(e).__name__}"
        return None, info
    un = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if un is None:
        info["bad"].append("unreadable")
        info["msg"] = "OpenCV 無法解碼"
        return None, info
    info["dtype"], info["channels"] = str(un.dtype), (1 if un.ndim == 2 else un.shape[2])
    del un
    if info["dtype"] != "uint8":
        info["bad"].append("dtype")
    if info["channels"] != 3:
        info["bad"].append("channels")
    if info["orientation"] not in (None, 1):
        info["bad"].append("orientation")
    try:
        rgb = decode(raw)
    except ValueError as e:
        info["bad"].append("unreadable")
        info["msg"] = f"decode：{e}"
        return None, info
    info["decoded"] = rgb.shape
    if rgb.shape != (info["h"], info["w"], 3):
        info["bad"].append("decoded_size")
    if info["bad"]:
        return None, info
    info["rgb_sha"] = hashlib.sha256(rgb.tobytes()).hexdigest()
    return rgb, info


def _read(source, name):
    """讀原始檔。I/O 錯誤（OSError，例如 FileNotFoundError、PermissionError）回傳 (None, 原因)，不偽造 hash 或尺寸。
    TimeoutError 也是 OSError 的子類別，但它代表 mutation 的逾時，照常往外丟。"""
    try:
        return source.read(name), None
    except TimeoutError:
        raise
    except OSError as e:
        return None, f"{type(e).__name__}: {e}"


def load_pair(source, pid, hazy_name, gt_name, decode=decode_rgb):
    """hazy 與 GT 走同一條讀取路徑（_read、inspect_image）。回傳 (hazy, gt, info)；info["status"] 是 ok、read_error、
    format 或 size_mismatch。不是 ok 時兩張影像都是 None：不計分，也不 resize、crop 或轉換。
    read_error：任一邊讀不到時，只記路徑與原因；兩邊都不做格式 gate，也不記 hash 或尺寸。"""
    hraw, hwhy = _read(source, hazy_name)
    graw, gwhy = _read(source, gt_name)
    if hwhy or gwhy:
        return None, None, {"id": pid, "status": "read_error",
                            **{k: ({"name": n, "read_error": why} if why else {"name": n, "skipped": True})
                               for k, n, why in (("hazy", hazy_name, hwhy), ("gt", gt_name, gwhy))}}
    hazy, hi = inspect_image(hraw, decode)
    gt, gi = inspect_image(graw, decode)
    del hraw, graw
    info = {"id": pid, "hazy": {"name": hazy_name, **hi}, "gt": {"name": gt_name, **gi}}
    if hi["bad"] or gi["bad"]:
        info["status"] = "format"
    elif hazy.shape != gt.shape:
        info["status"] = "size_mismatch"
    else:
        info["status"], info["h"], info["w"] = "ok", gt.shape[0], gt.shape[1]
    if info["status"] != "ok":
        hazy = gt = None
    return hazy, gt, info


def _dims(fi):
    return f"{fi.get('h', '?')}×{fi.get('w', '?')}"


def _file_status(fi):
    return "ok" if not fi["bad"] else "、".join(FORMAT_TEXT[b] for b in fi["bad"]) + (f"（{fi['msg']}）" if "msg" in fi else "")


def _pair_msg(info):
    if info["status"] == "read_error":
        return "；".join(f"{k} {info[k]['name']}：讀檔失敗 {info[k]['read_error']}" for k in ("hazy", "gt")
                        if "read_error" in info[k])
    if info["status"] == "format":
        return "；".join(f"{k} {info[k]['name']}：{_file_status(info[k])}" for k in ("hazy", "gt") if info[k]["bad"])
    if info["status"] == "size_mismatch":
        return f"hazy {_dims(info['hazy'])}、GT {_dims(info['gt'])}（H×W；不 resize、不 crop）"
    return ""


def _file_line(pid, role, fi):
    """pairs 與 ohaze 共用的逐檔一行。原始檔 SHA-256 與 decoded RGB SHA-256 都印完整值，分開兩欄。
    讀檔失敗或沒有做格式 gate 的檔案只印路徑與原因，不印 hash 或尺寸。"""
    if "read_error" in fi:
        return f"{pid:02d} {role:4} | {fi['name']} | 讀檔失敗：{fi['read_error']}（沒有 hash 或尺寸）"
    if fi.get("skipped"):
        return f"{pid:02d} {role:4} | {fi['name']} | 沒有做格式 gate（同一對的另一個檔案讀檔失敗）"
    o = fi.get("orientation", "?")
    dec = "×".join(map(str, fi["decoded"])) if "decoded" in fi else "-"
    return (f"{pid:02d} {role:4} | {fi['name']} | {fi['bytes']} bytes | {fi.get('format', '?')}×{fi.get('frames', '?')} | "
            f"H×W {_dims(fi)} | orientation {'無' if o is None else o} | {fi.get('dtype', '?')}／{fi.get('channels', '?')} ch | "
            f"decoded {dec} | {_file_status(fi)} | 原始 {fi['raw_sha']} | decoded RGB {fi.get('rgb_sha', '-')}")


def _listing_digest(entries):
    """清單識別：(路徑, 原始檔 SHA-256) 排序後，每行「路徑 SHA-256」串起來的 SHA-256。"""
    return hashlib.sha256("".join(f"{n} {h}\n" for n, h in sorted(entries)).encode()).hexdigest()


# ---------- FL／FX 的 golden 執行與每對一列（4.11.3、4.11.6、4.11.9） ----------

def fl_active():
    """浮點：ACTIVE0_DEFAULT、預設統計規則（tie first、th_round floor）。"""
    return golden.make_active(**golden.ACTIVE0_DEFAULT)


def fx_active():
    """定點候選：F = 20、定點 PALA、PTE T = 10、SD 候選 A（K = 10）；其餘和 FL 相同。格式是軟體暫定（W2–W6）。"""
    return golden.make_active(**golden.ACTIVE0_DEFAULT, inv_frac=20, pala_fixed=True, pte_frac=10, sd_mode="lut")


def run_mode(hazy, active, chunk_rows=0):
    """跑一次 static 排程（每次都由呼叫端給新的 active0）。只暫留 out、完整 stats 與 used；
    result 的其他中間值跟著 record 一起釋放。
    chunk_rows = 0：整張呼叫 golden.run_static。chunk_rows > 0：照 run_static 的順序，stats_pass 與 cpu_update
    對整張圖做，主路徑每 chunk_rows 列呼叫一次 apply_pass，每塊只留 out（4.11.6；golden 不改）。
    兩條路徑的 out、stats、used 由 Q16 比對。"""
    if type(chunk_rows) is not int or chunk_rows < 0:
        raise ValueError(f"chunk_rows 必須是 ≥ 0 的 Python int：{chunk_rows!r}")
    if chunk_rows == 0:
        rec = golden.run_static(hazy, active)
        return {"out": rec["result"]["out"], "stats": rec["stats"], "used": rec["used"]}
    stats = golden.stats_pass(hazy, active["t_h"], active["tie"])
    used, _ = golden.cpu_update(stats, active)
    parts = [golden.apply_pass(hazy[r:r + chunk_rows], used)["out"] for r in range(0, hazy.shape[0], chunk_rows)]
    return {"out": np.concatenate(parts, axis=0), "stats": stats, "used": used}


def fairness(fl, fx):
    """FL、FX 的同條件檢查：stats 每個欄位相等（ndarray 用 np.array_equal），used 的指定欄位相等。回傳不相等的欄位。"""
    a, b = fl["stats"], fx["stats"]
    if a.keys() != b.keys():
        return ["stats 的欄位"]
    bad = [f"stats.{k}" for k in a
           if not (np.array_equal(a[k], b[k]) if isinstance(a[k], np.ndarray) or isinstance(b[k], np.ndarray)
                   else a[k] == b[k])]
    return bad + [f"used.{k}" for k in USED_SAME if fl["used"][k] != fx["used"][k]]


def _stats_summary(m):
    s, u = m["stats"], m["used"]
    return {"t_h0": s["t_h_prev"], "gray_mean": s["gray_mean"], "a_far": u["a_far"], "a_near": u["a_near"],
            "n_far": s["far"]["count"], "n_near": s["near"]["count"]}


def pair_row(pid, gt, imgs, stats=None):
    """每對一列。imgs 是 {"H", "FL", "FX"} 三張 8-bit 影像，都和 gt 比較；任一個 metric 失敗，整對記 error。"""
    try:
        row = {"id": pid, "status": "ok", "h": gt.shape[0], "w": gt.shape[1], "stats": stats}
        for c in COLS:
            m = mse(imgs[c], gt)
            row[c] = {"mse": m, "psnr": psnr_from_mse(m), "ssim": ssim(imgs[c], gt)}
        row["d_psnr"] = delta(row["FX"]["psnr"], row["FL"]["psnr"])
        row["d_ssim"] = delta(row["FX"]["ssim"], row["FL"]["ssim"])
        row["fxfl"] = {**golden._diff_stats(imgs["FL"], imgs["FX"]), "psnr": psnr(imgs["FX"], imgs["FL"])}
        row["sha"] = {c: hashlib.sha256(imgs[c].tobytes()).hexdigest() for c in ("FL", "FX")}
        return row
    except Exception as e:      # 整對不計分，訊息列在報表
        return {"id": pid, "status": "error", "msg": f"metric：{type(e).__name__}: {e}"}


def run_pair(pid, hazy, gt, chunk_rows=0):
    """一對：FL、FX 依序跑，公平性比較後才釋放 stats 的 ndarray，再算 H、FL、FX 對 GT 的 metric。
    golden 失敗或公平性不通過，整對記 error、不計分（H、FL、FX 共用同一個成功集合 S）。
    chunk_rows 交給 run_mode（主路徑分塊；SSIM 不分塊）。ohaze 與 calib 都走這裡。"""
    t0 = time.perf_counter()
    try:
        fl = run_mode(hazy, fl_active(), chunk_rows)
        fx = run_mode(hazy, fx_active(), chunk_rows)
        bad = fairness(fl, fx)
        if bad:
            return {"id": pid, "status": "error", "msg": "公平性檢查不通過：" + "、".join(bad)}
        summary = _stats_summary(fl)
        imgs = {"H": hazy, "FL": fl["out"], "FX": fx["out"]}
        del fl, fx
    except Exception as e:
        return {"id": pid, "status": "error", "msg": f"golden：{type(e).__name__}: {e}"}
    row = pair_row(pid, gt, imgs, summary)
    row["sec"] = time.perf_counter() - t0
    return row


def summarize(rows):
    """合計，只用 status 為 ok 的配對（同一個 S）。不存在的統計量是 NA。"""
    ok = [r for r in rows if r["status"] == "ok"]
    s = {"n_total": len(rows), "n_ok": len(ok), "status": {}}
    for r in rows:
        s["status"].setdefault(r["status"], []).append(r["id"])
    for c in COLS:
        p = [r[c]["psnr"] for r in ok]
        s[c] = {"psnr": mean_psnr(ok, c), "ssim": column_mean([r[c]["ssim"] for r in ok]),
                "psnr_finite": finite_mean(p), "n_inf": sum(v == math.inf for v in p)}
    d = [r["d_psnr"] for r in ok]
    s["D"], s["mean_d_psnr"] = d_value(d), mean_delta(d)
    s["mean_d_ssim"] = mean_delta([r["d_ssim"] for r in ok])
    s["na_ids"] = [r["id"] for r in ok if isinstance(r["d_psnr"], NA)]
    s["extremes"] = delta_extremes([r["id"] for r in ok], d)
    f = [r["fxfl"]["psnr"] for r in ok]
    s["fxfl_inf"], s["fxfl_min"] = sum(v == math.inf for v in f), min_finite(f)
    return s


def _row_line(r):
    if r["status"] != "ok":
        return f"{r['id']:>3} | {r['status']} | {r.get('msg', '')}"
    st = r["stats"]
    cells = [f"{r['id']:>3}", f"{r['h']}×{r['w']}",
             "-" if st is None else (f"T_h0 {st['t_h0']} gray {st['gray_mean']} A_F {st['a_far']} "
                                     f"A_N {st['a_near']} 遠／近 {st['n_far']}／{st['n_near']}")]
    cells += [f"{c} {fmt(r[c]['psnr'], 4)} {fmt(r[c]['ssim'], 6)}" for c in COLS]
    cells.append(f"Δ {fmt(r['d_psnr'], 4)} {fmt(r['d_ssim'], 6)}")
    f = r["fxfl"]
    cells.append(f"{f['diff'] / f['n']:.4%} MAE {f['mae']:.6f} 最大 {f['max']} PSNR {fmt(f['psnr'], 4)}")
    cells.append(f"FL {r['sha']['FL'][:16]} FX {r['sha']['FX'][:16]}")
    cells.append(f"{r['sec']:.1f} 秒" if "sec" in r else "-")
    return " | ".join(cells)


def format_report(rows, s):
    """每對一列與合計（4.11.9 的欄位；報表開頭的設定標記與資料來源等 ohaze 子命令再加）。"""
    lines = ["每對一列（PSNR 小數 4 位、SSIM 6 位；FX 對 FL 是兩個模型輸出的差，不是畫質）",
             "ID | H×W | stats（active0 的 T_h、gray_mean、A_F、A_N、遠／近點數） | H 對 GT | FL 對 GT | FX 對 GT | "
             "Δ = FX − FL | FX 對 FL：不同的 channel 比例、MAE、最大差、PSNR | 輸出 SHA-256 前 16 碼 | 時間"]
    lines += [_row_line(r) for r in rows]
    lines.append(f"合計：N_ok = {s['n_ok']}／N_total = {s['n_total']}（每對權重相同的算術平均）")
    for c in COLS:
        fm, k = s[c]["psnr_finite"]
        lines.append(f"  {c} PSNR 平均 {fmt(s[c]['psnr'], 4)}（+inf {s[c]['n_inf']} 對；有限值的平均 {fmt(fm, 4)}，"
                     f"n = {k}／{s['n_ok']}，只當診斷）；SSIM 平均 {fmt(s[c]['ssim'], 6)}")
    lines.append(f"  D = |mean ΔPSNR| = {fmt(s['D'], 4)} dB（草案門檻 0.5 dB，本輪不判 PASS／FAIL）")
    lines.append(f"  帶正負號的 mean ΔPSNR {fmt(s['mean_d_psnr'], 4)}；mean ΔSSIM {fmt(s['mean_d_ssim'], 6)}")
    e = s["extremes"]
    if isinstance(e, NA):
        lines.append(f"  ΔPSNR 的最小值、|ΔPSNR| 的最大值、ΔPSNR < −0.5 的對數、最小的三對：{fmt(e, 4)}")
    else:
        lines.append(f"  ΔPSNR 最小 {fmt(e['min'][0], 4)}（ID {e['min'][1]}）；|ΔPSNR| 最大 {fmt(e['absmax'], 4)}；"
                     f"ΔPSNR < −0.5 的對數 {e['n_worse']}")
        lines.append("  ΔPSNR 最小的三對：" + "、".join(f"ID {i}（{fmt(v, 4)}）" for v, i in e["worst3"]))
    if s["na_ids"]:
        lines.append("  ΔPSNR 是 NA 的 ID：" + "、".join(str(i) for i in s["na_ids"]))
    lines.append(f"  FX 對 FL 的 PSNR（不是畫質）：+inf {s['fxfl_inf']} 對；有限值的最小值 {fmt(s['fxfl_min'], 4)}")
    st = s["status"]
    lines.append("  狀態：" + ("；".join(f"{k} {len(v)}" + ("" if k == "ok" else "（ID " + "、".join(map(str, v)) + "）")
                                     for k, v in sorted(st.items())) if st else "沒有配對"))
    return "\n".join(lines)


# ---------- 資源 gate（4.11.6） ----------

def resource_gate(rss_a, rss_b, rss_chunk_rows, chunk_rows, budget_gib, margin):
    """回傳每對允許的最大 pixel 數 P_max = floor((預算 / 餘裕 − a) / b)。
    a（bytes）、b（bytes/pixel）是 calib 在同一個分塊設定下兩點校準擬合的結果；預算（GiB）與餘裕要經使用者確認。
    本檔沒有任何預設值：12 GiB、15 分鐘、1.5 倍都只是提案。缺項、校準時的分塊設定和本次不同、數值不合理、
    或 P_max ≤ 0 時丟 ValueError。開了分塊不能代替這個 gate。"""
    missing = [k for k, v in (("--rss-a", rss_a), ("--rss-b", rss_b), ("--rss-chunk-rows", rss_chunk_rows),
                              ("--budget-gib", budget_gib), ("--margin", margin)) if v is None]
    if missing:
        raise ValueError("缺少資源條件：" + "、".join(missing))
    if rss_chunk_rows != chunk_rows:
        raise ValueError(f"校準時的分塊設定（{rss_chunk_rows} 列）和本次（{chunk_rows} 列）不同，校準值不適用")
    if not (all(math.isfinite(v) for v in (rss_a, rss_b, budget_gib, margin))
            and rss_a >= 0 and rss_b > 0 and budget_gib > 0 and margin >= 1):
        raise ValueError(f"資源條件不合理：a {rss_a}、b {rss_b}、預算 {budget_gib} GiB、餘裕 {margin}")
    p_max = math.floor((budget_gib * GIB / margin - rss_a) / rss_b)
    if p_max <= 0:
        raise ValueError(f"P_max = {p_max} ≤ 0：預算扣掉 a 之後，放不下任何一張圖")
    return p_max


# ---------- 報表開頭（4.11.4 的設定標記；4.11.9 的資料來源、環境與命令） ----------

def _sha_of(path):
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except (OSError, TypeError):
        return "（無法取得）"


def _dist_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "未安裝"


def _code_hashes():
    """golden、本檔、uv.lock、pyproject.toml 的 SHA-256（報表開頭與結果檔的 run 共用）。"""
    here = globals().get("__file__")
    base = os.path.dirname(os.path.abspath(here)) if here else None
    lock, proj = (os.path.join(base, n) if base else None for n in ("uv.lock", "pyproject.toml"))
    return {"golden": _sha_of(golden.__file__), "quality": _sha_of(here), "uv_lock": _sha_of(lock),
            "pyproject": _sha_of(proj)}


def header_lines(argv, settings=True):
    """設定值從 fl_active()、fx_active() 與 golden 的常數取，不另外寫死。settings=False 時只印 hash、環境與命令。"""
    h = _code_hashes()
    lines = [f"golden={h['golden']}；quality={h['quality']}；uv.lock={h['uv_lock']}；pyproject.toml={h['pyproject']}"]
    if settings:
        fl, fx = fl_active(), fx_active()
        lines += [f"rules: tie={fl['tie']}（L1 暫定）, th_round={fl['th_round']}（L2 暫定）, "
                  f"downsample=每 {golden.STEP} 列／行取左上角（L3 暫定）, "
                  f"active0=T_h {fl['t_h']}、A_F {fl['a_far']}、A_N {fl['a_near']}（L5／P7 暫定）",
                  f"schedule=static（每對獨立、active0 重設）；FL=浮點；FX=F{fx['inv_frac']}"
                  f"+PALA {'定點' if fx['pala_fixed'] else '浮點'}+T{fx['pte_frac']}+SD {fx['sd_mode']} K{golden.SD_LUT_K}"
                  "（W2–W6 暫定）",
                  "status=軟體暫定設定，不是團隊驗收"]
    vers = "、".join(f"{d} {_dist_version(d)}" for d in ("numpy", "opencv-python", "scikit-image", "scipy", "pillow"))
    lines += [f"環境：Python {platform.python_version()}、{vers}；{platform.platform()}",
              f"命令：{shlex.join(['ate_quality.py', *argv])}"]
    return lines


def _source_line():
    s = OHAZE_SOURCE
    return (f"資料來源（下載紀錄，README 與 run.log 第 10 節；本命令沒有重新驗證 zip）：{s['url']}；"
            f"zip SHA-256 {s['zip_sha256']}；ETag {s['etag']}；下載日期 {s['date']}")


# ---------- 子命令：pairs、ohaze、calib ----------

def pairs_command(source, argv):
    """只列配對清單與格式 gate：不呼叫 golden、不算畫質、只印 stdout、不寫檔。
    exit code：0 沒有異常；1 有 missing、read_error、format、size_mismatch 或檔名無法解析的影像；
    2 清單停止（duplicate、目錄不一致）。讀檔失敗的檔案只印路徑與原因，不中止。"""
    print("O-HAZE pairs：只列配對清單、格式 gate 與識別值（不呼叫 golden、不算畫質、不寫檔）")
    for ln in header_lines(argv, settings=False):
        print(ln)
    print(_source_line())
    print(f"資料目錄：{source.root}")
    try:
        lst = list_ohaze(source)
    except ValueError as e:
        print(f"停止：{e}")
        return 2
    print(f"清單：檔案 {lst['n_files']}；完整配對 {len(lst['pairs'])}；missing {len(lst['missing'])}"
          f"（{'、'.join(f'{i:02d}' for i in lst['missing']) or '無'}）；非影像 {len(lst['other'])}"
          f"（{'、'.join(lst['other']) or '無'}）；檔名無法解析的影像 {len(lst['unparsed'])}"
          f"（{'、'.join(lst['unparsed']) or '無'}）")
    print("逐對「ID 狀態」；逐檔：ID 角色 | 檔案 | bytes | Pillow format×frame 數 | H×W（frame 0） | orientation | "
          "原始 dtype／channel | decoded | 檔案狀態 | 原始檔 SHA-256 | decoded RGB SHA-256（C order）")
    status, seen = {}, []
    for pid, hn, gn in lst["pairs"]:
        _, _, info = load_pair(source, pid, hn, gn)      # 影像不保留，逐對釋放
        status.setdefault(info["status"], []).append(pid)
        print(f"{pid:02d} 狀態 {info['status']}" + (f"（{_pair_msg(info)}）" if info["status"] != "ok" else ""))
        for k in ("hazy", "gt"):
            print("   " + _file_line(pid, k, info[k]))
            if "raw_sha" in info[k]:            # 讀不到的檔案沒有 hash，不放進清單識別
                seen.append((info[k]["name"], info[k]["raw_sha"]))
    for i in lst["missing"]:
        print(f"{i:02d} 狀態 missing（只有 {'、'.join(lst['missing_files'][i])}；沒有做格式 gate）")
    print("配對狀態：" + "；".join(f"{k} {len(v)}" + ("" if k == "ok" else "（ID " + "、".join(f"{i:02d}" for i in v) + "）")
                              for k, v in sorted(status.items())) + f"；missing {len(lst['missing'])}")
    print(f"清單識別（配對檔案的「路徑 原始檔 SHA-256」）：{_listing_digest(seen)}")
    return 1 if lst["missing"] or lst["unparsed"] or set(status) - {"ok"} else 0


def _chunk_line(chunk_rows):
    return f"分塊：{'不分塊' if chunk_rows == 0 else f'主路徑每 {chunk_rows} 列'}（stats_pass、cpu_update 與 SSIM 都對整張圖）"


def _gate_line(gate, p_max):
    return (f"資源 gate：a {gate['rss_a']} bytes、b {gate['rss_b']} bytes/pixel（校準時每 {gate['rss_chunk_rows']} 列分塊）、"
            f"預算 {gate['budget_gib']} GiB、餘裕 {gate['margin']} → P_max {p_max} pixel")


def _list_line(lst):
    return (f"清單：檔案 {lst['n_files']}；完整配對 {len(lst['pairs'])}；missing {len(lst['missing'])}"
            f"（{'、'.join(f'{i:02d}' for i in lst['missing']) or '無'}）；非影像 {len(lst['other'])}"
            f"（{'、'.join(lst['other']) or '無'}）；檔名無法解析的影像 {len(lst['unparsed'])}")


def _print_listing(sel, pre, missing_files):
    """配對清單、逐檔 SHA-256 與清單識別。ohaze 與 ohaze-report 共用，兩者的報表開頭才會逐字相同（4.11.17）。"""
    print("配對清單（預檢；SHA-256 前 16 碼，完整值在下方）：ID | hazy | GT | H×W | 狀態 | hazy 原始 | GT 原始")
    for i in sel:
        if i in pre:
            p = pre[i]
            print(f"配對 ID {i:02d} | {p['hazy']['name']} | {p['gt']['name']} | {_dims(p) if p['status'] == 'ok' else '-'} | {p['status']} | "
                  f"{p['hazy'].get('raw_sha', '-')[:16]} | {p['gt'].get('raw_sha', '-')[:16]}")
        else:
            print(f"配對 ID {i:02d} | missing | 只有 {'、'.join(missing_files[i])}")
    print("完整 SHA-256（逐檔：原始檔、decoded RGB）：")
    for i in sel:
        for role in (("hazy", "gt") if i in pre else ()):
            print("   " + _file_line(i, role, pre[i][role]))
    print("清單識別（本次選取、有配對、讀得到的檔案）：" + _listing_digest(
        [(pre[i][k]["name"], pre[i][k]["raw_sha"]) for i in pre for k in ("hazy", "gt") if "raw_sha" in pre[i][k]]))


def _print_tail(rows, s):
    print(format_report(rows, s))
    print(f"和論文的對照（設定不同，僅供參考）：{PAPER_TABLE1}")
    print("峰值 RSS 看外部 /usr/bin/time -l 的 maximum resident set size（本命令不量）；重跑比對時排除時間欄。")


def ohaze_command(source, argv, ids, chunk_rows, gate, after_first, store=None):
    """O-HAZE 計分（4.11.3、4.11.6、4.11.9）：H、FL、FX 對同一張 GT，每對獨立 static 排程、重設 active0。
    命令存在不等於已獲准執行：資源校準、smoke、全部配對都要使用者另外批准（E Notes 4.11.12）。
    順序：資源 gate → 清單 → 結果檔的開始前檢查（給了 --record 時）→ 預檢（格式 gate、尺寸）
    → 逐對執行（golden 前檢查 pixel 上限；給了 --record 時每列寫入並讀回驗證）→ 報表。
    第一對完成後停止，除非明給 --after-first（只在使用者批准第二對之後的階段使用）。
    exit code：0 完成且 N_ok ≥ 1；1 N_ok = 0；2 拒絕開始（含預檢讀檔失敗、結果檔的開始前檢查）；3 遇到超過 pixel 上限
    的配對而停止（不跳過大圖繼續）；4 執行途中讀檔失敗而停止（保留已完成的結果並印報表；這一列不寫入結果檔）；
    5 結果檔寫入或讀回驗證失敗而停止（4.11.17；不刪除、不修改已寫的內容）。"""
    print(OHAZE_TITLE)
    for ln in header_lines(argv):
        print(ln)
    print(_source_line())
    print(f"資料目錄：{source.root}")
    print(_chunk_line(chunk_rows))
    try:
        p_max = resource_gate(chunk_rows=chunk_rows, **gate)
    except ValueError as e:
        print(f"拒絕開始（資源 gate）：{e}。開了分塊不能代替資源 gate；校準值、預算與餘裕都要經使用者確認（4.11.6）")
        return 2
    print(_gate_line(gate, p_max))
    try:
        lst = list_ohaze(source)
    except ValueError as e:
        print(f"拒絕開始（清單）：{e}")
        return 2
    print(_list_line(lst))
    if lst["unparsed"]:
        print("拒絕開始（清單）：有檔名無法解析的影像，不自行判定它屬於哪一對：" + "、".join(lst["unparsed"]))
        return 2
    complete = {p[0]: p for p in lst["pairs"]}
    known = sorted(set(complete) | set(lst["missing"]))
    sel = known if ids is None else sorted(set(ids))
    unknown = [i for i in sel if i not in known]
    if unknown:
        print("拒絕開始：清單中沒有這些 ID：" + "、".join(map(str, unknown)))
        return 2
    run = None
    if store is not None:
        run = record_run(source.root, chunk_rows, gate, p_max)
        why = record_precheck(store, run, sel)
        if why:
            print(f"拒絕開始（結果檔）：{why}。不修改既有內容（4.11.17）")
            return 2

    def keep(pinfo):
        """給了 --record 時，把剛加入的那一列寫入結果檔並讀回驗證；失敗時印原因、回傳 False（4.11.17）。"""
        if store is None:
            return True
        try:
            n = write_record(store, run, pinfo, rows[-1])
        except RecordError as e:
            print(f"停止：ID {rows[-1]['id']} 的紀錄寫入或讀回驗證失敗（{e}）。不刪除、不修改已寫的內容，不繼續下一對")
            return False
        print(f"紀錄已寫入並讀回驗證：ID {rows[-1]['id']:02d}、第 {n} 行")
        return True

    pre = {i: load_pair(source, *complete[i])[2] for i in sel if i in complete}     # 預檢：影像不保留
    _print_listing(sel, pre, lst["missing_files"])
    read_bad = [i for i in sel if i in pre and pre[i]["status"] == "read_error"]
    if read_bad:
        print("拒絕開始（預檢讀檔失敗）：" + "；".join(f"ID {i}（{_pair_msg(pre[i])}）" for i in read_bad)
              + "。不開始 golden，也不偽造讀不到的 hash 或尺寸")
        return 2
    fmt_bad = [i for i in sel if i in pre and pre[i]["status"] == "format"]
    if fmt_bad:
        print("拒絕開始（格式 gate）：ID " + "、".join(map(str, fmt_bad)) + " 不符合 4.11.6；"
              "不自動旋轉、resize、crop 或轉換，停下來問")
        return 2
    rows, ran, stop_rc = [], 0, 0
    for k, i in enumerate(sel):
        if i not in complete:
            rows.append({"id": i, "status": "missing", "msg": "只有 " + "、".join(lst["missing_files"][i])})
            if not keep({"id": i, "status": "missing", "missing_files": lst["missing_files"][i]}):
                stop_rc = 5
                break
            continue
        info = pre[i]
        if info["status"] != "ok":
            rows.append({"id": i, "status": info["status"], "msg": _pair_msg(info)})
            if not keep(info):
                stop_rc = 5
                break
            continue
        if info["h"] * info["w"] > p_max:
            print(f"停止：ID {i} 是 {info['h']}×{info['w']} = {info['h'] * info['w']} pixel，超過 P_max {p_max}。"
                  "不跳過大圖、不繼續下一對（4.11.6）")
            stop_rc = 3
            break
        hazy, gt, now = load_pair(source, *complete[i])
        if now["status"] == "read_error":
            rows.append({"id": i, "status": "read_error", "msg": _pair_msg(now)})
            print(f"停止：ID {i} 執行時讀檔失敗（{_pair_msg(now)}）。保留已完成的結果並印報表，不繼續下一對")
            stop_rc = 4
            break
        if now["status"] != "ok" or any(now[r]["raw_sha"] != info[r]["raw_sha"] for r in ("hazy", "gt")):
            rows.append({"id": i, "status": "error", "msg": "讀檔結果和預檢不同"})
        else:
            rows.append(run_pair(i, hazy, gt, chunk_rows))
        del hazy, gt
        ran += 1
        if not keep(info):
            stop_rc = 5
            break
        rest = [j for j in sel[k + 1:] if j in pre and pre[j]["status"] == "ok"]
        if ran == 1 and not after_first and rest:
            print("第一對完成後停止：還有 ID " + "、".join(map(str, rest)) + " 沒跑。看過這一對的時間與峰值 RSS、"
                  "經使用者另外批准後，才加 --after-first 繼續（4.11.6、4.11.12）")
            break
    s = summarize(rows)
    _print_tail(rows, s)
    if stop_rc:
        return stop_rc
    return 0 if s["n_ok"] > 0 else 1


def calib_command(argv, h, w, chunk_rows, seed):
    """RSS 校準用的單一尺寸執行（4.11.6）：在記憶體產生合成的 hazy、GT（uint8 隨機圖，不讀資料、不寫檔），
    走和 ohaze 相同的讀檔後路徑 run_pair（FL、FX、公平性、PSNR、SSIM，同一個分塊設定）。
    峰值 RSS 由外部的 /usr/bin/time -l 量，本命令只印時間；每個尺寸一個獨立程序。命令存在不等於已獲准執行。"""
    print("calib：RSS 校準用的合成圖執行（要在 /usr/bin/time -l 下執行；命令存在不等於已獲准執行）")
    for ln in header_lines(argv):
        print(ln)
    if h < 11 or w < 11:
        print(f"拒絕開始：H、W 都要 ≥ 11（SSIM 的 11×11 視窗）：{h}×{w}")
        return 2
    rng = np.random.default_rng(seed)
    hazy = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    gt = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    print(f"合成圖：{h}×{w}（{h * w} pixel），seed {seed}；"
          f"分塊：{'不分塊' if chunk_rows == 0 else f'主路徑每 {chunk_rows} 列'}（讀檔解碼不在校準裡）")
    row = run_pair(0, hazy, gt, chunk_rows)
    del hazy, gt
    print(_row_line(row))
    return 0 if row["status"] == "ok" else 1


# ---------- 結果檔：逐對紀錄、完成紀錄與彙整（E Notes 4.11.17） ----------
# 只寫這兩種紀錄的 writer 與 reader，不建立通用的 serialization 或 transaction 機制。

class RecordError(Exception):
    """結果檔寫入、寫入前驗證或讀回比對失敗（ohaze 的 exit 5、ohaze-complete 的 exit 5）。"""


class RecordFile:
    """結果檔：只讀全部 bytes 與在結尾追加；不刪除、不改寫既有內容。selfcheck 用 _MemRecord 替換。"""

    def __init__(self, path):
        self.path = os.path.abspath(path)

    def exists(self):
        return os.path.exists(self.path)

    def read(self):
        with open(self.path, "rb") as fh:
            return fh.read()

    def append(self, data):
        with open(self.path, "ab") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())


def _code_run():
    """run 裡和程式、環境有關的部分；ohaze-complete、ohaze-report 都拿它和執行中的程式比對。"""
    h = header_lines([])
    return {**_code_hashes(), "settings": h[1:4], "env": h[4], "resource_metric": RESOURCE_METRIC}


def record_run(data_dir, chunk_rows, gate, p_max):
    """一次評估的 run：版本、設定、資料目錄與資源 gate。同一個結果檔的每一行都必須逐值相同。"""
    return {**_code_run(), "data_dir": data_dir, "chunk_rows": chunk_rows,
            "gate": {**{k: gate[k] for k in ("rss_a", "rss_b", "rss_chunk_rows", "budget_gib", "margin")}, "p_max": p_max}}


def _enc(v):
    """寫入用的值：tuple 寫成 array；+inf 寫成字串 "+inf"；NA 寫成 {"NA": 原因}；有限浮點照 repr。其他非有限值不寫入。"""
    if isinstance(v, NA):
        return {"NA": v.reason}
    if type(v) is float:
        if v == math.inf:
            return "+inf"
        if not math.isfinite(v):
            raise ValueError(f"不能寫入的浮點數：{v!r}")
        return float(v)
    if isinstance(v, dict):
        return {k: _enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_enc(x) for x in v]
    if v is None or type(v) in (int, str, bool):
        return v
    raise ValueError(f"不能寫入的型別：{type(v).__name__}")


def _line_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _bad_token(tok):
    raise ValueError(f"不接受的數字 token：{tok}")


def _finite_float(tok):
    v = float(tok)
    if not math.isfinite(v):
        raise ValueError(f"數字溢位成非有限值：{tok}")
    return v


def _no_dup_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"重複的 key：{k}")
        out[k] = v
    return out


def _strict_json(text):
    """NaN、Infinity、-Infinity 與溢位成無限大的數字（例如 1e999）都在這一層拒絕；後面的欄位檢查不再重複檢查有限性。"""
    return json.loads(text, parse_constant=_bad_token, parse_float=_finite_float, object_pairs_hook=_no_dup_keys)


def _keys(d, where, need, opt=()):
    if type(d) is not dict:
        raise ValueError(f"{where} 不是 object")
    miss = [k for k in need if k not in d]
    if miss:
        raise ValueError(f"{where} 缺少必要的 key：{'、'.join(miss)}")
    extra = [k for k in d if k not in need and k not in opt]
    if extra:
        raise ValueError(f"{where} 有未定義的 key：{'、'.join(extra)}")


def _int(v, where, lo=0, hi=None):
    if type(v) is not int or v < lo or (hi is not None and v > hi):
        raise ValueError(f"{where} 必須是 {lo}–{'∞' if hi is None else hi} 的整數：{v!r}")
    return v


def _flt(v, where, lo=None):
    if type(v) is not float or (lo is not None and v < lo):
        raise ValueError(f"{where} 必須是浮點數{'' if lo is None else f'（≥ {lo}）'}：{v!r}")
    return v


def _str(v, where):
    if type(v) is not str:
        raise ValueError(f"{where} 必須是字串：{v!r}")
    return v


def _hex(v, where):
    if type(v) is not str or re.fullmatch(r"[0-9a-f]{64}", v) is None:
        raise ValueError(f"{where} 不是 SHA-256（64 個小寫十六進位字元）：{v!r}")
    return v


def _psnr_val(v, where):
    if v == "+inf":
        return math.inf
    return _flt(v, where)


def _ints3(v, where, lo, hi):
    if type(v) is not list or len(v) != 3:
        raise ValueError(f"{where} 必須是 3 個整數的 array：{v!r}")
    return tuple(_int(x, where, lo, hi) for x in v)


def _dec_run(r):
    _keys(r, "run", ("golden", "quality", "uv_lock", "pyproject", "settings", "env", "data_dir", "chunk_rows", "gate",
                     "resource_metric"))
    for k in ("golden", "quality", "uv_lock", "pyproject"):
        _hex(r[k], f"run.{k}")
    if type(r["settings"]) is not list or len(r["settings"]) != 3 or any(type(x) is not str for x in r["settings"]):
        raise ValueError("run.settings 必須是 3 個字串的 array")
    _str(r["env"], "run.env")
    _str(r["data_dir"], "run.data_dir")
    _int(r["chunk_rows"], "run.chunk_rows")
    g = r["gate"]
    _keys(g, "run.gate", ("rss_a", "rss_b", "rss_chunk_rows", "budget_gib", "margin", "p_max"))
    for k in ("rss_a", "rss_b", "budget_gib", "margin"):
        _flt(g[k], f"run.gate.{k}")
    _int(g["rss_chunk_rows"], "run.gate.rss_chunk_rows")
    _int(g["p_max"], "run.gate.p_max", 1)
    if r["resource_metric"] != RESOURCE_METRIC:
        raise ValueError(f"run.resource_metric 不是 {RESOURCE_METRIC!r}")
    return r


FILE_KEYS = ("name", "bytes", "raw_sha", "bad", "format", "frames", "h", "w", "orientation", "dtype", "channels",
             "decoded", "rgb_sha")


def _dec_file(f, where, pid, role):
    """寫入的配對都已通過格式 gate（format、read_error 的配對 runner 會拒絕開始），所以 bad 一定是空的、rgb_sha 一定存在。"""
    _keys(f, where, FILE_KEYS)
    name = _str(f["name"], f"{where}.name")
    if parse_ohaze(name) != (pid, role):
        raise ValueError(f"{where}.name 的 ID 或角色和紀錄不符：{name!r}")
    if f["bad"] != []:
        raise ValueError(f"{where}.bad 必須是空的：有格式問題的配對不會寫入")
    h, w = _int(f["h"], f"{where}.h", 1), _int(f["w"], f"{where}.w", 1)
    ori = f["orientation"]
    if not (ori is None or (type(ori) is int and ori == 1)) or f["dtype"] != "uint8":
        raise ValueError(f"{where} 的 orientation 或 dtype 和通過格式 gate 的值不符")
    _int(f["channels"], f"{where}.channels", 3, 3)
    out = {**f, "name": name, "bytes": _int(f["bytes"], f"{where}.bytes"), "raw_sha": _hex(f["raw_sha"], f"{where}.raw_sha"),
           "format": _str(f["format"], f"{where}.format"), "frames": _int(f["frames"], f"{where}.frames", 1),
           "decoded": _ints3(f["decoded"], f"{where}.decoded", 1, None), "rgb_sha": _hex(f["rgb_sha"], f"{where}.rgb_sha")}
    if out["decoded"] != (h, w, 3):
        raise ValueError(f"{where}.decoded 和 H×W 不符")
    return out


def _dec_pair(p):
    st = p.get("status") if type(p) is dict else None
    if st == "ok":
        _keys(p, "pair", ("id", "status", "hazy", "gt", "h", "w"))
    elif st == "size_mismatch":
        _keys(p, "pair", ("id", "status", "hazy", "gt"))
    elif st == "missing":
        _keys(p, "pair", ("id", "status", "missing_files"))
    else:
        raise ValueError(f"pair.status 不合法：{st!r}")
    pid = _int(p["id"], "pair.id", 1, 45)
    out = {"id": pid, "status": st}
    if st == "missing":
        mf = p["missing_files"]
        if type(mf) is not list or not mf or any(type(n) is not str or (parse_ohaze(n) or (None,))[0] != pid for n in mf):
            raise ValueError("pair.missing_files 必須是同一個 ID 的檔名 array")
        out["missing_files"] = list(mf)
        return out
    out["hazy"], out["gt"] = _dec_file(p["hazy"], "pair.hazy", pid, "hazy"), _dec_file(p["gt"], "pair.gt", pid, "gt")
    same = (out["hazy"]["h"], out["hazy"]["w"]) == (out["gt"]["h"], out["gt"]["w"])
    if st == "ok":
        out["h"], out["w"] = _int(p["h"], "pair.h", 1), _int(p["w"], "pair.w", 1)
        if not same or (out["h"], out["w"]) != (out["gt"]["h"], out["gt"]["w"]):
            raise ValueError("pair 是 ok，但 H×W 和兩個檔案不符")
    elif same:
        raise ValueError("pair 是 size_mismatch，但兩個檔案的 H×W 相同")
    return out


ROW_OK_KEYS = ("id", "status", "h", "w", "stats", "H", "FL", "FX", "d_psnr", "d_ssim", "fxfl", "sha", "sec")


def _dec_row(r, pair):
    st = r.get("status") if type(r) is dict else None
    allowed = {"ok": ("ok", "error"), "size_mismatch": ("size_mismatch",), "missing": ("missing",)}[pair["status"]]
    if st not in allowed:
        raise ValueError(f"row.status {st!r} 和 pair.status {pair['status']!r} 不相容")
    if st == "error":
        _keys(r, "row", ("id", "status", "msg"), ("sec",))
    elif st != "ok":
        _keys(r, "row", ("id", "status", "msg"))
    if st != "ok":
        if _int(r["id"], "row.id", 1, 45) != pair["id"]:
            raise ValueError("row.id 和 pair.id 不同")
        out = {"id": r["id"], "status": st, "msg": _str(r["msg"], "row.msg")}
        if "sec" in r:
            out["sec"] = _flt(r["sec"], "row.sec", 0.0)
        return out
    _keys(r, "row", ROW_OK_KEYS)
    if _int(r["id"], "row.id", 1, 45) != pair["id"]:
        raise ValueError("row.id 和 pair.id 不同")
    h, w = _int(r["h"], "row.h", 1), _int(r["w"], "row.w", 1)
    if (h, w) != (pair["h"], pair["w"]):
        raise ValueError("row 的 H×W 和 pair 不同")
    s = r["stats"]
    _keys(s, "row.stats", ("t_h0", "gray_mean", "a_far", "a_near", "n_far", "n_near"))
    stats = {"t_h0": _int(s["t_h0"], "row.stats.t_h0", 0, 255), "gray_mean": _int(s["gray_mean"], "row.stats.gray_mean", 0, 255),
             "a_far": _ints3(s["a_far"], "row.stats.a_far", 0, 255), "a_near": _ints3(s["a_near"], "row.stats.a_near", 0, 255),
             "n_far": _int(s["n_far"], "row.stats.n_far"), "n_near": _int(s["n_near"], "row.stats.n_near")}
    out = {"id": r["id"], "status": "ok", "h": h, "w": w, "stats": stats}
    for c in COLS:
        _keys(r[c], f"row.{c}", ("mse", "psnr", "ssim"))
        m = {"mse": _flt(r[c]["mse"], f"row.{c}.mse", 0.0), "psnr": _psnr_val(r[c]["psnr"], f"row.{c}.psnr"),
             "ssim": _flt(r[c]["ssim"], f"row.{c}.ssim")}
        if (m["psnr"] == math.inf) != (m["mse"] == 0):
            raise ValueError(f"row.{c}：PSNR 是 +inf 若且唯若 MSE 是 0")
        out[c] = m
    if math.isfinite(out["FX"]["psnr"]) and math.isfinite(out["FL"]["psnr"]):
        d_psnr = _flt(r["d_psnr"], "row.d_psnr")
        if d_psnr != out["FX"]["psnr"] - out["FL"]["psnr"]:
            raise ValueError("row.d_psnr 不等於 FX − FL")
    elif r["d_psnr"] == {"NA": "nonfinite"}:
        d_psnr = NA("nonfinite")
    else:
        raise ValueError("FL 或 FX 的 PSNR 是 +inf 時，row.d_psnr 必須是 {\"NA\": \"nonfinite\"}")
    d_ssim = _flt(r["d_ssim"], "row.d_ssim")
    if d_ssim != out["FX"]["ssim"] - out["FL"]["ssim"]:
        raise ValueError("row.d_ssim 不等於 FX − FL")
    out["d_psnr"], out["d_ssim"] = d_psnr, d_ssim
    f = r["fxfl"]
    _keys(f, "row.fxfl", ("n", "diff", "pix", "abs", "max", "mae", "psnr"))
    n, diff, pix = _int(f["n"], "row.fxfl.n", 1), _int(f["diff"], "row.fxfl.diff"), _int(f["pix"], "row.fxfl.pix")
    tot, mx = _int(f["abs"], "row.fxfl.abs"), _int(f["max"], "row.fxfl.max", 0, 255)
    mae, fp = _flt(f["mae"], "row.fxfl.mae", 0.0), _psnr_val(f["psnr"], "row.fxfl.psnr")
    if n != 3 * h * w:
        raise ValueError("row.fxfl.n 必須等於 3 × H × W（channel 總數）")
    if not (pix <= diff <= min(n, 3 * pix) and diff <= tot <= 255 * diff and (mx == 0) == (diff == 0)):
        raise ValueError("row.fxfl 的 diff（不同的 channel 數）、pix（不同的 pixel 數）、abs、max 互相矛盾")
    if mae != tot / n or (fp == math.inf) != (tot == 0):
        raise ValueError("row.fxfl 的 mae 或 psnr 和 abs 不一致")
    out["fxfl"] = {"n": n, "diff": diff, "pix": pix, "abs": tot, "max": mx, "mae": mae, "psnr": fp}
    _keys(r["sha"], "row.sha", ("FL", "FX"))
    out["sha"] = {c: _hex(r["sha"][c], f"row.sha.{c}") for c in ("FL", "FX")}
    out["sec"] = _flt(r["sec"], "row.sec", 0.0)
    return out


def decode_line(raw):
    """一行結果檔（不含結尾的 \\n）→ {"kind": "data", run, pair, row} 或 {"kind": "done", ...}。不合法就丟 ValueError。"""
    try:
        obj = _strict_json(raw.decode("utf-8"))
    except UnicodeDecodeError as e:
        raise ValueError(f"不是 UTF-8：{e}") from None
    sch = obj.get("schema") if type(obj) is dict else None
    if sch == REC_SCHEMA:
        _keys(obj, "資料行", ("schema", "run", "written_at", "pair", "row"))
        _str(obj["written_at"], "written_at")
        pair = _dec_pair(obj["pair"])
        return {"kind": "data", "run": _dec_run(obj["run"]), "pair": pair, "row": _dec_row(obj["row"], pair)}
    if sch == DONE_SCHEMA:
        _keys(obj, "完成紀錄", ("schema", "run", "written_at", "data_lines", "ids", "data_sha256"))
        _str(obj["written_at"], "written_at")
        if type(obj["ids"]) is not list:
            raise ValueError("完成紀錄的 ids 必須是 array")
        return {"kind": "done", "run": _dec_run(obj["run"]), "data_lines": _int(obj["data_lines"], "data_lines"),
                "ids": [_int(x, "ids", 1, 45) for x in obj["ids"]], "data_sha256": _hex(obj["data_sha256"], "data_sha256")}
    raise ValueError(f"schema 不合法：{sch!r}")


def read_records(data):
    """整個結果檔 → (資料行 list, 完成紀錄或 None, 完成紀錄之前的 bytes)。任何一行不合法、最後一行不完整、
    完成紀錄不在最後一行（含出現兩次）：丟 ValueError，附行號。"""
    if data and not data.endswith(b"\n"):
        raise ValueError("最後一行不完整（沒有以 \\n 結尾）")
    lines = data.split(b"\n")[:-1] if data else []
    recs, done, prefix = [], None, data
    for k, raw in enumerate(lines, 1):
        try:
            d = decode_line(raw)
        except ValueError as e:
            raise ValueError(f"第 {k} 行：{e}") from None
        if d["kind"] == "done":
            if k != len(lines):
                raise ValueError(f"第 {k} 行：完成紀錄不在最後一行")
            done, prefix = d, data[:len(data) - len(raw) - 1]
        else:
            recs.append(d)
    return recs, done, prefix


def _same(a, b):
    """逐值、逐型別相等（float 用 ==，+inf 相等；tuple 和 list 不算相同）。"""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def write_record(store, run, pair, row):
    """寫入一行資料（4.11.17 第 4 點）：寫入前用正式的 reader 解碼這一行，和記憶體中的 run、pair、row 逐值比較；
    追加後讀回整個檔，必須等於「原內容 + 這一行」。成功回傳行號；任何失敗丟 RecordError，不刪除、不修改已寫的內容。"""
    try:
        line = _line_bytes({"schema": REC_SCHEMA, "run": _enc(run), "written_at": _now(), "pair": _enc(pair), "row": _enc(row)})
        d = decode_line(line[:-1])
        if not (_same(d["run"], run) and _same(d["pair"], pair) and _same(d["row"], row)):
            raise RecordError("寫入前的驗證：解碼後的值和記憶體中的不同")
    except ValueError as e:
        raise RecordError(f"寫入前的驗證：{e}") from None
    try:
        before = store.read() if store.exists() else b""
        store.append(line)
        after = store.read()
    except OSError as e:
        raise RecordError(f"寫入失敗：{type(e).__name__}: {e}") from None
    if after != before + line:
        raise RecordError("讀回的檔案內容和寫入的不同")
    return after.count(b"\n")


def record_precheck(store, run, sel):
    """ohaze --record 的開始前檢查。回傳拒絕原因；可以開始時回傳 None。"""
    if not store.exists():
        return None
    try:
        recs, done, _ = read_records(store.read())
    except (OSError, ValueError) as e:
        return f"結果檔無法讀取或不合法：{e}"
    if done is not None:
        return "結果檔已有完成紀錄（已完成的檔不再追加）"
    bad = [k for k, r in enumerate(recs, 1) if not _same(r["run"], run)]
    if bad:
        return "第 " + "、".join(map(str, bad)) + " 行的 run 和本次不同（版本、設定、資料目錄或資源 gate）"
    dup = sorted({r["pair"]["id"] for r in recs} & set(sel))
    if dup:
        return "結果檔已有本次要跑的 ID：" + "、".join(f"{i:02d}" for i in dup)
    return None


def _id_problems(ids, expected):
    dup = sorted({i for i in ids if ids.count(i) > 1})
    miss = [i for i in expected if i not in ids]
    extra = sorted({i for i in ids if i not in expected})
    out = []
    if dup:
        out.append("重複的 ID：" + "、".join(f"{i:02d}" for i in dup))
    if miss:
        out.append("缺少的 ID：" + "、".join(f"{i:02d}" for i in miss))
    if extra:
        out.append("多餘的 ID：" + "、".join(f"{i:02d}" for i in extra))
    return out


def _code_part(run):
    return {k: run[k] for k in ("golden", "quality", "uv_lock", "pyproject", "settings", "env", "resource_metric")}


def _run_problems(recs, done):
    """各行（含完成紀錄）的 run 必須逐值相同，而且程式與環境的部分要和執行中的這份程式相同。"""
    run0 = recs[0]["run"]
    bad = [k for k, r in enumerate(recs, 1) if not _same(r["run"], run0)]
    out = ["第 " + "、".join(map(str, bad)) + " 行的 run 和第 1 行不同"] if bad else []
    if done is not None and not _same(done["run"], run0):
        out.append("完成紀錄的 run 和資料行不同")
    if not _same(_code_part(run0), _code_run()):
        out.append("紀錄的程式或環境（hash、設定標記、環境）和執行彙整的這份程式不同")
    return out


def complete_command(store, argv):
    """ohaze-complete（4.11.17）：外層迴圈在 45 對都通過事後檢查後才呼叫一次。檢查資料行的 ID 恰好是 01–45、run 一致，
    再追加完成紀錄（綁定前面全部資料行 bytes 的 SHA-256），寫入前驗證、寫入後讀回比對。
    exit code：0 寫入並驗證；2 拒絕（不修改既有內容）；5 寫入或讀回驗證失敗（不刪除、不修改已寫的內容）。"""
    print("ohaze-complete：追加完成紀錄（E Notes 4.11.17；只在外層迴圈的 45 對都通過事後檢查後執行）")
    for ln in header_lines(argv, settings=False):
        print(ln)
    try:
        data = store.read()
        recs, done, _ = read_records(data)
    except (OSError, ValueError) as e:
        print(f"拒絕（結果檔）：{e}。不修改既有內容")
        return 2
    probs = (["結果檔已有完成紀錄"] if done is not None else []) + (["結果檔沒有資料行"] if not recs else [])
    if recs:
        probs += _id_problems([r["pair"]["id"] for r in recs], EXPECTED_IDS)
        probs += _run_problems(recs, None)
    if probs:
        print("拒絕寫入完成紀錄：" + "；".join(probs) + "。不修改既有內容")
        return 2
    obj = {"schema": DONE_SCHEMA, "run": _enc(recs[0]["run"]), "written_at": _now(), "data_lines": len(recs),
           "ids": [r["pair"]["id"] for r in recs], "data_sha256": hashlib.sha256(data).hexdigest()}
    try:
        line = _line_bytes(obj)
        d = decode_line(line[:-1])
        if d["kind"] != "done" or d["data_sha256"] != hashlib.sha256(data).hexdigest() or d["ids"] != obj["ids"]:
            raise RecordError("寫入前的驗證：解碼後的完成紀錄和要寫的不同")
        store.append(line)
        if store.read() != data + line:
            raise RecordError("讀回的檔案內容和寫入的不同")
    except (ValueError, OSError, RecordError) as e:
        print(f"停止：完成紀錄寫入或讀回驗證失敗（{type(e).__name__}: {e}）。不刪除、不修改已寫的內容")
        return 5
    print(f"完成紀錄已寫入並讀回驗證：資料 {len(recs)} 行、SHA-256 {obj['data_sha256']}")
    return 0


def _done_problems(done, recs, prefix):
    if done is None:
        return ["沒有完成紀錄"]
    out = []
    if done["data_lines"] != len(recs):
        out.append(f"完成紀錄的 data_lines {done['data_lines']} 和資料行數 {len(recs)} 不同")
    if done["ids"] != [r["pair"]["id"] for r in recs]:
        out.append("完成紀錄的 ids 和資料行的 ID（依行的順序）不同")
    if done["data_sha256"] != hashlib.sha256(prefix).hexdigest():
        out.append("完成紀錄的 data_sha256 和前面資料行 bytes 的 SHA-256 不同")
    return out


def report_command(store, source, argv):
    """ohaze-report（4.11.17）：讀回結果檔，驗證完成紀錄、ID 集合、run、檔名與資料目錄後，用現有的 summarize、
    format_report 印出和 ohaze 同格式的報表。任何一項不符：只印原因與已保存的 ID，不印合計（exit 2）。"""
    try:
        data = store.read()
        recs, done, prefix = read_records(data)
    except (OSError, ValueError) as e:
        print(f"拒絕彙整（結果檔）：{e}")
        return 2
    try:
        lst = list_ohaze(source)
    except ValueError as e:
        print(f"拒絕彙整（資料目錄的清單）：{e}")
        return 2
    expected = EXPECTED_IDS
    probs = []
    probs += _done_problems(done, recs, prefix)
    if recs:
        probs += _id_problems([r["pair"]["id"] for r in recs], expected)
        probs += _run_problems(recs, done)
        run = recs[0]["run"]
        if run["data_dir"] != source.root:
            probs.append(f"紀錄的資料目錄 {run['data_dir']!r} 和本次的 {source.root!r} 不同")
        g = run["gate"]
        try:
            pm = resource_gate(g["rss_a"], g["rss_b"], g["rss_chunk_rows"], run["chunk_rows"], g["budget_gib"], g["margin"])
        except ValueError as e:
            pm = f"無法重算（{e}）"
        if pm != g["p_max"]:
            probs.append(f"紀錄的 P_max {g['p_max']} 和重算的 {pm} 不同")
    else:
        probs.append("結果檔沒有資料行")
    names = {p[0]: (p[1], p[2]) for p in lst["pairs"]}
    if tuple(sorted(names)) != tuple(expected) or lst["missing"] or lst["unparsed"]:
        probs.append("資料目錄的清單不是 01–45 的完整配對")
    off = [r["pair"]["id"] for r in recs
           if r["pair"]["status"] != "missing" and names.get(r["pair"]["id"]) != (r["pair"]["hazy"]["name"], r["pair"]["gt"]["name"])]
    if off:
        probs.append("紀錄的檔名不在資料目錄的清單裡：ID " + "、".join(f"{i:02d}" for i in off))
    if probs:
        saved = "、".join(f"{r['pair']['id']:02d}" for r in recs) or "無"
        print("資料已保存，但這個結果檔沒有通過整輪完成檢查，不印合計：" + "；".join(probs))
        print(f"已保存的 ID（{len(recs)} 行）：{saved}")
        return 2
    by_id = sorted(recs, key=lambda r: r["pair"]["id"])
    print(OHAZE_TITLE)
    for ln in header_lines(argv):
        print(ln)
    print(_source_line())
    print(f"資料目錄：{source.root}")
    print(_chunk_line(run["chunk_rows"]))
    print(_gate_line(run["gate"], run["gate"]["p_max"]))
    print(_list_line(lst))
    sel = [r["pair"]["id"] for r in by_id]
    _print_listing(sel, {r["pair"]["id"]: r["pair"] for r in by_id if r["pair"]["status"] != "missing"},
                   {r["pair"]["id"]: r["pair"]["missing_files"] for r in by_id if r["pair"]["status"] == "missing"})
    rows = [r["row"] for r in sorted(recs, key=lambda r: r["pair"]["id"])]
    s = summarize(rows)
    _print_tail(rows, s)
    return 0 if s["n_ok"] > 0 else 1


# ---------- selfcheck（4.11.10；預期值都由手算公式或獨立計算得到，不用被測 helper） ----------

def _raises(f, *args):
    try:
        f(*args)
    except ValueError:
        return True
    return False


def _ssim_oracle(x, y):
    """Q11 的獨立 oracle，直接照 Wang et al. 2004 的定義算，不呼叫 scikit-image 或 scipy：
    11×11 Gaussian 權重（σ = 1.5，正規化為 1），只取視窗完全在影像內的位置，
    加權平均、加權母體變異數與共變異數，最後取平均。回傳 (SSIM, 視窗數)。"""
    r = np.arange(-5, 6)
    g = np.exp(-(r * r) / (2 * 1.5 ** 2))
    w = np.outer(g, g)
    w /= w.sum()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    x, y = x.astype(np.float64), y.astype(np.float64)
    vals = []
    for i in range(x.shape[0] - 10):
        for j in range(x.shape[1] - 10):
            a, b = x[i:i + 11, j:j + 11], y[i:i + 11, j:j + 11]
            mx, my = (w * a).sum(), (w * b).sum()
            vx, vy = (w * (a - mx) ** 2).sum(), (w * (b - my) ** 2).sum()
            cxy = (w * (a - mx) * (b - my)).sum()
            vals.append((2 * mx * my + c1) * (2 * cxy + c2) / ((mx * mx + my * my + c1) * (vx + vy + c2)))
    return sum(vals) / len(vals), len(vals)


def _rgb(gray2d):
    """把同一張 2-D 影像複製到三個 channel（單 channel 的案例也經過正式的 RGB wrapper）。"""
    return np.repeat(gray2d[:, :, None], 3, axis=2)


def _near(a, b, tol):
    return isinstance(a, float) and isinstance(b, float) and abs(a - b) <= tol


def _psnr_ref(x, y):
    """Q19 的 PSNR 參考值：照 4.11.7 的公式用 float64 直接算，不呼叫被測的 mse、psnr。"""
    m = float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))
    return math.inf if m == 0 else 10 * math.log10(65025 / m)


def _ssim_rgb_ref(x, y):
    """Q19 的 SSIM 參考值：Q11 的 oracle 逐 channel 算，再取三個 channel 的算術平均（4.11.8）。"""
    return sum(_ssim_oracle(x[:, :, c], y[:, :, c])[0] for c in range(3)) / 3


def _test_parse(name):
    """只給 Q12、Q13 用的測試檔名（NN_hazy.png／NN_GT.png），不是 O-HAZE 的檔名規則。"""
    stem, dot, ext = name.partition(".")
    sid, _, role = stem.partition("_")
    if ext != "png" or not sid.isdigit() or role not in ("hazy", "GT"):
        return None
    return int(sid), role.lower()


class _MemSource:
    """selfcheck 的測試替身：記憶體中的資料目錄 {相對路徑: bytes}。不讀寫任何檔案。"""
    root = "（記憶體中的測試資料）"

    def __init__(self, files):
        self.files = dict(files)

    def names(self):
        return sorted(self.files)

    def open(self, name):
        return io.BytesIO(self.files[name])

    def read(self, name):
        return self.files[name]


class _MemRecord:
    """selfcheck 的結果檔替身：bytes 存在記憶體，不寫任何檔案。fail_append 模擬寫入失敗；corrupt_after 讓追加之後
    讀回的內容少一個 byte、多一個 X，模擬讀回不符。"""
    path = "（記憶體中的結果檔）"

    def __init__(self, data=None, fail_append=None, corrupt_after=False):
        self.data, self.fail_append, self.corrupt_after, self.appended = data, fail_append, corrupt_after, False

    def exists(self):
        return self.data is not None

    def read(self):
        if self.data is None:
            raise FileNotFoundError(2, "No such file or directory", self.path)
        return self.data[:-2] + b"X\n" if self.corrupt_after and self.appended else self.data

    def append(self, data):
        if self.fail_append is not None:
            raise self.fail_append
        self.data, self.appended = (self.data or b"") + data, True


def _png(arr_cv):
    """在記憶體編成 PNG（無損）。arr_cv 是 OpenCV 的順序（BGR、BGRA、灰階或 uint16）。"""
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(arr_cv))
    if not ok:
        raise ValueError("cv2.imencode 失敗")
    return buf.tobytes()


def _pil_bytes(img, fmt, **kw):
    """用 Pillow 在記憶體存成指定格式（MPO、帶 EXIF 的 JPEG）。"""
    b = io.BytesIO()
    img.save(b, fmt, **kw)
    return b.getvalue()


def _capture(argv):
    """在記憶體裡執行 main(argv)，回傳 (exit code, stdout 文字)。"""
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def selfcheck():
    import warnings
    from skimage.metrics import peak_signal_noise_ratio

    ck = golden._Checker()
    rng = np.random.default_rng(20261101)
    c1 = (0.01 * 255) ** 2
    q9 = c1 / (100 + c1)              # 常數圖 0 對 10：變異數、共變異數都是 0，SSIM = C1 / (0² + 10² + C1)

    print("PSNR（4.11.7）")
    x = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p = psnr(x, x.copy())
    ck.check("Q1 相同影像：PSNR = +inf", p == math.inf, f"{p}")
    ck.check("Q1 相同影像：沒有 warning", not w, f"{[str(m.message) for m in w]}")

    x2 = np.full((4, 4, 3), 10, np.uint8)
    y2 = np.zeros((4, 4, 3), np.uint8)
    y2[:2] = 30                       # 一半 pixel 是 30（差 −20）、一半是 0（差 10）：MSE = (400 + 100) / 2 = 250
    p = psnr(x2, y2)
    ck.check("Q2 常數誤差：MSE 250，PSNR = 10·log10(65025 / 250)", _near(p, 10 * math.log10(65025 / 250), 1e-9), f"{p}")

    x3 = np.zeros((4, 4, 3), np.uint8)
    y3 = x3.copy()
    y3[:, :, 0] = 5                   # 只有 R 差 5：MSE = 25 / 3
    p = psnr(x3, y3)
    ck.check("Q3 只有 R 差 5：MSE 25/3，PSNR = 10·log10(65025·3 / 25)",
             _near(p, 10 * math.log10(65025 * 3 / 25), 1e-9), f"{p}")

    e25, e100 = 10 * math.log10(65025 / 25), 10 * math.log10(65025 / 100)
    a4, b4 = np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 5, np.uint8)       # MSE 25
    a8, b8 = np.zeros((8, 8, 3), np.uint8), np.full((8, 8, 3), 10, np.uint8)      # MSE 100
    p4, p8 = psnr(a4, b4), psnr(a8, b8)
    ck.check("Q4 逐張 PSNR：4×4 MSE 25、8×8 MSE 100", _near(p4, e25, 1e-9) and _near(p8, e100, 1e-9), f"{p4}, {p8}")
    rows = [{"h": 4, "w": 4, "FL": {"mse": 25.0, "psnr": e25}}, {"h": 8, "w": 8, "FL": {"mse": 100.0, "psnr": e100}}]
    m = mean_psnr(rows, "FL")
    ck.check("Q4 資料集平均：逐張 PSNR 的算術平均 (e25 + e100) / 2", _near(m, (e25 + e100) / 2, 1e-9), f"{m}")

    rows = [{"h": 4, "w": 4, "FL": {"mse": 0.0, "psnr": math.inf}}, {"h": 4, "w": 4, "FL": {"mse": 25.0, "psnr": e25}}]
    m = mean_psnr(rows, "FL")
    fm, k = finite_mean([math.inf, e25])
    ck.check("Q5 有一對是 +inf：該欄平均是 +inf", m == math.inf, f"{m}")
    ck.check("Q5 有限值的平均 = e25，n = 1", _near(fm, e25, 1e-9) and k == 1, f"{fm}, {k}")

    ok = np.zeros((4, 4, 3), np.uint8)
    ck.check("Q6 dtype 不是 uint8：PSNR 丟 ValueError", _raises(psnr, ok.astype(np.float64), ok))
    ck.check("Q6 dtype 不是 uint8：SSIM 丟 ValueError", _raises(ssim, ok.astype(np.float64), ok))
    ck.check("Q6 形狀不同：丟 ValueError", _raises(psnr, ok, np.zeros((4, 5, 3), np.uint8)))
    ck.check("Q6 只有 2 維：丟 ValueError", _raises(psnr, ok[:, :, 0], ok[:, :, 0]))
    ck.check("Q6 最後一維不是 3：丟 ValueError", _raises(psnr, np.zeros((4, 4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8)))
    ck.check("Q6 H 是 0：丟 ValueError", _raises(psnr, np.zeros((0, 4, 3), np.uint8), np.zeros((0, 4, 3), np.uint8)))
    ck.check("Q6 W 是 0：丟 ValueError", _raises(psnr, np.zeros((4, 0, 3), np.uint8), np.zeros((4, 0, 3), np.uint8)))

    p_lib = float(peak_signal_noise_ratio(x2, y2, data_range=255))
    ck.check("Q7 和 scikit-image peak_signal_noise_ratio(data_range=255) 交叉核對", _near(psnr(x2, y2), p_lib, 1e-9),
             f"{psnr(x2, y2)} vs {p_lib}")

    z1, w1 = np.zeros((1, 1, 3), np.uint8), np.full((1, 1, 3), 255, np.uint8)
    m = mse(z1, w1)                   # MSE 錯時不再呼叫 psnr：負的 MSE 會讓 log10 丟例外，而例外不算抓到
    ck.check("Q18 平方不溢位：1×1 RGB 0 對 255 → MSE = 255² = 65025、PSNR = 0", m == 65025 and psnr(z1, w1) == 0.0,
             f"MSE {m}")

    print("SSIM（4.11.8，都經過正式的 RGB wrapper）")
    s = ssim(x, x.copy())
    ck.check("Q8 相同的非常數圖：SSIM = 1", _near(s, 1.0, 1e-12), f"{s}")

    zero, ten = np.zeros((16, 16), np.uint8), np.full((16, 16), 10, np.uint8)
    s = ssim(_rgb(zero), _rgb(ten))
    ck.check("Q9 常數圖 0 對 10：SSIM = C1 / (100 + C1)", _near(s, q9, 1e-9), f"{s} vs {q9}")

    gb = rng.integers(0, 256, (16, 16, 2), dtype=np.uint8)
    x10 = np.concatenate([zero[:, :, None], gb], axis=2)
    y10 = np.concatenate([ten[:, :, None], gb], axis=2)
    s = ssim(x10, y10)
    ck.check("Q10 只有 R 不同（0 對 10），G、B 相同：SSIM = (Q9 + 1 + 1) / 3", _near(s, (q9 + 2) / 3, 1e-9),
             f"{s} vs {(q9 + 2) / 3}")

    fr = np.random.default_rng(20261031)    # 4.11.10 的固定 fixture，順序不能換
    fx11 = fr.integers(0, 256, (16, 16), dtype=np.uint8)
    noise = fr.integers(-40, 41, (16, 16))
    fy11 = np.clip(fx11.astype(np.int64) + noise, 0, 255).astype(np.uint8)
    oracle, nwin = _ssim_oracle(fx11, fy11)
    s = ssim(_rgb(fx11), _rgb(fy11))
    ck.check("Q11 固定 fixture：和獨立 oracle（Wang 定義，36 個視窗）相同", nwin == 36 and _near(s, oracle, 1e-9),
             f"{s} vs {oracle}，視窗 {nwin}")
    print(f"  紀錄：Q11 oracle {oracle:.12f}，scikit-image {s:.12f}，差 {s - oracle:.3e}")

    print("配對、讀檔、公平性（4.11.6、4.11.3）")
    names = ["01_hazy.png", "01_GT.png", "02_hazy.png", "02_GT.png", "03_hazy.png",
             "04_hazy.png", "04_GT.png", "05_hazy.png", "05_GT.png", ".DS_Store"]
    res = pair_files(names, _test_parse)
    want = [(i, f"{i:02d}_hazy.png", f"{i:02d}_GT.png") for i in (1, 2, 4, 5)]
    ck.check("Q12 缺 GT 03：04 配 04、05 配 05，03 標 missing，.DS_Store 忽略",
             res["pairs"] == want and res["missing"] == [3] and res["ignored"] == [".DS_Store"], f"{res}")
    ck.check("Q13 同一編號有兩個 GT：停止並報錯", _raises(pair_files, names + ["4_GT.png"], _test_parse))

    rgb = np.array([[[200, 10, 60], [0, 128, 255]], [[255, 0, 1], [17, 99, 3]]], np.uint8)   # 每個 pixel 的 R ≠ B
    okc, buf = cv2.imencode(".png", np.ascontiguousarray(rgb[:, :, ::-1]))                    # cv2.imencode 吃 BGR
    got = decode_rgb(buf.tobytes()) if okc else None
    ck.check("Q14 RGB → BGR → PNG 位元組 → decode_rgb：解回原本的 RGB", got is not None and np.array_equal(got, rgb),
             f"{None if got is None else got.tolist()}")

    img = rng.integers(0, 256, (21, 22, 3), dtype=np.uint8)
    fl, fx = run_mode(img, fl_active()), run_mode(img, fx_active())
    sa, sb = fl["stats"], fx["stats"]
    same = sa.keys() == sb.keys() and all(
        np.array_equal(sa[k], sb[k]) if isinstance(sa[k], np.ndarray) else sa[k] == sb[k] for k in sa)
    ck.check("Q15 小圖 FL、FX：stats 全部相等（frame_down、dark 用 np.array_equal）", same)
    keys = ("t_h", "a_far", "a_near", "i_max", "tie", "th_round", "measured")
    ck.check("Q15 小圖 FL、FX：used 的 t_h、a_far、a_near、i_max、tie、th_round、measured 相等",
             all(fl["used"][k] == fx["used"][k] for k in keys))
    ck.check("Q15 FX 確實是定點候選（F20、PALA、T10、SD lut）",
             (fx["used"]["inv_frac"], fx["used"]["pala_fixed"], fx["used"]["pte_frac"], fx["used"]["sd_mode"])
             == (20, True, 10, "lut"))
    ck.check("Q15 公平性檢查：沒有不相等的欄位", fairness(fl, fx) == [], f"{fairness(fl, fx)}")
    def fx127():
        return golden.make_active(**{**golden.ACTIVE0_DEFAULT, "t_h": 127}, inv_frac=20, pala_fixed=True, pte_frac=10,
                                  sd_mode="lut")
    bad = run_mode(img, fx127())
    ck.check("Q15 公平性檢查：抓得到 active0 的 T_h 不同", "stats.t_h_prev" in fairness(fl, bad), f"{fairness(fl, bad)}")
    gt = rng.integers(0, 256, (21, 22, 3), dtype=np.uint8)
    row = run_pair(7, img, gt)
    ck.check("Q15 run_pair：小圖整對是 ok", row["status"] == "ok", row.get("msg", ""))
    rok = row["status"] == "ok"       # 不是 ok 時 Q19 只記 FAIL，不讀 row 的 metric（避免例外）
    if rok:
        f = row["fxfl"]
        print(f"  紀錄：小圖 FX 對 FL 不同的 channel {f['diff']}／{f['n']}，最大差 {f['max']}，"
              f"PSNR {fmt(f['psnr'], 4)}；FL 對 GT {fmt(row['FL']['psnr'], 4)} dB")

    # Q19：runner 的接線。預期值來自契約 4.11.3 指定的直接 golden 呼叫，再用獨立公式與 oracle 算，不用被測的 quality helper
    ck.check("Q19 FL 確實是浮點（inv_frac、pala_fixed、pte_frac、sd_mode 依序是 None、False、None、None）",
             (fl["used"]["inv_frac"], fl["used"]["pala_fixed"], fl["used"]["pte_frac"], fl["used"]["sd_mode"])
             == (None, False, None, None))
    direct = {"FL": golden.run_static(img, golden.make_active(**golden.ACTIVE0_DEFAULT))["result"]["out"],
              "FX": golden.run_static(img, golden.make_active(**golden.ACTIVE0_DEFAULT, inv_frac=20, pala_fixed=True,
                                                               pte_frac=10, sd_mode="lut"))["result"]["out"]}
    dd = np.abs(direct["FX"].astype(np.int64) - direct["FL"].astype(np.int64))
    ck.check("Q19 run_pair 的 FL、FX 兩列：PSNR、SSIM 等於直接呼叫 golden 的輸出對 GT 的值（fixture 的 FL、FX 輸出不同）",
             rok and dd.any() and all(_near(row[c]["psnr"], _psnr_ref(direct[c], gt), 1e-9)
                                      and _near(row[c]["ssim"], _ssim_rgb_ref(direct[c], gt), 1e-9) for c in ("FL", "FX")))
    want = {c: hashlib.sha256(bytes(direct[c].reshape(-1).tolist())).hexdigest() for c in ("FL", "FX")}
    ck.check("Q19 run_pair 的 FL、FX 輸出 SHA-256 等於直接呼叫 golden 的輸出", rok and row["sha"] == want)
    ck.check("Q19 run_pair 的 FX 對 FL：不同的 channel 數、MAE、最大差、PSNR 等於直接呼叫 golden 的輸出",
             rok and (row["fxfl"]["diff"], row["fxfl"]["max"]) == (int(np.count_nonzero(dd)), int(dd.max()))
             and _near(row["fxfl"]["mae"], float(dd.sum()) / dd.size, 1e-12)
             and _near(row["fxfl"]["psnr"], _psnr_ref(direct["FX"], direct["FL"]), 1e-9))

    # Q20：公平性負例。各改一個欄位，預期只報那一個；預期不同的量化欄位（inv_i_max 等）不要求相等
    dark = fx["stats"]["dark"].copy()
    dark.flat[0] ^= 1
    got = fairness(fl, {**fx, "stats": {**fx["stats"], "dark": dark}})
    ck.check("Q20 公平性負例：stats.dark 改一個值（ndarray 欄位），只報 stats.dark", got == ["stats.dark"], f"{got}")
    got = fairness(fl, {**fx, "used": {**fx["used"], "t_h": fx["used"]["t_h"] + 1}})
    ck.check("Q20 公平性負例：used.t_h 改成不同，只報 used.t_h", got == ["used.t_h"], f"{got}")
    g = globals()                     # 只在這個檢查裡讓 run_pair 的 FX 用 T_h 127，結束一定還原
    keep, g["fx_active"] = g["fx_active"], fx127
    try:
        row_bad = run_pair(8, img, gt)
    finally:
        g["fx_active"] = keep
    ck.check("Q20 run_pair 遇公平性失敗（FX 的 active0 T_h 127）：整對 error，只有 id、status、msg",
             row_bad["status"] == "error" and "stats.t_h_prev" in row_bad.get("msg", "")
             and set(row_bad) == {"id", "status", "msg"}, f"{row_bad['status']}：{row_bad.get('msg', '')}")

    print("主路徑分塊（4.11.6；Q15 的 21×22 小圖，預期值是不分塊的 run_static）")
    for c, act, whole in (("FL", fl_active, fl), ("FX", fx_active, fx)):
        same = all(np.array_equal(p["out"], whole["out"]) and fairness(whole, p) == [] and p["used"] == whole["used"]
                   for p in (run_mode(img, act(), r) for r in (1, 4, 7)))
        ck.check(f"Q16 分塊：{c} 每 1、4、7 列都和整張逐 byte 相同（stats、used 也相同）", same)
    calls = []
    real_apply, real_stats = golden.apply_pass, golden.stats_pass

    def spy_apply(frame, active):
        calls.append(("apply", frame.shape[0]))
        return real_apply(frame, active)

    def spy_stats(frame, t_h_prev, tie="first"):
        calls.append(("stats", frame.shape[0]))
        return real_stats(frame, t_h_prev, tie)
    golden.apply_pass, golden.stats_pass = spy_apply, spy_stats      # 只在這裡替換，結束一定還原
    try:
        row16 = run_pair(9, img, gt, 4)
    finally:
        golden.apply_pass, golden.stats_pass = real_apply, real_stats
    want = ([("stats", 21)] + [("apply", 4)] * 5 + [("apply", 1)]) * 2
    ck.check("Q16 run_pair(chunk_rows=4)：FL、FX 各做一次整張的 stats_pass（21 列），主路徑每次最多 4 列（4×5 + 1）",
             calls == want, f"{calls}")
    ck.check("Q16 run_pair(chunk_rows=4)：FL、FX 的輸出 SHA-256 和不分塊的 run_pair 相同",
             rok and row16["status"] == "ok" and row16["sha"] == row["sha"], row16.get("msg", ""))
    ck.check("Q16 分塊列數是負數：丟 ValueError", _raises(run_mode, img, fl_active(), -1))

    print("NA 規則（4.11.7）：helper 層直接檢查回傳值，報表層檢查文字")
    inf = math.inf
    for label, r, code in (("Δ(+inf, +inf)", delta(inf, inf), "nonfinite"),
                           ("Δ(+inf, 30.0)", delta(inf, 30.0), "nonfinite"),
                           ("Δ(30.0, +inf)", delta(30.0, inf), "nonfinite"),
                           ("D(空集合)", d_value([]), "empty"),
                           ("D([1.0, NA])", d_value([1.0, NA("nonfinite")]), "contains_na"),
                           ("有限值最小值([+inf, +inf])", min_finite([inf, inf]), "no_finite")):
        ck.check(f"Q17 helper 層：{label} 回傳 NA 物件，reason {code}", isinstance(r, NA) and r.reason == code, f"{r!r}")
    ck.check("Q17 helper 層：Δ(31.5, 30.0) = 1.5（有限值照常相減）", delta(31.5, 30.0) == 1.5, f"{delta(31.5, 30.0)!r}")
    ck.check("Q17 helper 層：D([1.0, −3.0]) = |mean Δ| = 1.0（不是 mean |Δ| = 2.0）", d_value([1.0, -3.0]) == 1.0,
             f"{d_value([1.0, -3.0])!r}")

    text_a = format_report([], summarize([]))
    line = {ln.split()[0]: ln for ln in text_a.splitlines() if ln.startswith("  ")}
    ck.check("Q17 報表層 (a) N_ok = 0：FL 平均與 D 都印 NA（N_ok = 0）",
             "NA（N_ok = 0）" in line.get("FL", "") and "NA（N_ok = 0）" in line.get("D", ""))

    g1 = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    g2, h2, f2 = (rng.integers(0, 256, (16, 16, 3), dtype=np.uint8) for _ in range(3))
    x2b = f2.copy()
    x2b[0, 0, 0] ^= 1
    rows_b = [pair_row(1, g1, {"H": rng.integers(0, 256, (16, 16, 3), dtype=np.uint8), "FL": g1.copy(), "FX": g1.copy()}),
              pair_row(2, g2, {"H": h2, "FL": f2, "FX": x2b})]
    text_b = format_report(rows_b, summarize(rows_b))
    row1 = next((ln for ln in text_b.splitlines() if ln.startswith("  1 |")), "")
    dline = next((ln for ln in text_b.splitlines() if ln.startswith("  D =")), "")
    ck.check("Q17 報表層 (b) 一對的 FL = FX = GT：該對 ΔPSNR 印 NA（非有限值）、D 印 NA（S 含 NA 的 Δ）",
             "Δ NA（非有限值）" in row1 and "NA（S 含 NA 的 Δ）" in dline, f"{row1} ／ {dline}")

    rows_c = [pair_row(i, g, {"H": g, "FL": o, "FX": o.copy()})
              for i, (g, o) in enumerate(((g1, f2), (g2, h2)), 1)]
    text_c = format_report(rows_c, summarize(rows_c))
    ck.check("Q17 報表層 (c) 所有對的 FX = FL：FX 對 FL 的 PSNR 最小值印 NA（沒有有限值）",
             "有限值的最小值 NA（沒有有限值）" in text_c)
    ck.check("Q17 報表層：(a)–(c) 三種報表都不含 nan（不分大小寫）",
             all("nan" not in t.lower() for t in (text_a, text_b, text_c)))

    print("Δ、合計與報表（4.11.3、4.11.9；預期值都是手算）")
    g0, f10, f20, h30 = (np.full((16, 16, 3), v, np.uint8) for v in (0, 10, 20, 30))
    r = pair_row(1, g0, {"H": h30, "FL": f10, "FX": f20})
    want = 10 * math.log10(65025 / 400) - 10 * math.log10(65025 / 100)     # FX 的 MSE 400、FL 的 MSE 100
    ck.check("Q21 每對 ΔPSNR：16×16 常數圖 GT 0、FL 10、FX 20 → 10·log10(65025/400) − 10·log10(65025/100)",
             r["status"] == "ok" and _near(r["d_psnr"], want, 1e-9), f"{r.get('d_psnr')} vs {want}")
    want = c1 / (400 + c1) - c1 / (100 + c1)                                 # 常數圖的 SSIM，同 Q9
    ck.check("Q21 每對 ΔSSIM：同一組 → C1/(400 + C1) − C1/(100 + C1)",
             r["status"] == "ok" and _near(r["d_ssim"], want, 1e-9), f"{r.get('d_ssim')} vs {want}")

    # 手算 rows（只當 summarize 的輸入）：S 上的 ΔPSNR = [−0.5, −2, 1, 4]，ΔSSIM = [0, −0.25, 0.125, 0]。
    # ID 15 是 error 但帶完整的數字：summarize 若納入它，平均與計數都會變。mse、h、w 只讓 QM4 的 pooled 改錯能執行。
    spec = [(3, "ok", (10.0, 20.0, 19.5), (0.25, 0.5, 0.5)),            # ID、狀態、H／FL／FX 的 PSNR、SSIM
            (8, "ok", (11.0, 22.0, 20.0), (0.25, 0.75, 0.5)),
            (15, "error", (5.0, 30.0, 20.0), (0.125, 0.875, 0.125)),
            (12, "ok", (12.0, 18.0, 19.0), (0.375, 0.625, 0.75)),
            (20, "ok", (13.0, 16.0, 20.0), (0.375, 0.875, 0.875))]
    rows_d = [{"id": i, "status": st, "h": 16, "w": 16,
               **{c: {"psnr": p, "ssim": v, "mse": 65025 / 10 ** (p / 10)} for c, p, v in zip(COLS, ps, ss)},
               "d_psnr": ps[2] - ps[1], "d_ssim": ss[2] - ss[1], "fxfl": {"psnr": 40.0}} for i, st, ps, ss in spec]
    s = summarize(rows_d)
    got = (s["n_ok"], s["n_total"], s["status"])
    ck.check("Q21 合計 S：N_ok 4／N_total 5；ok 是 ID 3、8、12、20，error 是 ID 15",
             got == (4, 5, {"ok": [3, 8, 12, 20], "error": [15]}), f"{got}")
    got = {c: (s[c]["psnr"], s[c]["ssim"]) for c in COLS}
    ck.check("Q21 合計平均：H、FL、FX 的 PSNR、SSIM 是 S 上的手算平均（例如 H 的 PSNR = (10 + 11 + 12 + 13) / 4）",
             got == {"H": (11.5, 0.3125), "FL": (19.0, 0.6875), "FX": (19.625, 0.65625)}, f"{got}")
    got = (s["mean_d_psnr"], s["D"], s["mean_d_ssim"])
    ck.check("Q21 合計 Δ：mean ΔPSNR = D = 0.625、mean ΔSSIM = −0.03125", got == (0.625, 0.625, -0.03125), f"{got}")
    e = s["extremes"]
    ok_e = isinstance(e, dict)
    ck.check("Q21 合計極值：ΔPSNR 最小 −2（ID 8）、|ΔPSNR| 最大 4、最小的三對依序是 ID 8、3、12",
             ok_e and (e["min"], e["absmax"], e["worst3"]) == ((-2.0, 8), 4.0, [(-2.0, 8), (-0.5, 3), (1.0, 12)]), f"{e}")
    ck.check("Q21 合計計數：ΔPSNR < −0.5 的對數是 1（剛好 −0.5 的 ID 3 不算）", ok_e and e["n_worse"] == 1, f"{e}")

    # Q22：H、FL、FX 都不同；FX 只在同一個 pixel 的 R、B 和 FL 不同（+3、−1）
    g5, h5 = (rng.integers(0, 256, (16, 16, 3), dtype=np.uint8) for _ in range(2))
    f5 = rng.integers(10, 246, (16, 16, 3), dtype=np.uint8)
    x5 = f5.copy()
    x5[5, 7, 0] += 3
    x5[5, 7, 2] -= 1
    r5 = pair_row(5, g5, {"H": h5, "FL": f5, "FX": x5})
    ok5 = r5["status"] == "ok"
    want = {c: hashlib.sha256(bytes(a.reshape(-1).tolist())).hexdigest() for c, a in (("FL", f5), ("FX", x5))}
    ck.check("Q22 輸出 SHA-256：FL、FX 各是 raw RGB（C order）位元組的完整 SHA-256", ok5 and r5["sha"] == want,
             f"{r5.get('sha')}")
    f = r5["fxfl"] if ok5 else {}
    ck.check("Q22 FX 對 FL：同一 pixel 有兩個 channel 不同 → 不同的 channel 2／768、MAE 4/768、最大差 3",
             ok5 and (f["diff"], f["n"], f["max"], f["mae"]) == (2, 768, 3, 4 / 768), f"{f}")
    line = next((ln for ln in format_report([r5], summarize([r5])).splitlines() if ln.startswith("  5 |")), "")
    ck.check("Q22 報表列：channel 比例 0.2604%（2/768，不是 pixel 比例 1/768 = 0.1302%）、MAE 0.005208、最大 3、"
             "SHA-256 前 16 碼", "| 0.2604% MAE 0.005208 最大 3 PSNR" in line
             and f"| FL {want['FL'][:16]} FX {want['FX'][:16]} |" in line, line)

    print("O-HAZE 的檔名、清單與配對（4.11.6 的實際規則；只用記憶體中的測試資料）")
    j16 = _png(rng.integers(0, 256, (16, 24, 3), dtype=np.uint8))
    files = {"hazy/01_outdoor_hazy.jpg": j16, "GT/01_outdoor_GT.jpg": j16,
             "hazy/02_outdoor_hazy.JPG": j16, "GT/02_outdoor_GT.JPG": j16, "hazy/03_outdoor_hazy.jpg": j16,
             "GT/.DS_Store": b"\x00\x00\x00\x01Bud1", "hazy/notes.txt": b"not an image", "hazy/IMG_0004.jpg": j16}
    lst = list_ohaze(_MemSource(files))
    ck.check("Q23 檔名規則：.jpg 與 .JPG 都接受，依編號配對 01、02；03 只有 hazy，標 missing",
             lst["pairs"] == [(1, "hazy/01_outdoor_hazy.jpg", "GT/01_outdoor_GT.jpg"),
                              (2, "hazy/02_outdoor_hazy.JPG", "GT/02_outdoor_GT.JPG")] and lst["missing"] == [3], f"{lst}")
    ck.check("Q23 非影像檔（.DS_Store、notes.txt）只計數，不當成場景",
             lst["other"] == ["GT/.DS_Store", "hazy/notes.txt"], f"{lst['other']}")
    ck.check("Q23 檔名無法解析的影像（hazy/IMG_0004.jpg）列出，不當成非影像忽略", lst["unparsed"] == ["hazy/IMG_0004.jpg"],
             f"{lst['unparsed']}")
    ck.check("Q23 檔名符合、但目錄不對或角色不一致（GT/ 下的 hazy、hazy/ 下的 GT、其他目錄、根目錄、小寫 gt/、子目錄）：丟 ValueError",
             all(_raises(parse_ohaze, n) for n in ("GT/05_outdoor_hazy.jpg", "hazy/05_outdoor_GT.jpg", "extra/05_outdoor_GT.jpg",
                                                   "05_outdoor_GT.jpg", "gt/05_outdoor_GT.jpg", "hazy/x/05_outdoor_hazy.jpg")))
    ck.check("Q23 不符合規則的檔名（Hazy 大寫、.jpeg、一位數編號）：回傳 None，交給清單分類",
             all(parse_ohaze(n) is None for n in ("hazy/01_outdoor_Hazy.jpg", "GT/01_outdoor_GT.jpeg", "GT/1_outdoor_GT.jpg")))
    ck.check("Q23 duplicate：編號 02 的 GT 同時有 .jpg 與 .JPG → 丟 ValueError，不自行挑選",
             _raises(list_ohaze, _MemSource({**files, "GT/02_outdoor_GT.jpg": j16})))

    print("讀檔與格式 gate（4.11.6「讀檔」、MPO 做法 A；fixture 都在記憶體編碼）")
    base = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
    png = _png(base[:, :, ::-1])
    got, inf_ = inspect_image(png)
    ck.check("Q24 一般 RGB（PNG，無損）：通過 gate，解回原本的陣列", got is not None and inf_["bad"] == []
             and np.array_equal(got, base), f"{inf_['bad']}")
    ck.check("Q24 兩種 hash 分開：原始檔 SHA-256 是檔案位元組，decoded RGB SHA-256 是解回陣列的 C order 位元組",
             inf_["raw_sha"] == hashlib.sha256(png).hexdigest()
             and inf_.get("rgb_sha") == hashlib.sha256(bytes(base.reshape(-1).tolist())).hexdigest()
             and inf_["raw_sha"] != inf_.get("rgb_sha"))
    prim = Image.new("RGB", (24, 16), (200, 30, 30))
    mpo = _pil_bytes(prim, "MPO", save_all=True, append_images=[Image.new("RGB", (12, 8), (30, 30, 200))])
    got, inf_ = inspect_image(mpo)
    ck.check("Q24 MPO（主圖 16×24 紅色、縮圖 8×12 藍色）：通過 gate，metadata 取主圖，frame 數 2",
             got is not None and inf_["bad"] == [] and (inf_["format"], inf_["frames"], inf_["h"], inf_["w"]) == ("MPO", 2, 16, 24),
             f"{inf_}")
    ck.check("Q24 MPO 只用主圖：解回 16×24 的紅色（R 平均 > 150、B 平均 < 80），不是縮圖",
             got is not None and got.shape == (16, 24, 3) and got[:, :, 0].mean() > 150 and got[:, :, 2].mean() < 80)
    for o, label in ((1, "EXIF orientation = 1：通過"), (6, "EXIF orientation = 6：不通過，原因含 orientation（不自動旋轉）")):
        ex = Image.Exif()
        ex[0x0112] = o
        got, inf_ = inspect_image(_pil_bytes(prim, "JPEG", exif=ex.tobytes()))
        ok24 = (got is not None and inf_["bad"] == []) if o == 1 else (got is None and "orientation" in inf_["bad"])
        ck.check(f"Q24 {label}", ok24, f"{inf_['bad']}")
    asym = Image.new("RGB", (24, 16), (20, 20, 220))
    asym.paste((220, 20, 20), (0, 0, 12, 16))       # 左半紅、右半藍：轉 180° 之後尺寸不變，只有內容左右顛倒
    ex = Image.Exif()
    ex[0x0112] = 3
    o3 = _pil_bytes(asym, "JPEG", exif=ex.tobytes())
    d3 = decode_rgb(o3)
    ck.check("Q24 orientation = 3 的 fixture（左紅右藍）：OpenCV 解碼後仍是 16×24，但內容轉了 180°（左半變藍），只看尺寸抓不到",
             d3.shape == (16, 24, 3) and d3[:, :12, 2].mean() > 150 and d3[:, :12, 0].mean() < 80)
    got, inf_ = inspect_image(o3)
    ck.check("Q24 EXIF orientation = 3：不通過，原因只有 orientation（尺寸相同，不是靠 decoded_size 擋下）",
             got is None and inf_["bad"] == ["orientation"], f"{inf_['bad']}")
    got, inf_ = inspect_image(_png(rng.integers(0, 256, (16, 24), dtype=np.uint8)))
    ck.check("Q24 灰階（1 channel）：不通過，原因含 channels（不默默轉成 3 channel）", got is None and "channels" in inf_["bad"],
             f"{inf_['bad']}")
    got, inf_ = inspect_image(_png(rng.integers(0, 256, (16, 24, 4), dtype=np.uint8)))
    ck.check("Q24 RGBA（4 channel）：不通過，原因含 channels", got is None and "channels" in inf_["bad"], f"{inf_['bad']}")
    got, inf_ = inspect_image(_png(rng.integers(0, 65536, (16, 24, 3)).astype(np.uint16)))
    ck.check("Q24 16-bit：不通過，原因含 dtype（不默默轉成 8-bit）", got is None and "dtype" in inf_["bad"], f"{inf_['bad']}")
    got, inf_ = inspect_image(png, decode=lambda raw: np.zeros((8, 12, 3), np.uint8))
    ck.check("Q24 decoded 尺寸和主圖 metadata 不同（測試替身解出 8×12）：不通過，原因只有 decoded_size",
             got is None and inf_["bad"] == ["decoded_size"], f"{inf_['bad']}")
    got, inf_ = inspect_image(b"not an image")
    ck.check("Q24 不是影像的位元組：不通過，原因是 unreadable", got is None and inf_["bad"] == ["unreadable"], f"{inf_['bad']}")

    print("一對的讀取（hazy、GT 同一條路徑；4.11.6 的狀態）")
    src = _MemSource({"h": png, "g_gray": _png(rng.integers(0, 256, (16, 24), dtype=np.uint8)),
                      "g_20": _png(base[:, :20, ::-1]), "g": png})
    _, _, inf_ = load_pair(src, 4, "h", "g_gray")
    ck.check("Q25 GT 也經過同一個格式 gate：GT 是灰階 → 整對 format", inf_["status"] == "format", inf_["status"])
    hz, g_, inf_ = load_pair(src, 5, "h", "g_20")
    ck.check("Q25 hazy 16×24、GT 16×20：整對 size_mismatch，兩張影像都不回傳（不 resize、不 crop）",
             inf_["status"] == "size_mismatch" and hz is None and g_ is None, inf_["status"])
    hz, g_, inf_ = load_pair(src, 6, "h", "g")
    ck.check("Q25 兩邊都通過、尺寸相同：ok，H×W 16×24", inf_["status"] == "ok" and (inf_["h"], inf_["w"]) == (16, 24)
             and hz is not None and hz.shape == (16, 24, 3), inf_["status"])

    print("資源 gate（4.11.6；下面都是測試用的假數字，不是校準結果）")
    ck.check("Q26 缺少校準值、預算或餘裕：丟 ValueError（本檔沒有預設預算）",
             _raises(resource_gate, None, None, None, 0, None, None))
    ck.check("Q26 分塊不能繞過：chunk_rows = 4、缺少資源條件，仍丟 ValueError",
             _raises(resource_gate, None, None, None, 4, None, None))
    ck.check("Q26 校準的分塊設定（0 列）和本次（4 列）不同：丟 ValueError", _raises(resource_gate, 1.0, 100.0, 0, 4, 12.0, 1.5))
    pm = resource_gate(float(GIB), 1000.0, 4, 4, 4.0, 2.0)
    ck.check("Q26 P_max = floor((4 GiB / 2 − 1 GiB) / 1000) = floor(2^30 / 1000) = 1073741", pm == 1073741, f"{pm}")
    ck.check("Q26 預算扣掉 a 之後不夠（P_max ≤ 0）：丟 ValueError", _raises(resource_gate, float(8 * GIB), 100.0, 0, 0, 4.0, 1.0))

    print("CLI 接線（測試替身：記憶體資料目錄；ohaze 只跑 16×16、16×20 的小圖；calib 的 run_pair 是不執行 golden 的替身）")
    g = globals()

    def cli(argv, src=None, pair_fn=None, rec=None):
        """在記憶體裡執行 main(argv)，回傳 (exit code, stdout, 例外)。src 是 {路徑: bytes} 或資料目錄替身；
        rec 是結果檔替身（_MemRecord），不寫任何檔案。
        呼叫丟出的例外不讓 selfcheck 中止，但依賴這次呼叫的檢查都經過 check_cli，FAIL 標籤會帶 EXC_MARK，
        mutation 不把它算成正常 FAIL。TimeoutError（mutation 的逾時）照常往外丟。"""
        keep = (g["DirSource"], g["run_pair"], g["RecordFile"])
        if src is not None:
            s = _MemSource(src) if isinstance(src, dict) else src
            g["DirSource"] = lambda root: s
        if pair_fn is not None:
            g["run_pair"] = pair_fn
        if rec is not None:
            g["RecordFile"] = lambda path: rec
        try:
            rc, out = _capture(argv)
            return rc, out, None
        except TimeoutError:
            raise
        except Exception as e:
            return None, "", f"{type(e).__name__}: {e}"
        finally:
            g["DirSource"], g["run_pair"], g["RecordFile"] = keep

    def check_cli(label, res, ok, detail="", pre=()):
        """res 是這個檢查直接依賴的 cli() 結果（沒有時給 None）；pre 是它另外依賴的前置步驟：cli() 結果，或 recs_of、
        attempt、rejects 回傳的例外字串（None 表示沒有例外）。任何一個丟過例外，或 ok（可呼叫時才延後計算）本身丟例外：
        FAIL 並帶 EXC_MARK，mutation 不算正常 FAIL。TimeoutError 照常往外丟。"""
        deps = ((res,) if res is not None else ()) + tuple(pre)
        err = next((e for e in (p[2] if isinstance(p, tuple) else p for p in deps) if e is not None), None)
        if err is None and callable(ok):
            try:
                ok = ok()
            except TimeoutError:
                raise
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        if err is not None:
            ck.check(f"{label} {EXC_MARK}", False, err)
        else:
            ck.check(label, ok, detail)

    def raise_timeout(*a, **k):
        raise TimeoutError("測試替身的逾時")

    try:
        cli(["calib", "16", "12"], None, raise_timeout)
        to_ok = False
    except TimeoutError:
        to_ok = True
    ck.check("Q27 CLI 測試 helper 不吞 TimeoutError（mutation 的逾時照常中止）", to_ok)

    a1 = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    pa1 = _png(a1[:, :, ::-1])
    mp = _pil_bytes(Image.new("RGB", (16, 16), (200, 30, 30)), "MPO", save_all=True,
                    append_images=[Image.new("RGB", (8, 8), (30, 30, 200))])
    pfiles = {"hazy/01_outdoor_hazy.jpg": pa1, "GT/01_outdoor_GT.jpg": _png(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)),
              "hazy/02_outdoor_hazy.JPG": mp, "GT/02_outdoor_GT.JPG": mp, "hazy/03_outdoor_hazy.jpg": pa1,
              "hazy/04_outdoor_hazy.jpg": pa1, "GT/04_outdoor_GT.jpg": _png(rng.integers(0, 256, (16, 16), dtype=np.uint8)),
              "GT/.DS_Store": b"\x00\x00\x00\x01Bud1"}
    gcalls = []
    real_static = golden.run_static
    golden.run_static = lambda *a: gcalls.append("run_static") or real_static(*a)
    golden.apply_pass = lambda *a: gcalls.append("apply_pass") or real_apply(*a)
    golden.stats_pass = lambda *a: gcalls.append("stats_pass") or real_stats(*a)
    try:
        res = cli(["pairs", "X"], pfiles)
    finally:
        golden.run_static, golden.apply_pass, golden.stats_pass = real_static, real_apply, real_stats
    rc, out, _ = res
    check_cli("Q27 pairs：不呼叫 golden（run_static、stats_pass、apply_pass 都沒有被呼叫）", res, gcalls == [], f"{gcalls}")
    check_cli("Q27 pairs：有 missing（03）與 format（04）時 exit code 1", res, rc == 1, f"{rc}")
    check_cli("Q27 pairs：逐對狀態 01 ok、02 ok（MPO×2）、04 format、03 missing；非影像 1 個", res,
              all(t in out for t in ("01 狀態 ok", "02 狀態 ok", "MPO×2", "04 狀態 format", "03 狀態 missing", "非影像 1")))
    check_cli("Q27 pairs：完整印出 01 hazy 的原始檔 SHA-256 與 decoded RGB SHA-256（兩者不同）", res,
              f"原始 {hashlib.sha256(pa1).hexdigest()} | decoded RGB {hashlib.sha256(bytes(a1.reshape(-1).tolist())).hexdigest()}"
              in out)

    o = [rng.integers(0, 256, s, dtype=np.uint8) for s in ((16, 16, 3), (16, 16, 3), (16, 20, 3), (16, 20, 3),
                                                           (16, 16, 3), (16, 16, 3))]
    ofiles = {f"{d}/{i:02d}_outdoor_{r}.jpg": _png(o[2 * (i - 1) + k][:, :, ::-1])
              for i in (1, 2, 3) for k, (d, r) in enumerate((("hazy", "hazy"), ("GT", "GT")))}
    gate300 = ["--rss-a", "0", "--rss-b", "1", "--rss-chunk-rows", "4", "--budget-gib", repr(300 / GIB), "--margin", "1"]
    seen = []
    real_pair = run_pair

    def rec_pair(pid, hazy, gt_, chunk_rows=0):
        seen.append((pid, chunk_rows))
        return real_pair(pid, hazy, gt_, chunk_rows)

    def ran():
        return [p for p, _ in seen]

    def row_ids(text):
        return [ln.split("|")[0].strip() for ln in text.splitlines() if re.match(r"\s+\d+ \| ", ln)]

    res = cli(["ohaze", "X", "--chunk-rows", "4"], ofiles, rec_pair)
    rc, out, _ = res
    check_cli("Q27 ohaze 缺少資源條件（已開分塊 4 列）：拒絕開始，exit code 2，沒有跑任何一對", res,
              rc == 2 and "拒絕開始（資源 gate）" in out and seen == [], f"{rc}、{seen}")
    seen.clear()
    res = cli(["ohaze", "X", "--chunk-rows", "4", *gate300], ofiles, rec_pair)
    rc, out, _ = res
    rid = row_ids(out)
    check_cli("Q27 ohaze 預設在第一對完成後停止：只跑 ID 1，報表只有 ID 1，exit code 0", res,
              rc == 0 and ran() == [1] and rid == ["1"] and "第一對完成後停止" in out, f"{rc}、{seen}、{rid}")
    check_cli("Q27 ohaze 把 --chunk-rows 4 傳給 run_pair", res, seen[:1] == [(1, 4)], f"{seen}")
    golden_sha = hashlib.sha256(open(golden.__file__, "rb").read()).hexdigest()
    check_cli("Q27 ohaze 報表開頭：golden hash、暫定規則、FX 設定、不是團隊驗收、資料來源、環境、命令、論文對照的註記", res,
              all(t in out for t in (f"golden={golden_sha}", "tie=first（L1 暫定）", "FX=F20+PALA 定點+T10+SD lut K10",
                                     "status=軟體暫定設定，不是團隊驗收", OHAZE_SOURCE["zip_sha256"], "環境：Python",
                                     "命令：ate_quality.py ohaze X --chunk-rows 4", "設定不同，僅供參考")))
    seen.clear()
    res = cli(["ohaze", "X", "--chunk-rows", "4", *gate300, "--after-first"], ofiles, rec_pair)
    rc, out, _ = res
    check_cli("Q27 ohaze 每對 golden 前檢查 pixel 上限：ID 2（16×20 = 320 > P_max 300）不跑，exit code 3", res,
              rc == 3 and 2 not in ran() and "超過 P_max 300" in out, f"{rc}、{seen}")
    check_cli("Q27 ohaze 超過上限時停止整次執行，不跳過大圖繼續跑 ID 3", res, 3 not in ran(), f"{seen}")
    seen.clear()
    gate0 = [*gate300[:5], "0", *gate300[6:]]       # 同一組假數字，只把校準時的分塊設定改成 0（和不分塊的本次一致）
    res = cli(["ohaze", "X", *gate0], {"hazy/01_outdoor_hazy.jpg": pa1}, rec_pair)
    rc, out, _ = res
    check_cli("Q27 ohaze N_ok = 0（只有 missing）：exit code 1，報表印 NA（N_ok = 0）", res,
              rc == 1 and seen == [] and "NA（N_ok = 0）" in out, f"{rc}")
    seen.clear()
    res = cli(["ohaze", "X", "--chunk-rows", "4", *gate300, "--after-first"],
              {**ofiles, "GT/03_outdoor_GT.jpg": _png(o[5][:, :, 0])}, rec_pair)
    rc, out, _ = res
    check_cli("Q27 ohaze 預檢遇到 format（ID 3 的 GT 是灰階）：拒絕開始，exit code 2，沒有跑任何一對", res,
              rc == 2 and "拒絕開始（格式 gate）" in out and seen == [], f"{rc}、{seen}")
    seen.clear()
    res = cli(["ohaze", "X", *gate0], {**ofiles, "hazy/IMG_9.jpg": pa1}, rec_pair)
    rc, out, _ = res
    check_cli("Q27 ohaze 有檔名無法解析的影像：拒絕開始，exit code 2，沒有跑任何一對", res,
              rc == 2 and "檔名無法解析的影像" in out and seen == [], f"{rc}、{seen}")

    cseen = []

    def stub_pair(pid, hazy, gt_, chunk_rows=0):
        cseen.append((hazy.shape, gt_.shape, hazy.dtype == gt_.dtype == np.uint8, chunk_rows))
        return {"id": pid, "status": "error", "msg": "測試替身：不執行 golden"}

    res = cli(["calib", "16", "12", "--chunk-rows", "4"], None, stub_pair)
    check_cli("Q27 calib：產生 16×12 的合成 hazy、GT（uint8），交給 run_pair（測試替身），分塊 4 列", res,
              cseen == [((16, 12, 3), (16, 12, 3), True, 4)] and res[0] == 1, f"{cseen}、{res[0]}")
    res = cli(["calib", "10", "12"], None, stub_pair)
    check_cli("Q27 calib：H 或 W < 11（SSIM 視窗）時拒絕開始，exit code 2，不呼叫 run_pair", res,
              res[0] == 2 and len(cseen) == 1, f"{res[0]}、{cseen}")

    print("讀檔的 I/O 錯誤（記憶體替身在第 k 次讀某個檔案時丟例外；不建立、刪除檔案，也不改權限）")

    class FailSource(_MemSource):
        def __init__(self, files, fail):
            super().__init__(files)
            self.fail, self.count = fail, {}

        def read(self, name):
            k = self.count[name] = self.count.get(name, 0) + 1
            if (name, k) in self.fail:
                raise self.fail[(name, k)]
            return super().read(name)

    g2 = "GT/02_outdoor_GT.jpg"
    perm = PermissionError(13, "Permission denied", g2)
    hz, g_, inf_ = load_pair(FailSource(ofiles, {(g2, 1): perm}), 2, "hazy/02_outdoor_hazy.jpg", g2)
    ck.check("Q28 load_pair 讀 GT 時 PermissionError：read_error，記路徑與原因，兩張都不回傳，也沒有 hash 或尺寸",
             inf_["status"] == "read_error" and hz is None and g_ is None
             and inf_["gt"].get("read_error", "").startswith("PermissionError") and inf_["gt"]["name"] == g2
             and not any(k in inf_[r] for r in ("hazy", "gt") for k in ("raw_sha", "rgb_sha", "h", "w", "bytes")), f"{inf_}")
    try:
        load_pair(FailSource(ofiles, {(g2, 1): TimeoutError("測試替身的逾時")}), 2, "hazy/02_outdoor_hazy.jpg", g2)
        to_ok = False
    except TimeoutError:
        to_ok = True
    ck.check("Q28 讀檔的 TimeoutError 照常往外丟，不當成讀檔失敗", to_ok)
    gate1000 = ["--rss-a", "0", "--rss-b", "1", "--rss-chunk-rows", "0", "--budget-gib", repr(1000 / GIB), "--margin", "1"]
    seen.clear()
    res = cli(["ohaze", "X", *gate1000, "--after-first"], FailSource(ofiles, {(g2, 1): perm}), rec_pair)
    rc, out, _ = res
    check_cli("Q28 ohaze 預檢讀檔失敗（ID 2 的 GT，PermissionError）：拒絕開始，exit code 2，沒有跑任何一對；印出 ID、路徑與原因，不印 hash",
              res, rc == 2 and seen == [] and "拒絕開始（預檢讀檔失敗）：ID 2" in out
              and f"02 gt   | {g2} | 讀檔失敗：PermissionError" in out, f"{rc}、{seen}")
    seen.clear()
    fnf = FileNotFoundError(2, "No such file or directory", g2)
    res = cli(["ohaze", "X", *gate1000, "--after-first"], FailSource(ofiles, {(g2, 2): fnf}), rec_pair)
    rc, out, _ = res
    rid = row_ids(out)
    check_cli("Q28 ohaze 執行途中讀檔失敗（ID 2 的 GT 第二次讀，FileNotFoundError）：保留 ID 1 的結果並印合計，ID 2 記 read_error，exit code 4",
              res, rc == 4 and rid == ["1", "2"] and "合計：N_ok = 1／N_total = 2" in out and "read_error 1（ID 2）" in out
              and "FileNotFoundError" in out, f"{rc}、{rid}")
    check_cli("Q28 ohaze 執行途中讀檔失敗後停止，不繼續跑 ID 3", res, ran() == [1], f"{seen}")
    res = cli(["pairs", "X"], FailSource(ofiles, {(g2, 1): perm}))
    rc, out, _ = res
    check_cli("Q28 pairs 遇到讀檔失敗：不中止，ID 2 列出路徑與原因，其他配對照常，exit code 1", res,
              rc == 1 and "02 狀態 read_error" in out and f"{g2} | 讀檔失敗：PermissionError" in out
              and "01 狀態 ok" in out and "03 狀態 ok" in out, f"{rc}")

    print("結果檔（4.11.17；結果檔與資料目錄都是記憶體替身，不寫任何檔案；預期值由 run_pair、load_pair 直接得到）")

    def strict_eq(a, b):
        """測試自己的逐值、逐型別比較（不用被測的 _same）。"""
        if type(a) is not type(b):
            return False
        if isinstance(a, dict):
            return set(a) == set(b) and all(strict_eq(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(strict_eq(x, y) for x, y in zip(a, b))
        return a == b

    def recs_of(data):
        """正式 reader 讀回的資料行，和例外字串。ValueError 是 reader 的正常拒絕：回傳 []、沒有例外，依賴的檢查正常 FAIL；
        其他例外回傳例外字串，依賴的檢查經 check_cli 帶 EXC_MARK；TimeoutError 照常往外丟。"""
        try:
            return read_records(data)[0], None
        except TimeoutError:
            raise
        except ValueError:
            return [], None
        except Exception as e:
            return [], f"{type(e).__name__}: {e}"

    def attempt(f, default=b""):
        """建構測試資料，回傳 (資料, 例外字串)。前一步沒有寫出紀錄時建構會失敗：回傳 default 和例外字串，依賴它的檢查
        經 check_cli 帶 EXC_MARK（不算正常 FAIL），selfcheck 照常跑完；TimeoutError 照常往外丟。"""
        try:
            return f(), None
        except TimeoutError:
            raise
        except Exception as e:
            return default, f"{type(e).__name__}: {e}"

    gate_r = ["--rss-a", "0", "--rss-b", "1", "--rss-chunk-rows", "0", "--budget-gib", repr(1000 / GIB), "--margin", "1"]
    got = []

    def keep_pair(pid, hazy, gt_, chunk_rows=0):
        r = real_pair(pid, hazy, gt_, chunk_rows)
        got.append(r)
        return r

    m1 = _MemRecord()
    res29 = cli(["ohaze", "X", *gate_r, "--after-first", "--record", "R"], ofiles, keep_pair, m1)
    rc, out, _ = res29
    check_cli("Q29 ohaze --record：3 對都寫入並讀回驗證，exit code 0", res29,
              rc == 0 and all(f"紀錄已寫入並讀回驗證：ID {i:02d}、第 {i} 行" in out for i in (1, 2, 3)), f"{rc}")
    recs1, e_r1 = recs_of(m1.data or b"")
    q29 = (res29, e_r1)          # Q29 的衍生檢查依賴寫入那次 cli 和讀回
    want = [load_pair(_MemSource(ofiles), i, f"hazy/{i:02d}_outdoor_hazy.jpg", f"GT/{i:02d}_outdoor_GT.jpg")[2]
            for i in (1, 2, 3)]
    check_cli("Q29 正式 writer → reader：還原的 row 和 run_pair 直接回傳的 row 逐值、逐型別相同（含 sec）", None,
              lambda: len(recs1) == 3 and len(got) == 3 and all(strict_eq(r["row"], x) for r, x in zip(recs1, got)),
              f"{len(recs1)}", pre=q29)
    check_cli("Q29 還原的 pair 和直接呼叫 load_pair 得到的預檢資訊逐值、逐型別相同（含 16×20 的 ID 2）", None,
              lambda: len(recs1) == 3 and all(strict_eq(r["pair"], p) for r, p in zip(recs1, want)), pre=q29)
    check_cli("Q29 還原的型別：stats.a_far 是 3 個 int 的 tuple、decoded 是 tuple、PSNR 與 sec 是 float", None,
              lambda: len(recs1) == 3 and type(recs1[0]["row"]["stats"]["a_far"]) is tuple
              and all(type(v) is int for v in recs1[0]["row"]["stats"]["a_far"])
              and type(recs1[0]["pair"]["hazy"]["decoded"]) is tuple
              and type(recs1[0]["row"]["FX"]["psnr"]) is float and type(recs1[0]["row"]["sec"]) is float, pre=q29)
    check_cli("Q29 run：golden hash、P_max 1000、resource_metric、3 行設定標記", None,
              lambda: len(recs1) == 3 and recs1[0]["run"]["golden"] == golden_sha and recs1[0]["run"]["gate"]["p_max"] == 1000
              and recs1[0]["run"]["resource_metric"] == "peak memory footprint (/usr/bin/time -l)"
              and len(recs1[0]["run"]["settings"]) == 3, pre=q29)

    hz5 = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    fl5 = run_mode(hz5, fl_active())["out"]
    nfiles = {"hazy/05_outdoor_hazy.jpg": _png(hz5[:, :, ::-1]), "GT/05_outdoor_GT.jpg": _png(fl5[:, :, ::-1])}
    m5 = _MemRecord()
    got.clear()
    res5 = cli(["ohaze", "X", *gate_r, "--record", "R"], nfiles, keep_pair, m5)
    check_cli("Q33 FL 等於 GT（FL 的 PSNR 是 +inf）：寫入並讀回驗證，exit code 0", res5,
              res5[0] == 0 and "紀錄已寫入並讀回驗證：ID 05" in res5[1], f"{res5[0]}")
    r5, e_r5 = recs_of(m5.data or b"")
    check_cli("Q33 同一列：d_psnr 是 NA(nonfinite)、d_ssim 是有限浮點、FX 的 PSNR 有限；還原後和原始 row 逐值相同", None,
              lambda: len(r5) == 1 and len(got) == 1 and r5[0]["row"]["d_psnr"] == NA("nonfinite")
              and type(r5[0]["row"]["d_ssim"]) is float and math.isfinite(r5[0]["row"]["d_ssim"])
              and math.isfinite(got[0]["FX"]["psnr"]) and got[0]["FL"]["psnr"] == math.inf
              and strict_eq(r5[0]["row"], got[0]), pre=(res5, e_r5))

    base, e_b = attempt(lambda: json.loads(m1.data.split(b"\n")[0]), {})
    base5, e_b5 = attempt(lambda: json.loads(m5.data.split(b"\n")[0]), {})

    def edited(fn, five=False, raw=None):
        """從 Q29（five 時是 Q33）寫出的第 1 行改出一行。回傳 (bytes, 前置步驟)：前置步驟是寫入那次 cli、取出那一行、
        這次修改，交給 check_cli 的 pre。"""
        def build():
            o_ = json.loads(json.dumps(base5 if five else base))
            fn(o_)
            text = json.dumps(o_, ensure_ascii=False)
            if raw is not None:
                text = text.replace('"__RAW__"', raw)
            return (text + "\n").encode("utf-8")
        data, e = attempt(build)
        return data, ((res5, e_b5) if five else (res29, e_b)) + (e,)

    def rejects(data, needle):
        """正式 reader 是否以 ValueError 拒絕、訊息含 needle。回傳 (結果, 例外字串)：ValueError 是正常拒絕；
        其他例外不算「沒拒絕」的正常結果，回傳例外字串，依賴的檢查經 check_cli 帶 EXC_MARK；TimeoutError 照常往外丟。"""
        try:
            read_records(data)
        except TimeoutError:
            raise
        except ValueError as e:
            return needle in str(e), None
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        return False, None

    def put(path, value):
        def fn(o_):
            d = o_
            for k in path[:-1]:
                d = d[k]
            if value is DEL:
                del d[path[-1]]
            else:
                d[path[-1]] = value
        return fn

    DEL = object()
    ok_line = edited(lambda o_: None)
    rj, e_rj = rejects(ok_line[0], "")
    r_ok, e_ok = recs_of(ok_line[0])
    check_cli("Q30 對照：未修改的一行可以讀回", None, lambda: not rj and len(r_ok) == 1, pre=(*ok_line[1], e_rj, e_ok))
    cases = [
        ("Q30 缺少必要的 key（row.sec）", edited(put(("row", "sec"), DEL)), "缺少必要的 key"),
        ("Q30 未定義的 key（row.extra）", edited(put(("row", "extra"), 1)), "未定義的 key"),
        ("Q30 整數寫成 1.0（row.h）", edited(put(("row", "h"), 16.0)), "的整數"),
        ("Q30 整數寫成 true（pair.id）", edited(put(("pair", "id"), True)), "的整數"),
        ("Q30 NaN token（row.H.ssim）", edited(put(("row", "H", "ssim"), "__RAW__"), raw="NaN"), "不接受的數字 token"),
        ("Q30 Infinity token（row.H.ssim）", edited(put(("row", "H", "ssim"), "__RAW__"), raw="Infinity"), "不接受的數字 token"),
        ("Q30 1e999 溢位成無限大（row.H.ssim）", edited(put(("row", "H", "ssim"), "__RAW__"), raw="1e999"), "溢位"),
        ("Q30 \"+inf\" 出現在不允許的位置（row.H.ssim）", edited(put(("row", "H", "ssim"), "+inf")), "必須是浮點數"),
        ("Q30 PSNR 是 +inf 但 MSE 不是 0", edited(put(("row", "H", "psnr"), "+inf")), "若且唯若"),
        ("Q30 NA 原因不是 nonfinite", edited(put(("row", "d_psnr"), {"NA": "empty"}), five=True), "row.d_psnr 必須是"),
        ("Q30 d_psnr 和 PSNR 的有限性不一致（兩個 PSNR 都有限卻是 NA）", edited(put(("row", "d_psnr"), {"NA": "nonfinite"})),
         "row.d_psnr 必須是浮點數"),
        ("Q30 d_ssim 是 NA", edited(put(("row", "d_ssim"), {"NA": "nonfinite"})), "row.d_ssim 必須是浮點數"),
        ("Q30 有限的 Δ 不等於 FX − FL", edited(lambda o_: o_["row"].update(d_psnr=o_["row"]["d_psnr"] + 0.5)), "不等於 FX − FL"),
        ("Q30 fxfl.n 不等於 3 × H × W", edited(lambda o_: o_["row"]["fxfl"].update(n=o_["row"]["fxfl"]["n"] + 3)), "3 × H × W"),
        ("Q30 SHA-256 格式錯誤", edited(put(("row", "sha", "FL"), "abc")), "不是 SHA-256"),
        ("Q30 row 的 H×W 和 pair 不同（pair 16×16、row 8×32：pixel 數相同，fxfl.n = 3 × 8 × 32 = 768 也成立）",
         edited(lambda o_: o_["row"].update(h=8, w=32)), "row 的 H×W 和 pair 不同"),
        ("Q30 最後一行不完整（沒有 \\n）", (ok_line[0][:-1], ok_line[1]), "最後一行不完整"),
        ("Q30 檔名的 ID 或角色不符（ID 1 的 GT 寫成 ID 2 的檔名）", edited(put(("pair", "gt", "name"), "GT/02_outdoor_GT.jpg")),
         "ID 或角色"),
        ("Q30 檔名的 ID 或角色不符（GT 寫成 hazy 的檔名）", edited(put(("pair", "gt", "name"), "hazy/01_outdoor_hazy.jpg")),
         "ID 或角色"),
        ("Q30 JSON 有重複的 key", (ok_line[0].replace(b'{"schema"', b'{"schema": "x", "schema"', 1), ok_line[1]), "重複的 key"),
        ("Q30 schema 不合法", edited(put(("schema",), "ohaze-record-0")), "schema 不合法"),
    ]
    for label, (data, deps), needle in cases:
        rj, e_rj = rejects(data, needle)
        check_cli(label + "：拒絕", None, rj, pre=(*deps, e_rj))
    check_cli("Q30 H×W 反例的前提：pair 是 16×16、fxfl.n 是 768（和 row 8×32 的 3 × H × W 相同，只有 H×W 的比對能拒絕）", None,
              lambda: (base["pair"]["h"], base["pair"]["w"], base["row"]["fxfl"]["n"]) == (16, 16, 768), pre=(res29, e_b))
    # reader 的每個必要 key 後面都會直接取值；不檢查缺少的 key 時，reader 路徑只會看到 KeyError（例外，不算正常 FAIL），
    # 所以 key 集合的檢查另外直接呼叫 _keys，看正常的回傳。
    try:
        _keys({"a": 1}, "測試物件", ("a", "b"))
        kmsg = None
    except ValueError as e:
        kmsg = str(e)
    ck.check("Q30 key 集合的檢查（直接呼叫 _keys）：缺少 b 時丟 ValueError，訊息列出 b", kmsg == "測試物件 缺少必要的 key：b",
             f"{kmsg!r}")

    def bad_pair(pid, hazy, gt_, chunk_rows=0):
        return {**real_pair(pid, hazy, gt_, chunk_rows), "extra": 1}

    for label, rec_, pfn, needle in (
            ("Q31 寫入失敗（OSError）：exit code 5，不印確認行，結果檔沒有內容",
             _MemRecord(fail_append=OSError(28, "No space left on device")), None, "寫入失敗"),
            ("Q31 讀回的內容和寫入的不同：exit code 5，不印確認行", _MemRecord(corrupt_after=True), None, "讀回的檔案內容和寫入的不同"),
            ("Q31 寫入前的驗證不通過（row 有未定義的 key）：exit code 5，不印確認行，結果檔沒有被寫入", _MemRecord(), bad_pair,
             "寫入前的驗證")):
        res = cli(["ohaze", "X", "--ids", "1", *gate_r, "--record", "R"], ofiles, pfn, rec_)
        written = rec_.data is not None and not rec_.corrupt_after
        check_cli(label, res, res[0] == 5 and "紀錄已寫入並讀回驗證" not in res[1] and needle in res[1] and not written,
                  f"{res[0]}、{res[1][-200:]!r}")
    keep1 = _MemRecord(m1.data)
    res = cli(["ohaze", "X", "--ids", "1", *gate_r, "--record", "R"], ofiles, None, keep1)
    check_cli("Q31 結果檔已有本次要跑的 ID：拒絕開始，exit code 2，不修改既有內容", res,
              res[0] == 2 and "已有本次要跑的 ID：01" in res[1] and keep1.data == m1.data, f"{res[0]}", pre=(res29,))
    gate_b = [*gate_r[:7], repr(999 / GIB), *gate_r[8:]]
    res = cli(["ohaze", "X", "--ids", "3", *gate_b, "--record", "R"], ofiles, None, keep1)
    check_cli("Q31 結果檔的 run 和本次不同（預算不同）：拒絕開始，exit code 2，不修改既有內容", res,
              res[0] == 2 and "run 和本次不同" in res[1] and keep1.data == m1.data, f"{res[0]}", pre=(res29,))
    m7 = _MemRecord()
    res = cli(["ohaze", "X", "--ids", "7", *gate_r, "--record", "R"], {"hazy/07_outdoor_hazy.jpg": pa1}, None, m7)
    r7, e_r7 = recs_of(m7.data or b"")
    check_cli("Q31 missing 的配對也寫入（pair 只有 missing_files），exit code 1（N_ok = 0）", res,
              res[0] == 1 and "紀錄已寫入並讀回驗證：ID 07" in res[1] and len(r7) == 1 and r7[0]["pair"]["status"] == "missing"
              and r7[0]["row"]["status"] == "missing", f"{res[0]}", pre=(e_r7,))

    def raise_static(*a):
        raise RuntimeError("測試替身：golden 失敗")

    m_g = _MemRecord()
    golden.run_static = raise_static
    try:
        res = cli(["ohaze", "X", "--ids", "1", *gate_r, "--record", "R"], ofiles, None, m_g)
    finally:
        golden.run_static = real_static
    rg, e_rg = recs_of(m_g.data or b"")
    check_cli("Q32 golden 失敗：寫入沒有 sec 的 error row，讀回驗證通過並印確認行，exit code 1", res,
              res[0] == 1 and "紀錄已寫入並讀回驗證：ID 01" in res[1] and len(rg) == 1 and rg[0]["row"]["status"] == "error"
              and "sec" not in rg[0]["row"] and rg[0]["row"]["msg"].startswith("golden："), f"{res[0]}", pre=(e_rg,))
    real_ssim = g["ssim"]

    def raise_ssim(*a):
        raise ValueError("測試替身：metric 失敗")

    m_m = _MemRecord()
    g["ssim"] = raise_ssim
    try:
        res = cli(["ohaze", "X", "--ids", "1", *gate_r, "--record", "R"], ofiles, None, m_m)
    finally:
        g["ssim"] = real_ssim
    rm, e_rm = recs_of(m_m.data or b"")
    check_cli("Q32 metric 失敗：寫入有 sec 的 error row，讀回驗證通過，exit code 1（不是 exit 5）", res,
              res[0] == 1 and "紀錄已寫入並讀回驗證：ID 01" in res[1] and len(rm) == 1 and rm[0]["row"]["status"] == "error"
              and type(rm[0]["row"].get("sec")) is float and rm[0]["row"]["msg"].startswith("metric："), f"{res[0]}",
              pre=(e_rm,))
    g1 = "GT/01_outdoor_GT.jpg"
    m_r = _MemRecord()
    res = cli(["ohaze", "X", "--ids", "1", *gate_r, "--record", "R"],
              FailSource(ofiles, {(g1, 2): FileNotFoundError(2, "No such file or directory", g1)}), None, m_r)
    check_cli("Q32 執行途中讀檔失敗：不寫入任何紀錄、不印確認行，exit code 4", res,
              res[0] == 4 and "紀錄已寫入並讀回驗證" not in res[1] and m_r.data is None, f"{res[0]}")

    print("整輪完成資格（4.11.17；expected IDs 固定 01–45，所以用 45 對 16×16 的合成配對）")
    f45 = {f"{d}/{i:02d}_outdoor_{r}.jpg": _png(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
           for i in EXPECTED_IDS for d, r in (("hazy", "hazy"), ("GT", "GT"))}
    res1 = cli(["ohaze", "X", *gate_r, "--after-first"], f45)
    m45 = _MemRecord()
    each, e45 = [], []
    for i in EXPECTED_IDS:
        r_ = cli(["ohaze", "X", "--ids", str(i), *gate_r, "--record", "R"], f45, None, m45)
        e45.append(r_[2])
        each.append(r_[0] == 0 and f"紀錄已寫入並讀回驗證：ID {i:02d}、第 {i} 行" in r_[1])
    check_cli("Q34 45 對各自一次 --record：都 exit code 0，確認行的 ID 與行號正確", None, all(each),
              f"{each.count(False)} 次不符", pre=e45)
    data45 = m45.data or b""
    res = cli(["ohaze-report", "R", "X"], f45, None, _MemRecord(data45))
    check_cli("Q34 45 行資料都通過但沒有完成紀錄：彙整拒絕正式合計（exit code 2），列出已保存的 ID", res,
              res[0] == 2 and "沒有完成紀錄" in res[1] and "合計：N_ok" not in res[1] and "已保存的 ID（45 行）" in res[1], f"{res[0]}",
              pre=e45)
    mdone = _MemRecord(data45)
    resc = cli(["ohaze-complete", "R"], None, None, mdone)
    check_cli("Q34 ohaze-complete：寫入完成紀錄並讀回驗證，exit code 0", resc,
              resc[0] == 0 and "完成紀錄已寫入並讀回驗證：資料 45 行" in resc[1], f"{resc[0]}、{resc[1][-200:]!r}", pre=e45)
    full = mdone.data or b""
    done_obj, e_d = attempt(lambda: json.loads(full[len(data45):]), {})
    p45 = (*e45, resc, e_d)      # 之後用到 data45、full、done_obj 的檢查都依賴這些前置步驟
    check_cli("Q34 完成紀錄：data_sha256 是前面 45 行資料 bytes 的 SHA-256，ids 是 1–45 依序，data_lines 45", None,
              lambda: done_obj["data_sha256"] == hashlib.sha256(data45).hexdigest()
              and done_obj["ids"] == list(range(1, 46)) and done_obj["data_lines"] == 45, pre=p45)
    res2 = cli(["ohaze-report", "R", "X"], f45, None, _MemRecord(full))

    def norm(text):
        out_ = []
        for ln in text.splitlines():
            if ln.startswith("命令："):
                continue
            out_.append(ln.rsplit(" | ", 1)[0] if re.match(r"\s+\d+ \| ", ln) else ln)
        return out_

    check_cli("Q34 正常完成：每列 status = ok，合計 N_ok = 45／N_total = 45，exit code 0", res2,
              res2[0] == 0 and "合計：N_ok = 45／N_total = 45" in res2[1] and "狀態：ok 45" in res2[1], f"{res2[0]}", pre=p45)
    check_cli("Q34 正常完成：彙整的報表和一個程序跑完的報表逐字相同（去掉時間欄與「命令：」一行）", res2,
              res1[0] == 0 and norm(res1[1]) == norm(res2[1]) and len(norm(res2[1])) > 100,
              f"{res1[0]}、{len(norm(res1[1]))}、{len(norm(res2[1]))}", pre=(res1, *p45))

    lines45 = data45.split(b"\n")[:-1]

    def with_done(data_lines, tweak=None):
        body = b"".join(ln + b"\n" for ln in data_lines)
        d = json.loads(json.dumps(done_obj))
        d.update(data_lines=len(data_lines), ids=[json.loads(ln)["pair"]["id"] for ln in data_lines],
                 data_sha256=hashlib.sha256(body).hexdigest())
        if tweak:
            tweak(d)
        return body + (json.dumps(d, ensure_ascii=False) + "\n").encode("utf-8")

    def rewrite(ln, fn):
        o_ = json.loads(ln)
        fn(o_)
        return json.dumps(o_, ensure_ascii=False).encode("utf-8")

    def report(data, files=f45):
        return cli(["ohaze-report", "R", "X"], files, None, _MemRecord(data))

    tail = full[len(data45):]
    moved, e_ = attempt(lambda: b"".join(ln + b"\n" for ln in [rewrite(lines45[0], lambda o_: o_.update(
        written_at="2000-01-01T00:00:00+00:00")), *lines45[1:]]) + tail)
    res = report(moved)
    check_cli("Q34 完成紀錄的 data_sha256 和資料不符（完成後改動第 1 行）：拒絕", res,
              res[0] == 2 and "data_sha256" in res[1] and "合計：N_ok" not in res[1], f"{res[0]}", pre=(*p45, e_))
    d_, e_ = attempt(lambda: with_done(lines45, lambda d: d.update(data_lines=44)))
    res = report(d_)
    check_cli("Q34 完成紀錄的 data_lines 不符：拒絕", res, res[0] == 2 and "data_lines" in res[1], f"{res[0]}", pre=(*p45, e_))
    d_, e_ = attempt(lambda: with_done(lines45, lambda d: d.update(ids=[2, 1, *range(3, 46)])))
    res = report(d_)
    check_cli("Q34 完成紀錄的 ids 和資料行的順序不同：拒絕", res, res[0] == 2 and "ids" in res[1], f"{res[0]}", pre=(*p45, e_))
    res = report(tail + data45)
    check_cli("Q34 完成紀錄不在最後一行：拒絕", res, res[0] == 2 and "完成紀錄不在最後一行" in res[1], f"{res[0]}", pre=p45)
    res = report(full + tail)
    check_cli("Q34 完成紀錄出現兩次：拒絕", res, res[0] == 2 and "完成紀錄不在最後一行" in res[1], f"{res[0]}", pre=p45)
    d_, e_ = attempt(lambda: with_done([*lines45, lines45[43]]))
    res = report(d_)
    check_cli("Q34 重複的 ID（ID 44 出現兩次）：拒絕", res, res[0] == 2 and "重複的 ID：44" in res[1] and "缺少的 ID" not in res[1],
              f"{res[0]}", pre=(*p45, e_))
    done44, e_44 = attempt(lambda: with_done(lines45[:44]))
    res = report(done44)
    check_cli("Q34 缺少的 ID（只有 01–44）：拒絕", res, res[0] == 2 and "缺少的 ID：45" in res[1] and "資料目錄" not in res[1],
              f"{res[0]}", pre=(*p45, e_44))
    f44 = {k: v for k, v in f45.items() if not k.split("/")[1].startswith("45_")}
    res = report(done44, f44)
    check_cli("Q34 資料目錄只有 44 對、紀錄也是 01–44：expected IDs 仍是 01–45，拒絕", res,
              res[0] == 2 and "缺少的 ID：45" in res[1] and "合計：N_ok" not in res[1], f"{res[0]}", pre=(*p45, e_44))
    res = report(full, f44)
    check_cli("Q34 資料目錄不是 01–45 的完整配對：拒絕", res, res[0] == 2 and "資料目錄的清單不是 01–45" in res[1], f"{res[0]}",
              pre=p45)
    other = _MemSource(f45)
    other.root = "（另一個記憶體資料目錄，檔名清單相同）"
    res = cli(["ohaze-report", "R", "X"], other, None, _MemRecord(full))
    check_cli("Q34 紀錄的資料目錄和本次不同（檔名清單相同）：拒絕，只列這一項原因", res,
              res[0] == 2 and f"不印合計：紀錄的資料目錄 {_MemSource.root!r} 和本次的 {other.root!r} 不同\n" in res[1], f"{res[0]}",
              pre=p45)
    pm_x, e_pm = attempt(lambda: [rewrite(ln, lambda o_: o_["run"]["gate"].update(p_max=1001)) for ln in lines45], [])
    d_, e_ = attempt(lambda: with_done(pm_x, lambda d: d["run"]["gate"].update(p_max=1001)))
    res = report(d_)
    check_cli("Q34 紀錄的 P_max 1001 和重算的 1000 不同（各行與完成紀錄一致）：拒絕，只列這一項原因", res,
              res[0] == 2 and "不印合計：紀錄的 P_max 1001 和重算的 1000 不同\n" in res[1], f"{res[0]}", pre=(*p45, e_pm, e_))
    env_x, e_env = attempt(lambda: [rewrite(ln, lambda o_: o_["run"].update(env="環境：測試用的另一個環境")) for ln in lines45], [])
    d_, e_ = attempt(lambda: with_done(env_x, lambda d: d["run"].update(env="環境：測試用的另一個環境")))
    res = report(d_)
    check_cli("Q34 紀錄的程式或環境和執行彙整的程式不同（各行一致）：拒絕", res,
              res[0] == 2 and "紀錄的程式或環境" in res[1] and "和第 1 行不同" not in res[1], f"{res[0]}", pre=(*p45, e_env, e_))
    m44 = _MemRecord(done44)
    res = cli(["ohaze", "X", "--ids", "45", *gate_r, "--record", "R"], f45, None, m44)
    check_cli("Q34 --record 遇到已有完成紀錄的檔：拒絕開始，exit code 2，不修改既有內容", res,
              res[0] == 2 and "已有完成紀錄" in res[1] and m44.data == done44, f"{res[0]}", pre=(*p45, e_44))
    res = cli(["ohaze-complete", "R"], None, None, _MemRecord(full))
    check_cli("Q34 ohaze-complete 遇到已有完成紀錄的檔：拒絕，exit code 2", res, res[0] == 2 and "已有完成紀錄" in res[1],
              f"{res[0]}", pre=p45)
    data44 = b"".join(ln + b"\n" for ln in lines45[:44])
    mc = _MemRecord(data44)
    res = cli(["ohaze-complete", "R"], None, None, mc)
    check_cli("Q34 ohaze-complete 遇到缺少 ID 45：拒絕，exit code 2，不修改既有內容", res,
              res[0] == 2 and "缺少的 ID：45" in res[1] and mc.data == data44, f"{res[0]}", pre=p45)
    d_, e_ = attempt(lambda: data45 + lines45[44] + b"\n")
    res = cli(["ohaze-complete", "R"], None, None, _MemRecord(d_))
    check_cli("Q34 ohaze-complete 遇到重複的 ID：拒絕，exit code 2", res, res[0] == 2 and "重複的 ID：45" in res[1], f"{res[0]}",
              pre=(*p45, e_))
    d_, e_ = attempt(lambda: b"".join(ln + b"\n" for ln in [env_x[0], *lines45[1:]]))
    res = cli(["ohaze-complete", "R"], None, None, _MemRecord(d_))
    check_cli("Q34 ohaze-complete 遇到各行的 run 不一致：拒絕，exit code 2", res, res[0] == 2 and "run" in res[1], f"{res[0]}",
              pre=(*p45, e_env, e_))
    mf = _MemRecord(data45, fail_append=OSError(28, "No space left on device"))
    res = cli(["ohaze-complete", "R"], None, None, mf)
    check_cli("Q34 ohaze-complete 寫入失敗：exit code 5，不印確認行，不修改既有內容", res,
              res[0] == 5 and "完成紀錄已寫入" not in res[1] and mf.data == data45, f"{res[0]}", pre=p45)
    mr = _MemRecord(data45, corrupt_after=True)
    res = cli(["ohaze-complete", "R"], None, None, mr)
    check_cli("Q34 ohaze-complete 讀回的內容和寫入的不同：exit code 5，不印確認行", res,
              res[0] == 5 and "讀回的檔案內容和寫入的不同" in res[1] and "完成紀錄已寫入" not in res[1], f"{res[0]}", pre=p45)

    print(f"結果：{ck.total - ck.failed}/{ck.total} PASS")
    return 0 if ck.failed == 0 else 1


# ---------- mutation（只改本檔的 production 函式；每項都要由指定的檢查正常 FAIL） ----------

MUTATIONS = [
    # (名稱, 函式, 原字串, 改成, 預期 FAIL 的檢查標籤含)
    ("QM1 拿掉相減前的 int64 轉型（相減與平方都在 uint8）", "mse",
     "x.astype(np.int64) - y.astype(np.int64)", "x - y", "Q2 "),
    ("QM2 PSNR 的 data_range 改成 1", "psnr_from_mse", "PEAK ** 2 / m", "1 ** 2 / m", "Q2 "),
    ("QM3 MSE 改成每 pixel 三色平方和的平均", "mse", "/ d.size", "/ (d.size // 3)", "Q3 "),
    ("QM4 資料集平均改成 pooled MSE", "mean_psnr", 'column_mean([r[col]["psnr"] for r in rows])',
     '(psnr_from_mse(math.fsum(r[col]["mse"] * r["h"] * r["w"] for r in rows) / sum(r["h"] * r["w"] for r in rows))'
     ' if rows else NA("empty"))', "Q4 "),
    ("QM5 平均前丟掉 +inf", "column_mean", "vals = list(values)", "vals = [v for v in values if math.isfinite(v)]",
     "Q5 "),
    ("QM6 SSIM 的 data_range 改成 1", "ssim", "data_range=PEAK", "data_range=1", "Q9 "),
    ("QM7 use_sample_covariance 改回 True", "ssim", "use_sample_covariance=False", "use_sample_covariance=True",
     "Q11 "),
    ("QM8 gaussian_weights 改成 False（7×7 均勻視窗）", "ssim", "gaussian_weights=True", "gaussian_weights=False",
     "Q11 "),
    ("QM9 SSIM 改成先轉灰階", "ssim", "x, y, channel_axis=2",
     "cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(y, cv2.COLOR_RGB2GRAY), channel_axis=None", "Q10 "),
    ("QM10 解碼不做 BGR → RGB", "decode_rgb", "cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)", "bgr", "Q14 "),
    ("QM11 配對改成依排序後的 list index", "pair_files",
     'pairs = [(i, table[i]["hazy"], table[i]["gt"]) for i in ids if len(table[i]) == 2]',
     'pairs = list(zip(ids, sorted(t["hazy"] for t in table.values() if "hazy" in t), '
     'sorted(t["gt"] for t in table.values() if "gt" in t)))', "Q12 "),
    ("QM12 FX 的 active0 改成 T_h 127", "fx_active", "**golden.ACTIVE0_DEFAULT,",
     '**{**golden.ACTIVE0_DEFAULT, "t_h": 127},', "Q15 "),
    ("QM13 Δ 直接相減，不檢查是否為有限值", "delta", "not (math.isfinite(fx) and math.isfinite(fl))", "False",
     "Q17 helper 層"),
    # QM14–QM28：2026-11-01 Codex code review 指出、原本 41 項都會通過的改錯
    ("QM14 MSE 的 int64 改成 int16（平方溢位）", "mse", "x.astype(np.int64) - y.astype(np.int64)",
     "x.astype(np.int16) - y.astype(np.int16)", "Q18 "),
    ("QM15 FL 改用 FX 的完整定點設定", "fl_active", "return golden.make_active(**golden.ACTIVE0_DEFAULT)",
     'return golden.make_active(**golden.ACTIVE0_DEFAULT, inv_frac=20, pala_fixed=True, pte_frac=10, sd_mode="lut")',
     "Q19 FL 確實是浮點"),
    ("QM16 run_pair 的 FX 改取 FL 的輸出", "run_pair", '"FX": fx["out"]', '"FX": fl["out"]', "Q19 run_pair 的 FL、FX 兩列"),
    ("QM17 公平性忽略 used", "fairness",
     'return bad + [f"used.{k}" for k in USED_SAME if fl["used"][k] != fx["used"][k]]', "return bad",
     "Q20 公平性負例：used"),
    ("QM18 公平性忽略 ndarray 欄位", "fairness", "np.array_equal(a[k], b[k])", "True", "Q20 公平性負例：stats"),
    ("QM19 run_pair 略過公平性 gate", "run_pair", "if bad:", "if False:", "Q20 run_pair"),
    ("QM20 ΔPSNR 反向（FL − FX）", "pair_row", 'delta(row["FX"]["psnr"], row["FL"]["psnr"])',
     'delta(row["FL"]["psnr"], row["FX"]["psnr"])', "Q21 每對 ΔPSNR"),
    ("QM21 ΔSSIM 反向（FL − FX）", "pair_row", 'delta(row["FX"]["ssim"], row["FL"]["ssim"])',
     'delta(row["FL"]["ssim"], row["FX"]["ssim"])', "Q21 每對 ΔSSIM"),
    ("QM22 合計丟掉最後一筆 ok", "summarize", 'ok = [r for r in rows if r["status"] == "ok"]',
     'ok = [r for r in rows if r["status"] == "ok"][:-1]', "Q21 合計 S"),
    ("QM23 合計納入 error", "summarize", 'ok = [r for r in rows if r["status"] == "ok"]', "ok = list(rows)",
     "Q21 合計 S"),
    ("QM24 |ΔPSNR| 的最大值固定 0", "delta_extremes", '"absmax": max(abs(v) for v in ds)', '"absmax": 0', "Q21 合計極值"),
    ("QM25 ΔPSNR < −0.5 的對數固定 0", "delta_extremes", '"n_worse": sum(v < WORSE for v in ds)', '"n_worse": 0',
     "Q21 合計計數"),
    ("QM26 ΔPSNR < −0.5 改成 ≤ −0.5", "delta_extremes", "v < WORSE", "v <= WORSE", "Q21 合計計數"),
    ("QM27 輸出 SHA-256 都改算 H", "pair_row", "hashlib.sha256(imgs[c].tobytes())", 'hashlib.sha256(imgs["H"].tobytes())',
     "Q22 輸出 SHA-256"),
    ("QM28 報表比例的分子改用 pix（pixel 數）", "_row_line", "f['diff'] / f['n']", "f['pix'] / f['n']", "Q22 報表列"),
    # QM29–QM52：2026-11-01 runner 階段新增（分塊、檔名、格式 gate、一對的讀取、資源 gate、CLI 接線）
    ("QM29 分塊漏掉最後不足一塊的列", "run_mode", "range(0, hazy.shape[0], chunk_rows)",
     "range(0, hazy.shape[0] - chunk_rows + 1, chunk_rows)", "Q16 分塊：FL"),
    ("QM30 分塊的主路徑改用 active0（不是 cpu_update 後的 used）", "run_mode",
     "golden.apply_pass(hazy[r:r + chunk_rows], used)", "golden.apply_pass(hazy[r:r + chunk_rows], active)", "Q16 分塊：FL"),
    ("QM31 run_pair 的 FL 沒把 chunk_rows 傳給 run_mode", "run_pair", "run_mode(hazy, fl_active(), chunk_rows)",
     "run_mode(hazy, fl_active())", "Q16 run_pair(chunk_rows=4)：FL、FX 各做一次"),
    ("QM32 副檔名只接受小寫 .jpg", "parse_ohaze", r"\.(jpg|JPG)", r"\.(jpg)", "Q23 檔名規則"),
    ("QM33 不檢查目錄與角色是否一致", "parse_ohaze", "if len(parts) != 2 or OHAZE_DIRS.get(parts[0]) != role:", "if False:",
     "Q23 檔名符合、但目錄不對"),
    ("QM34 檔名無法解析的影像當成非影像", "list_ohaze", "(unparsed if _is_image(source, n) else other).append(n)",
     "other.append(n)", "Q23 檔名無法解析的影像"),
    ("QM35 不檢查 EXIF orientation", "inspect_image", 'if info["orientation"] not in (None, 1):', "if False:",
     "Q24 EXIF orientation = 6"),
    ("QM36 不檢查 channel 數", "inspect_image", 'if info["channels"] != 3:', "if False:", "Q24 灰階"),
    ("QM37 不檢查 dtype", "inspect_image", 'if info["dtype"] != "uint8":', "if False:", "Q24 16-bit"),
    ("QM38 不檢查 decoded 尺寸", "inspect_image", 'if rgb.shape != (info["h"], info["w"], 3):', "if False:",
     "Q24 decoded 尺寸"),
    ("QM39 decoded RGB 的 hash 改成原始檔的 hash", "inspect_image", 'info["rgb_sha"] = hashlib.sha256(rgb.tobytes()).hexdigest()',
     'info["rgb_sha"] = info["raw_sha"]', "Q24 兩種 hash 分開"),
    ("QM40 metadata 改取 MPO 的最後一個 frame（縮圖）", "inspect_image", "            info.update(format=im.format",
     '            im.seek(getattr(im, "n_frames", 1) - 1)\n            info.update(format=im.format', "Q24 MPO（"),
    # QM41：2026-11-02 讀檔移到 _read 之後，替換字串跟著改；語意相同（GT 繞過格式 gate，借用 hazy 的結果）
    ("QM41 GT 不經過格式 gate", "load_pair", "gt, gi = inspect_image(graw, decode)",
     "gt, gi = decode(graw), dict(hi)", "Q25 GT 也經過同一個格式 gate"),
    ("QM42 不比較 hazy、GT 的尺寸", "load_pair", "elif hazy.shape != gt.shape:", "elif False:", "Q25 hazy 16×24、GT 16×20"),
    ("QM43 開了分塊就略過資源 gate", "resource_gate", "    missing = [k for k, v in",
     "    if chunk_rows:\n        return 1 << 62\n    missing = [k for k, v in", "Q26 分塊不能繞過"),
    ("QM44 不檢查校準時的分塊設定", "resource_gate", "if rss_chunk_rows != chunk_rows:", "if False:", "Q26 校準的分塊設定"),
    ("QM45 ohaze 不在第一對完成後停止", "ohaze_command", "if ran == 1 and not after_first and rest:", "if False:",
     "Q27 ohaze 預設在第一對完成後停止"),
    ("QM46 ohaze 不檢查 pixel 上限", "ohaze_command", 'if info["h"] * info["w"] > p_max:', "if False:",
     "Q27 ohaze 每對 golden 前檢查 pixel 上限"),
    ("QM47 超過 pixel 上限時跳過、繼續下一對", "ohaze_command", "            stop_rc = 3\n            break",
     "            stop_rc = 3\n            continue", "Q27 ohaze 超過上限時停止整次執行"),
    ("QM48 ohaze 沒把 --chunk-rows 傳給 run_pair", "ohaze_command", "run_pair(i, hazy, gt, chunk_rows)",
     "run_pair(i, hazy, gt)", "Q27 ohaze 把 --chunk-rows"),
    ("QM49 N_ok = 0 仍回傳 exit code 0", "ohaze_command", 'return 0 if s["n_ok"] > 0 else 1', "return 0",
     "Q27 ohaze N_ok = 0"),
    ("QM50 預檢遇到 format 仍開始", "ohaze_command", "if fmt_bad:", "if False:", "Q27 ohaze 預檢遇到 format"),
    ("QM51 ohaze 不理會檔名無法解析的影像", "ohaze_command", 'if lst["unparsed"]:', "if False:",
     "Q27 ohaze 有檔名無法解析的影像"),
    ("QM52 calib 沒把 --chunk-rows 傳給 run_pair", "calib_command", "run_pair(0, hazy, gt, chunk_rows)",
     "run_pair(0, hazy, gt)", "Q27 calib：產生"),
    # QM53–QM56：2026-11-02 Codex review 的三項 P2（讀檔的 I/O 錯誤、orientation = 3）
    ("QM53 預檢讀檔失敗仍開始", "ohaze_command", "if read_bad:", "if False:", "Q28 ohaze 預檢讀檔失敗"),
    ("QM54 執行途中讀檔失敗後改成繼續下一對", "ohaze_command", "            stop_rc = 4\n            break",
     "            stop_rc = 4\n            continue", "Q28 ohaze 執行途中讀檔失敗後停止"),
    ("QM55 讀檔把 TimeoutError 當成讀檔失敗吞掉", "_read", "    except TimeoutError:\n        raise\n", "",
     "Q28 讀檔的 TimeoutError"),
    ("QM56 orientation 允許 3（180° 旋轉）", "inspect_image", 'if info["orientation"] not in (None, 1):',
     'if info["orientation"] not in (None, 1, 3):', "Q24 EXIF orientation = 3：不通過"),
    # 以下是結果檔（E Notes 4.11.17 第 6 點）
    ("QM57 正式 writer 寫入前把浮點數四捨五入到 4 位", "_enc", "return float(v)", "return round(v, 4)",
     "Q29 ohaze --record：3 對都寫入並讀回驗證"),
    ("QM58 writer 把 +inf 寫成很大的數字", "_enc", 'return "+inf"', "return 1e308", "Q33 FL 等於 GT"),
    ("QM59 reader 把 \"+inf\" 讀成有限值", "_psnr_val", "return math.inf", "return 1e308", "Q33 FL 等於 GT"),
    ("QM60 reader 不檢查缺少的 key", "_keys", "if miss:", "if False:", "Q30 key 集合的檢查"),
    ("QM61 reader 不檢查未定義的 key", "_keys", "if extra:", "if False:", "Q30 未定義的 key"),
    ("QM62 reader 的整數欄位接受浮點數與 true", "_int", "type(v) is not int", "not isinstance(v, (int, float))",
     "Q30 整數寫成 1.0"),
    ("QM63 reader 接受 NaN、Infinity token", "_strict_json", "parse_constant=_bad_token", "parse_constant=float",
     "Q30 NaN token"),
    ("QM64 reader 接受溢位成無限大的數字", "_finite_float", "if not math.isfinite(v):", "if False:", "Q30 1e999"),
    ("QM65 reader 不比對檔名的 ID 與角色", "_dec_file", "if parse_ohaze(name) != (pid, role):", "if False:",
     "Q30 檔名的 ID 或角色不符（ID 1 的 GT"),
    ("QM66 reader 接受 d_ssim 是 NA", "_dec_row",
     '    d_ssim = _flt(r["d_ssim"], "row.d_ssim")\n    if d_ssim != out["FX"]["ssim"] - out["FL"]["ssim"]:\n'
     '        raise ValueError("row.d_ssim 不等於 FX − FL")',
     '    d_ssim = NA("nonfinite") if r["d_ssim"] == {"NA": "nonfinite"} else _flt(r["d_ssim"], "row.d_ssim")',
     "Q30 d_ssim 是 NA"),
    ("QM67 reader 拒絕有 sec 的 error row", "_dec_row", '_keys(r, "row", ("id", "status", "msg"), ("sec",))',
     '_keys(r, "row", ("id", "status", "msg"))', "Q32 metric 失敗"),
    ("QM68 寫入前不用 reader 驗證", "write_record",
     '        d = decode_line(line[:-1])\n'
     '        if not (_same(d["run"], run) and _same(d["pair"], pair) and _same(d["row"], row)):',
     "        if False:", "Q31 寫入前的驗證不通過"),
    ("QM69 寫入後不讀回比對", "write_record", "if after != before + line:", "if False:", "Q31 讀回的內容和寫入的不同"),
    ("QM70 --record 允許在已有完成紀錄的檔追加", "record_precheck", "if done is not None:", "if False:",
     "Q34 --record 遇到已有完成紀錄的檔"),
    ("QM71 ohaze-complete 不檢查 ID 集合", "complete_command",
     '        probs += _id_problems([r["pair"]["id"] for r in recs], EXPECTED_IDS)', "        probs += []",
     "Q34 ohaze-complete 遇到缺少 ID 45"),
    ("QM72 彙整略過完成紀錄的檢查", "report_command", "probs += _done_problems(done, recs, prefix)", "probs += []",
     "Q34 45 行資料都通過但沒有完成紀錄"),
    ("QM73 彙整不驗證 data_sha256", "_done_problems", 'if done["data_sha256"] != hashlib.sha256(prefix).hexdigest():',
     "if False:", "Q34 完成紀錄的 data_sha256 和資料不符"),
    ("QM74 彙整不檢查重複的 ID", "_id_problems", "dup = sorted({i for i in ids if ids.count(i) > 1})", "dup = []",
     "Q34 重複的 ID（ID 44"),
    ("QM75 彙整不檢查缺少的 ID", "_id_problems", "miss = [i for i in expected if i not in ids]", "miss = []",
     "Q34 缺少的 ID（只有 01–44）"),
    ("QM76 expected IDs 改由資料目錄推定", "report_command", "expected = EXPECTED_IDS",
     'expected = tuple(p[0] for p in lst["pairs"])', "Q34 資料目錄只有 44 對"),
    ("QM77 彙整不比對紀錄的程式與環境", "_run_problems", "if not _same(_code_part(run0), _code_run()):", "if False:",
     "Q34 紀錄的程式或環境"),
    ("QM78 彙整漏掉最後一行資料", "report_command",
     'rows = [r["row"] for r in sorted(recs, key=lambda r: r["pair"]["id"])]',
     'rows = [r["row"] for r in sorted(recs, key=lambda r: r["pair"]["id"])][:-1]', "Q34 正常完成：每列 status = ok"),
    ("QM79 完成紀錄的 digest 少算最後一行資料", "complete_command", '"data_sha256": hashlib.sha256(data).hexdigest()}',
     '"data_sha256": hashlib.sha256(data[: data[:-1].rfind(b"\\n") + 1]).hexdigest()}',
     "Q34 ohaze-complete：寫入完成紀錄並讀回驗證"),
    # QM80–QM83：2026-11-02 Codex code review 的 P2（完成紀錄寫後驗證、三項一致性檢查）
    ("QM80 ohaze-complete 寫入後不讀回比對", "complete_command", "if store.read() != data + line:", "if False:",
     "Q34 ohaze-complete 讀回的內容和寫入的不同"),
    ("QM81 reader 不比對 row 和 pair 的 H×W", "_dec_row", 'if (h, w) != (pair["h"], pair["w"]):', "if False:",
     "Q30 row 的 H×W 和 pair 不同"),
    ("QM82 彙整不比對紀錄的資料目錄", "report_command", 'if run["data_dir"] != source.root:', "if False:",
     "Q34 紀錄的資料目錄和本次不同"),
    ("QM83 彙整不比對紀錄的 P_max 和重算值", "report_command", 'if pm != g["p_max"]:', "if False:",
     "Q34 紀錄的 P_max 1001 和重算的 1000 不同"),
]


def _timeout(signum, frame):
    raise TimeoutError(f"selfcheck 超過 {MUTANT_TIMEOUT} 秒")


def _run_source(source):
    """沿用 golden 的 _run_selfcheck_source（warning 與 numpy 浮點錯誤都忽略，例外單獨回報），另加逾時。
    golden 的 namespace 沒有 __file__；結果檔的 run 要記本檔、uv.lock 與 pyproject.toml 的 hash（4.11.17），
    所以在原始碼最前面設定 __file__ 為本檔路徑（只影響 hash 的讀取位置，不改被測的程式）。"""
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(MUTANT_TIMEOUT)
    try:
        return golden._run_selfcheck_source(f"__file__ = {os.path.abspath(__file__)!r}\n" + source)
    finally:
        signal.alarm(0)


ONLY_EXC = "指定檢查只因呼叫例外而 FAIL，不算抓到"
EXC_STOP = "例外中止，預期項未 FAIL"


def _verdict(rc, fails, error, expect):
    """mutation 的判定。抓到 = selfcheck 跑完（沒有例外跑出 selfcheck）、exit code 1，而且指定檢查正常 FAIL。
    FAIL 標籤帶 EXC_MARK 的是 check_cli 記下的例外（cli 呼叫、前置步驟或條件本身丟的例外）：它讓 selfcheck 能跑完，
    但不算正常 FAIL。"""
    normal = [f for f in fails if expect in f and EXC_MARK not in f]
    only_exc = [f for f in fails if expect in f and EXC_MARK in f]
    if error is None and rc == 1 and normal:
        return "抓到", True
    if error is not None and (normal or only_exc):
        return "預期項 FAIL 後例外中止", False
    if error is not None:
        return EXC_STOP, False
    if only_exc:
        return ONLY_EXC, False
    if rc == 1:
        return "selfcheck 失敗，但預期項沒有 FAIL", False
    return "沒抓到", False


CONTROLS = [
    # 對照組（2026-11-02 兩次 Codex review 的重現）：只讓呼叫丟例外的改錯，都不能算抓到。
    # (名稱, 函式, 原字串, 改成, 指定檢查的標籤含, 預期判定, 預期的 selfcheck 例外開頭（None 表示 selfcheck 必須跑完）)
    ("QC1 ohaze 結尾改成丟 RuntimeError（只有例外）", "ohaze_command", 'return 0 if s["n_ok"] > 0 else 1',
     'raise RuntimeError("對照組：只有例外")', "Q27 ohaze 預設在第一對完成後停止", ONLY_EXC, None),
    ("QC2 同 QC1，指定 Q34 逐對 --record 的檢查", "ohaze_command", 'return 0 if s["n_ok"] > 0 else 1',
     'raise RuntimeError("對照組：只有例外")', "Q34 45 對各自一次 --record", ONLY_EXC, None),
    ("QC3 同 QC1，指定 Q34 報表逐字相同的檢查（依賴一個程序跑完的那次 cli）", "ohaze_command",
     'return 0 if s["n_ok"] > 0 else 1', 'raise RuntimeError("對照組：只有例外")',
     "Q34 正常完成：彙整的報表和一個程序跑完的報表逐字相同", ONLY_EXC, None),
    ("QC4 reader 一開始就丟 RuntimeError，指定 Q29 讀回的檢查", "read_records", 'if data and not data.endswith(b"\\n"):',
     'raise RuntimeError("對照組：reader 只丟例外")\n    if data and not data.endswith(b"\\n"):',
     "Q29 正式 writer → reader", ONLY_EXC, None),
    ("QC5 同 QC4，指定 Q30 拒絕的檢查", "read_records", 'if data and not data.endswith(b"\\n"):',
     'raise RuntimeError("對照組：reader 只丟例外")\n    if data and not data.endswith(b"\\n"):',
     "Q30 缺少必要的 key（row.sec）", ONLY_EXC, None),
    ("QC6 selfcheck 的 recs_of 呼叫 reader 時逾時（TimeoutError）", "selfcheck", "return read_records(data)[0], None",
     'raise TimeoutError("對照組：讀回時逾時")', "Q29 正式 writer → reader", EXC_STOP, "TimeoutError:"),
]


def mutation_command():
    import inspect

    path = __file__
    source = open(path, encoding="utf-8").read()
    print(f"原始碼：{path}（SHA-256 {hashlib.sha256(source.encode()).hexdigest()}）")
    print(f"ate_golden.py：SHA-256 {hashlib.sha256(open(golden.__file__, 'rb').read()).hexdigest()}")
    rc, fails, error, result = _run_source(source)
    print(f"baseline（未改）：exit code {rc}，{result}，FAIL {len(fails)}，例外 {error}")
    if rc != 0 or error:
        return 1

    def run(entry):
        name, func, old, new, expect = entry[:5]
        body = inspect.getsource(globals()[func])
        if body.count(old) != 1 or source.count(body) != 1:
            print(f"{name}：替換位置不唯一，無法執行")
            return None
        rc, fails, error, result = _run_source(source.replace(body, body.replace(old, new)))
        verdict, ok = _verdict(rc, fails, error, expect)
        side = [f for f in fails if EXC_MARK in f and expect not in f]
        return verdict, ok, side, (rc, fails, error, result)

    def show(name, entry, label, res):
        _, func, old, new, expect = entry[:5]
        rc, fails, error, result = res
        print(f"{name}［{label}］")
        print(f"  替換：{func}()：{old!r} → {new!r}")
        print(f"  預期失敗的檢查含：{expect!r}")
        print(f"  selfcheck exit code {rc}，{result}，例外 {error}")
        for f in fails:
            print(f"  FAIL {f}")

    caught = 0
    for entry in MUTATIONS:
        got = run(entry)
        if got is None:
            return 1
        verdict, ok, side, res = got
        caught += ok
        show(entry[0], entry, verdict + (f"；另有 {len(side)} 項非指定檢查的呼叫例外，見下面標 {EXC_MARK} 的 FAIL" if side else ""), res)
    controls = 0
    for entry in CONTROLS:
        got = run(entry)
        if got is None:
            return 1
        verdict, ok, side, res = got
        want, exc_head = entry[5], entry[6]
        error = res[2]
        good = not ok and verdict == want and (error is None if exc_head is None else (error or "").startswith(exc_head))
        controls += good
        show(entry[0], entry, f"對照組：{verdict}；預期「{want}」" + (f"、例外開頭 {exc_head!r}" if exc_head else "")
             + f"；{'符合預期（不算抓到）' if good else '不符合預期'}", res)
    rc, fails, error, result = _run_source(open(path, encoding="utf-8").read())
    print(f"baseline（結束後重跑）：exit code {rc}，{result}，例外 {error}")
    print(f"結果：{caught}/{len(MUTATIONS)} 個 mutation 在預期項目失敗；對照組 {controls}/{len(CONTROLS)} 依預期不算抓到")
    return 0 if caught == len(MUTATIONS) and controls == len(CONTROLS) and rc == 0 and error is None else 1


def _nonneg_int(text):
    v = int(text)
    if v < 0:
        raise argparse.ArgumentTypeError("必須 ≥ 0")
    return v


def _id_list(text):
    return [int(t) for t in text.split(",") if t.strip()]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(description="AVSD Final E 靜態畫質評估：metric、O-HAZE runner、selfcheck 與 mutation"
                                                 "（E Notes 4.11）。命令存在不等於已獲准執行")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selfcheck", help="Q1–Q34：只用記憶體中的小圖、記憶體編碼的 fixture 與測試替身（含結果檔替身），不讀資料、不寫檔")
    sub.add_parser("mutation", help="QM1–QM83：逐一改錯 production 函式，確認 selfcheck 在預期項目正常 FAIL；對照組 QC1–QC6 只丟例外或逾時，"
                                       "必須判定為不算抓到")
    pp = sub.add_parser("pairs", help="只讀資料：配對清單、格式 gate、尺寸與兩種 SHA-256；不呼叫 golden、不算畫質、不寫檔")
    pp.add_argument("dir", help="zip 解壓出的上層目錄（下面是 hazy/、GT/）")
    po = sub.add_parser("ohaze", help="計分（執行要使用者另外批准）：缺少資源條件時拒絕開始，預設第一對完成後停止")
    po.add_argument("dir", help="zip 解壓出的上層目錄（下面是 hazy/、GT/）")
    po.add_argument("--ids", type=_id_list, default=None, help="逗號分隔的場景編號；不給時是清單中全部的編號（依編號）")
    po.add_argument("--chunk-rows", type=_nonneg_int, default=0, help="主路徑每幾列一塊；0 是不分塊")
    po.add_argument("--rss-a", type=float, help="calib 兩點校準擬合的 a（bytes）")
    po.add_argument("--rss-b", type=float, help="calib 兩點校準擬合的 b（bytes/pixel）")
    po.add_argument("--rss-chunk-rows", type=_nonneg_int, help="校準時用的分塊設定，要和 --chunk-rows 相同")
    po.add_argument("--budget-gib", type=float, help="使用者確認的峰值 RSS 預算（GiB）；沒有預設值")
    po.add_argument("--margin", type=float, help="使用者確認的餘裕倍數（≥ 1）；沒有預設值")
    po.add_argument("--after-first", action="store_true", help="第一對之後繼續；只在使用者批准第二對之後的階段使用")
    po.add_argument("--record", help="結果檔（E Notes 4.11.17）：每對完成後追加一行並讀回驗證；不給時不寫任何檔")
    pk = sub.add_parser("ohaze-complete", help="追加完成紀錄（4.11.17；只在外層迴圈的 45 對都通過事後檢查後執行）")
    pk.add_argument("record", help="ohaze --record 寫的結果檔")
    pr = sub.add_parser("ohaze-report", help="讀回結果檔並彙整（4.11.17）：完成紀錄、ID、run、檔名與資料目錄都通過才印合計")
    pr.add_argument("record", help="ohaze --record 寫的結果檔")
    pr.add_argument("dir", help="zip 解壓出的上層目錄（只列檔名，不解碼）")
    pc = sub.add_parser("calib", help="RSS 校準用的合成圖執行（執行要使用者另外批准；要在 /usr/bin/time -l 下跑）")
    pc.add_argument("h", type=int)
    pc.add_argument("w", type=int)
    pc.add_argument("--chunk-rows", type=_nonneg_int, default=0, help="和之後 ohaze 要用的分塊設定相同")
    pc.add_argument("--seed", type=int, default=20261101)
    args = parser.parse_args(argv)
    if args.command == "selfcheck":
        return selfcheck()
    if args.command == "mutation":
        return mutation_command()
    if args.command == "calib":
        return calib_command(argv, args.h, args.w, args.chunk_rows, args.seed)
    if args.command == "ohaze-complete":
        return complete_command(RecordFile(args.record), argv)
    try:
        source = DirSource(args.dir)
    except ValueError as e:
        print(f"停止：{e}")
        return 2
    if args.command == "pairs":
        return pairs_command(source, argv)
    if args.command == "ohaze-report":
        return report_command(RecordFile(args.record), source, argv)
    gate = {"rss_a": args.rss_a, "rss_b": args.rss_b, "rss_chunk_rows": args.rss_chunk_rows,
            "budget_gib": args.budget_gib, "margin": args.margin}
    store = RecordFile(args.record) if args.record else None
    return ohaze_command(source, argv, args.ids, args.chunk_rows, gate, args.after_first, store)


if __name__ == "__main__":
    sys.exit(main())
