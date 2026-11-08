"""AVSD Final E golden model（階段 2 統計路徑、階段 3 主路徑、階段 4 cpu_update 與 frame 排程；浮點，軟體 v0 暫定）。

依 Upadhyay & Sarawadekar, IEEE TVLSI 2026（adaptive transmission estimation，
本檔名 ate 的由來）Eq.2、7、8 與 Algorithm 1 step 2–3，以及 E Notes 4.6 第 2 版
的暫定規則撰寫。規則是使用者 2026-10-29 批准的軟體 v0 暫定契約，尚未經 D 或
團隊同意為 RTL 介面。

暫定規則（對應 E Notes 4.6.6 的 P 編號）：
- 輸入是 RGB、uint8、形狀 H×W×3（P14 之前的軟體約定）。
- P5：抽樣，取從 0 起算、列與行都是 0、3、6… 的位置；影像內的位置全部保留，
  降採樣尺寸為 ceil(H/3) × ceil(W/3)。
- P11：dark channel 先取 RGB 最小值，再取以該位置為中心的 5×5 最小值；
  影像外補 255。A 取所選位置本身的 RGB。
- P19：dark >= T_h_prev 歸遠群（far），否則歸近群（near）。
- P8：同群並列最大時取 raster 順序第一個；空群組回報 valid=False，
  不以 RGB=0 表示。
- P12：gray_sum 累加 R、G、B 原值，累加前轉成 int64；
  平均灰階 = floor(gray_sum / (3N))。
- T_h_prev 由呼叫者明確提供（第一張 frame 的初值 P7 不在本階段）。

階段 3 主路徑（Eq.6、9–13）的暫定數值規則（AI 推薦，記為暫定，不是團隊決定）：
- P12：I_H、I_max 都用三色和（= 三色平均 × 3）；η 是比值，結果和用平均相同。
- P20：ω = 0.9375、γ = 2，取論文 p.6 的值，固定為常數。
- P10：A^c = 0 的顏色不參與 min_c(I^c/A^c)，三色都為 0 時該項取 0；
  min + max = 0（純黑）時 k = 1；t_m 截到 [1/1023, 1] 再進 SD。
- P13：浮點運算到最後，輸出時四捨五入（.5 進位）並截到 0–255。
- 單張展示的參數（不是團隊 cpu_update，沒有 step 4）：T_h_prev 用該圖自己的
  平均灰階；某一群為空時，用另一群的 A。

階段 4（cpu_update、影片與靜態排程）的暫定規則寫在「CPU 端與 frame 排程」段開頭。

不包含：定點、LUT、閃爍指標、PSNR／SSIM。

用法（在 dehaze-e/ 執行）：
    uv run --locked python -B ate_golden.py selfcheck
    uv run --locked python -B ate_golden.py stats <image> --t-h-prev 128 [--t-h-prev ...]
    uv run --locked python -B ate_golden.py demo <image> [--out-dir DIR] [--compare DCP輸出]
    uv run --locked python -B ate_golden.py video <image> [<image> ...] [--repeat K]
    uv run --locked python -B ate_golden.py static <image>
    uv run --locked python -B ate_golden.py video|static ... --inv-frac 16|20   # reciprocal 局部量化實驗
    uv run --locked python -B ate_golden.py video|static ... --inv-frac 16|20 --pala-fixed   # PALA 局部定點模型
    uv run --locked python -B ate_golden.py video|static ... --inv-frac 16|20 --pala-fixed --pte-frac 10|16   # PTE 理想定點 reference
    uv run --locked python -B ate_golden.py video|static ... --inv-frac 16|20 --pala-fixed --pte-frac 10|16 --sd ideal   # 理想整數 SD
    uv run --locked python -B ate_golden.py video|static ... --inv-frac 16|20 --pala-fixed --pte-frac 10 --sd lut       # SD 候選 A（直接 LUT）
    uv run --locked python -B ate_golden.py video|static ... [--tie last] [--th-round ceil]   # 統計規則的影響（和預設規則比較）
    uv run --locked python -B ate_golden.py mutation
"""

import argparse
import hashlib
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

STEP = 3      # P5：抽樣間隔
WINDOW = 5    # Eq.2 在降採樣圖上的視窗大小（論文 p.3）
PAD = 255     # P11：影像外的值
TIE_RULES = ("first", "last")    # P8：並列時取 raster 第一個（預設）或最後一個（Fig.4 的 ≤ 可能是這樣，判讀）
TH_ROUNDS = ("floor", "ceil")    # 下一張 T_h：floor（預設）或 ceil；ceil 配 dark >= T_h 等於 Eq.7 的實數比較
OMEGA = 0.9375    # P20：論文 p.6
GAMMA = 2         # P20：論文 p.6
T_MIN = 1 / 1023  # P10：進 SD 前 t_m 的下限


def _check_frame(frame):
    if not (isinstance(frame, np.ndarray) and frame.dtype == np.uint8
            and frame.ndim == 3 and frame.shape[2] == 3
            and frame.shape[0] > 0 and frame.shape[1] > 0):
        raise ValueError("frame 必須是 uint8、形狀 H×W×3 的 RGB 陣列")


def downsample(frame):
    """P5：取列、行都是 3 的倍數的位置 → ceil(H/3) × ceil(W/3) × 3。"""
    _check_frame(frame)
    return frame[::STEP, ::STEP]


def dark_channel(frame_down):
    """Eq.2：RGB 最小值，再取中心對齊 5×5 最小值，外圍補 255（P11）。"""
    half = WINDOW // 2
    padded = np.pad(frame_down.min(axis=2), half, constant_values=PAD)
    return sliding_window_view(padded, (WINDOW, WINDOW)).min(axis=(2, 3))


def gray_sum(frame_down):
    """P12：R、G、B 原值總和；先轉 int64，避免 uint8 相加溢位。"""
    return int(frame_down.astype(np.int64).sum())


def _group(mask, dark, frame_down, tie="first"):
    count = int(mask.sum())
    if count == 0:
        return {"valid": False, "count": 0, "max_dark": None, "pos": None, "rgb": None}
    # 群外設為 -1，dark 為 0 的群組也能被選到；argmax 取第一個最大值（raster 順序），tie = "last" 時取最後一個
    masked = np.where(mask, dark.astype(np.int16), -1)
    idx = int(masked.argmax()) if tie == "first" else masked.size - 1 - int(masked.ravel()[::-1].argmax())
    r, c = np.unravel_index(idx, dark.shape)
    return {"valid": True, "count": count, "max_dark": int(dark[r, c]),
            "pos": (int(r), int(c)), "rgb": tuple(int(v) for v in frame_down[r, c])}


def group_max(dark, frame_down, t_h_prev, tie="first"):
    """Eq.8：依 T_h_prev 分遠群、近群，各取 dark 最大位置的 RGB（P8、P19）。"""
    far = dark >= t_h_prev
    return {"far": _group(far, dark, frame_down, tie), "near": _group(~far, dark, frame_down, tie)}


def stats_pass(frame, t_h_prev, tie="first"):
    """統計路徑一次掃描：降採樣、dark channel、灰階總和、遠近兩群最大點。tie 是並列規則（TIE_RULES）。"""
    if not (type(t_h_prev) is int and 0 <= t_h_prev <= 255):   # 排除 bool 與 numpy 整數
        raise ValueError("t_h_prev 必須是 0–255 的 Python int")
    if type(tie) is not str or tie not in TIE_RULES:       # 和 make_active 相同：只接受 Python str
        raise ValueError(f"tie 必須是 {TIE_RULES} 之一的 Python str：{tie!r}")
    down = downsample(frame)
    dark = dark_channel(down)
    total = gray_sum(down)
    n = down.shape[0] * down.shape[1]
    return {"t_h_prev": t_h_prev, "shape_down": down.shape[:2], "n": n,
            "gray_sum": total, "gray_mean": total // (3 * n),
            **group_max(dark, down, t_h_prev, tie), "frame_down": down, "dark": dark}


# ---------------------------------------------------------------- 主路徑（階段 3）

def pala(frame, a_far, a_near, inv_i_max):
    """Eq.9–10：灰階 >= I_max 用 A_F，否則 (1−η)·A_F + η·A_N（P12：灰階用三色和）。

    η = (I_max − I_H) × inv_i_max，inv_i_max 是 CPU 寫入的 1/I_max（P20），主路徑不自己做除法。
    η 只在加權時有意義；灰階 >= I_max 的 pixel 回報 η = 0。
    """
    s = frame.astype(np.int64).sum(axis=2)
    s_max = sum(a_far)
    use_far = s >= s_max                     # I_max = 0 時全部為 True，不會用到 inv_i_max
    eta = np.where(use_far, 0.0, (s_max - s) * inv_i_max)
    if np.any(eta > 1):                      # 理想 1/I_max 不會發生；量化後的 1/I_max 才可能
        raise ValueError(f"η 超過 1（最大 {float(eta.max())!r}）；saturation 規則待 C 決定（E Notes 5.4 第 12 題）")
    af, an = np.array(a_far, float), np.array(a_near, float)
    a_map = np.where(use_far[..., None], af, (1 - eta)[..., None] * af + eta[..., None] * an)
    return eta, a_map


def pala_fixed(frame, a_far, a_near, inv_q, inv_frac):
    """PALA 局部定點模型（軟體暫定契約，不是論文或團隊的硬體規格）。只用 int64 整數運算。

    輸入：frame RGB 8-bit；A_F、A_N 各 8-bit；inv_q = floor(2^F / I_max)，F = 16 或 20。
    S = R + G + B（0–765，10-bit）；I_max = A_F 三色和（0–765）。
    S < I_max：delta = I_max − S（1–765）；product = delta × inv_q <= I_max × inv_q <= 2^F
               （F + 1 bit：F=16 時 17-bit、F=20 時 21-bit）；eta_q = product >> (F − 8)，floor，0–256（9-bit）。
    S >= I_max（含 I_max = 0）：eta_q = 0（診斷值），不做乘法。
    A_q[c] = (256 − eta_q) × A_F[c] + eta_q × A_N[c]：兩項各 <= 256 × 255，和 <= 65280（16-bit），
             A_q / 256 是 A，保留 8 個小數 bit，不提前捨成 8-bit。
    """
    s = frame.astype(np.int64).sum(axis=2)
    i_max = sum(a_far)
    below = s < i_max                          # I_max = 0 時全部是 False
    delta = np.where(below, i_max - s, 0)
    product = delta * np.int64(inv_q)
    eta_q = product >> (inv_frac - 8)          # 非 below 的 product 是 0，eta_q 也是 0
    af, an = np.array(a_far, np.int64), np.array(a_near, np.int64)
    a_q = (256 - eta_q)[..., None] * af + eta_q[..., None] * an
    return {"product": product, "eta_q": eta_q, "a_q": a_q}


def pte(frame, a_map):
    """Eq.12–13：t_m = 1 − ω·min_c(I^c/A^c)·k，k = 1 − (min/(min+max))^γ（P10 處理零分母）。"""
    i = frame.astype(float)
    lo, hi = i.min(axis=2), i.max(axis=2)
    ratio = np.divide(lo, lo + hi, out=np.zeros_like(lo), where=(lo + hi) > 0)
    k = 1.0 - ratio ** GAMMA
    q = np.divide(i, a_map, out=np.full_like(i, np.inf), where=a_map > 0)
    m = q.min(axis=2)
    m = np.where(np.isinf(m), 0.0, m)        # 三色的 A 都是 0
    return k, m, 1.0 - OMEGA * m * k


PTE_FRACS = (10, 16)        # 定點 PTE 的 t 小數位數 T（E Notes 4.9，軟體暫定）
T_MIN_FIXED = 1 / 1024      # 定點 PTE 的 t 下限 = t_min_q / 2^T，t_min_q = 2^(T − 10)；不等於浮點的 1/1023
M_SAT = (1 << 17) - 1       # m_q 的飽和值（m < 2）


def pte_fixed(frame, a_q, t_frac):
    """PTE 理想定點 reference（E Notes 4.9，軟體暫定契約，不是 LUT 硬體算法或 RTL 規格）。只用 int64 整數運算。

    輸入：frame RGB 8-bit；a_q 是定點 PALA 的 A_q（0–65280，A = A_q / 256）；t_frac = T ∈ {10, 16}。
    中間值小數 W = 16。各步的位寬由值域推導，不依賴 int64 的餘裕：
    - lo、hi = 三色最小、最大（8-bit）；s = lo + hi（0–510，9-bit）。
    - r_q = floor(lo·2^16 / s)，s = 0（純黑，lo = 0）時 0；r = r_q / 2^16，0–32768（0.5），16-bit。
    - r2_q = (r_q·r_q) >> 16，floor；乘積 <= 2^30（31-bit），結果 0–16384（15-bit）。
    - k_q = 2^16 − r2_q，49152–65536（0.75–1.0），17-bit。
    - m_c = floor(I^c·2^24 / A_q[c])，A_q[c] > 0；I/A = 256·I / A_q，分子 <= 255·2^24 < 2^32，分母 16-bit。
      A_q[c] = 0 的 channel 不參加（診斷值 −1）。
    - m_q = min(min_c m_c, 2^17 − 1)，三色的 A_q 都是 0 時 0；0–131071（< 2），17-bit。
      m >= 1.4209 時 t 一定截到下限，所以飽和不改變 t。
    - p = 15·m_q·k_q，= ω·m·k·2^36 的精確值（ω = 15/16 不捨入），< 15·2^17·2^16 < 2^37（37-bit）。
    - u = p >> (36 − T)，floor；u < 1.875·2^T（T + 1 bit）。
    - t_q = u >= 2^T − t_min_q ? t_min_q : 2^T − u，t_min_q = 2^(T − 10)；t_min_q–2^T（T + 1 bit），1.0 = 2^T。
      相對於已量化的 m_q、k_q，未截斷的 t_q = 2^T − floor(ω·m·k·2^T) = ceil((1 − ω·m·k)·2^T)，
      也就是 t 向上取整；不是「全部向下取整」。
    """
    i = frame.astype(np.int64)
    lo, hi = i.min(axis=2), i.max(axis=2)
    s = lo + hi
    r_q = np.where(s > 0, (lo << 16) // np.maximum(s, 1), 0)
    r2_q = (r_q * r_q) >> 16
    k_q = (1 << 16) - r2_q
    a_q = np.asarray(a_q, np.int64)
    valid = a_q > 0
    m_c = np.where(valid, (i << 24) // np.maximum(a_q, 1), -1)
    m_q = np.where(valid.any(axis=2), np.minimum(np.where(valid, m_c, M_SAT).min(axis=2), M_SAT), 0)
    p = 15 * m_q * k_q
    u = p >> (36 - t_frac)
    one, t_min_q = 1 << t_frac, 1 << (t_frac - 10)
    t_q = np.where(u >= one - t_min_q, t_min_q, one - u)      # 先比較再相減，不產生負數
    return {"r_q": r_q, "r2_q": r2_q, "k_q": k_q, "m_c": m_c, "m_q": m_q, "p": p, "u": u, "t_q": t_q}


SD_MODES = ("ideal", "lut")   # 定點 SD 的模式（E Notes 4.10，軟體暫定）：理想整數 reference；候選 A 直接 LUT
SD_LUT_K = 10                 # 候選 A 的 reciprocal 小數位數 K（第一輪只做 K = 10、T = 10）


def sd_fixed(frame, a_q, t_q, t_frac, mode, k=SD_LUT_K):
    """SD 定點（E Notes 4.10，軟體暫定契約，不是論文或團隊的硬體規格）。只用 int64 整數運算。

    輸入：frame RGB 8-bit；a_q（0–65280，A = A_q / 256）；t_q（2^(T−10)–2^T，t = t_q / 2^T，三色共用）。
    輸出 out = clip(floor(J + 1/2), 0, 255)，J = (I − A)/t + A；只在最後 rounding 一次，不加其他中間截位。
    - ideal：N = (256·I − A_q)·2^T + A_q·t_q，D = 256·t_q，rounded = floor((2N + D) / (2D))。
      N：T = 10 時 27-bit signed、T = 16 時 33-bit signed；D <= 2^24。
    - lut（候選 A，只用於 T = 10）：inv_t_q = floor(2^(K+T) / t_q + 1/2)（2^K–2^(K+10)，K + 11 bit），
      P = (256·I − A_q)·inv_t_q + A_q·2^K（K + 27-bit signed），rounded = (P + 2^(K+7)) >> (K + 8)（floor）。
    負數的 rounding 用 floor；J < 0 時 clip 後一律是 0。sd_j 是 clip 前的 J（float64：ideal 是 N/D 的最接近值，lut 的 P/2^(K+8) 精確）；
    sd_tie 標出 J 剛好是 k + 1/2 的 channel。
    """
    i = frame.astype(np.int64)
    a_q = np.asarray(a_q, np.int64)
    tq = np.asarray(t_q, np.int64)[..., None]
    delta = 256 * i - a_q
    if mode == "ideal":
        n = delta * (1 << t_frac) + a_q * tq
        d = 256 * tq
        rounded = (2 * n + d) // (2 * d)
        fx = {"sd_n": n, "sd_d": d[..., 0], "sd_j": n / d, "sd_tie": (2 * n + d) % (2 * d) == 0}
    else:
        inv = ((1 << (k + t_frac + 1)) + tq) // (2 * tq)       # LUT 內容：nearest，tie 向上
        p = delta * inv + (a_q << k)
        rounded = (p + (1 << (k + 7))) >> (k + 8)
        fx = {"sd_inv_q": inv[..., 0], "sd_p": p, "sd_j": p / (1 << (k + 8)), "sd_tie": (p + (1 << (k + 7))) % (1 << (k + 8)) == 0}
    out = np.clip(rounded, 0, 255).astype(np.uint8)
    return {**fx, "sd_rounded": rounded, "out": out}


def sd(frame, a_map, t_raw, t_min=T_MIN):
    """Eq.6：J = (I − A)/t + A，t 先截到 [t_min, 1]；輸出四捨五入並截到 0–255（P13）。

    t_min 預設是浮點的 T_MIN = 1/1023；定點 PTE 模式傳 1/1024，避免把定點的下限悄悄改回 1/1023。
    """
    t = np.clip(t_raw, t_min, 1.0)
    j = (frame.astype(float) - a_map) / t[..., None] + a_map
    if not np.all(np.isfinite(j)):           # 非有限值不得轉成 uint8
        raise ValueError("J 含 inf 或 NaN，不轉成 8-bit 輸出")
    out = np.clip(np.floor(j + 0.5), 0, 255).astype(np.uint8)
    return t, j, out


def _as_int(v, name):
    """先檢查是 0–255 的整數（Python int 或 numpy 整數，不含 bool），再轉成 Python int。"""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not 0 <= v <= 255:
        raise ValueError(f"{name} 必須是 0–255 的整數：{v!r}")
    return int(v)


def _as_rgb(a, name):
    """把 A 轉成 3 個 Python int（0–255）；避免 numpy uint8 在相加時溢位。"""
    values = tuple(a)                        # 只讀一次，iterator 也能檢查每個元素
    if len(values) != 3:
        raise ValueError(f"{name} 必須是 3 個 0–255 的整數")
    return tuple(_as_int(v, name) for v in values)


def apply_pass(frame, active):
    """主路徑一次處理：PALA → PTE → SD，使用 active 的 A_F、A_N、1/I_max。回傳輸出與所有中間值。"""
    _check_frame(frame)
    a_far, a_near = _as_rgb(active["a_far"], "a_far"), _as_rgb(active["a_near"], "a_near")
    inv = active["inv_i_max"]
    if isinstance(inv, (bool, np.bool_)) or not isinstance(inv, (int, float, np.floating)) \
            or not np.isfinite(inv) or not 0 <= inv <= 1:
        # I_max >= 1 時 1/I_max <= 1；只限制範圍，不要求精確等於 1/I_max（之後要量化）
        raise ValueError(f"inv_i_max 必須是 0–1 的有限數：{inv!r}")
    fx = {}
    if active["pala_fixed"]:
        frac, q = active["inv_frac"], active["inv_i_max_q"]
        if frac not in INV_FRACS or type(q) is not int or not 0 <= q <= 1 << frac:
            raise ValueError(f"定點 PALA 需要 F ∈ {INV_FRACS} 與 0–2^F 的 Python int q：{frac!r}, {q!r}")
        fx = pala_fixed(frame, a_far, a_near, q, frac)
        if np.any(fx["eta_q"] > 256):
            raise ValueError("eta_q 超過 256（1.0）")
        # 轉接到浮點 PTE／SD：eta = eta_q / 256、A = A_q / 256（float64 可精確表示）
        eta, a_map = fx["eta_q"] / 256, fx["a_q"] / 256
    else:
        eta, a_map = pala(frame, a_far, a_near, float(inv))
    tf = active["pte_frac"]
    if tf is not None:
        if tf not in PTE_FRACS or not active["pala_fixed"] or active["t_min"] != T_MIN_FIXED:
            raise ValueError(f"定點 PTE 需要 T ∈ {PTE_FRACS}、定點 PALA 與 t 下限 1/1024：{tf!r}, {active['t_min']!r}")
        fx.update(pte_fixed(frame, fx["a_q"], tf))
        # 轉接到浮點 SD：k、m、t 都是 2 的次方分母，float64 可精確表示；t_q 已截斷，sd 的截斷不改值
        k, m, t_raw = fx["k_q"] / (1 << 16), fx["m_q"] / (1 << 16), fx["t_q"] / (1 << tf)
    else:
        k, m, t_raw = pte(frame, a_map)
    t, j, out = sd(frame, a_map, t_raw, active["t_min"])
    sm = active["sd_mode"]
    if sm is not None:
        if tf is None or sm not in SD_MODES or (sm == "lut" and tf != 10):
            raise ValueError(f"定點 SD 需要定點 PTE；lut 只用於 T = 10：{sm!r}, {tf!r}")
        fx.update(sd_fixed(frame, fx["a_q"], fx["t_q"], tf, sm))
        j, out = fx.pop("sd_j"), fx.pop("out")      # j 改成定點 SD 的 J（clip 前），out 是定點 SD 的輸出
    return {"eta": eta, "a_map": a_map, "k": k, "m": m, "t_raw": t_raw, "t": t, "j": j, "out": out, **fx}


def single_frame_params(frame):
    """單張展示用的參數（暫定，不是團隊 cpu_update）：T_h_prev 取該圖自己的平均灰階。"""
    t_h = stats_pass(frame, 0)["gray_mean"]   # gray_mean 和 T_h_prev 無關
    s = stats_pass(frame, t_h)
    far, near = s["far"], s["near"]          # 兩群不會同時為空
    a_far = far["rgb"] if far["valid"] else near["rgb"]
    a_near = near["rgb"] if near["valid"] else far["rgb"]
    return s, a_far, a_near


def dehaze_single(frame):
    s, a_far, a_near = single_frame_params(frame)
    return s, a_far, a_near, apply_pass(frame, make_active(s["t_h_prev"], a_far, a_near, measured=True))


# ---------------------------------------------------------------- CPU 端與 frame 排程（階段 4）
#
# 暫定規則（AI 推薦，記為暫定，不是團隊決定；對應 E Notes 4.6.3、4.6.6）：
# - active：一張 frame 處理時生效的參數，即團隊草稿 shadow 切換後的內容
#   （T_h、A_F、A_N、I_max 與 1/I_max）。
# - P12、P19：下一張的 T_h = 本張 stats 的 gray_mean = floor(gray_sum / (3N))。
# - P9（step 4）：A_F、A_N 分開判斷；三色各自和「上一個 active」比，任一色的
#   絕對差 > 3 就三色都取 floor((本張 + 上一個) / 2)，否則直接用本張。
# - P8：某一群為空時沿用上一個 active 的值。
# - P7：active0 是 CPU 寫入的預設參數，標記 measured=False。第一次更新時，上一個
#   值不是量測結果，所以不做 step 4；若某一群為空，改用另一群本張的值。
# - P20：I_max = step 4 之後 A_F 的三色和；1/I_max 為浮點，I_max = 0 時寫 0 當哨兵值。

ACTIVE0_DEFAULT = {"t_h": 128, "a_far": (255, 255, 255), "a_near": (255, 255, 255)}   # P7 暫定

# reciprocal 局部量化實驗（E Notes 4.8，軟體暫定，不是 RTL 規格）：只把 1/I_max 換成
# q / 2^F，q = floor(2^F / I_max)；I_max = 0 時 q = 0；I_max = 1 時 q = 2^F（恰好 1.0）。
# PALA／PTE／SD 仍是浮點。inv_frac = None 是原本的浮點 1/I_max。
INV_FRACS = (16, 20)


def inv_imax_q(i_max, inv_frac):
    """q = floor(2^F / I_max)，只用 Python int；I_max = 0 時回傳 0（哨兵值）。"""
    if type(i_max) is not int or not 0 <= i_max <= 765 or type(inv_frac) is not int or inv_frac not in INV_FRACS:
        raise ValueError(f"I_max 必須是 0–765 的 Python int，F 必須是 {INV_FRACS} 之一的 Python int：{i_max!r}, {inv_frac!r}")
    return (1 << inv_frac) // i_max if i_max else 0


def make_active(t_h, a_far, a_near, measured=False, inv_frac=None, pala_fixed=False, pte_frac=None, t_min=None,
                sd_mode=None, tie="first", th_round="floor"):
    """inv_frac 是 None 時 1/I_max 用浮點；是 16 或 20 時用 q / 2^F（float64 可精確表示）。

    pala_fixed、pte_frac、t_min、sd_mode 是模型設定（不是暫存器）：
    - sd_mode：None 是浮點 SD；"ideal" 是理想整數 SD；"lut" 是候選 A（直接 LUT，只用於 T = 10）。需要 pte_frac。
    - tie、th_round：統計路徑的並列規則與下一張 T_h 的取整（TIE_RULES、TH_ROUNDS），用來比較規則的影響。
    - pala_fixed：True 時主路徑的 PALA 用 pala_fixed，需要 inv_frac。
    - pte_frac：10 或 16 時 PTE 用 pte_fixed（t 的小數位數 T），需要 pala_fixed。
    - t_min：SD 前 t 的下限。None 時浮點 PTE 用 T_MIN（1/1023）、定點 PTE 用 T_MIN_FIXED（1/1024）；
      浮點 PTE 也可明確指定 T_MIN_FIXED，只供比較 T_MIN 修改的影響。
    """
    if not (type(t_h) is int and 0 <= t_h <= 255):
        raise ValueError("t_h 必須是 0–255 的 Python int")
    if type(pala_fixed) is not bool or (pala_fixed and inv_frac is None):
        raise ValueError(f"pala_fixed 必須是 bool，且 True 時需要 inv_frac：{pala_fixed!r}, {inv_frac!r}")
    if pte_frac is not None and (type(pte_frac) is not int or pte_frac not in PTE_FRACS or not pala_fixed):
        raise ValueError(f"pte_frac 必須是 {PTE_FRACS} 之一的 Python int，且需要 pala_fixed：{pte_frac!r}")
    if t_min is None:
        t_min = T_MIN if pte_frac is None else T_MIN_FIXED
    if type(t_min) is not float or t_min not in (T_MIN, T_MIN_FIXED) or (pte_frac is not None and t_min != T_MIN_FIXED):
        raise ValueError(f"t_min 必須是 1/1023 或 1/1024；定點 PTE 只能用 1/1024：{t_min!r}")
    if sd_mode is not None and (type(sd_mode) is not str or sd_mode not in SD_MODES or pte_frac is None
                                or (sd_mode == "lut" and pte_frac != 10)):
        raise ValueError(f"sd_mode 必須是 {SD_MODES} 之一，需要 pte_frac；lut 只用於 T = 10：{sd_mode!r}, {pte_frac!r}")
    if type(tie) is not str or tie not in TIE_RULES or type(th_round) is not str or th_round not in TH_ROUNDS:
        raise ValueError(f"tie 必須是 {TIE_RULES} 之一、th_round 必須是 {TH_ROUNDS} 之一：{tie!r}, {th_round!r}")
    a_far, a_near = _as_rgb(a_far, "a_far"), _as_rgb(a_near, "a_near")
    i_max = sum(a_far)
    if inv_frac is None:
        q, inv = None, 1 / i_max if i_max > 0 else 0.0
    else:
        q = inv_imax_q(i_max, inv_frac)
        inv = q / (1 << inv_frac)
    return {"t_h": t_h, "a_far": a_far, "a_near": a_near, "i_max": i_max,
            "inv_i_max": inv, "inv_frac": inv_frac, "inv_i_max_q": q, "pala_fixed": pala_fixed, "pte_frac": pte_frac,
            "t_min": t_min, "sd_mode": sd_mode, "tie": tie, "th_round": th_round, "measured": measured}


def _step4(raw, prev):
    diff = max(abs(r - p) for r, p in zip(raw, prev))
    if diff > 3:
        return tuple((r + p) // 2 for r, p in zip(raw, prev)), True, diff
    return raw, False, diff


def cpu_update(stats, active):
    """由本張 stats 與目前的 active 算出下一張的 active（D 的 ISR 在 frame 之間做的事）。"""
    raw = {g: _as_rgb(stats[g]["rgb"], g) if stats[g]["valid"] else None for g in ("far", "near")}
    prev = {"far": active["a_far"], "near": active["a_near"]}
    # T_h：floor 是 gray_mean；ceil 配 dark >= T_h 等於和實數平均 gray_sum / (3N) 比較（Eq.7）
    t_next = stats["gray_mean"] if active["th_round"] == "floor" else -(-stats["gray_sum"] // (3 * stats["n"]))
    used, info = {}, {"t_h_next": t_next}
    for g, other in (("far", "near"), ("near", "far")):
        triggered, diff = False, None
        if raw[g] is None and not active["measured"]:
            used[g], action = raw[other], "空群組，首次更新：用另一群本張的值"
        elif raw[g] is None:
            used[g], action = prev[g], "空群組：沿用上一個"
        elif not active["measured"]:
            used[g], action = raw[g], "首次更新：直接採用"
        else:
            used[g], triggered, diff = _step4(raw[g], prev[g])
            action = f"差 {diff} > 3：與上一個平均" if triggered else f"差 {diff} <= 3：採用本張"
        info[g] = {"raw": raw[g], "used": used[g], "action": action, "diff": diff, "triggered": triggered}
    return make_active(_as_int(t_next, "gray_mean"), used["far"], used["near"], measured=True,
                       inv_frac=active["inv_frac"], pala_fixed=active["pala_fixed"],
                       pte_frac=active["pte_frac"], t_min=active["t_min"], sd_mode=active["sd_mode"],
                       tie=active["tie"], th_round=active["th_round"]), info


def _frame_record(n, frame, active):
    """統計一張 frame（用 active 的 T_h 分群），再算出下一個 active。

    stats_active 是統計 pass 用的 active；used 是除霧 pass 用的 active，由排程填入。
    """
    stats = stats_pass(frame, active["t_h"], active["tie"])
    nxt, info = cpu_update(stats, active)
    return {"n": n, "frame": frame, "stats_active": active, "stats": stats, "update": info, "next": nxt}


def run_video(frames, active0):
    """影片排程：frame n 用 active(n) 除霧並統計；active(n+1) = cpu_update(stats(n), active(n))。"""
    records, active = [], active0
    for n, frame in enumerate(frames):
        rec = _frame_record(n, frame, active)
        rec["used"] = active                 # 影片：統計與除霧是同一次讀取、同一個 active
        rec["result"] = apply_pass(frame, rec["used"])
        records.append(rec)
        active = rec["next"]
    return records


def run_static(image, active0):
    """靜態排程（團隊草稿 §5.3）：第一次讀取只統計（用 active0 的 T_h 分群），
    CPU 更新後，第二次讀取用新的 active 除霧。"""
    rec = _frame_record(0, image, active0)
    rec["used"] = rec["next"]                # 靜態：除霧用 CPU 更新後的 active
    rec["result"] = apply_pass(image, rec["used"])
    return rec


def _fmt_group(stats, g):
    s = stats[g]
    return f"{s['rgb']}（{s['count']} 點）" if s["valid"] else "空"


def format_trace(records):
    """逐 frame 表格：統計用的 T_h、除霧用的 A → 本張統計 → step 4 → 下一個參數。"""
    lines = ["| frame | 統計用 T_h | 除霧用 A_F | 除霧用 A_N | 統計 gray_mean | 本張遠群 | 本張近群 "
             "| A_F 處理 | A_N 處理 | 下一個 T_h | 下一個 A_F | 下一個 A_N | 下一個 I_max |",
             "|" + "---|" * 13]
    for r in records:
        u, s, x, nx = r["used"], r["stats"], r["update"], r["next"]
        lines.append(f"| {r['n']} | {r['stats_active']['t_h']} | {u['a_far']} | {u['a_near']} | {s['gray_mean']} "
                     f"| {_fmt_group(s, 'far')} | {_fmt_group(s, 'near')} "
                     f"| {x['far']['action']} | {x['near']['action']} "
                     f"| {nx['t_h']} | {nx['a_far']} | {nx['a_near']} | {nx['i_max']} |")
    return lines


# ---------------------------------------------------------------- selfcheck

def _ref_stats(frame, t_h_prev):
    """參考實作：逐點迴圈的不同寫法，只用 Python int，不呼叫上面的函式。"""
    h_full, w_full, _ = frame.shape
    rows, cols = range(0, h_full, STEP), range(0, w_full, STEP)
    down = [[tuple(int(v) for v in frame[r, c]) for c in cols] for r in rows]
    h, w = len(down), len(down[0])

    def pixel_min(i, j):
        return min(down[i][j]) if 0 <= i < h and 0 <= j < w else PAD

    half = WINDOW // 2
    dark = [[min(pixel_min(i + di, j + dj)
                 for di in range(-half, half + 1) for dj in range(-half, half + 1))
             for j in range(w)] for i in range(h)]
    total = 0
    for row in down:
        for p in row:
            total += p[0] + p[1] + p[2]
    groups = {}
    for name, belongs in (("far", lambda d: d >= t_h_prev), ("near", lambda d: d < t_h_prev)):
        best, count = None, 0
        for i in range(h):
            for j in range(w):
                if belongs(dark[i][j]):
                    count += 1
                    if best is None or dark[i][j] > dark[best[0]][best[1]]:
                        best = (i, j)
        groups[name] = ({"valid": False, "count": 0, "max_dark": None, "pos": None, "rgb": None}
                        if best is None else
                        {"valid": True, "count": count, "max_dark": dark[best[0]][best[1]],
                         "pos": best, "rgb": down[best[0]][best[1]]})
    return {"shape_down": (h, w), "n": h * w, "gray_sum": total,
            "gray_mean": total // (3 * h * w), "dark": dark, "down": down, **groups}


class _Checker:
    def __init__(self):
        self.failed = 0
        self.total = 0

    def check(self, label, ok, detail=""):
        self.total += 1
        if not ok:
            self.failed += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"（{detail}）" if detail and not ok else ""))


def _summary(s):
    return {k: s[k] for k in ("shape_down", "n", "gray_sum", "gray_mean", "far", "near")}


def _compare_with_ref(ck, frame, t_h_prev):
    """被測結果和參考實作逐值比較，再做性質檢查。回傳 stats_pass 結果。"""
    s = stats_pass(frame, t_h_prev)
    ref = _ref_stats(frame, t_h_prev)
    ck.check("降採樣圖和參考實作逐值相同",
             s["frame_down"].tolist() == [[list(p) for p in row] for row in ref["down"]])
    ck.check("dark channel 和參考實作逐值相同", s["dark"].tolist() == ref["dark"])
    for key in ("shape_down", "n", "gray_sum", "gray_mean", "far", "near"):
        ck.check(f"{key} 和參考實作相同", tuple(s[key]) == tuple(ref[key]) if key == "shape_down"
                 else s[key] == ref[key], f"{s[key]} vs {ref[key]}")
    # 性質檢查
    dark, far = s["dark"], s["dark"] >= t_h_prev
    ck.check("兩群 pixel 數相加等於降採樣 pixel 數", s["far"]["count"] + s["near"]["count"] == s["n"])
    sums = s["frame_down"].astype(np.int64).sum(axis=2)
    ck.check("floor(最小灰階) <= 平均灰階 <= 最大灰階",
             int(sums.min()) // 3 <= s["gray_mean"] and 3 * s["gray_mean"] <= int(sums.max()))
    for name in ("far", "near"):
        g = s[name]
        if g["valid"]:
            r, c = g["pos"]
            ck.check(f"{name}：所選位置在群內、dark 等於最大值、RGB 取自該位置",
                     bool(far[r, c] == (name == "far")) and int(dark[r, c]) == g["max_dark"]
                     and g["rgb"] == tuple(int(v) for v in s["frame_down"][r, c]))
        else:
            ck.check(f"{name}：空群組的欄位都是 None，不是 RGB=0",
                     g["count"] == 0 and g["pos"] is None and g["rgb"] is None and g["max_dark"] is None)
    again = stats_pass(frame.copy(), t_h_prev)
    ck.check("重跑一次：數值與陣列逐值相同",
             _summary(again) == _summary(s) and np.array_equal(again["dark"], s["dark"])
             and np.array_equal(again["frame_down"], s["frame_down"]))
    return s


def _filled(h, w, rgb):
    frame = np.empty((h, w, 3), dtype=np.uint8)
    frame[:] = rgb
    return frame


def _put_down(frame, i, j, rgb):
    """在降採樣位置 (i, j)，即原圖 (3i, 3j)，寫入 RGB。"""
    frame[STEP * i, STEP * j] = rgb


def selfcheck():
    ck = _Checker()
    rng = np.random.default_rng(20261029)

    print("案例 a：21×21 一般圖（降採樣 7×7，測 5×5 視窗內部）")
    frame = rng.integers(0, 256, size=(21, 21, 3), dtype=np.uint8)
    ref_dark = sorted(v for row in _ref_stats(frame, 0)["dark"] for v in row)
    t = ref_dark[len(ref_dark) // 2]          # 取參考實作 dark 的中位數，讓兩群都非空
    s = _compare_with_ref(ck, frame, t)
    ck.check("手算：降採樣尺寸 7×7、N = 49", s["shape_down"] == (7, 7) and s["n"] == 49)
    ck.check("兩群都非空", s["far"]["valid"] and s["near"]["valid"])
    print(f"  紀錄：T_h_prev = {t}，far = {s['far']}，near = {s['near']}")

    print("案例 b：20×22（不是 3 的倍數，測尾端與視窗邊界）")
    frame = rng.integers(0, 256, size=(20, 22, 3), dtype=np.uint8)
    frame[18, 21] = (1, 2, 3)                 # 最後一個抽樣位置
    frame[19, 21] = frame[18, 20] = (250, 250, 250)   # 不是抽樣位置，不得出現在降採樣圖
    s = _compare_with_ref(ck, frame, 128)
    ck.check("手算：降採樣尺寸 ceil(20/3)×ceil(22/3) = 7×8、N = 56",
             s["shape_down"] == (7, 8) and s["n"] == 56)
    shape_ok = s["shape_down"] == (7, 8)      # 尺寸錯時下面的索引會越界，直接判 FAIL，不中止 selfcheck
    ck.check("手算：原圖 (18, 21) 落在降採樣 (6, 7)",
             shape_ok and tuple(int(v) for v in s["frame_down"][6, 7]) == (1, 2, 3))
    # 角落的 5×5 視窗只有 3×3 在影像內：外圍補 255 時，角落 dark = 影像內 3×3 的最小值；
    # 若誤補 0，角落 dark 會變成 0
    tl = int(s["frame_down"][:3, :3].min())
    br = int(s["frame_down"][4:7, 5:8].min()) if shape_ok else None
    ck.check("手算：角落 dark 等於影像內 3×3 的最小值（左上、右下）；右下 <= 1",
             shape_ok and int(s["dark"][0, 0]) == tl and int(s["dark"][6, 7]) == br and br <= 1,
             f"dark[0,0]={int(s['dark'][0, 0])}/{tl}, 右下 {br}" if shape_ok else f"降採樣尺寸 {s['shape_down']}")
    ck.check("手算：左上角影像內 3×3 的最小值不是 0（否則無法分辨補 0 或補 255）", tl > 0, str(tl))

    print("案例 c：dark 恰好等於 T_h_prev（9×9 單色 (120, 130, 140)）")
    frame = _filled(9, 9, (120, 130, 140))
    s = _compare_with_ref(ck, frame, 120)
    ck.check("手算：dark 全為 120，T_h_prev = 120 → 9 點都歸遠群",
             s["far"] == {"valid": True, "count": 9, "max_dark": 120, "pos": (0, 0), "rgb": (120, 130, 140)})
    ck.check("手算：近群為空，valid=False", s["near"]["valid"] is False and s["near"]["count"] == 0)
    ck.check("手算：gray_sum = 9 × 390 = 3510，平均 = floor(3510 / 27) = 130",
             s["gray_sum"] == 3510 and s["gray_mean"] == 130)

    print("案例 d：遠群兩點並列最大（13×13，降採樣 5×5）")
    frame = np.zeros((13, 13, 3), dtype=np.uint8)   # 非抽樣位置全為 0，不得影響結果
    for i in range(5):
        for j in range(5):
            high = (i <= 2 and j <= 2) or (i >= 2 and j >= 2)
            _put_down(frame, i, j, (200, 200, 200) if high else (10, 10, 10))
    _put_down(frame, 0, 0, (200, 210, 220))
    _put_down(frame, 4, 4, (220, 200, 210))
    s = _compare_with_ref(ck, frame, 150)
    ck.check("手算：只有 (0,0)、(4,4) 的 dark = 200，並列時取 raster 第一個 (0,0)",
             s["far"] == {"valid": True, "count": 2, "max_dark": 200, "pos": (0, 0), "rgb": (200, 210, 220)},
             str(s["far"]))
    ck.check("手算：其餘 23 點 dark = 10，近群取第一個 (0,1)，RGB (200,200,200)",
             s["near"] == {"valid": True, "count": 23, "max_dark": 10, "pos": (0, 1), "rgb": (200, 200, 200)},
             str(s["near"]))
    ck.check("手算：gray_sum = 8×30 + 15×600 + 630 + 630 = 10500，平均 = floor(10500/75) = 140",
             s["gray_sum"] == 10500 and s["gray_mean"] == 140)

    print("案例 e：近群最大 dark 為 0（15×15，降採樣中心有一色為 0）")
    frame = _filled(15, 15, (50, 60, 70))
    _put_down(frame, 2, 2, (0, 90, 90))
    s = _compare_with_ref(ck, frame, 1)
    ck.check("手算：每個 5×5 視窗都涵蓋 (2,2)，dark 全為 0；近群仍 valid，取 (0,0)",
             s["near"] == {"valid": True, "count": 25, "max_dark": 0, "pos": (0, 0), "rgb": (50, 60, 70)},
             str(s["near"]))
    ck.check("手算：遠群為空", s["far"]["valid"] is False)
    ck.check("手算：gray_sum = 24×180 + 180 = 4500，平均 = 60", s["gray_sum"] == 4500 and s["gray_mean"] == 60)

    print("案例 f：遠群為空（同案例 c，T_h_prev = 121）")
    frame = _filled(9, 9, (120, 130, 140))
    s = _compare_with_ref(ck, frame, 121)
    ck.check("手算：dark 120 < 121 → 遠群為空，valid=False、rgb 是 None",
             s["far"]["valid"] is False and s["far"]["rgb"] is None)
    ck.check("手算：近群 9 點，取 (0,0)",
             s["near"] == {"valid": True, "count": 9, "max_dark": 120, "pos": (0, 0), "rgb": (120, 130, 140)})

    print("累加溢位：全白 1080×1920（降採樣 360×640）")
    frame = _filled(1080, 1920, (255, 255, 255))
    s = stats_pass(frame, 255)
    ck.check("手算：N = 230400，gray_sum = 230400 × 765 = 176256000，平均 = 255",
             s["n"] == 230400 and s["gray_sum"] == 176256000 and s["gray_mean"] == 255)
    ck.check("單一 pixel 的 R+G+B = 765，超過 uint8 範圍時仍正確",
             gray_sum(frame[:1, :1]) == 765)
    ck.check("手算：dark 全為 255，T_h_prev = 255 → 全部歸遠群",
             s["far"]["count"] == 230400 and s["near"]["valid"] is False)

    print("輸入檢查")
    for label, bad in (("BGRA 四通道", np.zeros((3, 3, 4), np.uint8)),
                       ("uint16", np.zeros((3, 3, 3), np.uint16))):
        try:
            stats_pass(bad, 0)
            ck.check(f"{label} 應拒絕", False)
        except ValueError:
            ck.check(f"{label} 應拒絕", True)
    for t in (-1, 256, 1.5, True, np.int64(5)):
        try:
            stats_pass(_filled(3, 3, (0, 0, 0)), t)
            ck.check(f"T_h_prev = {t} 應拒絕", False)
        except ValueError:
            ck.check(f"T_h_prev = {t} 應拒絕", True)

    _selfcheck_main_path(ck, rng)
    _selfcheck_schedule(ck)
    _selfcheck_inv_quant(ck)
    _selfcheck_pala_fixed(ck)
    _selfcheck_pte_fixed(ck)
    _selfcheck_sd_fixed(ck)
    _selfcheck_rules(ck)

    print(f"結果：{ck.total - ck.failed}/{ck.total} PASS")
    return 0 if ck.failed == 0 else 1


def _ref_pixel(rgb, a_far, a_near):
    """主路徑參考實作：單一 pixel、以 Fraction 精確計算，不呼叫上面的函式。"""
    from fractions import Fraction as F

    i = [F(int(v)) for v in rgb]            # 先轉 Python int，避免 numpy uint8 溢位
    s, s_max = sum(i), sum(F(int(v)) for v in a_far)
    if s >= s_max:
        eta, a = F(0), [F(v) for v in a_far]
    else:
        eta = (s_max - s) / s_max
        a = [(1 - eta) * f + eta * n for f, n in zip(a_far, a_near)]
    lo, hi = min(i), max(i)
    k = F(1) if lo + hi == 0 else 1 - (lo / (lo + hi)) ** GAMMA
    ratios = [ic / ac for ic, ac in zip(i, a) if ac > 0]
    m = min(ratios) if ratios else F(0)
    t_raw = 1 - F(OMEGA) * m * k
    t = min(max(t_raw, F(1, 1023)), F(1))
    j = [(ic - ac) / t + ac for ic, ac in zip(i, a)]
    out = [min(max(int((jc + F(1, 2)).__floor__()), 0), 255) for jc in j]
    return {"eta": eta, "a": a, "k": k, "m": m, "t_raw": t_raw, "t": t, "j": j, "out": out}


_FLOAT_KEYS = ("eta", "a_map", "k", "m", "t_raw", "t", "j")


def _ref_errors(frame, r, a_far, a_near):
    """主路徑結果和 Fraction 參考實作比較。回傳 (所有中間值有限, 最大絕對差, 誤差都 <= 1e-9, 輸出相同)。

    用 `d <= 1e-9` 判斷每一個差，NaN 一定判為不符合；不靠 max() 累積。
    """
    finite = all(bool(np.all(np.isfinite(r[key]))) for key in _FLOAT_KEYS)
    diffs, out_same = [], True
    for y in range(frame.shape[0]):
        for x in range(frame.shape[1]):
            ref = _ref_pixel(frame[y, x], a_far, a_near)
            diffs += [abs(float(r[key][y, x]) - float(ref[key])) for key in ("eta", "k", "m", "t_raw", "t")]
            diffs += [abs(float(r["a_map"][y, x, c]) - float(ref["a"][c])) for c in range(3)]
            diffs += [abs(float(r["j"][y, x, c]) - float(ref["j"][c])) for c in range(3)]
            out_same &= [int(v) for v in r["out"][y, x]] == ref["out"]
    within = all(d <= 1e-9 for d in diffs)
    worst = max((d for d in diffs if d == d), default=0.0)
    return finite, worst, within, out_same


def _failed_result(frame):
    """apply_pass 報錯時的替代結果：浮點值全為 NaN、輸出全為 −1，後續的手算比較都會 FAIL，不會中止 selfcheck。"""
    h, w = frame.shape[:2]
    r = {key: np.full((h, w), np.nan) for key in ("eta", "k", "m", "t_raw", "t")}
    r.update(a_map=np.full((h, w, 3), np.nan), j=np.full((h, w, 3), np.nan), out=np.full((h, w, 3), -1))
    return r


def _check_pixels_with_ref(ck, frame, a_far, a_near, label):
    try:
        r = apply_pass(frame, make_active(0, a_far, a_near, measured=True))
        ck.check(f"{label}：apply_pass 沒有報錯", True)
    except ValueError as e:
        ck.check(f"{label}：apply_pass 沒有報錯", False, str(e))
        r = _failed_result(frame)
    finite, worst, within, out_same = _ref_errors(frame, r, a_far, a_near)
    ck.check(f"{label}：所有中間值都是有限值（finite，沒有 inf／NaN）", finite)
    ck.check(f"{label}：η、A、k、t、J 和 Fraction 參考實作的絕對差 <= 1e-9", within, f"{worst:.3g}")
    ck.check(f"{label}：8-bit 輸出和參考實作逐值相同", out_same)
    return r


def _selfcheck_main_path(ck, rng):
    af, an = (200, 200, 200), (100, 100, 100)

    print("主路徑 m1：逐步手算（I = (50, 100, 150)，A_F = 200、A_N = 100）")
    r = _check_pixels_with_ref(ck, np.array([[[50, 100, 150]]], np.uint8), af, an, "m1")
    # 三色和 300 < 600 → η = 0.5 → A = 150；min(I/A) = 1/3；
    # k = 1 − (50/200)^2 = 0.9375；t = 1 − 0.9375 × (1/3) × 0.9375 = 0.70703125
    # J = (I − 150)/t + 150 = (8.5635…, 79.2818…, 150) → (9, 79, 150)
    ck.check("手算：η = 0.5、A = 150、k = 0.9375、t = 0.70703125",
             abs(r["eta"][0, 0] - 0.5) <= 1e-12 and np.allclose(r["a_map"][0, 0], 150, rtol=0, atol=1e-9)
             and r["k"][0, 0] == 0.9375 and abs(r["t"][0, 0] - 0.70703125) <= 1e-12)
    ck.check("手算：輸出 (9, 79, 150)", r["out"][0, 0].tolist() == [9, 79, 150], str(r["out"][0, 0]))

    print("主路徑 m2：灰階 >= I_max 直接用 A_F（I = (210, 220, 230)）")
    r = _check_pixels_with_ref(ck, np.array([[[210, 220, 230]]], np.uint8), af, an, "m2")
    # 三色和 660 >= 600 → A = A_F；k = 1 − (210/440)^2 ≈ 0.77221；t ≈ 0.239855；
    # J_R = 10/t + 200 ≈ 241.69 → 242；G、B 超過 255 → 255
    ck.check("手算：A = A_F、η = 0、輸出 (242, 255, 255)",
             np.all(r["a_map"][0, 0] == 200) and r["eta"][0, 0] == 0
             and r["out"][0, 0].tolist() == [242, 255, 255], str(r["out"][0, 0]))

    print("主路徑 m3：純黑 pixel（min + max = 0 → k = 1）")
    r = _check_pixels_with_ref(ck, np.zeros((1, 1, 3), np.uint8), af, an, "m3")
    ck.check("手算：η = 1、A = A_N、k = 1、t = 1、輸出 (0, 0, 0)",
             abs(r["eta"][0, 0] - 1) <= 1e-12 and np.allclose(r["a_map"][0, 0], 100, rtol=0, atol=1e-9)
             and r["k"][0, 0] == 1
             and r["t"][0, 0] == 1 and r["out"][0, 0].tolist() == [0, 0, 0])

    print("主路徑 m4：A 有一色為 0（A_N = (0, 100, 100)，純黑 pixel）")
    r = _check_pixels_with_ref(ck, np.zeros((1, 1, 3), np.uint8), af, (0, 100, 100), "m4")
    ck.check("手算：R 不參與 min，min = 0、t = 1，輸出 (0, 0, 0)，沒有 inf／nan",
             r["t"][0, 0] == 1 and r["out"][0, 0].tolist() == [0, 0, 0]
             and all(np.all(np.isfinite(r[key])) for key in ("eta", "a_map", "k", "m", "t", "j")))

    print("主路徑 m5：I_max = 0（A_F = A_N = (0, 0, 0)）")
    frame = rng.integers(0, 256, size=(5, 6, 3), dtype=np.uint8)
    r = _check_pixels_with_ref(ck, frame, (0, 0, 0), (0, 0, 0), "m5")
    ck.check("手算：A 全為 0 → t = 1 → 輸出等於輸入", np.array_equal(r["out"], frame))

    print("主路徑 m6：t_m 低於下限（A_F = A_N = (10, 10, 10)，白色 pixel）")
    r = _check_pixels_with_ref(ck, np.full((1, 1, 3), 255, np.uint8), (10, 10, 10), (10, 10, 10), "m6")
    # min(I/A) = 25.5，k = 0.75 → t_raw = 1 − 0.9375 × 25.5 × 0.75 ≈ −16.93 → 截到 1/1023
    ck.check("手算：t_raw < 0，t = 1/1023，輸出 255",
             r["t_raw"][0, 0] < 0 and r["t"][0, 0] == T_MIN and r["out"][0, 0].tolist() == [255] * 3)

    print("主路徑 m7：隨機 6×7 圖，逐點和參考實作比較並做性質檢查")
    frame = rng.integers(0, 256, size=(6, 7, 3), dtype=np.uint8)
    af7, an7 = (230, 235, 240), (120, 110, 100)
    r = _check_pixels_with_ref(ck, frame, af7, an7, "m7")
    s = frame.astype(np.int64).sum(axis=2)
    blend = s < sum(af7)
    lo_a, hi_a = np.minimum(af7, an7), np.maximum(af7, an7)
    tol = 1e-12
    ck.check("k ∈ [0.75, 1]", bool(np.all((r["k"] >= 0.75 - tol) & (r["k"] <= 1 + tol))))
    ck.check("η ∈ [0, 1]", bool(np.all((r["eta"] >= -tol) & (r["eta"] <= 1 + tol))))
    ck.check("灰階 >= I_max 的 pixel，A = A_F", bool(np.all(r["a_map"][~blend] == np.array(af7, float))))
    ck.check("其他 pixel 的 A 介於 A_N、A_F 之間",
             bool(np.all((r["a_map"][blend] >= lo_a - tol) & (r["a_map"][blend] <= hi_a + tol))))
    ck.check("截斷前 t_m >= 1 − ω·min(I/A)（k <= 1）",
             bool(np.all(r["t_raw"] >= 1 - OMEGA * r["m"] - tol)))
    recon = r["j"] * r["t"][..., None] + r["a_map"] * (1 - r["t"][..., None])
    ck.check("Eq.1 反推：J·t + A·(1 − t) 和輸入的差 <= 1e-9",
             bool(np.max(np.abs(recon - frame)) <= 1e-9), f"{np.max(np.abs(recon - frame)):.3g}")
    again = apply_pass(frame.copy(), make_active(0, af7, an7, measured=True))
    ck.check("重跑一次：所有中間值與輸出逐值相同",
             all(np.array_equal(again[key], r[key]) for key in r))

    print("主路徑 m8：接上統計單元（單張展示參數）")
    frame = rng.integers(0, 256, size=(21, 21, 3), dtype=np.uint8)
    frame[:9] = rng.integers(200, 256, size=(9, 21, 3), dtype=np.uint8)   # 上方像霧的亮區
    s, a_far, a_near, r = dehaze_single(frame)
    ref_t = sum(int(v) for v in frame[::3, ::3].reshape(-1)) // (3 * 49)
    ck.check("T_h_prev = 該圖自己的平均灰階（另寫的加總）", s["t_h_prev"] == ref_t, f"{s['t_h_prev']} vs {ref_t}")
    ck.check("兩群都非空，A_F、A_N 取自 stats_pass 的兩群 RGB",
             s["far"]["valid"] and s["near"]["valid"]
             and a_far == s["far"]["rgb"] and a_near == s["near"]["rgb"])
    _check_pixels_with_ref(ck, frame, a_far, a_near, "m8")
    frame = _filled(9, 9, (120, 130, 140))
    s, a_far, a_near, r = dehaze_single(frame)
    # 平均灰階 130 > dark 120 → 9 點都在近群，遠群為空
    ck.check("遠群為空時 A_F 改用 A_N（單色圖），輸出不含 inf／nan",
             s["far"]["valid"] is False and a_far == a_near == (120, 130, 140)
             and bool(np.all(np.isfinite(r["j"]))))

    print("主路徑 m9：A 三色不同、η ≠ 0.5 的逐步手算")
    af9, an9 = (220, 200, 180), (100, 120, 140)
    r = _check_pixels_with_ref(ck, np.array([[[60, 90, 120]]], np.uint8), af9, an9, "m9")
    # 三色和 270 < 600 → η = 330/600 = 0.55；A = 0.45·A_F + 0.55·A_N = (154, 156, 158)
    # min(I/A) = 60/154（R）；k = 1 − (60/180)^2 = 8/9；t = 1 − (15/16)(60/154)(8/9) = 52/77
    # J = (I − A)·77/52 + A = (14.81, 58.27, 101.73) → (15, 58, 102)
    ck.check("手算：η = 0.55、A = (154, 156, 158)、k = 8/9、t = 52/77",
             abs(r["eta"][0, 0] - 0.55) <= 1e-12 and np.allclose(r["a_map"][0, 0], (154, 156, 158), atol=1e-9)
             and abs(r["k"][0, 0] - 8 / 9) <= 1e-12 and abs(r["t"][0, 0] - 52 / 77) <= 1e-12)
    ck.check("手算：輸出 (15, 58, 102)", r["out"][0, 0].tolist() == [15, 58, 102], str(r["out"][0, 0]))

    print("主路徑 m10：A 有一色為 0、而 I 的該色 > 0（I = (50, 60, 70)，A = (0, 100, 100)）")
    r = _check_pixels_with_ref(ck, np.array([[[50, 60, 70]]], np.uint8), (0, 100, 100), (0, 100, 100), "m10")
    # R 不參與 min → min(60/100, 70/100) = 0.6；k = 1 − (50/120)^2 = 119/144；
    # t = 1 − (15/16)(0.6)(119/144) = 1233/2304；若把 R 的比值當 0，t 會變成 1、輸出等於輸入
    ck.check("手算：min = 0.6、t = 1233/2304 ≈ 0.5352、輸出 (93, 25, 44)",
             abs(r["m"][0, 0] - 0.6) <= 1e-12 and abs(r["t"][0, 0] - 1233 / 2304) <= 1e-12
             and r["out"][0, 0].tolist() == [93, 25, 44], str(r["out"][0, 0]))

    print("主路徑 m11：A 用 numpy uint8 傳入，不得溢位")
    px = np.array([[[50, 100, 150]]], np.uint8)
    r = apply_pass(px, make_active(0, tuple(np.uint8(v) for v in af), tuple(np.uint8(v) for v in an)))
    ck.check("uint8 的 A 和 Python int 的 A 結果相同：(9, 79, 150)", r["out"][0, 0].tolist() == [9, 79, 150])
    for label, bad in (("A 只有 2 個值", (1, 2)), ("A 超過 255", (256, 0, 0)), ("A 含 bool", (True, 0, 0)),
                       ("A 含小數 100.9", (100.9, 120, 140)), ("A 含負小數 −0.5", (-0.5, 120, 140)),
                       ("A 是數字字串", ("100", "120", "140")), ("A 是含 bool 的 iterator", iter([True, 120, 140])),
                       ("A 含 numpy 浮點數", (np.float64(7.0), 1, 2))):
        try:
            make_active(0, bad, an)
            ck.check(f"{label} 應拒絕", False)
        except ValueError:
            ck.check(f"{label} 應拒絕", True)
    good = make_active(0, af, an)
    for label, inv in (("1/I_max 是 NaN", float("nan")), ("1/I_max 是負數", -1.0), ("1/I_max 是 bool", True),
                       ("1/I_max 是 1e308（有限，但 600 × 1e308 會溢位）", 1e308), ("1/I_max 是 1.5", 1.5)):
        try:
            apply_pass(px, dict(good, inv_i_max=inv))
            ck.check(f"{label} 應拒絕", False)
        except ValueError:
            ck.check(f"{label} 應拒絕", True)
    white = np.full((1, 1, 3), 255, np.uint8)   # 三色和 765 >= I_max 600 → η 固定為 0，後面的 η > 1 檢查擋不到
    try:
        apply_pass(white, dict(good, inv_i_max=1.5))
        ck.check("白色 pixel（η = 0）、1/I_max 是 1.5 應拒絕：由輸入範圍檢查擋下", False)
    except ValueError as e:
        ck.check("白色 pixel（η = 0）、1/I_max 是 1.5 應拒絕：由輸入範圍檢查擋下", "inv_i_max" in str(e), str(e))

    print("主路徑 m12：SD 的捨入與截斷（直接給 A 與 t）")
    one = np.full((1, 1, 3), 100, np.uint8)
    t_half = np.array([[0.5]])
    _, j, out = sd(one, np.full((1, 1, 3), 99.5), t_half)
    # J = (100 − 99.5)/0.5 + 99.5 = 100.5 → 四捨五入（.5 進位）= 101；round-half-even 會得到 100
    ck.check("手算：J = 100.5 → 101（.5 進位）", float(j[0, 0, 0]) == 100.5 and out[0, 0].tolist() == [101] * 3)
    _, j, out = sd(np.zeros((1, 1, 3), np.uint8), np.full((1, 1, 3), 200.0), np.array([[0.25]]))
    ck.check("手算：J = (0 − 200)/0.25 + 200 = −600 → 截到 0", float(j[0, 0, 0]) == -600 and out[0, 0].tolist() == [0] * 3)
    _, j, out = sd(np.full((1, 1, 3), 255, np.uint8), np.zeros((1, 1, 3)), t_half)
    ck.check("手算：J = 255/0.5 = 510 → 截到 255", float(j[0, 0, 0]) == 510 and out[0, 0].tolist() == [255] * 3)

    print("主路徑 m13：主路徑實際使用 active 的 1/I_max（同 m9 的 pixel，1/I_max 放大 2^-8）")
    px9 = np.array([[[60, 90, 120]]], np.uint8)
    base = make_active(0, af9, an9)
    near = dict(base, inv_i_max=base["inv_i_max"] * (1 + 2 ** -8))   # 相對誤差 0.39%，接近定點量化的大小
    r0, r1 = apply_pass(px9, base), apply_pass(px9, near)
    # 三色和 270 < I_max 600 → 走加權分支；η = 330 × 1/I_max，A_F ≠ A_N，所以 A 也會改變
    ck.check("1/I_max 改變 → η = (600 − 270) × 新的 1/I_max，和原本不同",
             r1["eta"][0, 0] == 330 * near["inv_i_max"] and r1["eta"][0, 0] != r0["eta"][0, 0],
             f"{r0['eta'][0, 0]!r} → {r1['eta'][0, 0]!r}")
    ck.check("1/I_max 改變 → 三色的 A 都改變", bool(np.all(r1["a_map"] != r0["a_map"])))
    print(f"  紀錄：η {r0['eta'][0, 0]:.6f} → {r1['eta'][0, 0]:.6f}；輸出 {r0['out'][0, 0].tolist()} → {r1['out'][0, 0].tolist()}")

    print("主路徑 m14：checker 本身拒絕 NaN（只改 η，輸出不變）")
    px2 = np.array([[[210, 220, 230]]], np.uint8)
    r = apply_pass(px2, make_active(0, af, an))
    r["eta"] = r["eta"].copy()
    r["eta"][0, 0] = np.nan                  # 灰階 >= I_max 的 pixel，η 不影響 A 與輸出
    finite, _, within, out_same = _ref_errors(px2, r, af, an)
    ck.check("η 注入 NaN：checker 判為非有限值且誤差不合格，輸出仍相同",
             not finite and not within and out_same)

    print("主路徑 m15：η 超過 1 先報錯（saturation 規則待 C 決定）；理想 1/I_max 不會觸發")
    black = np.zeros((1, 1, 3), np.uint8)
    b = make_active(0, (200, 200, 200), (0, 0, 0))
    try:
        apply_pass(black, dict(b, inv_i_max=b["inv_i_max"] * (1 + 2 ** -8)))
        ck.check("純黑 pixel、A_N = 0、1/I_max 放大 2^-8：η = 257/256 > 1 應報錯", False)
    except ValueError as e:
        ck.check("純黑 pixel、A_N = 0、1/I_max 放大 2^-8：η = 257/256 > 1 應報錯", "saturation" in str(e), str(e))
    ok = True
    for i_max in range(1, 766):              # 把 I_max 拆成三色，每色 <= 255
        a = (min(i_max, 255), min(max(i_max - 255, 0), 255), max(i_max - 510, 0))
        try:
            ok &= abs(float(apply_pass(black, make_active(0, a, (0, 0, 0)))["eta"][0, 0]) - 1) <= 1e-12
        except ValueError:
            ok = False
    ck.check("理想 1/I_max：I_max = 1–765、純黑 pixel 都不報錯，η 和 1 的差 <= 1e-12", ok)
    try:
        sd(black, np.full((1, 1, 3), np.nan), np.array([[1.0]]))
        ck.check("SD 遇到 NaN 的 A 應報錯，不轉成 uint8", False)
    except ValueError:
        ck.check("SD 遇到 NaN 的 A 應報錯，不轉成 uint8", True)


def _fake_stats(far_rgb, near_rgb, gray_mean):
    return {"far": {"valid": far_rgb is not None, "rgb": far_rgb},
            "near": {"valid": near_rgb is not None, "rgb": near_rgb}, "gray_mean": gray_mean}


def _expect_trace(ck, records, expected, label):
    """影片排程用。expected：每張 frame 一列
    (統計用 T_h, 除霧用 A_F, 除霧用 A_N, F 觸發, N 觸發, 下一張 T_h, 下一張 A_F, 下一張 A_N)。"""
    got = [(r["stats_active"]["t_h"], r["used"]["a_far"], r["used"]["a_near"],
            r["update"]["far"]["triggered"], r["update"]["near"]["triggered"],
            r["next"]["t_h"], r["next"]["a_far"], r["next"]["a_near"]) for r in records]
    for n, (g, e) in enumerate(zip(got, expected)):
        ck.check(f"{label} frame {n}：本張參數、step 4、下一張參數都和手算相同", g == e, f"{g} vs {e}")
    ck.check(f"{label}：frame 數相同", len(got) == len(expected))
    ck.check(f"{label}：frame n 的參數 = frame n−1 算出的下一張參數",
             all(records[n]["used"] is records[n - 1]["next"] for n in range(1, len(records))))
    ck.check(f"{label}：影片的統計與除霧用同一個 active，stats 的 T_h_prev 等於它的 T_h",
             all(r["stats_active"] is r["used"] and r["stats"]["t_h_prev"] == r["used"]["t_h"] for r in records))
    ck.check(f"{label}：每張輸出都用本張的參數（和直接呼叫 apply_pass 相同）",
             all(np.array_equal(r["result"]["out"], apply_pass(r["frame"], r["used"])["out"]) for r in records))


def _selfcheck_schedule(ck):
    prev = make_active(120, (200, 200, 200), (100, 100, 100), measured=True)

    print("cpu_update c1：差 3 不觸發、差 4 觸發，A_F 和 A_N 分開判斷")
    nxt, info = cpu_update(_fake_stats((203, 200, 200), (104, 100, 100), 117), prev)
    ck.check("手算：A_F 差 3 → 用本張 (203, 200, 200)", nxt["a_far"] == (203, 200, 200) and not info["far"]["triggered"])
    ck.check("手算：A_N 差 4 → floor((104 + 100)/2) = (102, 100, 100)",
             nxt["a_near"] == (102, 100, 100) and info["near"]["triggered"])
    ck.check("手算：下一張 T_h = gray_mean = 117；I_max = 603；1/I_max = 1/603",
             nxt["t_h"] == 117 and nxt["i_max"] == 603 and nxt["inv_i_max"] == 1 / 603 and nxt["measured"])

    print("cpu_update c2：負方向的差 3、差 4")
    nxt, info = cpu_update(_fake_stats((196, 200, 200), (97, 100, 100), 117), prev)
    ck.check("手算：A_F 差 4 → (198, 200, 200)；A_N 差 3 → (97, 100, 100)",
             nxt["a_far"] == (198, 200, 200) and nxt["a_near"] == (97, 100, 100)
             and info["far"]["triggered"] and not info["near"]["triggered"])

    print("cpu_update c3：平均時三色都取 floor，總和是奇數")
    nxt, _ = cpu_update(_fake_stats((205, 199, 200), (100, 100, 100), 117), prev)
    ck.check("手算：差 5 → (floor(405/2), floor(399/2), 200) = (202, 199, 200)", nxt["a_far"] == (202, 199, 200))

    print("cpu_update c4：stats 的 RGB 是 numpy uint8，平均不得溢位")
    prev_hi = make_active(120, (250, 250, 250), (100, 100, 100), measured=True)
    nxt, _ = cpu_update(_fake_stats(tuple(np.uint8(255) for _ in range(3)), (100, 100, 100), 117), prev_hi)
    ck.check("手算：floor((255 + 250)/2) = 252（uint8 相加會溢位成 124）", nxt["a_far"] == (252, 252, 252))

    print("cpu_update c5：空群組沿用上一個 active")
    nxt, info = cpu_update(_fake_stats(None, (101, 100, 100), 90), prev)
    ck.check("手算：遠群為空 → A_F 沿用 (200, 200, 200)，不觸發；近群差 1 → (101, 100, 100)",
             nxt["a_far"] == (200, 200, 200) and not info["far"]["triggered"]
             and nxt["a_near"] == (101, 100, 100) and nxt["i_max"] == 600)

    print("cpu_update c6：第一次更新（active0 不是量測值）")
    a0 = make_active(128, (255, 255, 255), (255, 255, 255))
    nxt, info = cpu_update(_fake_stats((240, 240, 240), None, 130), a0)
    ck.check("手算：和預設值差 15 也不做 step 4 → A_F = (240, 240, 240)",
             nxt["a_far"] == (240, 240, 240) and not info["far"]["triggered"])
    ck.check("手算：近群為空且是第一次 → A_N 用遠群本張的值 (240, 240, 240)", nxt["a_near"] == (240, 240, 240))

    print("cpu_update c7：I_max = 0")
    nxt, _ = cpu_update(_fake_stats((0, 0, 0), (0, 0, 0), 0), a0)
    ck.check("手算：A_F = (0, 0, 0) → I_max = 0，1/I_max 寫 0", nxt["i_max"] == 0 and nxt["inv_i_max"] == 0.0)

    print("cpu_update c8：gray_mean 先驗證再轉型")
    for gm in (117.5, True, -1, 256):
        try:
            cpu_update(_fake_stats((200, 200, 200), (100, 100, 100), gm), prev)
            ck.check(f"gray_mean = {gm!r} 應拒絕", False)
        except ValueError:
            ck.check(f"gray_mean = {gm!r} 應拒絕", True)
    nxt, _ = cpu_update(_fake_stats((200, 200, 200), (100, 100, 100), np.int64(117)), prev)
    ck.check("gray_mean 是 numpy 整數時接受，轉成 Python int 117", type(nxt["t_h"]) is int and nxt["t_h"] == 117)

    a0 = make_active(**ACTIVE0_DEFAULT)

    print("排程 S1：第一個 frame 與時間關係（單色 (120, 130, 140) × 3）")
    frames = [_filled(9, 9, (120, 130, 140)) for _ in range(3)]
    recs = run_video(frames, a0)
    c = (120, 130, 140)
    _expect_trace(ck, recs, [
        (128, (255,) * 3, (255,) * 3, False, False, 130, c, c),   # 首次：遠群空 → 用近群本張的值
        (130, c, c, False, False, 130, c, c),                     # 遠群空 → 沿用；近群差 0
        (130, c, c, False, False, 130, c, c),
    ], "S1")
    ck.check("S1：frame 0 用預設的 A = 255 除霧", np.array_equal(
        recs[0]["result"]["out"], apply_pass(frames[0], make_active(128, (255,) * 3, (255,) * 3))["out"]))

    print("排程 S2：近群差 3／差 4（單色序列，遠群一直是空的）")
    colors = [(120, 130, 140), (123, 130, 140), (127, 130, 140), (127, 130, 140)]
    frames = [_filled(9, 9, col) for col in colors]
    recs = run_video(frames, a0)
    _expect_trace(ck, recs, [
        (128, (255,) * 3, (255,) * 3, False, False, 130, c, c),
        (130, c, c, False, False, 131, c, (123, 130, 140)),                       # 差 3：採用本張
        (131, c, (123, 130, 140), False, True, 132, c, (125, 130, 140)),          # 差 4：floor((127+123)/2)
        (132, c, (125, 130, 140), False, False, 132, c, (127, 130, 140)),         # 差 2：採用本張
    ], "S2")
    ck.check("S2：遠群每張都是空的，A_F 一直沿用 (120, 130, 140)、I_max = 390",
             all(not r["stats"]["far"]["valid"] and r["next"]["i_max"] == 390 for r in recs))

    print("排程 S3：遠群變空、近群逐張收斂（(130,130,130) × 2 → (100,110,120) × 4）")
    colors = [(130, 130, 130)] * 2 + [(100, 110, 120)] * 4
    frames = [_filled(9, 9, col) for col in colors]
    recs = run_video(frames, a0)
    g, b = (130, 130, 130), (100, 110, 120)
    _expect_trace(ck, recs, [
        (128, (255,) * 3, (255,) * 3, False, False, 130, g, g),        # dark 130 >= 128 → 遠群；近群空 → 用遠群
        (130, g, g, False, False, 130, g, g),                          # dark 130 = T_h 130 → 仍歸遠群
        (130, g, g, False, True, 110, g, (115, 120, 125)),             # 換色：近群差 30 → 平均；遠群空 → 沿用
        (110, g, (115, 120, 125), False, True, 110, g, (107, 115, 122)),
        (110, g, (107, 115, 122), False, True, 110, g, (103, 112, 121)),
        (110, g, (103, 112, 121), False, False, 110, g, b),            # 差 3 → 採用本張，收斂
    ], "S3")

    print("排程 S4：靜態排程（第一次只統計，第二次用新的參數除霧）")
    img = _filled(9, 9, (100, 110, 120))
    rec = run_static(img, a0)
    ck.check("手算：用預設 T_h 128 分群 → 近群 (100, 110, 120)；遠群空 → 用近群；下一個 T_h = 110",
             rec["stats_active"] is a0 and rec["stats"]["t_h_prev"] == 128
             and rec["next"]["a_far"] == rec["next"]["a_near"] == b and rec["next"]["t_h"] == 110)
    ck.check("S4：除霧用 CPU 更新後的 active，輸出和直接呼叫 apply_pass 相同",
             rec["used"] is rec["next"] and np.array_equal(rec["result"]["out"], apply_pass(img, rec["used"])["out"]))
    ck.check("手算：A = I → J = I，輸出等於輸入", np.array_equal(rec["result"]["out"], img))

    # 兩區人工圖 X（30×9）：上 15 列 Q，下 15 列 P；降採樣 10×3，列 0–4 是 Q、列 5–9 是 P。
    # dark：列 0–6 的 5×5 視窗碰到 Q → 60；列 7–9 只有 P → 120。
    # gray_sum = 15×210 + 15×375 = 8775，平均 = floor(8775/90) = 97。
    q, p = (60, 70, 80), (120, 125, 130)
    x_img = np.concatenate([_filled(15, 9, q), _filled(15, 9, p)])

    print("排程 S5：靜態排程必須用 active0 的 T_h 分群（T_h 128 與 97 會選出不同的 A_N）")
    rec = run_static(x_img, a0)
    # T_h 128：30 點都歸近群，最大 dark 120 第一個在 (7, 0) → A_N = P；遠群空 → A_F 也用 P。
    # 若誤用該圖自己的平均 97：遠群 = 列 7–9（A_F = P），近群 = 列 0–6，第一個 (0, 0) → A_N = Q。
    ck.check("手算 S5：T_h_prev = 128，近群 30 點、最大 dark 120 → A_F = A_N = P；下一個 T_h = 97",
             rec["stats"]["t_h_prev"] == 128 and rec["stats"]["near"]["count"] == 30
             and not rec["stats"]["far"]["valid"] and rec["used"]["a_far"] == rec["used"]["a_near"] == p
             and rec["used"]["t_h"] == 97 and rec["stats"]["gray_mean"] == 97, str(rec["used"]))

    print("排程 S6：同一張圖重複 10 張（影片），手算收斂過程與收斂後的穩定")
    recs = run_video([x_img] * 10, a0)
    # frame 1 起 T_h = 97：遠群 P（差 0）；近群 Q，和上一個 A_N 的差 60 → 30 → 15 → 7 → 3
    _expect_trace(ck, recs, [
        (128, (255,) * 3, (255,) * 3, False, False, 97, p, p),          # 首次：遠群空 → 用近群本張 P
        (97, p, p, False, True, 97, p, (90, 97, 105)),                  # 差 60：floor((Q + P)/2)
        (97, p, (90, 97, 105), False, True, 97, p, (75, 83, 92)),       # 差 30
        (97, p, (75, 83, 92), False, True, 97, p, (67, 76, 86)),        # 差 15
        (97, p, (67, 76, 86), False, True, 97, p, (63, 73, 83)),        # 差 7
        (97, p, (63, 73, 83), False, False, 97, p, q),                  # 差 3：採用本張，收斂
    ] + [(97, p, q, False, False, 97, p, q)] * 4, "S6")
    ck.check("S6：frame 1 起每張的 stats（gray_mean、兩群）都相同",
             all(_summary(r["stats"]) == _summary(recs[1]["stats"]) for r in recs[1:]))
    ck.check("S6：收斂後（frame 6–9）不再觸發，除霧用的 A 不變",
             all(not r["update"]["far"]["triggered"] and not r["update"]["near"]["triggered"]
                 and r["used"]["a_far"] == p and r["used"]["a_near"] == q for r in recs[6:]))

    print("排程 S7：active0 的 T_h 等於該圖自己的 T_h（97），frame 1 起 A 不變")
    recs = run_video([x_img] * 10, make_active(97, (255,) * 3, (255,) * 3))
    _expect_trace(ck, recs, [(97, (255,) * 3, (255,) * 3, False, False, 97, p, q)]
                  + [(97, p, q, False, False, 97, p, q)] * 9, "S7")


def _selfcheck_inv_quant(ck):
    """reciprocal 局部量化實驗：預期值用手算數字、整數不等式與 Fraction，不呼叫被測公式重算。"""
    from fractions import Fraction as F
    from math import floor

    print("量化 R1：q 的端點（手算）")
    hand = {16: {0: 0, 1: 65536, 2: 32768, 3: 21845, 7: 9362, 375: 174, 390: 168, 600: 109, 765: 85},
            20: {0: 0, 1: 1048576, 2: 524288, 3: 349525, 7: 149796, 375: 2796, 390: 2688, 600: 1747, 765: 1370}}
    for frac, table in hand.items():
        got = {i: inv_imax_q(i, frac) for i in table}
        ck.check(f"F={frac}：q(0, 1, 2, 3, 7, 375, 390, 600, 765) = {list(table.values())}", got == table, str(got))
        ck.check(f"F={frac}：I_max = 1 → q = 2^F，active 的 1/I_max 恰好是 1.0",
                 make_active(0, (1, 0, 0), (0, 0, 0), inv_frac=frac)["inv_i_max"] == 1.0)

    print("量化 R2：I_max = 1–765 全範圍（整數不等式與 Fraction）")
    for frac in INV_FRACS:
        one = 1 << frac
        qs = {i: inv_imax_q(i, frac) for i in range(1, 766)}
        ck.check(f"F={frac}：全部滿足 I_max·q <= 2^F < I_max·(q + 1)",
                 all(i * qv <= one < i * (qv + 1) for i, qv in qs.items()))
        ck.check(f"F={frac}：q 等於 Fraction(2^F, I_max) 取 floor", all(qv == floor(F(one, i)) for i, qv in qs.items()))
        errs = [F(1, i) - F(qv, one) for i, qv in qs.items()]
        ck.check(f"F={frac}：reciprocal 誤差 0 <= 1/I_max − q/2^F < 2^-F（不會高估）", all(0 <= e < F(1, one) for e in errs))
        worst = max(errs)
        rel = max(e * i for e, i in zip(errs, qs))
        print(f"  紀錄：F={frac} reciprocal 最大絕對誤差 {float(worst):.3e}，最大相對誤差 {float(rel):.3e}")
        black = np.zeros((1, 1, 3), np.uint8)
        ok, exact = True, True
        for i, qv in qs.items():
            a = (min(i, 255), min(max(i - 255, 0), 255), max(i - 510, 0))
            act = make_active(0, a, (0, 0, 0), inv_frac=frac)
            exact &= F(act["inv_i_max"]) == F(qv, one)
            try:
                eta = F(float(apply_pass(black, act)["eta"][0, 0]))
                ok &= eta == F(i * qv, one) and eta <= 1
            except ValueError:
                ok = False
        ck.check(f"F={frac}：active 的 1/I_max 精確等於 q/2^F（Fraction 比較）", exact)
        ck.check(f"F={frac}：純黑 pixel（最大 η 分支）的 η = I_max·q/2^F <= 1，沒有報錯", ok)

    print("量化 R3：m9 的 pixel（I = (60, 90, 120)，I_max = 600，三色和 270）")
    px9 = np.array([[[60, 90, 120]]], np.uint8)
    af9, an9 = (220, 200, 180), (100, 120, 140)
    r_float = apply_pass(px9, make_active(0, af9, an9))
    for frac, eta_hand in ((16, F(330 * 109, 1 << 16)), (20, F(330 * 1747, 1 << 20))):
        r = apply_pass(px9, make_active(0, af9, an9, inv_frac=frac))
        ck.check(f"F={frac}：主路徑使用量化的 1/I_max：η = 330·q/2^F = {float(eta_hand):.9f}（手算 q）",
                 F(float(r["eta"][0, 0])) == eta_hand, f"{r['eta'][0, 0]!r}")
        print(f"  紀錄：F={frac} η {float(r_float['eta'][0, 0]):.9f} → {float(r['eta'][0, 0]):.9f}；"
              f"輸出 {r_float['out'][0, 0].tolist()} → {r['out'][0, 0].tolist()}")

    print("量化 R4：排程（S3 序列、S6 的兩區圖 X × 10）T_h、A、step 4 和浮點版相同，只有 reciprocal 不同")
    q_c, p_c = (60, 70, 80), (120, 125, 130)
    x_img = np.concatenate([_filled(15, 9, q_c), _filled(15, 9, p_c)])
    s3 = [_filled(9, 9, col) for col in [(130, 130, 130)] * 2 + [(100, 110, 120)] * 4]
    for name, frames, i_maxes in (("S3", s3, None), ("S6", [x_img] * 10, [765] + [375] * 9)):
        ref = run_video(frames, make_active(**ACTIVE0_DEFAULT))
        for frac in INV_FRACS:
            recs = run_video(frames, make_active(**ACTIVE0_DEFAULT, inv_frac=frac))
            ck.check(f"{name} F={frac}：T_h、A、step 4 觸發旗標逐張和浮點版相同",
                     [_trace_key(r) for r in recs] == [_trace_key(r) for r in ref])
            ck.check(f"{name} F={frac}：每張的 active 都帶 F，q 滿足 I_max·q <= 2^F < I_max·(q + 1)",
                     all(r["used"]["inv_frac"] == frac and r["next"]["inv_frac"] == frac
                         and r["used"]["i_max"] * r["used"]["inv_i_max_q"] <= (1 << frac)
                         < r["used"]["i_max"] * (r["used"]["inv_i_max_q"] + 1) for r in recs))
            if i_maxes:
                ck.check(f"{name} F={frac}：每張的 I_max 與 q 和手算相同（frame 0 為 765，之後 375）",
                         [(r["used"]["i_max"], r["used"]["inv_i_max_q"]) for r in recs]
                         == [(i, hand[frac][i]) for i in i_maxes])
            diff = sum(int(np.count_nonzero(a["result"]["out"] != b["result"]["out"])) for a, b in zip(ref, recs))
            print(f"  紀錄：{name} F={frac} 和浮點版輸出不同的 channel 值 {diff} 個（共 {3 * sum(f.size // 3 for f in frames)} 個）")

    print("量化 R5：輸入檢查（必須丟 ValueError；其他例外記成 FAIL）")
    for label, call in (("F = 12", lambda: inv_imax_q(600, 12)), ("I_max = 766", lambda: inv_imax_q(766, 16)),
                        ("I_max 是 bool", lambda: inv_imax_q(True, 16)),
                        ("I_max 是 numpy 整數", lambda: inv_imax_q(np.int64(600), 16)),
                        ("F = 16.0", lambda: inv_imax_q(600, 16.0)),
                        ("F 是 numpy 整數", lambda: inv_imax_q(600, np.int64(16))),
                        ("make_active 的 F = 16.0", lambda: make_active(0, (200,) * 3, (100,) * 3, inv_frac=16.0)),
                        ("make_active 的 F 是 numpy 整數",
                         lambda: make_active(0, (200,) * 3, (100,) * 3, inv_frac=np.int64(20)))):
        try:
            call()
            ck.check(f"{label} 應拒絕", False)
        except ValueError:
            ck.check(f"{label} 應拒絕", True)
        except Exception as e:
            ck.check(f"{label} 應拒絕", False, f"丟出 {type(e).__name__}，不是 ValueError")

    print("量化 R6：比較報表的計算（手算）")
    a = np.array([[[0, 0, 5], [7, 7, 7]]], np.uint8)
    b = np.array([[[3, 0, 3], [7, 7, 7]]], np.uint8)     # 第一個 pixel 差 (3, 0, 2)，第二個沒有差
    s1 = _diff_stats(a, b)
    ck.check("R6 單張：1×2 圖，不同的 channel 值 2、有差異的 pixel 1、絕對差總和 5、MAE = 5/6、最大差 3",
             s1 == {"n": 6, "diff": 2, "pix": 1, "abs": 5, "max": 3, "mae": 5 / 6}, str(s1))
    s2 = _diff_stats(np.full((1, 1, 3), 10, np.uint8), np.array([[[11, 9, 11]]], np.uint8))
    tot = _sum_stats([s1, s2])
    # 依 channel 數加權：(5 + 3) / (6 + 3) = 8/9；直接平均兩張的 MAE 會是 (5/6 + 1)/2 = 11/12
    ck.check("R6 合計：再加 1×1、差 (1, 1, 1) 的圖 → MAE = 8/9（依 channel 數加權，不是 11/12）",
             tot == {"n": 9, "diff": 5, "pix": 2, "abs": 8, "max": 3, "mae": 8 / 9}, str(tot))
    j = np.array([[[254.5, 255.5, -0.5], [-0.6, 255.49, 0.0]]])
    # 捨入（.5 進位）後：255、256、0、−1、255、0 → 超過 255 的 1 個、低於 0 的 1 個
    ck.check("R6 clipping：J = 254.5、255.49 捨入成 255 不算；255.5 算；−0.5 捨入成 0 不算；−0.6 算 → (1, 1)",
             _clip_counts(j) == (1, 1), str(_clip_counts(j)))


def _rgb_sum(s):
    """三色和為 s（0–765）的一個 RGB。"""
    return (min(s, 255), min(max(s - 255, 0), 255), max(s - 510, 0))


def _selfcheck_pala_fixed(ck):
    """PALA 局部定點模型：預期值是手算數字、整數不等式，或另一種寫法 256·A_F − η·(A_F − A_N)。"""
    def fx_of(pixels, af, an, frac):
        frame = np.array([pixels], np.uint8)
        return pala_fixed(frame, af, an, inv_imax_q(sum(af), frac), frac)

    def try_check(label, fn):
        try:
            ck.check(label, fn())
        except ValueError as e:
            ck.check(label, False, f"ValueError：{e}")

    print("PALA P1：m9 的 pixel（I = (60, 90, 120)，S = 270；A_F = (220, 200, 180)，I_max = 600；A_N = (100, 120, 140)）")
    px9, af9, an9 = (60, 90, 120), (220, 200, 180), (100, 120, 140)
    # delta = 330。F=16：q = 109，product = 35970，eta_q = floor(35970/256) = 140；
    # F=20：q = 1747，product = 576510，eta_q = floor(576510/4096) = 140。
    # A_q = 116·A_F + 140·A_N = (25520 + 14000, 23200 + 16800, 20880 + 19600) = (39520, 40000, 40480)
    for frac, prod in ((16, 35970), (20, 576510)):
        r = fx_of([px9], af9, an9, frac)
        ck.check(f"P1 F={frac}：product = {prod}，eta_q = 140（手算）",
                 int(r["product"][0, 0]) == prod and int(r["eta_q"][0, 0]) == 140, f"{r['product'][0, 0]}, {r['eta_q'][0, 0]}")
        ck.check(f"P1 F={frac}：A_q = (39520, 40000, 40480)，即 A = (154.375, 156.25, 158.125)，保留小數",
                 r["a_q"][0, 0].tolist() == [39520, 40000, 40480], str(r["a_q"][0, 0]))
        act = make_active(0, af9, an9, inv_frac=frac, pala_fixed=True)
        try_check(f"P1 F={frac}：轉接到浮點 PTE／SD 時 eta = eta_q/256、A = A_q/256（逐值相等）",
                  lambda: (lambda o: float(o["eta"][0, 0]) == 140 / 256
                           and o["a_map"][0, 0].tolist() == [39520 / 256, 40000 / 256, 40480 / 256]
                           and o["eta_q"][0, 0] == 140)(apply_pass(np.array([[px9]], np.uint8), act)))

    print("PALA P2：分支與端點（手算）")
    af, an = (200, 200, 200), (100, 50, 0)
    pix = [(0, 0, 0), _rgb_sum(599), (200, 200, 200), _rgb_sum(700)]     # S = 0、599、600 = I_max、700
    # S = 0：delta 600。F=16：product = 600·109 = 65400，eta_q = 255；F=20：600·1747 = 1048200，eta_q = 255
    #        A_q = 1·200 + 255·(100, 50, 0) = (25700, 12950, 200)
    # S = 599：delta 1，product = q，eta_q = 0 → A_q = 256·200 = 51200；S >= I_max → eta_q 0、A_q 51200
    for frac, p0, p1 in ((16, 65400, 109), (20, 1048200, 1747)):
        r = fx_of(pix, af, an, frac)
        ck.check(f"P2 F={frac}：S 小於／等於／大於 I_max 的 product = ({p0}, {p1}, 0, 0)、eta_q = (255, 0, 0, 0)",
                 r["product"][0].tolist() == [p0, p1, 0, 0] and r["eta_q"][0].tolist() == [255, 0, 0, 0])
        ck.check(f"P2 F={frac}：A_q = (25700, 12950, 200)，其餘三個 pixel 是 A_F·256 = 51200",
                 r["a_q"][0].tolist() == [[25700, 12950, 200]] + [[51200] * 3] * 3, str(r["a_q"][0].tolist()))
        r = fx_of([(0, 0, 0), (5, 5, 5)], (0, 0, 0), (10, 20, 30), frac)
        ck.check(f"P2 F={frac}：I_max = 0 → 全部走 A_F 分支，eta_q = 0、A_q = 0、不做乘法",
                 r["eta_q"][0].tolist() == [0, 0] and r["a_q"][0].tolist() == [[0, 0, 0]] * 2 and r["product"][0].tolist() == [0, 0])
        r = fx_of([(0, 0, 0), (1, 0, 0)], (1, 0, 0), (10, 20, 30), frac)
        ck.check(f"P2 F={frac}：I_max = 1 → 純黑 pixel 的 product = 2^F、eta_q = 256（1.0），A_q = 256·A_N；S = 1 走 A_F",
                 r["product"][0].tolist() == [1 << frac, 0] and r["eta_q"][0].tolist() == [256, 0]
                 and r["a_q"][0].tolist() == [[2560, 5120, 7680], [256, 0, 0]])
        e765 = {16: 254, 20: 255}[frac]          # 765·85 = 65025 → 254；765·1370 = 1048050 → 255
        r = fx_of([(0, 0, 0)], (255, 255, 255), (0, 0, 0), frac)
        ck.check(f"P2 F={frac}：I_max = 765、純黑 pixel → eta_q = {e765}，A_q = (256 − {e765})·255",
                 int(r["eta_q"][0, 0]) == e765 and r["a_q"][0, 0].tolist() == [(256 - e765) * 255] * 3)
        r = fx_of([(0, 0, 0), (100, 50, 25)], (255, 255, 255), (255, 255, 255), frac)
        ck.check(f"P2 F={frac}：A_F = A_N = 255 → A_q = 65280（最大合法值），不論 eta_q", r["a_q"][0].tolist() == [[65280] * 3] * 2)
        r = fx_of([(0, 0, 0), (40, 50, 60)], (30, 60, 90), (30, 60, 90), frac)
        ck.check(f"P2 F={frac}：A_F = A_N → A_q = 256·A（eta_q 不影響）", r["a_q"][0].tolist() == [[7680, 15360, 23040]] * 2)

    print("PALA P3：截位邊界（floor，不是四捨五入）")
    full, zero = (255, 255, 255), (0, 0, 0)
    # F=16、I_max 765、q 85：S = 509 → 256·85 = 21760 → 85；S = 506 → 259·85 = 22015 = 86·256 − 1 → 85；S = 505 → 22100 → 86
    r = fx_of([_rgb_sum(509), _rgb_sum(506), _rgb_sum(505)], full, zero, 16)
    ck.check("P3 F=16：product (21760, 22015, 22100) → eta_q (85, 85, 86)",
             r["product"][0].tolist() == [21760, 22015, 22100] and r["eta_q"][0].tolist() == [85, 85, 86])
    # F=20、I_max 765、q 1370：S = 182 → 583·1370 = 798710 = 195·4096 − 10 → 194；S = 181 → 800080 → 195
    r = fx_of([_rgb_sum(182), _rgb_sum(181)], full, zero, 20)
    ck.check("P3 F=20：product (798710, 800080) → eta_q (194, 195)",
             r["product"][0].tolist() == [798710, 800080] and r["eta_q"][0].tolist() == [194, 195])
    # I_max = 256（A_F = (86, 85, 85)）：q16 = 256、q20 = 4096，product 剛好是 2^(F−8) 的倍數 → eta_q = delta
    for frac in INV_FRACS:
        r = fx_of([(0, 0, 0), _rgb_sum(1), _rgb_sum(100)], (86, 85, 85), (0, 0, 0), frac)
        ck.check(f"P3 F={frac}：I_max = 256 → eta_q = delta = (256, 255, 156)", r["eta_q"][0].tolist() == [256, 255, 156])

    print("PALA P4：I_max = 1–765、S = 0–765 全範圍（整數不等式與另一種寫法）")
    frame = np.array([[_rgb_sum(s) for s in range(766)]], np.uint8)
    s_all = np.arange(766, dtype=np.int64)
    an = (255, 0, 128)
    for frac in INV_FRACS:
        one, step = 1 << frac, 1 << (frac - 8)
        ok_p = ok_e = ok_a = True
        mx = [0, 0, 0]
        for i_max in range(1, 766):
            af = _rgb_sum(i_max)
            qv = inv_imax_q(i_max, frac)
            r = pala_fixed(frame, af, an, qv, frac)
            prod, eta, a_q = r["product"][0], r["eta_q"][0], r["a_q"][0]
            below = s_all < i_max
            ok_p &= bool(np.all(prod[below] == (i_max - s_all[below]) * qv) and np.all(prod[~below] == 0)
                         and np.all(prod <= one))
            ok_e &= bool(np.all((eta[below] * step <= prod[below]) & (prod[below] < (eta[below] + 1) * step))
                         and np.all(eta[~below] == 0) and np.all((eta >= 0) & (eta <= 256)))
            afv, anv = np.array(af, np.int64), np.array(an, np.int64)
            ok_a &= bool(np.all(a_q == 256 * afv - eta[:, None] * (afv - anv))
                         and np.all((a_q >= 256 * np.minimum(afv, anv)) & (a_q <= 256 * np.maximum(afv, anv)))
                         and np.all((a_q >= 0) & (a_q <= 65280)))
            mx = [max(mx[0], int(prod.max())), max(mx[1], int(eta.max())), max(mx[2], int(a_q.max()))]
        ck.check(f"P4 F={frac}：product = (I_max − S)·q（S < I_max）、其餘為 0，且 <= 2^F", ok_p)
        ck.check(f"P4 F={frac}：eta_q·2^(F−8) <= product < (eta_q + 1)·2^(F−8)（floor），eta_q 在 0–256", ok_e)
        ck.check(f"P4 F={frac}：A_q = 256·A_F − eta_q·(A_F − A_N)，各色介於 A_F·256 與 A_N·256 之間，0–65280", ok_a)
        print(f"  紀錄：F={frac} 實測最大 product {mx[0]}（{mx[0].bit_length()} bit）、eta_q {mx[1]}（{mx[1].bit_length()} bit）、"
              f"A_q {mx[2]}（{mx[2].bit_length()} bit）")

    print("PALA P5：排程（S3 序列、S6 的兩區圖 X × 10）三種模式的 T_h、A、step 4 相同")
    q_c, p_c = (60, 70, 80), (120, 125, 130)
    x_img = np.concatenate([_filled(15, 9, q_c), _filled(15, 9, p_c)])
    s3 = [_filled(9, 9, col) for col in [(130, 130, 130)] * 2 + [(100, 110, 120)] * 4]
    for name, frames in (("S3", s3), ("S6", [x_img] * 10)):
        ref = run_video(frames, make_active(**ACTIVE0_DEFAULT))
        for frac in INV_FRACS:
            recip = run_video(frames, make_active(**ACTIVE0_DEFAULT, inv_frac=frac))
            try:
                recs = run_video(frames, make_active(**ACTIVE0_DEFAULT, inv_frac=frac, pala_fixed=True))
            except ValueError as e:
                ck.check(f"{name} F={frac}：定點 PALA 排程執行完成", False, str(e))
                continue
            ck.check(f"{name} F={frac}：T_h、A_F、A_N、step 4 觸發旗標逐張和浮點版相同",
                     [_trace_key(r) for r in recs] == [_trace_key(r) for r in ref])
            ck.check(f"{name} F={frac}：每張的 active 都是定點 PALA，且帶 F",
                     all(r["used"]["pala_fixed"] and r["next"]["pala_fixed"] and r["used"]["inv_frac"] == frac for r in recs))
            d1 = sum(int(np.count_nonzero(a["result"]["out"] != b["result"]["out"])) for a, b in zip(recip, recs))
            d2 = sum(int(np.count_nonzero(a["result"]["out"] != b["result"]["out"])) for a, b in zip(ref, recs))
            print(f"  紀錄：{name} F={frac} 輸出不同的 channel 值：只量化 reciprocal → 加定點 PALA {d1} 個；"
                  f"理想浮點 → 加定點 PALA {d2} 個（共 {sum(f.size for f in frames)} 個）")

    print("PALA P6：輸入檢查")
    good = make_active(0, af9, an9, inv_frac=16, pala_fixed=True)
    px = np.array([[px9]], np.uint8)
    for label, call in (("pala_fixed=True 但沒有 inv_frac", lambda: make_active(0, af9, an9, pala_fixed=True)),
                        ("pala_fixed 不是 bool", lambda: make_active(0, af9, an9, inv_frac=16, pala_fixed=1)),
                        ("q 超過 2^F", lambda: apply_pass(px, dict(good, inv_i_max_q=(1 << 16) + 1))),
                        ("q 是浮點數", lambda: apply_pass(px, dict(good, inv_i_max_q=109.0)))):
        try:
            call()
            ck.check(f"{label} 應拒絕", False)
        except ValueError:
            ck.check(f"{label} 應拒絕", True)
        except Exception as e:
            ck.check(f"{label} 應拒絕", False, f"丟出 {type(e).__name__}，不是 ValueError")

    print("PALA P7：_compare_pala 報表的 η、A 誤差欄（手算，對照印出的字串）")
    # m9 的 pixel、F=16：只量化 reciprocal 的 η = 35970/65536，定點 PALA 的 η = 140/256 = 35840/65536 → 差 130/65536；
    # A = A_F − η·(A_F − A_N)，A_F − A_N = (120, 80, 40) → A 最大差 = 120·130/65536 = 0.238037109375
    import contextlib
    import io

    one = [np.array([[px9]], np.uint8)]
    buf = io.StringIO()
    try:
        recip = run_video(one, make_active(0, af9, an9, inv_frac=16))
        recs = run_video(one, make_active(0, af9, an9, inv_frac=16, pala_fixed=True))
        with contextlib.redirect_stdout(buf):
            same = _compare_pala(recip, recs, "P7")
        row = next(line for line in buf.getvalue().splitlines() if line.startswith("| 0 |"))
        cols = [c.strip() for c in row.split("|")[1:-1]]
    except ValueError as e:
        same, cols = False, ["", "", str(e)]
    ck.check("P7 報表的 η 最大差欄 = 1.984e-03（130/65536）", same and cols[1] == "1.984e-03", str(cols[1:3]))
    ck.check("P7 報表的 A 最大差欄 = 2.380e-01（120·130/65536）", same and cols[2] == "2.380e-01", str(cols[1:3]))


def _pte_oracle(rgb, aq, t_frac):
    """PTE 定點契約的 Fraction 寫法（單一 pixel，不呼叫 pte_fixed）：每一步依數學定義取 floor，
    t 用 ceil((1 − ω·m·k)·2^T) 再和 t_min_q 取大，回傳 (r_q, k_q, m_q, t_q)。"""
    import math
    from fractions import Fraction as F

    i, a = [int(v) for v in rgb], [int(v) for v in aq]
    lo, hi = min(i), max(i)
    r = math.floor(F(lo, lo + hi) * 2 ** 16) if lo + hi else 0
    k = 2 ** 16 - math.floor(F(r * r, 2 ** 16))
    ms = [math.floor(F(256 * x, y) * 2 ** 16) for x, y in zip(i, a) if y > 0]   # I/A = 256·I/A_q
    m = min(min(ms), 2 ** 17 - 1) if ms else 0
    t_up = math.ceil((1 - F(15, 16) * F(m, 2 ** 16) * F(k, 2 ** 16)) * 2 ** t_frac)
    return r, k, m, max(t_up, 2 ** (t_frac - 10))


def _selfcheck_pte_fixed(ck):
    """PTE 理想定點 reference：預期值是手算數字、整數不等式，或 Fraction 寫法（_pte_oracle）。"""
    def fx_of(pixels, aqs, frac):
        return pte_fixed(np.array([pixels], np.uint8), np.array([aqs], np.int64), frac)

    def at(r, key, n=0):
        return r[key][0, n].tolist()

    px9, aq9 = (60, 90, 120), (39520, 40000, 40480)
    print("PTE X1–X6：手算（T = 10 ／ 16；A = A_q/256，m 的分母是 A_q，分子 I·2^24）")
    # X1：lo 60、hi 120、s 180 → r_q = floor(60·65536/180) = 21845；r_q² = 477204025 → >>16 = 7281；k_q = 58255
    #     m_c = floor(I·2^24/A_q) = (25471, 37748, 49734)，m_q 25471；p = 15·25471·58255 = 22257196575
    #     T=10：p/2^26 = 331.66 → u 331 → t_q 693；T=16：p/2^20 = 21226.6 → u 21226 → t_q 44310
    for frac, u1, t1 in ((10, 331, 693), (16, 21226, 44310)):
        r = fx_of([px9], [aq9], frac)
        ck.check(f"X1 T={frac}：r_q、k_q = 21845、58255", (at(r, "r_q"), at(r, "r2_q"), at(r, "k_q")) == (21845, 7281, 58255))
        ck.check(f"X1 T={frac}：m_c = (25471, 37748, 49734)、m_q 25471", at(r, "m_c") == [25471, 37748, 49734] and at(r, "m_q") == 25471,
                 str(at(r, "m_c")))
        ck.check(f"X1 T={frac}：u、t_q = {u1}、{t1}（p = 22257196575）", (at(r, "p"), at(r, "u"), at(r, "t_q")) == (22257196575, u1, t1),
                 str((at(r, "p"), at(r, "u"), at(r, "t_q"))))
    for frac, t2 in ((10, 664), (16, 42496)):
        # X2：灰色、A = 200：r = 0.5、k = 0.75、m = 0.5；t = 1 − 15/16·0.5·0.75 = 0.6484375，可精確表示
        r = fx_of([(100, 100, 100)], [(51200,) * 3], frac)
        ck.check(f"X2 T={frac}：灰色 pixel → r_q 32768、k_q 49152、m_q 32768、t_q {t2}（= 0.6484375·2^T，和浮點相同）",
                 (at(r, "r_q"), at(r, "k_q"), at(r, "m_q"), at(r, "t_q")) == (32768, 49152, 32768, t2) and t2 / (1 << frac) == 0.6484375)
        # X3：純黑 → s = 0、r_q 0、k_q 65536、m_q 0 → t_q = 2^T；X3b：lo = 0 的彩色 pixel → k = 1、m = 0 → t = 1
        r = fx_of([(0, 0, 0), (0, 100, 200)], [aq9] * 2, frac)
        ck.check(f"X3 T={frac}：純黑與 lo = 0 的 pixel → r_q 0、k_q 65536、m_q 0、t_q = 2^T（1.0）",
                 r["r_q"][0].tolist() == [0, 0] and r["k_q"][0].tolist() == [65536] * 2
                 and r["m_q"][0].tolist() == [0, 0] and r["t_q"][0].tolist() == [1 << frac] * 2)
    for frac, u4, t4 in ((10, 1439, 1), (16, 92159, 64)):
        # X4：I = 255、A = 1 → m_c = 255·2^24/256 = 16711680 → 飽和 131071；u = floor(15·131071·49152/2^(36−T))
        r = fx_of([(255, 255, 255)], [(256,) * 3], frac)
        ck.check(f"X4 T={frac}：m_c 16711680 → m_q 飽和 131071，u {u4} → t_q = t_min_q = {t4}（t = 1/1024）",
                 at(r, "m_c") == [16711680] * 3 and (at(r, "m_q"), at(r, "u"), at(r, "t_q")) == (131071, u4, t4))
    for frac, t5 in ((10, 935), (16, 59777)):
        # X5：A_q[R] = 0 不參加；m_q = floor(20·2^24/51200) = 6553（6553.6）；浮點 t·2^T 剛好 934 ／ 59776，差異來自 m 的 floor
        r = fx_of([(10, 20, 30)], [(0, 51200, 51200)], frac)
        ck.check(f"X5 T={frac}：A_q[R] = 0 → m_c[R] 不參加（−1），m_q 6553，t_q {t5}",
                 at(r, "m_c")[0] == -1 and at(r, "m_q") == 6553 and at(r, "t_q") == t5, str((at(r, "m_c"), at(r, "t_q"))))
        r = fx_of([(10, 20, 30)], [(0, 0, 0)], frac)
        ck.check(f"X6 T={frac}：三色的 A_q 都是 0 → m_q 0、t_q = 2^T", at(r, "m_q") == 0 and at(r, "t_q") == 1 << frac)

    print("PTE T2：截斷與最大合法輸入的邊界（手算）")
    # 灰色 pixel 的 k_q = 49152，15·49152 = 45·2^14。T=10：u = floor(45·m/4096)；T=16：u = floor(45·m/64)
    # T=10，界線 2^T − t_min_q = 1023：A_q 45945 → m 93115 → u 1022 → t_q 2；45900 → 93206 → 1023 → 1；45856 → 93296 → 1024 → 截成 1
    # T=16，界線 65472：A_q 45945 → u 65471 → 65；45944 → m 93117 → 65472 → 64；I 253、A_q 45583 → m 93118 → 65473 → 截成 64（不是 63）
    for frac, cases in ((10, [(255, 45945, 1022, 2), (255, 45900, 1023, 1), (255, 45856, 1024, 1)]),
                        (16, [(255, 45945, 65471, 65), (255, 45944, 65472, 64), (253, 45583, 65473, 64)])):
        r = fx_of([(v, v, v) for v, _, _, _ in cases], [(a,) * 3 for _, a, _, _ in cases], frac)
        ck.check(f"T2 T={frac}：截斷邊界 u = 2^T − t_min_q − 1、= 、+ 1 → t_q = {[c[3] for c in cases]}"
                 f"（t_min_q = {1 << (frac - 10)} LSB = 1/1024）",
                 r["u"][0].tolist() == [c[2] for c in cases] and r["t_q"][0].tolist() == [c[3] for c in cases],
                 str((r["u"][0].tolist(), r["t_q"][0].tolist())))
    r = fx_of([(255, 255, 255), (255, 255, 255)], [(1,) * 3, (65280,) * 3], 16)
    ck.check("T2 最大合法輸入：I 255、A_q 1 → m_c = 255·2^24 = 4278190080（< 2^32）→ 飽和；A_q 65280 → m_c 65536（1.0）",
             at(r, "m_c") == [4278190080] * 3 and at(r, "m_q") == M_SAT and at(r, "m_c", 1) == [65536] * 3)

    print("PTE T3：性質（整數不等式、Fraction 寫法、位寬）")
    lo_hi = [(lo, hi) for lo in range(256) for hi in range(lo, 256)]
    frame_r = np.array([[(lo, hi, lo) for lo, hi in lo_hi]], np.uint8)
    lo_a, s_a = np.array([lo for lo, _ in lo_hi]), np.array([lo + hi for lo, hi in lo_hi])
    r = pte_fixed(frame_r, np.full(frame_r.shape, 51200, np.int64), 10)
    rq, r2, kq = r["r_q"][0], r["r2_q"][0], r["k_q"][0]
    ok_r = bool(np.all(np.where(s_a > 0, (rq * s_a <= lo_a << 16) & (lo_a << 16 < (rq + 1) * s_a), rq == 0)) and rq.max() == 32768)
    ok_k = bool(np.all((r2 << 16 <= rq * rq) & (rq * rq < (r2 + 1) << 16)) and np.all(kq == 65536 - r2)
                and kq.min() == 49152 and kq.max() == 65536)
    ck.check(f"T3 lo <= hi 全部 {len(lo_hi)} 組：r_q·s <= lo·2^16 < (r_q + 1)·s，s = 0 時 0，最大 32768", ok_r)
    ck.check("T3 同上：r2_q·2^16 <= r_q² < (r2_q + 1)·2^16，k_q = 2^16 − r2_q，範圍 49152–65536", ok_k)
    aqs = sorted(set(range(1, 65281, 97)) | {1, 2, 255, 256, 257, 25600, 45900, 65279, 65280})
    iv = np.repeat(np.arange(256), len(aqs))
    av = np.tile(np.array(aqs, np.int64), 256)
    r = pte_fixed(np.repeat(iv.astype(np.uint8)[None, :, None], 3, axis=2), np.repeat(av[None, :, None], 3, axis=2), 16)
    mc, mq, num = r["m_c"][0, :, 0], r["m_q"][0], iv.astype(np.int64) << 24
    ok_m = bool(np.all((mc * av <= num) & (num < (mc + 1) * av)) and np.all(mq == np.minimum(mc, M_SAT)) and mc.max() < 1 << 32)
    ck.check(f"T3 I = 0–255 × {len(aqs)} 個 A_q（1–65280）：m_c·A_q <= I·2^24 < (m_c + 1)·A_q，m_q = min(m_c, 2^17 − 1)，m_c < 2^32", ok_m)
    rng = np.random.default_rng(20261030)
    pix = rng.integers(0, 256, size=(1, 600, 3), dtype=np.uint8)
    aq_r = rng.integers(0, 65281, size=(1, 600, 3)).astype(np.int64)
    aq_r[rng.random((1, 600, 3)) < 0.1] = 0                         # 約 10% 的 channel 是 A_q = 0
    pix[0, :20] = rng.integers(200, 256, size=(20, 3))              # 加入會截到下限的 pixel
    aq_r[0, :20] = rng.integers(1, 30000, size=(20, 3))
    for frac in PTE_FRACS:
        r = pte_fixed(pix, aq_r, frac)
        got = list(zip(r["r_q"][0].tolist(), r["k_q"][0].tolist(), r["m_q"][0].tolist(), r["t_q"][0].tolist()))
        want = [_pte_oracle(pix[0, n], aq_r[0, n], frac) for n in range(pix.shape[1])]
        ck.check(f"T3 T={frac}：600 個隨機 pixel 的 r_q、k_q、m_q、t_q 和 Fraction 寫法（t = ceil((1 − ω·m·k)·2^T)）相同",
                 got == want, f"{sum(g != w for g, w in zip(got, want))} 個不同")
        allr = [pte_fixed(f, a, frac) for f, a in ((frame_r, np.full(frame_r.shape, 51200, np.int64)),
                                                    (np.repeat(iv.astype(np.uint8)[None, :, None], 3, axis=2),
                                                     np.repeat(av[None, :, None], 3, axis=2)), (pix, aq_r))]
        mx = {key: max(int(x[key].max()) for x in allr) for key in ("r_q", "k_q", "m_c", "m_q", "p", "u", "t_q")}
        mx["r_q²"] = mx["r_q"] ** 2
        one, tmin = 1 << frac, 1 << (frac - 10)
        ok_w = (mx["r_q"] <= 1 << 15 and mx["r_q²"] < 1 << 31 and mx["k_q"] <= 1 << 16 and mx["m_c"] < 1 << 32
                and mx["m_q"] <= M_SAT and mx["p"] < 1 << 37 and mx["u"] < 1 << (frac + 1)
                and all(bool(np.all((x["t_q"] >= tmin) & (x["t_q"] <= one))) for x in allr)
                and all(bool(np.all((x["t_q"] == one) == (x["m_q"] == 0))) for x in allr))
        ck.check(f"T3 T={frac}：位寬 r_q <= 2^15、r_q² < 2^31、k_q <= 2^16、m_c < 2^32、m_q <= 2^17 − 1、p < 2^37、"
                 f"u < 2^{frac + 1}；t_q 在 {tmin}–{one}；t_q = 2^T 若且唯若 m_q = 0", ok_w, str(mx))
        print(f"  紀錄：T={frac} 實測最大值 " + "、".join(f"{key} {v}（{v.bit_length()} bit）" for key, v in mx.items()))

    print("PTE T4：接到 apply_pass（A_q 進 PTE、t 轉接、SD 的下限）與輸入檢查")
    for frac, t1 in ((10, 693), (16, 44310)):
        act = make_active(0, (220, 200, 180), (100, 120, 140), inv_frac=16, pala_fixed=True, pte_frac=frac)
        try:
            o = apply_pass(np.array([[px9]], np.uint8), act)
            ok = (at(o, "t_q") == t1 and float(o["t"][0, 0]) == t1 / (1 << frac)
                  and float(o["k"][0, 0]) == 58255 / 65536 and float(o["m"][0, 0]) == 25471 / 65536)
        except ValueError as e:
            ok = False
        ck.check(f"T4 T={frac}：m9 經定點 PALA 得 A_q = (39520, 40000, 40480)，t_q = {t1}，t = t_q/2^T、k、m 逐值相等", ok)
        act = make_active(0, (10, 10, 10), (10, 10, 10), inv_frac=16, pala_fixed=True, pte_frac=frac)
        try:
            o = apply_pass(np.array([[(200, 200, 200)]], np.uint8), act)
            ok = at(o, "t_q") == 1 << (frac - 10) and float(o["t"][0, 0]) == 1 / 1024
        except ValueError:
            ok = False
        ck.check(f"T4 T={frac}：t_q = t_min_q 的 pixel 進 SD 後 t 仍是 1/1024（不是 1/1023）", ok)
    af9, an9 = (220, 200, 180), (100, 120, 140)
    for label, call in (("pte_frac = 12", lambda: make_active(0, af9, an9, inv_frac=16, pala_fixed=True, pte_frac=12)),
                        ("pte_frac = 10.0", lambda: make_active(0, af9, an9, inv_frac=16, pala_fixed=True, pte_frac=10.0)),
                        ("pte_frac 沒有 pala_fixed", lambda: make_active(0, af9, an9, inv_frac=16, pte_frac=10)),
                        ("定點 PTE 用 t_min 1/1023", lambda: make_active(0, af9, an9, inv_frac=16, pala_fixed=True, pte_frac=10,
                                                                     t_min=T_MIN)),
                        ("t_min = 0.001", lambda: make_active(0, af9, an9, t_min=0.001))):
        try:
            call()
            ck.check(f"{label} 應拒絕", False)
        except ValueError:
            ck.check(f"{label} 應拒絕", True)
        except Exception as e:
            ck.check(f"{label} 應拒絕", False, f"丟出 {type(e).__name__}，不是 ValueError")

    print("PTE T5：排程（S3、S6）的浮點、浮點 PTE＋下限 1/1024、定點 PTE 三種模式 trace 相同，設定逐張傳遞")
    q_c, p_c = (60, 70, 80), (120, 125, 130)
    x_img = np.concatenate([_filled(15, 9, q_c), _filled(15, 9, p_c)])
    s3 = [_filled(9, 9, col) for col in [(130, 130, 130)] * 2 + [(100, 110, 120)] * 4]
    for name, frames in (("S3", s3), ("S6", [x_img] * 10)):
        ref = run_video(frames, make_active(**ACTIVE0_DEFAULT))
        try:
            p1024 = run_video(frames, make_active(**ACTIVE0_DEFAULT, inv_frac=16, pala_fixed=True, t_min=T_MIN_FIXED))
        except ValueError as e:
            ck.check(f"{name}：浮點 PTE＋下限 1/1024 的排程執行完成", False, str(e))
            continue
        ck.check(f"{name}：浮點 PTE＋下限 1/1024 的模式，每張 active 的 t_min 都是 1/1024、pte_frac 是 None",
                 all(r[k]["t_min"] == T_MIN_FIXED and r[k]["pte_frac"] is None for r in p1024 for k in ("used", "next")))
        for frac in PTE_FRACS:
            try:
                recs = run_video(frames, make_active(**ACTIVE0_DEFAULT, inv_frac=16, pala_fixed=True, pte_frac=frac))
            except ValueError as e:
                ck.check(f"{name} T={frac}：定點 PTE 排程執行完成", False, str(e))
                continue
            key = [_trace_key(r) for r in ref]
            ck.check(f"{name} T={frac}：T_h、A_F、A_N、step 4 觸發旗標逐張和浮點版、下限 1/1024 版相同",
                     [_trace_key(r) for r in recs] == key and [_trace_key(r) for r in p1024] == key)
            ck.check(f"{name} T={frac}：每張 active 都是定點 PTE（pte_frac = {frac}、t_min 1/1024、定點 PALA）",
                     all(r[k]["pte_frac"] == frac and r[k]["t_min"] == T_MIN_FIXED and r[k]["pala_fixed"]
                         for r in recs for k in ("used", "next")))
            d = sum(int(np.count_nonzero(a["result"]["out"] != b["result"]["out"])) for a, b in zip(p1024, recs))
            print(f"  紀錄：{name} T={frac} 浮點 PTE → 定點 PTE（下限都是 1/1024）輸出不同的 channel 值 {d} 個"
                  f"（共 {sum(f.size for f in frames)} 個）")
    for frac in PTE_FRACS:
        try:
            rec = run_static(x_img, make_active(**ACTIVE0_DEFAULT, inv_frac=16, pala_fixed=True, pte_frac=frac))
            ok = rec["used"]["pte_frac"] == frac and rec["used"]["t_min"] == T_MIN_FIXED and "t_q" in rec["result"]
        except ValueError:
            ok = False
        ck.check(f"靜態 T={frac}：除霧用的 active（CPU 更新後）仍是定點 PTE，結果帶 t_q", ok)

    print("PTE T6：_compare_pala 的 t 欄位（手算，對照印出的字串）")
    # 1×2：m9 的 pixel 與 (200, 200, 200)；A_F = A_N = 100 → A_q 都是 25600，F = 16、T = 10
    # m9：浮點 t = 1 − 15/16·0.6·8/9 = 0.5；定點 m_q = floor(60·2^24/25600) = 39321，t_q = 513 → 差 1/1024 → 9.766e-04
    # (200, 200, 200)：m = 2 → 兩邊都截到下限 1/1024，差 0；下限命中 1／1
    # T_MIN 對照（浮點 PTE，1/1023 → 1/1024）：只有下限的 pixel 不同，差 1/1023 − 1/1024 = 1/1047552 → 9.546e-07
    import contextlib
    import io

    two = [np.array([[px9, (200, 200, 200)]], np.uint8)]
    rows = {}
    try:
        base = dict(t_h=0, a_far=(100, 100, 100), a_near=(100, 100, 100), inv_frac=16, pala_fixed=True)
        pq = run_video(two, make_active(**base))
        p1024 = run_video(two, make_active(**base, t_min=T_MIN_FIXED))
        pt = run_video(two, make_active(**base, pte_frac=10))
        for name, a, b in (("PTE", p1024, pt), ("T_MIN", pq, p1024)):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                same = _compare_pala(a, b, name, t_cols=True)
            row = next(line for line in buf.getvalue().splitlines() if line.startswith("| 0 |"))
            rows[name] = (same, [c.strip() for c in row.split("|")[1:-1]])
    except ValueError as e:
        rows = {n: (False, [str(e)] * 13) for n in ("PTE", "T_MIN")}
    for name, d_t in (("PTE 量化", "9.766e-04"), ("T_MIN 對照", "9.546e-07")):
        same, cols = rows[name.split()[0]]
        ck.check(f"T6 {name}報表的 t 最大差欄 = {d_t}", same and len(cols) == 13 and cols[3] == d_t, str(cols[3:5]))
        ck.check(f"T6 {name}報表的下限命中欄 = 1／1", same and len(cols) == 13 and cols[4] == "1／1", str(cols[3:5]))


def _sd_oracle(rgb, aq, tq, t_frac, mode, k=SD_LUT_K):
    """SD 定點契約的 Fraction 寫法（單一 pixel，不呼叫 sd_fixed）：直接從 J = (I − A)/t + A 計算。

    lut 模式的 reciprocal 在這裡自己依 nearest（tie 向上）的定義算出。回傳 (三色的 J, 三色的 out)。
    """
    import math
    from fractions import Fraction as F

    t = F(int(tq), 2 ** t_frac)
    r = 1 / t if mode == "ideal" else F(math.floor(F(2 ** (k + t_frac), int(tq)) + F(1, 2)), 2 ** k)
    js = [(int(i) - F(int(a), 256)) * r + F(int(a), 256) for i, a in zip(rgb, aq)]
    return js, [min(max(math.floor(j + F(1, 2)), 0), 255) for j in js]


def _selfcheck_sd_fixed(ck):
    """SD 定點（E Notes 4.10）：預期值是手算數字、整數不等式、獨立性質，或 Fraction 寫法（_sd_oracle）。"""
    import contextlib
    import io
    from fractions import Fraction as F

    def run(pixels, aqs, tqs, frac, mode):
        return sd_fixed(np.array([pixels], np.uint8), np.array([aqs], np.int64), np.array([tqs], np.int64), frac, mode)

    def try_check(label, fn):
        try:
            ck.check(label, bool(fn()))
        except Exception as e:           # mutant 可能丟例外：記成 FAIL 後繼續，不中止 selfcheck
            ck.check(label, False, f"{type(e).__name__}：{e}")

    def v(r, key, n=0):
        return r[key][0, n].tolist()

    print("SD D1–D7：手算（E Notes 4.10.9；理想 SD 與候選 A，K = 10）")
    px9, aq9 = (60, 90, 120), (39520, 40000, 40480)
    try_check("SD D1 T=10 ideal：N = (2647520, 10352960, 18058400)、D = 177408、out (15, 58, 102)",
              lambda: (lambda r: v(r, "sd_n") == [2647520, 10352960, 18058400] and v(r, "sd_d") == 177408
                       and v(r, "out") == [15, 58, 102])(run([px9], [aq9], [693], 10, "ideal")))
    try_check("SD D1 T=10 lut：inv_t_q = 1513、P = (3914400, 15299520, 26684640)、out (15, 58, 102)",
              lambda: (lambda r: v(r, "sd_inv_q") == 1513 and v(r, "sd_p") == [3914400, 15299520, 26684640]
                       and v(r, "out") == [15, 58, 102])(run([px9], [aq9], [693], 10, "lut")))
    try_check("SD D1 T=16 ideal：N = (167781440, 660909440, 1154037440)、D = 11343360、out (15, 58, 102)",
              lambda: (lambda r: v(r, "sd_n") == [167781440, 660909440, 1154037440] and v(r, "sd_d") == 11343360
                       and v(r, "out") == [15, 58, 102])(run([px9], [aq9], [44310], 16, "ideal")))
    # (標籤, I, A_q, t_q, 理想的 rounded, 理想 out, 候選 A out)；三色相同
    cases = (("D2 .5 tie（J = 99.5）", 100, 25728, 512, 100, 100, 100),
             ("D3 負 J（J = −100）", 0, 25600, 512, -100, 0, 0),
             ("D4 超過 255（J = 510）", 255, 0, 512, 510, 255, 255),
             ("D5 t = 1（J = I = 37）", 37, 51200, 1024, 37, 37, 37),
             ("D6 t = 1/1024（J = 96 + 1/256）", 100, 25601, 1, 96, 96, 96),
             ("D7 LUT 改變輸出（J = 165.4949…）", 16, 0, 99, 165, 165, 166))
    for label, i, aq, tq, rnd, o_i, o_l in cases:
        try_check(f"SD {label.split()[0]} ideal：rounded = {rnd}、out = {o_i}",
                  lambda: (lambda r: v(r, "sd_rounded") == [rnd] * 3 and v(r, "out") == [o_i] * 3)(
                      run([(i,) * 3], [(aq,) * 3], [tq], 10, "ideal")))
        try_check(f"SD {label.split()[0]} lut：out = {o_l}",
                  lambda: v(run([(i,) * 3], [(aq,) * 3], [tq], 10, "lut"), "out") == [o_l] * 3)
    try_check("SD D2 ideal：sd_tie 標出 J = 99.5；D1 不是 tie",
              lambda: v(run([(100,) * 3], [(25728,) * 3], [512], 10, "ideal"), "sd_tie") == [True] * 3
              and v(run([px9], [aq9], [693], 10, "ideal"), "sd_tie") == [False] * 3)
    try_check("SD D7 lut：inv_t_q = 10592（2^20/99 = 10591.68 → nearest），J = 331/2",
              lambda: (lambda r: v(r, "sd_inv_q") == 10592 and F(v(r, "sd_p")[0], 1 << 18) == F(331, 2))(
                  run([(16,) * 3], [(0,) * 3], [99], 10, "lut")))
    try_check("SD D7 sd_tie：ideal 不是 tie（J = 16384/99）、lut 是 tie（J = 331/2）",
              lambda: v(run([(16,) * 3], [(0,) * 3], [99], 10, "ideal"), "sd_tie") == [False] * 3
              and v(run([(16,) * 3], [(0,) * 3], [99], 10, "lut"), "sd_tie") == [True] * 3)
    # D8：I = 100，A_q = 25472（A = 99.5），t_q = 512（t = 1/2）→ J = 0.5/0.5 + 99.5 = 100.5；
    # 契約是 tie 向上 → 101；round-half-even 會得到 100。候選 A 的 inv_t_q = 2^20/512 = 2048 精確，J 相同
    for mode in SD_MODES:
        try_check(f"SD D8 {mode}：J = 100.5 → rounded = 101、out = 101（tie 向上，不是 half-even），sd_tie 為 True",
                  lambda: (lambda r: v(r, "sd_rounded") == [101] * 3 and v(r, "out") == [101] * 3
                           and v(r, "sd_tie") == [True] * 3)(run([(100,) * 3], [(25472,) * 3], [512], 10, mode)))

    print("SD S1：Fraction 寫法（直接從 J = (I − A)/t + A 計算，不呼叫 sd_fixed）")
    rng = np.random.default_rng(20261031)
    pix = rng.integers(0, 256, size=(1, 400, 3), dtype=np.uint8)
    aq_r = rng.integers(0, 65281, size=(1, 400, 3)).astype(np.int64)
    aq_r[0, :30] = (0, 65280, 25600)                                  # A 的端點
    for frac, modes in ((10, SD_MODES), (16, ("ideal",))):
        lo_t = 1 << (frac - 10)
        tq_r = rng.integers(lo_t, (1 << frac) + 1, size=(1, 400)).astype(np.int64)
        tq_r[0, :10], tq_r[0, 10:20] = lo_t, 1 << frac                # t 的端點
        for mode in modes:
            def oracle_ok():
                r = sd_fixed(pix, aq_r, tq_r, frac, mode)
                bad = 0
                for n in range(pix.shape[1]):
                    js, outs = _sd_oracle(pix[0, n], aq_r[0, n], tq_r[0, n], frac, mode)
                    got = ([F(int(x), int(r["sd_d"][0, n])) for x in r["sd_n"][0, n]] if mode == "ideal"
                           else [F(int(x), 1 << (SD_LUT_K + 8)) for x in r["sd_p"][0, n]])
                    ties = [j.denominator == 2 for j in js]       # 約分後分母是 2 = 整數 + 1/2（含負值）
                    bad += got != js or r["out"][0, n].tolist() != outs or r["sd_tie"][0, n].tolist() != ties
                return bad == 0
            try_check(f"SD S1 T={frac} {mode}：400 個 pixel 的 J（精確分數）、out 與 sd_tie 和 Fraction 寫法相同", oracle_ok)

    print("SD S2：LUT 內容（候選 A，T = 10、K = 10，全部 t_q = 1–1024）")
    tq_all = np.arange(1, 1025, dtype=np.int64)[None, :]
    def lut_ok():
        inv = sd_fixed(np.zeros((1, 1024, 3), np.uint8), np.zeros((1, 1024, 3), np.int64), tq_all, 10, "lut")["sd_inv_q"][0]
        top = 1 << (SD_LUT_K + 10 + 1)
        return (bool(np.all(((2 * inv - 1) * tq_all[0] <= top) & (top < (2 * inv + 1) * tq_all[0])))
                and bool(np.all(np.diff(inv) <= 0)) and int(inv[0]) == 1 << 20 and int(inv[-1]) == 1 << 10
                and int(inv.max()).bit_length() == 21)
    try_check("SD S2：(2·inv − 1)·t_q <= 2^21 < (2·inv + 1)·t_q（nearest、tie 向上）；單調不增；t = 1/1024 → 2^20、t = 1 → 2^10；最大 21 bit",
              lut_ok)

    print("SD S3：獨立性質與 ≤ 1 的 guard")
    for frac, mode in ((10, "ideal"), (16, "ideal"), (10, "lut")):
        try_check(f"SD S3 T={frac} {mode}：t = 1 時 out = I（400 個 pixel、隨機 A_q）",
                  lambda: np.array_equal(sd_fixed(pix, aq_r, np.full((1, 400), 1 << frac), frac, mode)["out"], pix))
        try_check(f"SD S3 T={frac} {mode}：256·I = A_q 時 out = I（任意 t_q）",
                  lambda: np.array_equal(sd_fixed(pix, 256 * pix.astype(np.int64),
                                                  rng.integers(1 << (frac - 10), (1 << frac) + 1, size=(1, 400)), frac, mode)["out"], pix))
        def mono_ok():
            ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
            for _ in range(40):
                aq1 = np.full((1, 256, 3), int(rng.integers(0, 65281)), np.int64)
                tq1 = np.full((1, 256), int(rng.integers(1 << (frac - 10), (1 << frac) + 1)), np.int64)
                if np.any(np.diff(sd_fixed(ramp, aq1, tq1, frac, mode)["out"][0, :, 0].astype(int)) < 0):
                    return False
            return True
        try_check(f"SD S3 T={frac} {mode}：固定 A_q、t_q 時 out 隨 I 單調不減（40 組）", mono_ok)
    guard = {}
    def guard_ok():
        p = rng.integers(0, 256, size=(1, 20000, 3), dtype=np.uint8)
        a = rng.integers(0, 65281, size=(1, 20000, 3)).astype(np.int64)
        t = rng.integers(1, 1025, size=(1, 20000)).astype(np.int64)
        d = np.abs(sd_fixed(p, a, t, 10, "lut")["out"].astype(int) - sd_fixed(p, a, t, 10, "ideal")["out"].astype(int))
        guard["n"] = int(np.count_nonzero(d))
        return int(d.max()) <= 1
    try_check("SD S3 guard：60,000 個 channel 值，候選 A 和理想 SD 的差 <= 1（E Notes 4.10.6 的推導）", guard_ok)
    print(f"  紀錄：其中不同的 channel 值 {guard.get('n')} 個")

    print("SD S4：位寬（端點組合）")
    for frac in PTE_FRACS:
        def width_ok():
            combo = [(i, a, t) for i in (0, 255) for a in (0, 65280) for t in (1 << (frac - 10), 1 << frac)]
            r = sd_fixed(np.array([[(i,) * 3 for i, _, _ in combo]], np.uint8), np.array([[(a,) * 3 for _, a, _ in combo]], np.int64),
                         np.array([[t for _, _, t in combo]], np.int64), frac, "ideal")
            n, d = r["sd_n"], r["sd_d"][..., None]
            return (int(n.max()) == 65280 << frac and int(n.min()) == -65280 * ((1 << frac) - (1 << (frac - 10)))
                    and int(d.max()) == 1 << (frac + 8) and int(np.abs(n).max()) < 1 << (frac + 16)
                    and int((2 * n + d).max()) < 1 << (frac + 17) and int((2 * n + d).min()) >= -(1 << (frac + 17)))
        try_check(f"SD S4 T={frac} ideal：N 最大 65280·2^T、最小 −65280·(2^T − 2^(T−10))、|N| < 2^(T+16)、D 最大 2^(T+8)、"
                  f"2N + D 在 ±2^(T+17)", width_ok)
    def p_ok():
        combo = [(i, a, t) for i in (0, 255) for a in (0, 65280) for t in (1, 1024)]
        r = sd_fixed(np.array([[(i,) * 3 for i, _, _ in combo]], np.uint8), np.array([[(a,) * 3 for _, a, _ in combo]], np.int64),
                     np.array([[t for _, _, t in combo]], np.int64), 10, "lut")
        top = 1 << (SD_LUT_K + 26)
        return int(r["sd_p"].max()) == 65280 << (SD_LUT_K + 10) and int(np.abs(r["sd_p"]).max()) + (1 << (SD_LUT_K + 7)) < top
    try_check("SD S4 T=10 lut：P 最大 65280·2^(K+10)，|P| + 2^(K+7) < 2^(K+26)（K + 27-bit signed）", p_ok)

    print("SD S5：接到 apply_pass（A_q、t_q 不變，只換 SD）與輸入檢查")
    af9, an9 = (220, 200, 180), (100, 120, 140)
    for frac, mode in ((10, "ideal"), (16, "ideal"), (10, "lut")):
        def wire_ok():
            base = apply_pass(np.array([[px9]], np.uint8), make_active(0, af9, an9, inv_frac=16, pala_fixed=True, pte_frac=frac))
            o = apply_pass(np.array([[px9]], np.uint8), make_active(0, af9, an9, inv_frac=16, pala_fixed=True, pte_frac=frac,
                                                                   sd_mode=mode))
            key = "sd_n" if mode == "ideal" else "sd_inv_q"
            return (o["out"][0, 0].tolist() == [15, 58, 102] and key in o
                    and all(np.array_equal(o[k], base[k]) for k in ("eta_q", "a_q", "t_q", "m_q", "k_q"))
                    and np.array_equal(np.floor(o["j"] + 0.5), o["sd_rounded"]))
        try_check(f"SD 轉接 T={frac} {mode}：m9 經定點 PALA、PTE 後 out (15, 58, 102)；PALA／PTE 中間值和不用定點 SD 時逐值相同", wire_ok)
    two = np.array([[(7, 35, 49), (100, 100, 100)]], np.uint8)
    a100 = (100, 100, 100)
    def two_ok():
        outs = [apply_pass(two, make_active(0, a100, a100, inv_frac=16, pala_fixed=True, pte_frac=10, sd_mode=m))
                for m in (None, "ideal", "lut")]
        return ([int(o["t_q"][0, 0]) for o in outs] == [958] * 3 and [int(o["out"][0, 0, 1]) for o in outs] == [31, 31, 30]
                and all(o["out"][0, 1].tolist() == [100] * 3 for o in outs))
    try_check("SD 轉接 lut：兩個 pixel，t_q = 958 的 G 在浮點 SD／理想／候選 A 是 31／31／30；I = A 的 pixel 都是 100", two_ok)
    for label, call in (("sd_mode lut 用 T = 16", lambda: make_active(0, af9, an9, inv_frac=16, pala_fixed=True, pte_frac=16,
                                                                       sd_mode="lut")),
                        ("sd_mode 沒有 pte_frac", lambda: make_active(0, af9, an9, inv_frac=16, pala_fixed=True, sd_mode="ideal")),
                        ("sd_mode = 'float'", lambda: make_active(0, af9, an9, inv_frac=16, pala_fixed=True, pte_frac=10,
                                                                   sd_mode="float")),
                        ("apply_pass 收到 sd_mode lut、T = 16", lambda: apply_pass(
                            np.array([[px9]], np.uint8),
                            dict(make_active(0, af9, an9, inv_frac=16, pala_fixed=True, pte_frac=16), sd_mode="lut")))):
        try:
            call()
            ck.check(f"{label} 應拒絕", False)
        except ValueError:
            ck.check(f"{label} 應拒絕", True)
        except Exception as e:
            ck.check(f"{label} 應拒絕", False, f"丟出 {type(e).__name__}，不是 ValueError")

    print("SD S6：排程（S3、靜態）的 trace 不變，sd_mode 逐張傳遞；理想 SD 只在 .5 tie 可能和浮點 SD 不同")
    s3 = [_filled(9, 9, col) for col in [(130, 130, 130)] * 2 + [(100, 110, 120)] * 4]
    x_img = np.concatenate([_filled(15, 9, (60, 70, 80)), _filled(15, 9, (120, 125, 130))])
    ref = run_video(s3, make_active(**ACTIVE0_DEFAULT))
    for frac, mode in ((10, "ideal"), (16, "ideal"), (10, "lut")):
        try:
            base = run_video(s3, make_active(**ACTIVE0_DEFAULT, inv_frac=16, pala_fixed=True, pte_frac=frac))
            recs = run_video(s3, make_active(**ACTIVE0_DEFAULT, inv_frac=16, pala_fixed=True, pte_frac=frac, sd_mode=mode))
        except ValueError as e:
            ck.check(f"SD S3 T={frac} {mode}：排程執行完成", False, str(e))
            continue
        ck.check(f"SD S3 T={frac} {mode}：T_h、A_F、A_N、step 4 觸發旗標逐張和浮點版相同",
                 [_trace_key(r) for r in recs] == [_trace_key(r) for r in ref])
        ck.check(f"SD S3 T={frac} {mode}：每張 active 都帶 sd_mode = {mode}",
                 all(r[k]["sd_mode"] == mode for r in recs for k in ("used", "next")))
        if mode == "ideal":
            mis = [a["result"]["out"] != b["result"]["out"] for a, b in zip(base, recs)]
            try_check(f"SD S3 T={frac} ideal：和浮點 SD 不同的 channel 值都是 J 剛好 .5 的 tie",
                      lambda: all(bool(np.all(b["result"]["sd_tie"][m])) for m, b in zip(mis, recs)))
            ties = sum(int(b["result"]["sd_tie"].sum()) for b in recs if "sd_tie" in b["result"])
            print(f"  紀錄：S3 T={frac} 理想 SD 和浮點 SD 不同的 channel 值 {sum(int(m.sum()) for m in mis)} 個，tie {ties} 個")
        try:
            rec = run_static(x_img, make_active(**ACTIVE0_DEFAULT, inv_frac=16, pala_fixed=True, pte_frac=frac, sd_mode=mode))
            ok = rec["used"]["sd_mode"] == mode and "sd_rounded" in rec["result"]
        except ValueError:
            ok = False
        ck.check(f"靜態 T={frac} {mode}：除霧用的 active（CPU 更新後）仍是定點 SD，結果帶 sd_rounded", ok)

    print("SD S7：_compare_sd 的欄位（手算，對照印出的字串）")
    # 兩個 pixel、A = 100：t_q = 958 的 G 從 31（理想）變 30（候選 A），其餘相同 → 不同 1／6，signed −1/6，最大差 1
    # J 最大差在 R：93·|1095/1024 − 1024/958| = 93·434/980992 = 4.114e-02（inv_t_q = RN(2^20/958) = 1095）
    # e：t_q = 958 → e = 1；(100, 100, 100) 的 t_q = 304 → e = 2
    rows = {}
    try:
        b2 = dict(t_h=0, a_far=a100, a_near=a100, inv_frac=16, pala_fixed=True, pte_frac=10)
        ri, rl = run_video([two], make_active(**b2, sd_mode="ideal")), run_video([two], make_active(**b2, sd_mode="lut"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            same = _compare_sd(ri, rl, "SD")
        lines = buf.getvalue().splitlines()
        rows = {"same": same, "row": [c.strip() for c in next(x for x in lines if x.startswith("| 0 |")).split("|")[1:-1]],
                "e": [x for x in lines if x.startswith("| 1 |") or x.startswith("| 2 |")]}
    except Exception as e:
        rows = {"same": False, "row": [str(e)] * 10, "e": []}
    row = rows["row"]
    ck.check("SD 報表的不同 channel、pixel、最大差欄 = 1、1、1", rows["same"] and len(row) == 10 and (row[1], row[3], row[5]) == ("1", "1", "1"),
             str(row))
    ck.check("SD 報表的 signed 平均欄 = −1/6 = -0.166667", len(row) == 10 and row[4] == "-0.166667", str(row[4:5]))
    ck.check("SD 報表的 J 最大差欄 = 4.114e-02", len(row) == 10 and row[6] == "4.114e-02", str(row[6:7]))
    ck.check("SD 報表的 e 分組：e = 1 有 1 個 pixel、1 個不同（−1/3）；e = 2 有 1 個 pixel、0 個不同",
             rows["e"] == ["| 1 | 1 | 1 | -0.333333 |", "| 2 | 1 | 0 | +0.000000 |"], str(rows["e"]))

    print("SD S8：_compare_sd 的 clip 欄、tie 欄與合計 signed 平均（人工 records，手算）")
    # 結構沿用 run_video 的紀錄（frame 0 一個 pixel、frame 1 兩個 pixel），只換 out、j、sd_tie；out = clip(floor(j + 1/2))
    # frame 0：前 j (255.5, 10, −0.75) → out (255, 10, 0)，> 255 一個、< 0 一個、tie 一個（255.5）
    #          後 j (254.4, 10, −0.5)  → out (254, 10, 0)，> 255、< 0 都沒有（−0.5 → 0）、tie 一個（−0.5）；差 (−1, 0, 0)
    # frame 1：前 j (100, 0.4, 20 | 254.5, −0.5, 300) → out (100, 0, 20 | 255, 0, 255)，> 255 一個（300；254.5 → 255 不算）、
    #          < 0 沒有、tie 兩個（254.5、−0.5）
    #          後 j (101, 1, 21 | 255.5, −0.75, 300) → out (101, 1, 21 | 255, 0, 255)，> 255 兩個、< 0 一個、tie 一個；
    #          差 (+1, +1, +1, 0, 0, 0)
    # 合計 signed = (−1 + 3) / 9 = +2/9 = +0.222222（不是兩張平均 (−1/3 + 1/2)/2）
    def fake(rec, js, outs, ties):
        res = dict(rec["result"], j=np.array([js], float), out=np.array([outs], np.uint8), sd_tie=np.array([ties]))
        return dict(rec, result=res)
    s8 = {}
    try:
        recs = run_video([np.array([[(7, 35, 49)]], np.uint8), two], make_active(**b2, sd_mode="ideal"))
        f, t = (False, False, False), (True, False, False)
        before = [fake(recs[0], [(255.5, 10, -0.75)], [(255, 10, 0)], [t]),
                  fake(recs[1], [(100, 0.4, 20), (254.5, -0.5, 300)], [(100, 0, 20), (255, 0, 255)], [f, (True, True, False)])]
        after = [fake(recs[0], [(254.4, 10, -0.5)], [(254, 10, 0)], [(False, False, True)]),
                 fake(recs[1], [(101, 1, 21), (255.5, -0.75, 300)], [(101, 1, 21), (255, 0, 255)], [f, t])]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s8["same"] = _compare_sd(before, after, "SD S8")
        lines = buf.getvalue().splitlines()
        for n in (0, 1):
            s8[n] = [c.strip() for c in next(x for x in lines if x.startswith(f"| {n} |")).split("|")[1:-1]]
        s8["sum"] = next(x for x in lines if x.startswith("合計"))
    except Exception as e:
        s8 = {"same": False, 0: [str(e)] * 10, 1: [str(e)] * 10, "sum": str(e)}
    r0, r1 = s8[0], s8[1]
    ck.check("SD S8 報表的 clip 欄：frame 0 是 > 255「1／0」、< 0「1／0」；frame 1 是「1／2」、「0／1」（254.5、−0.5 不算）",
             s8["same"] and len(r0) == len(r1) == 10 and (r0[7], r0[8], r1[7], r1[8]) == ("1／0", "1／0", "1／2", "0／1"),
             str((r0[7:9], r1[7:9])))
    ck.check("SD S8 報表的 tie 欄：frame 0 是「1／1」、frame 1 是「2／1」",
             len(r0) == len(r1) == 10 and (r0[9], r1[9]) == ("1／1", "2／1"), str((r0[9:], r1[9:])))
    ck.check("SD S8 報表的逐 frame signed 平均：-0.333333、+0.500000",
             len(r0) == len(r1) == 10 and (r0[4], r1[4]) == ("-0.333333", "+0.500000"), str((r0[4:5], r1[4:5])))
    ck.check("SD S8 報表的合計 signed 平均 = +2/9 = +0.222222（總 signed ÷ 總 channel 數）",
             "signed 平均 +0.222222；" in s8["sum"], s8["sum"])


def _selfcheck_rules(ck):
    """統計規則的選項（並列 tie、T_h 取整 th_round）：預期值是手算數字或整數不等式。"""
    import contextlib
    import io

    def try_check(label, fn):
        try:
            ck.check(label, bool(fn()))
        except Exception as e:           # mutant 可能丟例外：記成 FAIL 後繼續
            ck.check(label, False, f"{type(e).__name__}：{e}")

    # 9×9 → 降採樣 3×3，每點 RGB 最小值都是 50，dark 處處是 50；(0, 0) 與 (2, 2) 的 RGB 不同
    tie_img = _filled(9, 9, (50, 50, 50))
    _put_down(tie_img, 0, 0, (50, 60, 70))
    _put_down(tie_img, 2, 2, (50, 80, 90))
    print("規則 G1：並列取第一個或最後一個（dark 處處 50，9 點並列）")
    for th, g in ((0, "far"), (128, "near")):
        try_check(f"規則 G1 first：T_h = {th}，{g} 取 (0, 0)、RGB (50, 60, 70)",
                  lambda: (lambda x: x["pos"] == (0, 0) and x["rgb"] == (50, 60, 70) and x["count"] == 9)(stats_pass(tie_img, th)[g]))
        try_check(f"規則 G1 last：T_h = {th}，{g} 取 (2, 2)、RGB (50, 80, 90)",
                  lambda: (lambda x: x["pos"] == (2, 2) and x["rgb"] == (50, 80, 90) and x["count"] == 9)(
                      stats_pass(tie_img, th, "last")[g]))

    print("規則 G2：下一張 T_h 的取整（手算）")
    # 3×3 → 降採樣 1 點：(100, 100, 101) 的 gray_sum = 301，N = 1 → floor 100、ceil 101；(100, 100, 100) 兩者都是 100
    for rgb, want in (((100, 100, 101), (100, 101)), ((100, 100, 100), (100, 100))):
        for rnd, t in zip(TH_ROUNDS, want):
            try_check(f"規則 G2 {rnd}：{rgb} 的 gray_sum = {sum(rgb)} → 下一張 T_h = {t}",
                      lambda: (lambda o: o[0]["t_h"] == t and o[1]["t_h_next"] == t)(
                          cpu_update(stats_pass(_filled(3, 3, rgb), 128), make_active(**ACTIVE0_DEFAULT, th_round=rnd))))

    print("規則 G3：th_round = ceil 時，下一張的分群等於和實數平均比較（3N·dark >= gray_sum，Eq.7）")
    # 9×9 單色 (114, 114, 115)：降採樣 9 點、dark 都是 114，gray_sum = 9·343 = 3087，3N = 27 → 平均 114.33
    # floor 114：dark 114 >= 114 → 9 點都在遠群；ceil 115：0 點；實數比較 27·114 = 3078 < 3087 → 0 點
    img = _filled(9, 9, (114, 114, 115))
    s0 = stats_pass(img, 128)
    real_far = int(np.sum(3 * s0["n"] * s0["dark"].astype(np.int64) >= s0["gray_sum"]))
    for rnd, want in (("ceil", 0), ("floor", 9)):
        try_check(f"規則 G3 {rnd}：第 2 張的遠群 pixel 數 = {want}（同一張圖連放兩張）",
                  lambda: run_video([img, img], make_active(**ACTIVE0_DEFAULT, th_round=rnd))[1]["stats"]["far"]["count"] == want)
    try_check("規則 G3：實數比較（3N·dark >= gray_sum）的遠群也是 0 點，和 ceil 相同", lambda: real_far == 0)

    print("規則 G4：選項逐張傳遞，統計路徑用的是 active 的 tie")
    try:
        recs = run_video([tie_img] * 3, make_active(**ACTIVE0_DEFAULT, tie="last", th_round="ceil"))
        dflt = run_video([tie_img] * 3, make_active(**ACTIVE0_DEFAULT))
    except ValueError as e:
        recs = dflt = []
        ck.check("規則 G4：排程執行完成", False, str(e))
    try_check("規則 G4：每張 active 都帶 tie = last",
              lambda: len(recs) == 3 and all(r[k]["tie"] == "last" for r in recs for k in ("used", "next")))
    try_check("規則 G4：每張 active 都帶 th_round = ceil",
              lambda: len(recs) == 3 and all(r[k]["th_round"] == "ceil" for r in recs for k in ("used", "next")))
    try_check("規則 G4：tie = last 時每張統計都取 (2, 2)；預設時都取 (0, 0)",
              lambda: [r["stats"]["near"]["pos"] for r in recs] == [(2, 2)] * 3
              and [r["stats"]["near"]["pos"] for r in dflt] == [(0, 0)] * 3)
    try_check("規則 G4：預設的 active 是 tie = first、th_round = floor",
              lambda: all(r[k]["tie"] == "first" and r[k]["th_round"] == "floor" for r in dflt for k in ("used", "next")))
    for label, call in (("tie = 'middle'", lambda: make_active(**ACTIVE0_DEFAULT, tie="middle")),
                        ("th_round = 'round'", lambda: make_active(**ACTIVE0_DEFAULT, th_round="round")),
                        ("tie = None", lambda: make_active(**ACTIVE0_DEFAULT, tie=None)),
                        ("stats_pass 的 tie = 'middle'", lambda: stats_pass(tie_img, 0, "middle")),
                        ("stats_pass 的 tie = np.str_('last')", lambda: stats_pass(tie_img, 0, np.str_("last"))),
                        ("stats_pass 的 tie = np.array(['last'])", lambda: stats_pass(tie_img, 0, np.array(["last"]))),
                        ("stats_pass 的 tie = np.array('last')", lambda: stats_pass(tie_img, 0, np.array("last")))):
        try:
            call()
            ck.check(f"規則 G5：{label} 應拒絕", False)
        except ValueError:
            ck.check(f"規則 G5：{label} 應拒絕", True)
        except Exception as e:
            ck.check(f"規則 G5：{label} 應拒絕", False, f"丟出 {type(e).__name__}，不是 ValueError")

    print("規則 G6：_compare_rules 的欄位（手算，對照印出的字串）")
    # tie_img 連放兩張。frame 0 用 active0（T_h 128、A 255）→ 兩種規則輸出相同；dark 50 < 128，全部在近群、遠群空
    # 第一次更新：A_N 用近群，A_F 用另一群（近群）→ first 是 (50, 60, 70)、last 是 (50, 80, 90)
    # frame 0 的 gray_sum = 7·150 + 180 + 220 = 1450，3N = 27 → 53.70 → floor 53、ceil 54
    rows = {}
    try:
        for name, kw in (("tie", {"tie": "last"}), ("th", {"th_round": "ceil"})):
            a = run_video([tie_img] * 2, make_active(**ACTIVE0_DEFAULT))
            b = run_video([tie_img] * 2, make_active(**ACTIVE0_DEFAULT, **kw))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _compare_rules(a, b, "G6")
            lines = buf.getvalue().splitlines()
            rows[name] = [[c.strip() for c in next(x for x in lines if x.startswith(f"| {n} |")).split("|")[1:-1]] for n in (0, 1)]
    except Exception as e:
        rows = {k: [[str(e)] * 9] * 2 for k in ("tie", "th")}
    t0, t1 = rows["tie"]
    ck.check("規則 G6 tie：frame 0 的 T_h「128／128」、A_F「(255, 255, 255)／(255, 255, 255)」、不同 0",
             len(t0) == 9 and (t0[1], t0[2], t0[4]) == ("128／128", "(255, 255, 255)／(255, 255, 255)", "0"), str(t0[:5]))
    ck.check("規則 G6 tie：frame 1 的 T_h「53／53」、A_F 與 A_N 都是「(50, 60, 70)／(50, 80, 90)」",
             len(t1) == 9 and (t1[1], t1[2], t1[3]) == ("53／53", "(50, 60, 70)／(50, 80, 90)", "(50, 60, 70)／(50, 80, 90)"),
             str(t1[:4]))
    h1 = rows["th"][1]
    ck.check("規則 G6 th_round：frame 1 的 T_h「53／54」，dark 50 兩種門檻都在近群，輸出不同 0",
             len(h1) == 9 and (h1[1], h1[4]) == ("53／54", "0"), str(h1[:5]))

    print("規則 G7：_compare_rules 的數量欄與合計列（人工 records，手算）")
    # 結構沿用 run_video 的紀錄（frame 0 一個 pixel、frame 1 兩個 pixel），只換 out
    # frame 0：前 (10, 20, 30)、後 (15, 22, 30) → 不同 2、比例 2/3、pixel 1、MAE 7/3、最大差 5
    # frame 1：前 (0, 0, 0 | 5, 5, 5)、後 (0, 0, 0 | 6, 8, 9) → 不同 3、比例 3/6、pixel 1、MAE 8/6、最大差 4
    # 合計：不同 5／9、pixel 2／3、MAE (7 + 8)/9 = 1.666667（不是兩張平均 1.833333）、最大差 5（在 frame 0）
    # 每列的不同 channel 數、pixel 數、總 pixel 數兩兩不同，最後一張的值也不等於合計，欄位或變數用錯都會改變字串
    def fake_out(rec, outs):
        return dict(rec, result=dict(rec["result"], out=np.array([outs], np.uint8)))
    g7 = {}
    try:
        base = run_video([np.array([[(7, 35, 49)]], np.uint8), np.array([[(7, 35, 49), (100, 100, 100)]], np.uint8)],
                         make_active(**ACTIVE0_DEFAULT))
        before = [fake_out(base[0], [(10, 20, 30)]), fake_out(base[1], [(0, 0, 0), (5, 5, 5)])]
        after = [fake_out(base[0], [(15, 22, 30)]), fake_out(base[1], [(0, 0, 0), (6, 8, 9)])]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _compare_rules(before, after, "G7")
        lines = buf.getvalue().splitlines()
        for n in (0, 1):
            g7[n] = [c.strip() for c in next(x for x in lines if x.startswith(f"| {n} |")).split("|")[1:-1]]
        g7["sum"] = next(x for x in lines if x.startswith("合計"))
    except Exception as e:
        g7 = {0: [str(e)] * 9, 1: [str(e)] * 9, "sum": str(e)}
    for n, want in ((0, ("2", "66.6667%", "1", "2.333333", "5")), (1, ("3", "50.0000%", "1", "1.333333", "4"))):
        row = g7[n]
        ck.check(f"規則 G7：frame {n} 的不同 channel、比例、pixel、MAE、最大差 = {'、'.join(want)}",
                 len(row) == 9 and tuple(row[4:9]) == want, str(row[4:9]))
    ck.check("規則 G7：合計列 = 不同 5／9（55.5556%）、pixel 2／3、MAE 1.666667（總絕對差 ÷ 總 channel 數）、最大差 5",
             g7["sum"] == "合計：不同的 channel 值 5／9（55.5556%）；有差異的 pixel 2／3；MAE 1.666667（channel 值）；最大絕對差 5",
             g7["sum"])


def _trace_key(r):
    """排程比較用：統計用 T_h、除霧用 A、step 4 觸發旗標、下一個 T_h 與 A。"""
    u = r["update"]
    return (r["stats_active"]["t_h"], r["used"]["a_far"], r["used"]["a_near"], u["far"]["triggered"],
            u["near"]["triggered"], r["next"]["t_h"], r["next"]["a_far"], r["next"]["a_near"])


# ---------------------------------------------------------------- stats

def stats_command(path, t_values):
    import cv2  # 只在讀圖時需要

    try:
        raw, frame = _read_rgb(path)
    except ValueError as e:
        print(e)
        return 1
    print(f"輸入：{path}")
    print(f"SHA-256：{hashlib.sha256(raw).hexdigest()}")
    print(f"尺寸 H×W：{frame.shape[0]}×{frame.shape[1]}；OpenCV {cv2.__version__} 解碼，轉為 RGB")
    for t in t_values:
        if t == "self":
            t = stats_pass(frame, 0)["gray_mean"]
            label = f"{t}（該圖自己的平均灰階）"
        else:
            t = int(t)
            label = str(t)
        s = stats_pass(frame, t)
        ref = _ref_stats(frame, t)
        same = (all(s[k] == ref[k] for k in ("n", "gray_sum", "gray_mean", "far", "near"))
                and s["dark"].tolist() == ref["dark"])
        h, w = s["shape_down"]
        print(f"T_h_prev = {label}")
        print(f"  降採樣 H×W：{h}×{w}，N = {s['n']}")
        print(f"  gray_sum = {s['gray_sum']}，平均灰階 = {s['gray_mean']}")
        for name in ("far", "near"):
            g = s[name]
            if g["valid"]:
                r, c = g["pos"]
                print(f"  {name}：count = {g['count']}，max_dark = {g['max_dark']}，"
                      f"降採樣位置 ({r}, {c}) = 原圖 ({STEP * r}, {STEP * c})，RGB = {g['rgb']}")
            else:
                print(f"  {name}：空群組（valid=False）")
        print(f"  和逐點參考實作比較：{'相同' if same else '不同'}")
        if not same:
            return 1
    return 0


def _read_rgb(path):
    import cv2

    raw = open(path, "rb").read()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"無法讀取：{path}")
    return raw, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def demo_command(path, out_dir, compare):
    """單張浮點展示：印出參數與數值摘要；有 --out-dir 才寫 PNG。"""
    import os

    import cv2

    raw, frame = _read_rgb(path)
    s, a_far, a_near, r = dehaze_single(frame)
    print(f"輸入：{path}（SHA-256 {hashlib.sha256(raw).hexdigest()}），H×W {frame.shape[0]}×{frame.shape[1]}，RGB")
    print(f"統計：降採樣 {s['shape_down'][0]}×{s['shape_down'][1]}，平均灰階 {s['gray_mean']} → 當作 T_h_prev")
    for name in ("far", "near"):
        g = s[name]
        print(f"  {name}：" + (f"count {g['count']}，max_dark {g['max_dark']}，"
                               f"原圖 ({STEP * g['pos'][0]}, {STEP * g['pos'][1]})，RGB {g['rgb']}"
                               if g["valid"] else "空群組（改用另一群的 A）"))
    print(f"使用的 A_F = {a_far}，A_N = {a_near}，I_max（三色和）= {sum(a_far)}；ω = {OMEGA}，γ = {GAMMA}")
    n = frame.shape[0] * frame.shape[1]
    t_raw, j, out = r["t_raw"], r["j"], r["out"]
    print(f"k：min {r['k'].min():.4f}，max {r['k'].max():.4f}")
    print(f"t_m（截斷前）：min {t_raw.min():.4f}，mean {t_raw.mean():.4f}，max {t_raw.max():.4f}；"
          f"< 1/1023 的 pixel {int((t_raw < T_MIN).sum())}，> 1 的 pixel {int((t_raw > 1).sum())}")
    use_far = int((frame.astype(np.int64).sum(axis=2) >= sum(a_far)).sum())
    print(f"灰階 >= I_max（直接用 A_F）的 pixel：{use_far}／{n}")
    rounded = np.floor(j + 0.5)               # 和 sd 相同的捨入；截斷 = 捨入後超出 0–255
    print(f"J（截斷前）：min {j.min():.2f}，max {j.max():.2f}；"
          f"捨入後 < 0 而截到 0 的數值 {int((rounded < 0).sum())}，"
          f"捨入後 > 255 而截到 255 的數值 {int((rounded > 255).sum())}（共 {3 * n} 個）")
    gray_in = frame.astype(float).mean(axis=2)
    gray_out = out.astype(float).mean(axis=2)
    print(f"整張灰階：平均 {gray_in.mean():.1f} → {gray_out.mean():.1f}；標準差 {gray_in.std():.1f} → {gray_out.std():.1f}")
    print(f"輸出陣列 SHA-256：{hashlib.sha256(out.tobytes()).hexdigest()}")
    if out_dir is None:
        print("未指定 --out-dir：不寫檔")
        return 0
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{stem}_ate_float.png")
    if not cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR)):
        print(f"寫入失敗：{out_path}")
        return 1
    _, back = _read_rgb(out_path)
    same = np.array_equal(back, out)
    print(f"寫出：{out_path}；讀回後和輸出陣列逐值相同：{same}")
    if not same:
        return 1
    panels, labels = [frame], ["input (hazy)"]
    if compare:
        _, dcp = _read_rgb(compare)
        panels.append(dcp)
        labels.append("DCP (He-Zhang, reference only)")
    panels.append(out)
    labels.append("ATE float v0 (this model)")
    tiles = []
    for img, label in zip(panels, labels):
        tile = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bar = np.full((40, tile.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(np.vstack([bar, tile]))
    gap = np.full((tiles[0].shape[0], 10, 3), 255, np.uint8)
    row = tiles[0]
    for tile in tiles[1:]:
        row = np.hstack([row, gap, tile])
    cmp_path = os.path.join(out_dir, f"{stem}_compare.png")
    if not cv2.imwrite(cmp_path, row):
        print(f"寫入失敗：{cmp_path}")
        return 1
    print(f"寫出：{cmp_path}（{' | '.join(labels)}）")
    return 0


def _run_schedule(mode, frames, a0):
    return run_video(frames, a0) if mode == "video" else [run_static(frames[0], a0)]


def _diff_stats(out_a, out_b):
    """兩張 8-bit 輸出的差異。n、diff、abs、mae 的單位是 channel 值；pix 是至少一個 channel 不同的 pixel 數。"""
    d = np.abs(out_b.astype(np.int64) - out_a.astype(np.int64))
    n_abs = int(d.sum())
    return {"n": d.size, "diff": int(np.count_nonzero(d)), "pix": int(np.count_nonzero(d.max(axis=2))),
            "abs": n_abs, "max": int(d.max()), "mae": n_abs / d.size}


def _sum_stats(stats):
    """多張合計：MAE = 總絕對差 ÷ 總 channel 數（依大小加權），不是平均各張的 MAE。"""
    tot = {k: sum(s[k] for s in stats) for k in ("n", "diff", "pix", "abs")}
    tot["max"] = max(s["max"] for s in stats)
    tot["mae"] = tot["abs"] / tot["n"]
    return tot


def _clip_counts(j):
    """和 sd 相同的捨入（.5 進位）後，(> 255 的 channel 值個數, < 0 的 channel 值個數)。"""
    r = np.floor(j + 0.5)
    return int((r > 255).sum()), int((r < 0).sum())


def _compare_inv(ref, recs, inv_frac):
    """浮點 1/I_max（ref）與量化 1/I_max（recs）逐 frame 比較；只記錄，不設門檻。"""
    same = len(ref) == len(recs) and [_trace_key(r) for r in ref] == [_trace_key(r) for r in recs]
    print(f"reciprocal 局部量化實驗 F={inv_frac}：和浮點 1/I_max 比較（PALA／PTE／SD 仍是浮點）")
    print(f"T_h、A、step 4 觸發旗標逐張和浮點版相同：{same}")
    print("| frame | I_max | q | reciprocal 相對誤差 | η 最大絕對差（pixel） | 不同的 channel 值 | 比例（channel 值） "
          "| 有差異的 pixel | MAE（channel 值） | 最大絕對差 | 捨入後 > 255（浮點／量化，channel 值） "
          "| 捨入後 < 0（浮點／量化，channel 值） |")
    print("|" + "---|" * 12)
    stats = []
    for a, b in zip(ref, recs):
        i_max, qv = b["used"]["i_max"], b["used"]["inv_i_max_q"]
        rel = (1 / i_max - qv / (1 << inv_frac)) * i_max if i_max else 0.0
        d_eta = float(np.max(np.abs(b["result"]["eta"] - a["result"]["eta"])))
        s = _diff_stats(a["result"]["out"], b["result"]["out"])
        stats.append(s)
        clip = [_clip_counts(r["result"]["j"]) for r in (a, b)]
        print(f"| {b['n']} | {i_max} | {qv} | {rel:.3e} | {d_eta:.3e} | {s['diff']} | {s['diff'] / s['n']:.4%} | {s['pix']} "
              f"| {s['mae']:.6f} | {s['max']} | {clip[0][0]}／{clip[1][0]} | {clip[0][1]}／{clip[1][1]} |")
    tot = _sum_stats(stats)
    print(f"合計：不同的 channel 值 {tot['diff']}／{tot['n']}（{tot['diff'] / tot['n']:.4%}）；有差異的 pixel {tot['pix']}；"
          f"MAE {tot['mae']:.6f}（channel 值）；最大絕對差 {tot['max']}")
    return same


def _compare_pala(ref, recs, title, t_cols=False):
    """兩種模式逐 frame 比較 η、A 與輸出；只記錄，不設門檻。

    t_cols 時多兩欄：t（截斷後、進 SD 的值）的最大絕對差，以及 t 等於各自下限（used 的 t_min）的 pixel 數。
    """
    same = len(ref) == len(recs) and [_trace_key(r) for r in ref] == [_trace_key(r) for r in recs]
    print(title)
    print(f"T_h、A_F、A_N、step 4 觸發旗標逐張相同：{same}")
    t_head = "| t 最大絕對差（pixel，實數） | t 等於下限的 pixel（前／後） " if t_cols else ""
    print("| frame | η 最大絕對差（pixel） | A 最大絕對差（channel 值，8-bit 單位） " + t_head + "| 不同的 channel 值 | 比例（channel 值） "
          "| 有差異的 pixel | 比例（pixel） | MAE（channel 值） | 最大絕對差 | 捨入後 > 255（前／後，channel 值） "
          "| 捨入後 < 0（前／後，channel 值） |")
    print("|" + "---|" * (13 if t_cols else 11))
    stats, hit_tot = [], [0, 0]
    for a, b in zip(ref, recs):
        d_eta = float(np.max(np.abs(b["result"]["eta"] - a["result"]["eta"])))
        d_a = float(np.max(np.abs(b["result"]["a_map"] - a["result"]["a_map"])))
        t_cell = ""
        if t_cols:
            d_t = float(np.max(np.abs(b["result"]["t"] - a["result"]["t"])))
            hits = [int((r["result"]["t"] <= r["used"]["t_min"]).sum()) for r in (a, b)]
            hit_tot = [hit_tot[0] + hits[0], hit_tot[1] + hits[1]]
            t_cell = f" | {d_t:.3e} | {hits[0]}／{hits[1]}"
        s = _diff_stats(a["result"]["out"], b["result"]["out"])
        stats.append(s)
        clip = [_clip_counts(r["result"]["j"]) for r in (a, b)]
        print(f"| {b['n']} | {d_eta:.3e} | {d_a:.3e}{t_cell} | {s['diff']} | {s['diff'] / s['n']:.4%} | {s['pix']} "
              f"| {s['pix'] / (s['n'] // 3):.4%} | {s['mae']:.6f} | {s['max']} "
              f"| {clip[0][0]}／{clip[1][0]} | {clip[0][1]}／{clip[1][1]} |")
    tot = _sum_stats(stats)
    print(f"合計：不同的 channel 值 {tot['diff']}／{tot['n']}（{tot['diff'] / tot['n']:.4%}）；"
          f"有差異的 pixel {tot['pix']}／{tot['n'] // 3}（{tot['pix'] / (tot['n'] // 3):.4%}）；"
          f"MAE {tot['mae']:.6f}（channel 值）；最大絕對差 {tot['max']}"
          + (f"；t 等於下限的 pixel 前 {hit_tot[0]}、後 {hit_tot[1]}" if t_cols else ""))
    return same


def _compare_sd(ref, recs, title):
    """SD 本身的影響：兩種 SD 模式在同一組 A_q、t_q 下逐 frame 比較；只記錄，不設門檻。

    signed 平均 = Σ(後 − 前) ÷ channel 值個數；J 是 clip 前的值；最後依 e = T − floor(log2 t_q) 分組合計
    （t 在 [2^−e, 2^(1−e))，e = 0 是 t = 1）。
    """
    same = (len(ref) == len(recs) and [_trace_key(r) for r in ref] == [_trace_key(r) for r in recs]
            and all(np.array_equal(a["result"][key], b["result"][key]) for a, b in zip(ref, recs) for key in ("a_q", "t_q")))
    print(title)
    print(f"T_h、A_F、A_N、step 4 觸發旗標逐張相同，SD 的輸入 A_q、t_q 逐值相同：{same}")
    print("| frame | 不同的 channel 值 | 比例（channel 值） | 有差異的 pixel | signed 平均（後 − 前，channel 值） | 最大絕對差 "
          "| J 最大絕對差（clip 前） | 捨入後 > 255（前／後，channel 值） | 捨入後 < 0（前／後，channel 值） "
          "| J 剛好是 .5 的 channel 值（前／後） |")
    print("|" + "---|" * 10)
    stats, signed, by_e = [], 0, {}
    for a, b in zip(ref, recs):
        d = b["result"]["out"].astype(np.int64) - a["result"]["out"].astype(np.int64)
        s = _diff_stats(a["result"]["out"], b["result"]["out"])
        stats.append(s)
        signed += int(d.sum())
        d_j = float(np.max(np.abs(b["result"]["j"] - a["result"]["j"])))
        clip = [_clip_counts(r["result"]["j"]) for r in (a, b)]
        ties = [str(int(r["result"]["sd_tie"].sum())) if "sd_tie" in r["result"] else "—" for r in (a, b)]
        e = b["used"]["pte_frac"] - (np.frexp(b["result"]["t_q"].astype(float))[1] - 1)   # 整數的 frexp 精確
        for ev in np.unique(e).tolist():
            sel = e == ev
            row = by_e.setdefault(ev, [0, 0, 0])
            row[0] += int(sel.sum())
            row[1] += int(np.count_nonzero(d[sel]))
            row[2] += int(d[sel].sum())
        print(f"| {b['n']} | {s['diff']} | {s['diff'] / s['n']:.4%} | {s['pix']} | {int(d.sum()) / s['n']:+.6f} | {s['max']} "
              f"| {d_j:.3e} | {clip[0][0]}／{clip[1][0]} | {clip[0][1]}／{clip[1][1]} | {ties[0]}／{ties[1]} |")
    tot = _sum_stats(stats)
    print(f"合計：不同的 channel 值 {tot['diff']}／{tot['n']}（{tot['diff'] / tot['n']:.4%}）；"
          f"有差異的 pixel {tot['pix']}／{tot['n'] // 3}；signed 平均 {signed / tot['n']:+.6f}；最大絕對差 {tot['max']}")
    print("依 e 分組（t 在 [2^−e, 2^(1−e))）：")
    print("| e | pixel | 不同的 channel 值 | signed 平均（channel 值） |")
    print("|---|---|---|---|")
    for ev in sorted(by_e):
        px, nd, sg = by_e[ev]
        print(f"| {ev} | {px} | {nd} | {sg / (3 * px):+.6f} |")
    return same


def _compare_rules(ref, recs, title):
    """統計規則（並列、T_h 取整）不同的兩次排程逐 frame 比較：統計用 T_h、除霧用 A 與輸出；只記錄，不設門檻。"""
    print(title)
    print("| frame | 統計用 T_h（前／後） | 除霧用 A_F（前／後） | 除霧用 A_N（前／後） | 不同的 channel 值 | 比例（channel 值） "
          "| 有差異的 pixel | MAE（channel 值） | 最大絕對差 |")
    print("|" + "---|" * 9)
    stats = []
    for a, b in zip(ref, recs):
        s = _diff_stats(a["result"]["out"], b["result"]["out"])
        stats.append(s)
        print(f"| {b['n']} | {a['stats_active']['t_h']}／{b['stats_active']['t_h']} | {a['used']['a_far']}／{b['used']['a_far']} "
              f"| {a['used']['a_near']}／{b['used']['a_near']} | {s['diff']} | {s['diff'] / s['n']:.4%} | {s['pix']} "
              f"| {s['mae']:.6f} | {s['max']} |")
    tot = _sum_stats(stats)
    print(f"合計：不同的 channel 值 {tot['diff']}／{tot['n']}（{tot['diff'] / tot['n']:.4%}）；有差異的 pixel {tot['pix']}／{tot['n'] // 3}；"
          f"MAE {tot['mae']:.6f}（channel 值）；最大絕對差 {tot['max']}")


def schedule_command(mode, paths, repeat=1, inv_frac=None, pala_fixed=False, pte_frac=None, sd_mode=None,
                     tie="first", th_round="floor"):
    """實際圖片的逐 frame 表格；只印到畫面，不寫任何檔案。

    inv_frac 是 16 或 20 時，表格與 hash 是量化 1/I_max 的結果，最後再和浮點 1/I_max 比較。
    """
    frames = []
    for path in paths:
        raw, frame = _read_rgb(path)
        print(f"輸入：{path}（SHA-256 {hashlib.sha256(raw).hexdigest()[:16]}…），H×W {frame.shape[0]}×{frame.shape[1]}")
        frames += [frame] * repeat
    a0 = make_active(**ACTIVE0_DEFAULT, inv_frac=inv_frac, pala_fixed=pala_fixed, pte_frac=pte_frac, sd_mode=sd_mode,
                     tie=tie, th_round=th_round)
    rules = (tie, th_round) != ("first", "floor")
    sd_text = {None: "，轉接到浮點 SD", "ideal": "，SD 用理想整數 reference", "lut": f"，SD 用候選 A（直接 LUT，K = {SD_LUT_K}）"}
    print(f"active0（P7 暫定）：T_h {a0['t_h']}，A_F {a0['a_far']}，A_N {a0['a_near']}"
          + ("" if inv_frac is None else f"，1/I_max = q/2^{inv_frac}，q = {a0['inv_i_max_q']}")
          + ("，PALA 用定點（eta_q/256、A_q/256 轉接到浮點 PTE／SD）" if pala_fixed else "")
          + (f"，PTE 用理想定點 reference（T = {pte_frac}，t = t_q/2^{pte_frac}，下限 1/1024{sd_text[sd_mode]}）"
             if pte_frac else "")
          + (f"；統計規則：並列取{'第一個' if tie == 'first' else '最後一個'}、下一張 T_h 取 {th_round}" if rules else ""))
    records = _run_schedule(mode, frames, a0)
    print(f"排程：{'影片' if mode == 'video' else '靜態（第一次讀取只統計，第二次讀取用 CPU 更新後的參數除霧，即表中「除霧用」欄）'}，共 {len(records)} 張")
    for line in format_trace(records):
        print(line)
    for r in records:
        out = r["result"]["out"]
        print(f"frame {r['n']} 輸出：SHA-256 {hashlib.sha256(out.tobytes()).hexdigest()[:16]}…，"
              f"灰階平均 {out.astype(float).mean():.1f}")
    if rules:
        # 只比較統計規則：同一數值模式，預設規則（並列取第一個、T_h 取 floor）→ 指定規則；不做數值模式的比較
        ref = _run_schedule(mode, frames, make_active(**ACTIVE0_DEFAULT, inv_frac=inv_frac, pala_fixed=pala_fixed,
                                                      pte_frac=pte_frac, sd_mode=sd_mode))
        _compare_rules(ref, records, f"統計規則的影響：並列取第一個、T_h 取 floor → 並列取{'第一個' if tie == 'first' else '最後一個'}、"
                                     f"T_h 取 {th_round}（同一數值模式）")
        return 0
    if inv_frac is None:
        return 0
    ref = _run_schedule(mode, frames, make_active(**ACTIVE0_DEFAULT))
    if sd_mode is not None:
        # SD 本身的影響：同一組 A_q、t_q（同排程、同上游定點），只換 SD；再看和理想浮點的合計
        fixed = dict(inv_frac=inv_frac, pala_fixed=True, pte_frac=pte_frac)
        base = _run_schedule(mode, frames, make_active(**ACTIVE0_DEFAULT, **fixed))
        tag = f"F={inv_frac}，T={pte_frac}"
        if sd_mode == "lut":
            ideal = _run_schedule(mode, frames, make_active(**ACTIVE0_DEFAULT, **fixed, sd_mode="ideal"))
            ok = _compare_sd(base, ideal, f"理想 SD 的影響：浮點 SD → 理想整數 SD（{tag}）")
            ok &= _compare_sd(ideal, records, f"候選 A 的影響：理想整數 SD → 直接 LUT（K = {SD_LUT_K}，{tag}）")
        else:
            ok = _compare_sd(base, records, f"理想 SD 的影響：浮點 SD → 理想整數 SD（{tag}）")
        ok &= _compare_pala(ref, records, f"合計影響：理想浮點（1/1023）→ reciprocal＋定點 PALA＋定點 PTE＋定點 SD（{tag}，SD {sd_mode}）",
                            t_cols=True)
        return 0 if ok else 1
    if pte_frac is not None:
        # 分開比較：T_MIN 修改（浮點 PTE，1/1023 → 1/1024）、純 PTE 量化（同為 1/1024）、和理想浮點的合計
        pala_q = _run_schedule(mode, frames, make_active(**ACTIVE0_DEFAULT, inv_frac=inv_frac, pala_fixed=True))
        pala_1024 = _run_schedule(mode, frames, make_active(**ACTIVE0_DEFAULT, inv_frac=inv_frac, pala_fixed=True,
                                                            t_min=T_MIN_FIXED))
        ok = _compare_pala(pala_q, pala_1024, f"T_MIN 修改的影響：定點 PALA＋浮點 PTE，t 下限 1/1023 → 1/1024（F={inv_frac}）",
                           t_cols=True)
        ok &= _compare_pala(pala_1024, records, f"PTE 量化的影響：定點 PALA、t 下限都是 1/1024，浮點 PTE → 定點 PTE"
                                                f"（F={inv_frac}，T={pte_frac}）", t_cols=True)
        ok &= _compare_pala(ref, records, f"合計影響：理想浮點（1/1023）→ reciprocal＋定點 PALA＋定點 PTE"
                                          f"（F={inv_frac}，T={pte_frac}）", t_cols=True)
        return 0 if ok else 1
    if not pala_fixed:
        return 0 if _compare_inv(ref, records, inv_frac) else 1
    # 三種模式分開比較：理想浮點 → 只量化 reciprocal → 再加定點 PALA
    recip = _run_schedule(mode, frames, make_active(**ACTIVE0_DEFAULT, inv_frac=inv_frac))
    ok = _compare_inv(ref, recip, inv_frac)
    ok &= _compare_pala(recip, records, f"PALA 定點的新增影響：只量化 reciprocal → 加上定點 PALA（F={inv_frac}）")
    ok &= _compare_pala(ref, records, f"合計影響：理想浮點 → reciprocal 加定點 PALA（F={inv_frac}）")
    return 0 if ok else 1


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("必須 >= 1")
    return value


# ---------------------------------------------------------------- mutation

# 每項：(名稱, 被改的 production 函式, 函式內的原文, 改成, 預期失敗的檢查標籤片段)。
# 只改 production 函式的原始碼；參考實作、手算預期與 selfcheck 本身不動。
MUTATIONS = [
    ("外圍補 0", "dark_channel", "constant_values=PAD", "constant_values=0", "角落 dark"),
    ("dark 等於 T_h 歸近群", "group_max", "far = dark >= t_h_prev", "far = dark > t_h_prev",
     "dark 全為 120，T_h_prev = 120"),
    ("並列取最後一個", "_group", "int(masked.argmax())", "masked.size - 1 - int(masked.ravel()[::-1].argmax())",
     "並列時取 raster 第一個"),
    ("逐 pixel 用 uint8 相加", "gray_sum", "int(frame_down.astype(np.int64).sum())",
     "int(frame_down.sum(axis=2, dtype=np.uint8).astype(np.int64).sum())", "R+G+B = 765"),
    ("抽樣從 1 開始", "downsample", "frame[::STEP, ::STEP]", "frame[1::STEP, 1::STEP]", "降採樣圖和參考實作逐值相同"),
    ("ω 改成 0.95", "pte", "1.0 - OMEGA * m * k", "1.0 - 0.95 * m * k", "t = 0.70703125"),
    ("γ 改成 1", "pte", "ratio ** GAMMA", "ratio ** 1", "k = 0.9375"),
    ("拿掉 min + max = 0 的處理", "pte", "np.divide(lo, lo + hi, out=np.zeros_like(lo), where=(lo + hi) > 0)",
     "lo / (lo + hi)", "m3：所有中間值都是有限值"),
    ("A^c = 0 的比值當 0", "pte", "out=np.full_like(i, np.inf)", "out=np.zeros_like(i)", "min = 0.6"),
    ("A_F、A_N 對調", "pala", "af, an = np.array(a_far, float)", "an, af = np.array(a_far, float)", "A = A_F、η = 0"),
    ("η 與 1−η 對調", "pala", "(1 - eta)[..., None] * af + eta[..., None] * an",
     "eta[..., None] * af + (1 - eta)[..., None] * an", "η = 0.55"),
    ("主路徑自己做除法，不用 1/I_max", "pala", "(s_max - s) * inv_i_max", "(s_max - s) / max(s_max, 1)",
     "1/I_max 改變 → η"),
    ("灰階 >= I_max 的 pixel，η 回報 NaN", "pala", "np.where(use_far, 0.0,", "np.where(use_far, np.nan,",
     "m2：所有中間值都是有限值"),
    ("q 改成四捨五入", "inv_imax_q", "(1 << inv_frac) // i_max if i_max", "((1 << inv_frac) + i_max // 2) // i_max if i_max",
     "全部滿足 I_max·q <= 2^F"),
    ("I_max = 0 時 q 不是 0", "inv_imax_q", "if i_max else 0", "if i_max else 1 << inv_frac", "q(0, 1, 2, 3"),
    ("主路徑忽略量化的 1/I_max", "make_active", "inv = q / (1 << inv_frac)", "inv = 1 / i_max if i_max > 0 else 0.0",
     "主路徑使用量化的 1/I_max"),
    ("cpu_update 沒有傳遞 F", "cpu_update", 'inv_frac=active["inv_frac"]', "inv_frac=None", "每張的 active 都帶 F"),
    ("MAE 改成不同的 channel 數 ÷ 總數", "_diff_stats", '"mae": n_abs / d.size', '"mae": int(np.count_nonzero(d)) / d.size',
     "R6 單張"),
    ("合計 MAE 改成平均各張的 MAE", "_sum_stats", 'tot["mae"] = tot["abs"] / tot["n"]',
     'tot["mae"] = sum(s["mae"] for s in stats) / len(stats)', "R6 合計"),
    ("截斷條件改成 >= 255", "_clip_counts", "int((r > 255).sum())", "int((r >= 255).sum())", "R6 clipping"),
    ("eta_q 的 shift 少 1 bit", "pala_fixed", "product >> (inv_frac - 8)", "product >> (inv_frac - 9)", "P1 F=16：product"),
    ("A_q 的 η 與 1−η 權重對調", "pala_fixed", "(256 - eta_q)[..., None] * af + eta_q[..., None] * an",
     "eta_q[..., None] * af + (256 - eta_q)[..., None] * an", "P1 F=16：A_q"),
    ("A_q 提前捨成 8-bit 整數", "pala_fixed", '"a_q": a_q}', '"a_q": (a_q >> 8) << 8}', "P1 F=20：A_q"),
    ("轉接時改用浮點 PALA", "apply_pass", 'eta, a_map = fx["eta_q"] / 256, fx["a_q"] / 256',
     "eta, a_map = pala(frame, a_far, a_near, float(inv))", "P1 F=16：轉接"),
    ("cpu_update 沒有傳遞 pala_fixed", "cpu_update", 'pala_fixed=active["pala_fixed"]', "pala_fixed=False",
     "每張的 active 都是定點 PALA"),
    ("報表的 A 誤差固定回報 0", "_compare_pala", 'd_a = float(np.max(np.abs(b["result"]["a_map"] - a["result"]["a_map"])))',
     "d_a = 0.0", "P7 報表的 A 最大差欄"),
    ("m 的分子少乘 256（把 A_q 當成 8-bit A）", "pte_fixed", "(i << 24) // np.maximum(a_q, 1)", "(i << 16) // np.maximum(a_q, 1)",
     "X1 T=10：m_c"),
    ("k 改成 1 − r（γ = 1）", "pte_fixed", "k_q = (1 << 16) - r2_q", "k_q = (1 << 16) - r_q", "X1 T=10：r_q、k_q"),
    ("u 改成四捨五入", "pte_fixed", "u = p >> (36 - t_frac)", "u = (p + (1 << (35 - t_frac))) >> (36 - t_frac)",
     "X1 T=10：u、t_q"),
    ("t_min_q 一律是 1 LSB", "pte_fixed", "one, t_min_q = 1 << t_frac, 1 << (t_frac - 10)", "one, t_min_q = 1 << t_frac, 1",
     "T2 T=16：截斷邊界"),
    ("A_q = 0 的 channel 當成 m_c = 0", "pte_fixed", "np.where(valid, m_c, M_SAT)", "np.where(valid, m_c, 0)", "X5 T=10"),
    ("SD 仍用 1/1023 截 t", "apply_pass", 'sd(frame, a_map, t_raw, active["t_min"])', "sd(frame, a_map, t_raw)",
     "T4 T=10：t_q = t_min_q 的 pixel 進 SD"),
    ("cpu_update 沒有傳遞 pte_frac", "cpu_update", 'pte_frac=active["pte_frac"]', "pte_frac=None",
     "S3 T=10：每張 active 都是定點 PTE"),
    ("報表的 t 誤差固定回報 0", "_compare_pala", 'd_t = float(np.max(np.abs(b["result"]["t"] - a["result"]["t"])))',
     "d_t = 0.0", "T6 PTE 量化報表的 t 最大差欄"),
    ("報表的下限命中數固定回報 0", "_compare_pala", 'hits = [int((r["result"]["t"] <= r["used"]["t_min"]).sum()) for r in (a, b)]',
     "hits = [0, 0]", "T6 PTE 量化報表的下限命中欄"),
    ("拿掉 1/I_max <= 1 的上限", "apply_pass", "or not 0 <= inv <= 1:", "or not 0 <= inv:", "白色 pixel（η = 0）"),
    ("拿掉 η > 1 的報錯", "pala", "if np.any(eta > 1):", "if False:", "η = 257/256 > 1 應報錯"),
    ("拿掉 J 的有限值檢查", "sd", "if not np.all(np.isfinite(j)):", "if False:", "SD 遇到 NaN 的 A 應報錯"),
    ("輸出改成捨去", "sd", "np.floor(j + 0.5)", "np.floor(j)", "輸出 (9, 79, 150)"),
    ("輸出改成 round-half-even", "sd", "np.floor(j + 0.5)", "np.round(j)", "J = 100.5 → 101"),
    ("A 先轉 int 再驗證", "_as_rgb", "return tuple(_as_int(v, name) for v in values)",
     "return tuple(_as_int(int(v), name) for v in values)", "A 含小數 100.9"),
    ("step 4 門檻改成 >= 3", "_step4", "if diff > 3:", "if diff >= 3:", "A_F 差 3"),
    ("step 4 平均改成進位", "_step4", "(r + p) // 2 for", "(r + p + 1) // 2 for", "(202, 199, 200)"),
    ("step 4 只平均差 > 3 的那一色", "_step4", "tuple((r + p) // 2 for r, p in zip(raw, prev))",
     "tuple((r + p) // 2 if abs(r - p) > 3 else r for r, p in zip(raw, prev))", "S3 frame"),
    ("第一次更新也做 step 4", "cpu_update", 'elif not active["measured"]:', "elif False:", "和預設值差 15"),
    ("空群組寫 RGB 0", "cpu_update", 'used[g], action = prev[g], "空群組：沿用上一個"',
     'used[g], action = (0, 0, 0), "空群組：沿用上一個"', "遠群為空 → A_F 沿用"),
    ("影片排程本張就用新參數", "run_video", 'rec["used"] = active ', 'rec["used"] = rec["next"] ',
     "frame n 的參數 = frame n−1"),
    ("靜態排程用該圖自己的平均分群", "run_static", "rec = _frame_record(0, image, active0)",
     'rec = _frame_record(0, image, dict(active0, t_h=stats_pass(image, 0)["gray_mean"]))', "手算 S5"),
    ("理想 SD 的 rounding 改成 floor", "sd_fixed", "rounded = (2 * n + d) // (2 * d)", "rounded = n // d", "SD D2 ideal"),
    ("候選 A 的 rounding 改成 floor", "sd_fixed", "rounded = (p + (1 << (k + 7))) >> (k + 8)", "rounded = p >> (k + 8)",
     "SD D2 lut"),
    ("SD 的 clip 改成 uint8 wraparound", "sd_fixed", "np.clip(rounded, 0, 255).astype(np.uint8)", "rounded.astype(np.uint8)",
     "SD D3 ideal"),
    ("SD 的 delta 少乘 256（把 A_q 當成 8-bit A）", "sd_fixed", "delta = 256 * i - a_q", "delta = i - a_q", "SD D1 T=10 ideal"),
    ("候選 A 的 LUT 改用 floor", "sd_fixed", "inv = ((1 << (k + t_frac + 1)) + tq) // (2 * tq)", "inv = (1 << (k + t_frac)) // tq",
     "SD D7 lut"),
    ("cpu_update 沒有傳遞 sd_mode", "cpu_update", 'sd_mode=active["sd_mode"]', "sd_mode=None", "SD S3 T=10 ideal：每張 active"),
    ("apply_pass 沒有換成定點 SD 的輸出", "apply_pass", 'j, out = fx.pop("sd_j"), fx.pop("out")', 'fx.pop("sd_j"), fx.pop("out")',
     "SD 轉接 lut：兩個 pixel"),
    ("SD 報表的 signed 平均固定回報 0", "_compare_sd", "int(d.sum()) / s['n']", "0 / s['n']", "SD 報表的 signed 平均欄"),
    ("SD 報表的 e 少減 1", "_compare_sd", 'np.frexp(b["result"]["t_q"].astype(float))[1] - 1', 'np.frexp(b["result"]["t_q"].astype(float))[1]',
     "SD 報表的 e 分組"),
    ("理想 SD 的 rounding 改成 round-half-even", "sd_fixed", "rounded = (2 * n + d) // (2 * d)",
     "rounded = np.round(n / d).astype(np.int64)", "SD D8 ideal"),
    ("候選 A 的 rounding 改成 round-half-even", "sd_fixed", "rounded = (p + (1 << (k + 7))) >> (k + 8)",
     "rounded = np.round(p / (1 << (k + 8))).astype(np.int64)", "SD D8 lut"),
    ("候選 A 的 sd_tie 固定為 False", "sd_fixed", '"sd_tie": (p + (1 << (k + 7))) % (1 << (k + 8)) == 0}',
     '"sd_tie": np.zeros(p.shape, bool)}', "SD D7 sd_tie"),
    ("SD 報表的 clip 數固定回報 0", "_compare_sd", 'clip = [_clip_counts(r["result"]["j"]) for r in (a, b)]',
     "clip = [(0, 0), (0, 0)]", "SD S8 報表的 clip 欄"),
    ("SD 報表的合計 signed 平均固定回報 0", "_compare_sd", "signed / tot['n']", "0 / tot['n']", "SD S8 報表的合計 signed 平均"),
    ("tie = last 仍取第一個", "_group", "masked.size - 1 - int(masked.ravel()[::-1].argmax())", "int(masked.ravel().argmax())",
     "規則 G1 last"),
    ("th_round = ceil 仍用 floor", "cpu_update", '-(-stats["gray_sum"] // (3 * stats["n"]))', 'stats["gray_sum"] // (3 * stats["n"])',
     "規則 G2 ceil"),
    ("cpu_update 沒有傳遞 tie", "cpu_update", 'tie=active["tie"]', 'tie="first"', "規則 G4：每張 active 都帶 tie"),
    ("cpu_update 沒有傳遞 th_round", "cpu_update", 'th_round=active["th_round"]', 'th_round="floor"', "規則 G4：每張 active 都帶 th_round"),
    ("統計時沒有用 active 的 tie", "_frame_record", 'stats_pass(frame, active["t_h"], active["tie"])', 'stats_pass(frame, active["t_h"])',
     "規則 G4：tie = last 時每張統計"),
    ("規則報表的 A_F 前後印成同一個", "_compare_rules", "{a['used']['a_far']}／{b['used']['a_far']}",
     "{b['used']['a_far']}／{b['used']['a_far']}", "規則 G6 tie：frame 1"),
    ("規則報表逐 frame 的不同 channel 數固定 0", "_compare_rules", "| {s['diff']} |", "| 0 |", "規則 G7：frame 0"),
    ("規則報表合計的不同 channel 數固定 0", "_compare_rules", "{tot['diff']}／{tot['n']}", "0／{tot['n']}", "規則 G7：合計列"),
    ("規則報表合計的 MAE 固定 0", "_compare_rules", "MAE {tot['mae']:.6f}", "MAE {0:.6f}", "規則 G7：合計列"),
    ("規則報表合計的最大差固定 0", "_compare_rules", "最大絕對差 {tot['max']}", "最大絕對差 0", "規則 G7：合計列"),
    ("stats_pass 接受非 Python str 的 tie", "stats_pass", "if type(tie) is not str or tie not in TIE_RULES:", "if tie not in TIE_RULES:",
     "規則 G5：stats_pass 的 tie = np.str_"),
    ("規則報表逐 frame 的 pixel 欄印成不同 channel 數", "_compare_rules", "| {s['pix']} \"", "| {s['diff']} \"", "規則 G7：frame 0"),
    ("規則報表逐 frame 的比例分子改成 pixel 數", "_compare_rules", "{s['diff'] / s['n']:.4%}", "{s['pix'] / s['n']:.4%}",
     "規則 G7：frame 0"),
    ("規則報表合計的 pixel 印成不同 channel 數", "_compare_rules", "有差異的 pixel {tot['pix']}／", "有差異的 pixel {tot['diff']}／",
     "規則 G7：合計列"),
    ("規則報表合計的 pixel 分子改成總 pixel 數", "_compare_rules", "{tot['pix']}／{tot['n'] // 3}", "{tot['n'] // 3}／{tot['n'] // 3}",
     "規則 G7：合計列"),
    ("規則報表合計的比例分子改成 pixel 數", "_compare_rules", "{tot['diff'] / tot['n']:.4%}", "{tot['pix'] / tot['n']:.4%}",
     "規則 G7：合計列"),
    ("規則報表合計的最大差用最後一張", "_compare_rules", "最大絕對差 {tot['max']}", "最大絕對差 {s['max']}", "規則 G7：合計列"),
]


def _run_selfcheck_source(source):
    """在新的 namespace 執行一份原始碼的 selfcheck。回傳 (exit code 或 None, FAIL 標籤, 例外, 結果行)。"""
    import contextlib
    import io
    import warnings

    ns = {"__name__": "ate_golden_mutant"}
    buf = io.StringIO()
    rc, error = None, None
    with contextlib.redirect_stdout(buf), warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            exec(compile(source, "<mutant>", "exec"), ns)
            rc = ns["selfcheck"]()
        except Exception as e:      # 例外不算抓到規則，單獨回報
            error = f"{type(e).__name__}: {e}"
    lines = buf.getvalue().splitlines()
    fails = [line.strip()[len("FAIL"):].strip() for line in lines if line.strip().startswith("FAIL")]
    result = next((line for line in reversed(lines) if line.startswith("結果：")), "（沒有結果行）")
    return rc, fails, error, result


def mutation_command():
    import inspect

    path = __file__
    source = open(path, encoding="utf-8").read()
    print(f"原始碼：{path}（SHA-256 {hashlib.sha256(source.encode()).hexdigest()}）")
    rc, fails, error, result = _run_selfcheck_source(source)
    print(f"baseline（未改）：exit code {rc}，{result}，FAIL {len(fails)}，例外 {error}")
    if rc != 0 or error:
        return 1
    caught = 0
    for n, (name, func, old, new, expect) in enumerate(MUTATIONS, 1):
        body = inspect.getsource(globals()[func])
        if body.count(old) != 1 or source.count(body) != 1:
            print(f"M{n} {name}：替換位置不唯一，無法執行")
            return 1
        mutated = source.replace(body, body.replace(old, new))
        rc, fails, error, result = _run_selfcheck_source(mutated)
        hit = [f for f in fails if expect in f]
        if error is None and rc == 1 and hit:
            verdict, ok = "抓到", True
        elif error is not None and hit:
            verdict, ok = "預期項 FAIL 後例外中止", False
        elif error is not None:
            verdict, ok = "例外中止，預期項未 FAIL", False
        elif rc == 1:
            verdict, ok = "selfcheck 失敗，但預期項沒有 FAIL", False
        else:
            verdict, ok = "沒抓到", False
        caught += ok
        print(f"M{n} {name}［{verdict}］")
        print(f"  替換：{func}()：{old!r} → {new!r}")
        print(f"  預期失敗的檢查含：{expect!r}")
        print(f"  selfcheck exit code {rc}，{result}，例外 {error}")
        for f in fails:
            print(f"  FAIL {f}")
    rc, fails, error, result = _run_selfcheck_source(open(path, encoding="utf-8").read())
    print(f"baseline（結束後重跑）：exit code {rc}，{result}，例外 {error}")
    print(f"結果：{caught}/{len(MUTATIONS)} 個 mutation 在預期項目失敗")
    return 0 if caught == len(MUTATIONS) and rc == 0 and error is None else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="AVSD Final E golden model：階段 2 統計路徑")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selfcheck", help="執行人工案例與性質檢查")
    p = sub.add_parser("stats", help="讀一張圖，印出統計值，不寫任何檔案")
    p.add_argument("image")
    p.add_argument("--t-h-prev", action="append", required=True,
                   help="0–255 的整數，或 self（用該圖自己的平均灰階）；可重複")
    p = sub.add_parser("demo", help="單張浮點除霧展示；有 --out-dir 才寫 PNG")
    p.add_argument("image")
    p.add_argument("--out-dir")
    p.add_argument("--compare", help="並列顯示的另一張輸出（例如 DCP 結果）")
    p = sub.add_parser("video", help="影片排程：依序處理多張圖，印逐 frame 表格，不寫檔")
    p.add_argument("images", nargs="+")
    p.add_argument("--repeat", type=_positive_int, default=1, help="每張圖重複的 frame 數（>= 1）")
    p.add_argument("--inv-frac", type=int, choices=INV_FRACS, help="reciprocal 局部量化實驗：1/I_max 用 floor(2^F / I_max)")
    p.add_argument("--pala-fixed", action="store_true", help="PALA 局部定點模型（需要 --inv-frac）")
    p.add_argument("--pte-frac", type=int, choices=PTE_FRACS, help="PTE 理想定點 reference，t 的小數位數 T（需要 --pala-fixed）")
    p.add_argument("--sd", choices=SD_MODES, help="定點 SD：ideal 理想整數 reference；lut 候選 A 直接 LUT（需要 --pte-frac，lut 只用 10）")
    p.add_argument("--tie", choices=TIE_RULES, default="first", help="並列規則：first 取 raster 第一個（預設）；last 取最後一個")
    p.add_argument("--th-round", choices=TH_ROUNDS, default="floor", help="下一張 T_h 的取整：floor（預設）；ceil（等於 Eq.7 的實數比較）")
    p = sub.add_parser("static", help="靜態排程：同一張圖讀兩次，印表格，不寫檔")
    p.add_argument("image")
    p.add_argument("--inv-frac", type=int, choices=INV_FRACS, help="reciprocal 局部量化實驗：1/I_max 用 floor(2^F / I_max)")
    p.add_argument("--pala-fixed", action="store_true", help="PALA 局部定點模型（需要 --inv-frac）")
    p.add_argument("--pte-frac", type=int, choices=PTE_FRACS, help="PTE 理想定點 reference，t 的小數位數 T（需要 --pala-fixed）")
    p.add_argument("--sd", choices=SD_MODES, help="定點 SD：ideal 理想整數 reference；lut 候選 A 直接 LUT（需要 --pte-frac，lut 只用 10）")
    p.add_argument("--tie", choices=TIE_RULES, default="first", help="並列規則：first 取 raster 第一個（預設）；last 取最後一個")
    p.add_argument("--th-round", choices=TH_ROUNDS, default="floor", help="下一張 T_h 的取整：floor（預設）；ceil（等於 Eq.7 的實數比較）")
    sub.add_parser("mutation", help="固定的 mutation 清單：逐一改錯 production 函式，確認 selfcheck 在預期項目失敗")
    args = parser.parse_args(argv)
    if getattr(args, "pala_fixed", False) and args.inv_frac is None:
        parser.error("--pala-fixed 需要 --inv-frac 16 或 20")
    if getattr(args, "pte_frac", None) is not None and not args.pala_fixed:
        parser.error("--pte-frac 需要 --pala-fixed 與 --inv-frac")
    if getattr(args, "sd", None) is not None and (args.pte_frac is None or (args.sd == "lut" and args.pte_frac != 10)):
        parser.error("--sd 需要 --pte-frac（以及 --pala-fixed、--inv-frac）；--sd lut 只用於 --pte-frac 10")
    if args.command == "selfcheck":
        return selfcheck()
    if args.command == "mutation":
        return mutation_command()
    if args.command == "video":
        return schedule_command("video", args.images, args.repeat, args.inv_frac, args.pala_fixed, args.pte_frac, args.sd,
                                args.tie, args.th_round)
    if args.command == "static":
        return schedule_command("static", [args.image], inv_frac=args.inv_frac, pala_fixed=args.pala_fixed,
                                pte_frac=args.pte_frac, sd_mode=args.sd, tie=args.tie, th_round=args.th_round)
    if args.command == "demo":
        return demo_command(args.image, args.out_dir, args.compare)
    return stats_command(args.image, args.t_h_prev)


if __name__ == "__main__":
    sys.exit(main())
