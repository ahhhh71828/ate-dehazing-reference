"""AVSD Final E：PTE reciprocal LUT 第一輪（E Notes 4.13；軟體候選，不是 RTL 規格）。

PTE 的兩種除法改成「查倒數表＋乘法」，定義照 E Notes 4.13.4–4.13.6（第二稿，2026-11-07 Codex 針對性複審通過）：
- ratio：s = lo + hi 當 index，E[s] = RN(2^23 / s)（s = 2–510；E[0] = E[1] = 0），r = (lo·E[s]) >> 7，floor。
- I/A：1 ≤ A_q ≤ 127 不查表（I = 0 時 0，否則 M_SAT）；A_q ≥ 128 時 n = bit_length(A_q) − 1，
  n ≤ B：j = (A_q << (B − n)) − 2^B；n > B：j = RS(A_q, n − B) − 2^B（四捨五入，.5 進位），
  R[j] = RN(2^(17+B) / (2^B + j))，m_raw = (I·R[j]) >> (n − 7)，m_c = min(m_raw, M_SAT)。
- 三色各自查表、乘、shift、飽和，再取 min；A_q = 0 的 channel 不參加（m_c = −1）。
- 下游 r2、k、p、u、t 的公式和 ate_golden.pte_fixed 相同；只接受 T = 10。

使用者 2026-11-07 批准第一輪：新檔、B = 8 與 10 都做、既有程式與依賴不變。B、K_r = 23、L = 17 都是草案提案，
C 與團隊都還沒同意；這是可以和理想定點 reference 逐值比對的候選版本，不是論文作者的算法，也不是最終 golden。

本檔 import ate_golden（只讀取使用 pte_fixed、run_static、sd_fixed、make_active 與常數），不修改它。
只印到螢幕，不寫任何檔案。

用法（在 dehaze-e/ 執行；Python 一律 -B；讀影像時加 -I）：
    .venv/bin/python -B ate_pte_lut.py selfcheck
    .venv/bin/python -B ate_pte_lut.py mutation
    .venv/bin/python -B ate_pte_lut.py measure            # 窮舉表與單元誤差、100 萬組隨機 pixel，不讀影像
    .venv/bin/python -B -I ate_pte_lut.py compare <image> # 一張影像，FX 設定（靜態排程）
"""

import argparse
import hashlib
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np

# 用 -I 執行時，Python 不會把本檔所在的目錄放進 sys.path；只加入本檔自己的目錄（和 ate_quality.py 相同的做法）
_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else None
if _HERE and _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ate_golden as golden  # noqa: E402

L_INV = 17                  # I/A 倒數表的 scale（4.13.5）
B_CHOICES = (8, 10)         # I/A 表 index 取 A_q 最高位之後的 bit 數（4.13.5）
T_FRAC = 10                 # 查表模式只接受 T = 10（4.13.3）
A_Q_MAX = 255 * 256         # 定點 PALA 的 A_q 上限 65280
SMALL_A_Q = 127             # 1 ≤ A_q ≤ 127 走特殊分支（4.13.5 步驟 2）
M_SAT = golden.M_SAT        # 131071
T_GUARD = {8: 3, 10: 1}     # 4.13.6：和 pte_fixed 比，|Δt_q| 的上界
SEED = 20261107             # measure 的隨機 pixel（使用者 2026-11-07 批准 100 萬組）
N_RANDOM = 1_000_000
RESOURCE_HAND = {"ratio": 11_753, 8: 4_626, 10: 18_450, "three8": 13_878, "three10": 55_350}   # 4.13.7 手算的 logical bits

EXC_MARK = "［呼叫丟出例外，不是正常 FAIL］"   # 檢查本身丟例外時加在 FAIL 標籤上；mutation 不把它算成抓到
ONLY_EXC = "指定檢查只因呼叫例外而 FAIL，不算抓到"
EXC_STOP = "例外中止，預期項未 FAIL"


# ---------------------------------------------------------------- 表與查表

def ratio_table(kr=23):
    """4.13.4：E[s] = RN(2^kr / s) = floor((2^(kr+1) + s) / (2s))，s = 2–510；E[0] = E[1] = 0。int64，長度 511。"""
    s = np.arange(511, dtype=np.int64)
    e = ((1 << (kr + 1)) + s) // (2 * np.maximum(s, 1))
    return np.where(s >= 2, e, 0)


def _ratio_lut(lo, hi):
    """4.13.4：r_lut = (lo·E[s]) >> (K_r − 16)，floor。s ≤ 1 時 lo = 0，所以 r = 0。"""
    kr = 23
    return (lo * ratio_table(kr)[lo + hi]) >> (kr - 16)


def inv_a_table(b):
    """4.13.5 步驟 5：R[j] = RN(2^(L+b) / (2^b + j)) = floor((2^(L+b+1) + d) / (2d))，d = 2^b + j，j = 0–2^b。"""
    if type(b) is not int or b not in B_CHOICES:
        raise ValueError(f"b 必須是 {B_CHOICES} 之一的 Python int：{b!r}")
    d = (1 << b) + np.arange((1 << b) + 1, dtype=np.int64)
    return ((1 << (L_INV + b + 1)) + d) // (2 * d)


def _bit_length(x):
    """非負 int64（< 2^16）的 bit_length；0 → 0。"""
    out = np.zeros_like(x)
    for k in range(16):
        out += (x >> k) > 0
    return out


def inv_a_lookup(i, a_q, b):
    """4.13.5 步驟 3–7（A_q ≥ 128 的查表路徑）。i、a_q 是同形狀的 int64。

    回傳 (n, j, m_raw)：n = bit_length(A_q) − 1（A_q = 0 時 −1）；A_q ≤ 127 的位置不查表，j、m_raw 都是 −1。
    m_raw 是飽和前的值。"""
    big = a_q > SMALL_A_Q
    n = _bit_length(a_q) - 1
    a = np.where(big, a_q, 1 << 7)               # 不查表的位置代入 128，結果最後丟掉
    nn = np.where(big, n, 7)
    shl = np.maximum(b - nn, 0)
    shr = np.maximum(nn - b, 0)
    half = np.where(shr > 0, np.left_shift(1, np.maximum(shr - 1, 0)), 0)
    j_le = (a << shl) - (1 << b)
    j_gt = ((a + half) >> shr) - (1 << b)
    j = np.where(nn <= b, j_le, j_gt)
    m_raw = (i * inv_a_table(b)[j]) >> (nn - 7)
    return n, np.where(big, j, -1), np.where(big, m_raw, -1)


def _check_inputs(frame, a_q, b, t_frac):
    golden._check_frame(frame)
    if type(b) is not int or b not in B_CHOICES:
        raise ValueError(f"b 必須是 {B_CHOICES} 之一的 Python int：{b!r}")
    if type(t_frac) is not int or t_frac != T_FRAC:
        raise ValueError(f"查表模式只接受 T = {T_FRAC}（4.13.3）：{t_frac!r}")
    a = np.asarray(a_q)
    if a.shape != frame.shape or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"a_q 必須是和 frame 同形狀 {frame.shape} 的整數陣列：{a.shape}, {a.dtype}")
    if a.min() < 0 or a.max() > A_Q_MAX:
        raise ValueError(f"a_q 必須在 0–{A_Q_MAX}：{a.min()}–{a.max()}")


def pte_lut(frame, a_q, b, t_frac=T_FRAC):
    """PTE 查表版（4.13.4–4.13.6）。回傳和 pte_fixed 相同的欄位，另加診斷用的 n、j。

    同名的 m_c 意義不同：這裡是步驟 8 飽和後的值；pte_fixed 是未飽和的診斷值。A_q = 0 的 channel 兩版都是 −1。"""
    _check_inputs(frame, a_q, b, t_frac)
    i = frame.astype(np.int64)
    a = np.asarray(a_q, np.int64)
    lo, hi = i.min(axis=2), i.max(axis=2)
    r_q = _ratio_lut(lo, hi)
    r2_q = (r_q * r_q) >> 16
    k_q = (1 << 16) - r2_q
    n, j, m_raw = inv_a_lookup(i, a, b)
    valid = a > 0
    small = valid & (a <= SMALL_A_Q)
    m_sat = np.where(small, np.where(i == 0, 0, M_SAT), np.minimum(m_raw, M_SAT))
    m_c = np.where(valid, m_sat, -1)
    m_q = np.where(valid.any(axis=2), np.where(valid, m_c, M_SAT).min(axis=2), 0)
    p = 15 * m_q * k_q
    u = p >> (36 - t_frac)
    one, t_min_q = 1 << t_frac, 1 << (t_frac - 10)
    t_q = np.where(u >= one - t_min_q, t_min_q, one - u)
    return {"r_q": r_q, "r2_q": r2_q, "k_q": k_q, "m_c": m_c, "m_q": m_q, "p": p, "u": u, "t_q": t_q, "n": n, "j": j}


# ---------------------------------------------------------------- selfcheck

_SAT_LITERAL = 131071   # selfcheck 用的契約字面值（4.9.2：M_SAT = 2^17 − 1）；刻意不引用 M_SAT，M_SAT 改錯時才抓得到


def _rn(x):
    """RN(x) = floor(x + 1/2)，x 是非負 Fraction。"""
    return math.floor(x + Fraction(1, 2))


def _oracle_pixel(rgb, aq, b):
    """單一 pixel 的參考實作：Python int 與 Fraction 逐步照 4.13.4–4.13.6 計算，和向量化版本分開寫。"""
    lo, hi = min(rgb), max(rgb)
    s = lo + hi
    e = 0 if s < 2 else _rn(Fraction(1 << 23, s))
    r = (lo * e) >> 7
    r2 = (r * r) >> 16
    k = (1 << 16) - r2
    ns, js, ms = [], [], []
    for val, a in zip(rgb, aq):
        if a == 0:
            ns.append(-1), js.append(-1), ms.append(-1)
            continue
        n = a.bit_length() - 1
        if a <= SMALL_A_Q:
            ns.append(n), js.append(-1), ms.append(0 if val == 0 else _SAT_LITERAL)
            continue
        j = (a * (1 << b)) // (1 << n) - (1 << b) if n <= b else _rn(Fraction(a, 1 << (n - b))) - (1 << b)
        r_j = _rn(Fraction(1 << (L_INV + b), (1 << b) + j))
        ns.append(n), js.append(j), ms.append(min((val * r_j) // (1 << (n - 7)), _SAT_LITERAL))
    used = [m for m, a in zip(ms, aq) if a > 0]
    m_q = min(used) if used else 0
    p = 15 * m_q * k
    u = p // (1 << 26)
    t = 1 if u >= 1023 else 1024 - u
    return {"r_q": r, "r2_q": r2, "k_q": k, "m_c": ms, "m_q": m_q, "p": p, "u": u, "t_q": t, "n": ns, "j": js}


def _px(pixels):
    return np.array(pixels, np.uint8).reshape(-1, 1, 3)


def _aq(values):
    return np.array(values, np.int64).reshape(-1, 1, 3)


def _at(r, key, idx=0):
    v = r[key][idx, 0]
    return v.tolist() if np.ndim(v) else int(v)


def _try(ck, label, fn):
    """執行一項檢查；fn 回傳 (ok, detail)。fn 丟例外時記成帶 EXC_MARK 的 FAIL，讓 selfcheck 能跑完。"""
    try:
        ok, detail = fn()
    except Exception as e:      # 例外不算正常 FAIL（mutation 判定會排除）
        ck.check(label + EXC_MARK, False, f"{type(e).__name__}: {e}")
        return
    ck.check(label, bool(ok), detail)


def _selfcheck_tables(ck):
    print("T：表內容（整數不等式；預期值不由被測函式產生）")
    e = ratio_table()
    s = np.arange(511, dtype=np.int64)
    ck.check("T1 E 表：長度 511、E[0] = E[1] = 0、E[2] = 2^22", len(e) == 511 and e[0] == 0 and e[1] == 0 and e[2] == 1 << 22,
             str(e[:3].tolist()))
    ss, es = s[2:], e[2:]
    ck.check("T1 E 表：s = 2–510 的 RN 不等式 |2·E[s]·s − 2^24| < s", bool(np.all(np.abs(2 * es * ss - (1 << 24)) < ss)))
    ck.check("T1 E 表：s = 2–510 單調不增", bool(np.all(np.diff(es) <= 0)))
    ck.check("T1 E 表：最大位寬 23 bit", int(e.max()).bit_length() == 23, str(int(e.max())))
    for b in B_CHOICES:
        r = inv_a_table(b)
        d = (1 << b) + np.arange((1 << b) + 1, dtype=np.int64)
        ck.check(f"T2 B={b} R 表：長度 2^B + 1、R[0] = 2^17、R[2^B] = 2^16",
                 len(r) == (1 << b) + 1 and r[0] == 1 << 17 and r[-1] == 1 << 16, str((len(r), int(r[0]), int(r[-1]))))
        ck.check(f"T2 B={b} R 表：每一項的 RN 不等式 |2·R[j]·d − 2^(L+B+1)| < d", bool(np.all(np.abs(2 * r * d - (1 << (L_INV + b + 1))) < d)))
        ck.check(f"T2 B={b} R 表：單調不增", bool(np.all(np.diff(r) <= 0)))
        ck.check(f"T2 B={b} R 表：最大位寬 18 bit", int(r.max()).bit_length() == 18, str(int(r.max())))


def _selfcheck_hand(ck):
    print("H：手算例（4.13.4–4.13.6、4.13.10；預期值另外用 Fraction 獨立算過，寫成常數）")
    j200 = {8: 144, 10: 576}
    j51200 = {8: 144, 10: 576}
    j51713 = {8: 148, 10: 592}
    j51327 = {8: 145, 10: 580}
    for b in B_CHOICES:
        gray = lambda: pte_lut(_px([(100, 100, 100)]), _aq([(51200,) * 3]), b)  # noqa: E731
        _try(ck, f"H1 B={b}：灰色 pixel 的 r_q、r2_q、k_q = 32767、16383、49153",
             lambda: (lambda r: ((_at(r, "r_q"), _at(r, "r2_q"), _at(r, "k_q")) == (32767, 16383, 49153),
                                 str((_at(r, "r_q"), _at(r, "r2_q"), _at(r, "k_q")))))(gray()))
        _try(ck, f"H1 B={b}：灰色 pixel 的 n、j、m_c、m_q = 15、{j51200[b]}、32767、32767",
             lambda: (lambda r: (_at(r, "n") == [15] * 3 and _at(r, "j") == [j51200[b]] * 3
                                 and _at(r, "m_c") == [32767] * 3 and _at(r, "m_q") == 32767,
                                 str((_at(r, "n"), _at(r, "j"), _at(r, "m_c"), _at(r, "m_q")))))(gray()))
        _try(ck, f"H1 B={b}：灰色 pixel 的 p、u、t_q = 24158945265、359、665（理想 664）",
             lambda: (lambda r: ((_at(r, "p"), _at(r, "u"), _at(r, "t_q")) == (24158945265, 359, 665),
                                 str((_at(r, "p"), _at(r, "u"), _at(r, "t_q")))))(gray()))
        _try(ck, f"H2 B={b}：純黑 pixel → r_q 0、k_q 65536、m_q 0、t_q 1024",
             lambda: (lambda r: ((_at(r, "r_q"), _at(r, "k_q"), _at(r, "m_q"), _at(r, "t_q")) == (0, 65536, 0, 1024), ""))(
                 pte_lut(_px([(0, 0, 0)]), _aq([(51200,) * 3]), b)))
        _try(ck, f"H3 B={b}：I = 255、A_q = 256 → n 8、j 0、m_c 飽和成 131071、u 1440、t_q 1",
             lambda: (lambda r: (_at(r, "n") == [8] * 3 and _at(r, "j") == [0] * 3 and _at(r, "m_c") == [_SAT_LITERAL] * 3
                                 and (_at(r, "m_q"), _at(r, "u"), _at(r, "t_q")) == (_SAT_LITERAL, 1440, 1),
                                 str((_at(r, "m_c"), _at(r, "u"), _at(r, "t_q")))))(
                 pte_lut(_px([(255, 255, 255)]), _aq([(256,) * 3]), b)))
        _try(ck, f"X5 B={b}：A_q[R] = 0 不參加 → m_c (−1, 6553, 9830)、m_q 6553、k_q 61441、t_q 935",
             lambda: (lambda r: (_at(r, "m_c") == [-1, 6553, 9830] and (_at(r, "m_q"), _at(r, "k_q"), _at(r, "t_q")) == (6553, 61441, 935),
                                 str((_at(r, "m_c"), _at(r, "k_q"), _at(r, "t_q")))))(
                 pte_lut(_px([(10, 20, 30)]), _aq([(0, 51200, 51200)]), b)))
        _try(ck, f"X6 B={b}：三色 A_q 都是 0 → m_c 都是 −1、m_q 0、t_q 1024",
             lambda: (lambda r: (_at(r, "m_c") == [-1] * 3 and (_at(r, "m_q"), _at(r, "t_q")) == (0, 1024), ""))(
                 pte_lut(_px([(10, 20, 30)]), _aq([(0, 0, 0)]), b)))
        _try(ck, f"H5 B={b}：A_q = 127 不查表、128 查表 → n (6, 7, −1)、j (−1, 0, −1)、m_c (131071, 131071, −1)、t_q 1",
             lambda: (lambda r: (_at(r, "n") == [6, 7, -1] and _at(r, "j") == [-1, 0, -1] and _at(r, "m_c") == [_SAT_LITERAL, _SAT_LITERAL, -1]
                                 and _at(r, "t_q") == 1, str((_at(r, "n"), _at(r, "j"), _at(r, "m_c")))))(
                 pte_lut(_px([(1, 1, 1)]), _aq([(127, 128, 0)]), b)))
        _try(ck, f"H5 B={b}：I = 0、A_q = 127 → m_c (0, −1, −1)、m_q 0、t_q 1024",
             lambda: (lambda r: (_at(r, "m_c") == [0, -1, -1] and (_at(r, "m_q"), _at(r, "t_q")) == (0, 1024), str(_at(r, "m_c"))))(
                 pte_lut(_px([(0, 5, 5)]), _aq([(127, 0, 0)]), b)))
        _try(ck, f"H6 B={b}：I = 1、A_q = 200（n = 7 ≤ B）→ n 7、j {j200[b]}、m_c 83886、t_q 103",
             lambda: (lambda r: (_at(r, "n")[0] == 7 and _at(r, "j")[0] == j200[b] and _at(r, "m_c") == [83886, -1, -1]
                                 and _at(r, "t_q") == 103, str((_at(r, "j"), _at(r, "m_c"), _at(r, "t_q")))))(
                 pte_lut(_px([(1, 1, 1)]), _aq([(200, 0, 0)]), b)))
        _try(ck, f"H7 B={b}：三色各查後取 min → j ({j51200[b]}, {j51713[b]}, −1)、m_c (32767, 32768, −1)、m_q 32767、t_q 598",
             lambda: (lambda r: (_at(r, "j") == [j51200[b], j51713[b], -1] and _at(r, "m_c") == [32767, 32768, -1]
                                 and (_at(r, "m_q"), _at(r, "t_q")) == (32767, 598), str((_at(r, "m_c"), _at(r, "m_q"), _at(r, "t_q")))))(
                 pte_lut(_px([(100, 101, 199)]), _aq([(51200, 51713, 0)]), b)))
        _try(ck, f"H8 B={b}：lo = 0 的反例 → r_q 0、k_q 65536、m_q 32767、t_q 545（理想 32768、544）",
             lambda: (lambda r: ((_at(r, "r_q"), _at(r, "k_q"), _at(r, "m_q"), _at(r, "t_q")) == (0, 65536, 32767, 545),
                                 str((_at(r, "m_q"), _at(r, "t_q")))))(
                 pte_lut(_px([(0, 100, 100)]), _aq([(0, 51200, 51200)]), b)))
        _try(ck, f"H9 B={b}：index 四捨五入（A_q = 51327、I = 100）→ j {j51327[b]}、m_c 32686",
             lambda: (lambda r: (_at(r, "j")[0] == j51327[b] and _at(r, "m_c")[0] == 32686, str((_at(r, "j"), _at(r, "m_c")))))(
                 pte_lut(_px([(100, 100, 100)]), _aq([(51327, 0, 0)]), b)))
    print("R：同一批手算例的理想版（ate_golden.pte_fixed，T = 10），確認反例的另一邊")
    r = golden.pte_fixed(_px([(100, 100, 100), (100, 101, 199), (0, 100, 100), (1, 1, 1)]),
                         _aq([(51200,) * 3, (51200, 51713, 0), (0, 51200, 51200), (127, 128, 0)]), T_FRAC)
    ck.check("R1 理想版：灰色 t_q 664；H7 m_c (32768, 32767, −1)；H8 m_q 32768、t_q 544；H5 m_c (132104, 131072, −1)",
             _at(r, "t_q", 0) == 664 and _at(r, "m_c", 1) == [32768, 32767, -1] and (_at(r, "m_q", 2), _at(r, "t_q", 2)) == (32768, 544)
             and _at(r, "m_c", 3) == [132104, 131072, -1])


def _selfcheck_equivalence(ck):
    print("Q：逐欄位的等價（只比推導保證的欄位）")
    mid, hi = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    keep = mid <= hi
    px = np.stack([np.zeros(keep.sum(), np.int64), mid[keep], hi[keep]], axis=1).astype(np.uint8).reshape(-1, 1, 3)
    aq = np.full(px.shape, 51200, np.int64)
    ref = golden.pte_fixed(px, aq, T_FRAC)
    vals = np.arange(256)
    grid_i, grid_a = np.meshgrid(vals, np.arange(128), indexing="ij")
    px2 = np.repeat(grid_i.reshape(-1, 1, 1), 3, axis=2).astype(np.uint8)
    aq2 = np.repeat(grid_a.reshape(-1, 1, 1), 3, axis=2).astype(np.int64)
    rng = np.random.default_rng(4130)
    px3 = np.zeros((1000, 1, 3), np.uint8)
    aq3 = rng.integers(SMALL_A_Q + 1, A_Q_MAX + 1, size=(1000, 1, 3))
    ref2, ref3 = golden.pte_fixed(px2, aq2, T_FRAC), golden.pte_fixed(px3, aq3, T_FRAC)

    def channel_ok(r, rf, aq_):
        want = np.where(aq_ > 0, np.minimum(rf["m_c"], _SAT_LITERAL), -1)
        return np.array_equal(r["m_c"], want), f"不同 {int((r['m_c'] != want).sum())} 個"

    for b in B_CHOICES:
        _try(ck, f"Q1 B={b}：lo = 0 的 {px.shape[0]} 個 pixel，r_q、r2_q、k_q 和 pte_fixed 相同",
             lambda: (lambda r: (all(np.array_equal(r[k], ref[k]) for k in ("r_q", "r2_q", "k_q")), ""))(pte_lut(px, aq, b)))
        _try(ck, f"Q2 B={b}：I = 0–255 × A_q = 0–127 的每個 channel，m_c 等於 min(理想 m_c, M_SAT)；A_q = 0 兩版都是 −1",
             lambda: channel_ok(pte_lut(px2, aq2, b), ref2, aq2))
        _try(ck, f"Q2 B={b}：I = 0、A_q ≥ 128（1000 組隨機）→ m_c 兩版都是 0",
             lambda: channel_ok(pte_lut(px3, aq3, b), ref3, aq3))


def _selfcheck_oracle(ck):
    print("F：向量化版本和 Fraction 單一 pixel 參考版本逐值比對")
    rng = np.random.default_rng(4131)
    px = rng.integers(0, 256, size=(2000, 1, 3)).astype(np.uint8)
    aq = rng.integers(0, A_Q_MAX + 1, size=(2000, 1, 3))
    aq[:200] = rng.integers(0, 600, size=(200, 1, 3))          # 小 A_q 與 0 的比例提高
    edge_a = [0, 1, 127, 128, 129, 255, 256, 257, 511, 512, 1023, 1024, 32767, 32768, 51200, 51327, 51713, 65279, 65280]
    edge_px = _px([(v, v, v) for v in (0, 1, 2, 100, 254, 255)] + [(0, 100, 100), (100, 101, 199), (1, 0, 255)])
    ea = np.array(edge_a, np.int64)
    edge_aq = np.stack([ea, np.roll(ea, 3), np.roll(ea, 7)], axis=1).reshape(-1, 1, 3)
    px = np.concatenate([px, np.repeat(edge_px, len(edge_a), axis=0)])
    aq = np.concatenate([aq, np.tile(edge_aq, (edge_px.shape[0], 1, 1))])
    keys = ("r_q", "r2_q", "k_q", "m_c", "m_q", "p", "u", "t_q", "n", "j")
    for b in B_CHOICES:
        def run(b=b):
            r = pte_lut(px, aq, b)
            bad = 0
            for idx in range(px.shape[0]):
                o = _oracle_pixel(tuple(int(v) for v in px[idx, 0]), tuple(int(v) for v in aq[idx, 0]), b)
                if any(_at(r, k, idx) != o[k] for k in keys):
                    bad += 1
            return bad == 0, f"不同 {bad} 個 pixel"
        _try(ck, f"F1 B={b}：向量化版和 Fraction 參考版的 10 個欄位逐值相同（{px.shape[0]} 個 pixel）", run)


def _selfcheck_bounds(ck):
    print("B：誤差上界（4.13.5 的 m 上界用 Fraction；4.13.6 的 t 上界和 pte_fixed 比）")
    rng = np.random.default_rng(4132)
    a_vals = np.concatenate([np.array([128, 129, 255, 256, 257, 511, 512, 1023, 1024, 2047, 32767, 32768, 51200,
                                       51327, 51713, 65279, 65280]), rng.integers(SMALL_A_Q + 1, A_Q_MAX + 1, 3000)])
    i_vals = np.array([1, 2, 3, 7, 100, 128, 200, 254, 255])
    ii, aa = np.meshgrid(i_vals, a_vals, indexing="ij")
    ii, aa = ii.ravel().astype(np.int64), aa.ravel().astype(np.int64)
    for b in B_CHOICES:
        def m_bound(b=b):
            _, _, m_raw = inv_a_lookup(ii, aa, b)
            eps = Fraction(1, 1 << (b + 1)) + Fraction(1, 1 << 17) + Fraction(1, 1 << (b + 18))
            bad = 0
            for iv, av, mv in zip(ii.tolist(), aa.tolist(), m_raw.tolist()):
                m_star = Fraction(iv << 24, av)
                if abs(mv - math.floor(m_star)) > m_star * eps + 1:
                    bad += 1
            return bad == 0, f"超過上界 {bad} 組"
        _try(ck, f"Bm B={b}：m 的誤差上界（飽和前，A_q ≥ 128，{len(ii)} 組）", m_bound)
    px = rng.integers(0, 256, size=(20000, 1, 3)).astype(np.uint8)
    aq = rng.integers(0, A_Q_MAX + 1, size=(20000, 1, 3))
    ref = golden.pte_fixed(px, aq, T_FRAC)["t_q"]
    for b in B_CHOICES:
        _try(ck, f"Bt B={b}：t 的上界 |Δt_q| ≤ {T_GUARD[b]}（20,000 個隨機 pixel）",
             lambda b=b: (lambda d: (int(np.abs(d).max()) <= T_GUARD[b], f"最大 {int(np.abs(d).max())}"))(
                 pte_lut(px, aq, b)["t_q"] - ref))


def _selfcheck_inputs(ck):
    print("K、V：回傳欄位與輸入檢查")
    one, a1 = _px([(1, 2, 3)]), _aq([(256, 256, 256)])
    ref_keys = set(golden.pte_fixed(one, a1, T_FRAC))
    _try(ck, "K pte_lut 回傳的欄位 = pte_fixed 的欄位 + n、j",
         lambda: (lambda r: (set(r) == ref_keys | {"n", "j"}, str(sorted(r))))(pte_lut(one, a1, 8)))

    def raises(*calls):
        """每個呼叫都要丟 ValueError；其他例外交給 _try 記成帶 EXC_MARK 的 FAIL。"""
        for fn in calls:
            try:
                fn()
            except ValueError:
                continue
            return False, "沒有報錯"
        return True, ""
    _try(ck, "V1 b = 9 報錯", lambda: raises(lambda: pte_lut(one, a1, 9)))
    _try(ck, "V2 t_frac = 16 報錯", lambda: raises(lambda: pte_lut(one, a1, 8, 16)))
    _try(ck, "V3 a_q 形狀不符報錯", lambda: raises(lambda: pte_lut(one, _aq([(256, 256, 256), (1, 1, 1)]), 8)))
    _try(ck, "V4 a_q 超出 0–65280 報錯", lambda: raises(lambda: pte_lut(one, _aq([(65281, 0, 0)]), 8),
                                                     lambda: pte_lut(one, _aq([(-1, 0, 0)]), 8)))
    _try(ck, "V5 frame 不是 uint8 報錯", lambda: raises(lambda: pte_lut(one.astype(np.int64), a1, 8)))


def _selfcheck_compare(ck):
    """compare_command 的接線與報表：記憶體合成影像＋讀檔替身，預期值由 golden 與 pte_lut 另外計算，不經過 compare 的程式碼。"""
    import contextlib
    import io

    print("CP：compare 的接線與報表（合成影像 40×56、seed 4133，讀檔替身；預期值另外計算）")
    rng = np.random.default_rng(4133)
    h, w = 40, 56
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([60 + 3 * yy + 2 * xx, 80 + 2 * yy + xx, 100 + yy + 3 * xx], axis=2)
    frame = np.clip(base + rng.integers(-40, 41, (h, w, 3)), 0, 255).astype(np.uint8)
    raw = b"ate_pte_lut selfcheck synthetic\n" + frame.tobytes()
    box = {}

    def run_compare():
        """只跑一次 compare_command；它丟出的例外每次都重新丟出，讓每一項都記成 EXC_MARK。"""
        if "error" in box:
            raise box["error"]
        if "lines" not in box:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    box["rc"] = compare_command("合成影像", read_rgb=lambda p: (raw, frame))
            except Exception as e:
                box["error"] = e
                raise
            box["lines"] = buf.getvalue().splitlines()
        return box["rc"], box["lines"]

    def sha(a):
        return hashlib.sha256(a.tobytes()).hexdigest()

    # 預期值：FX 設定用字面值另外建立（不用 compare 裡的 ACTIVE0_DEFAULT、T_FRAC），走 golden 的 run_static。
    # 預期值也在 _try 裡才計算：計算時丟出的例外（前置失敗）記成 EXC_MARK，不算正常 FAIL，selfcheck 也能跑完。
    def expect_fx():
        if "fx" not in box:
            active = golden.make_active(128, (255, 255, 255), (255, 255, 255), inv_frac=20, pala_fixed=True, pte_frac=10,
                                        sd_mode="lut")
            box["fx"] = golden.run_static(frame, active)
        return box["fx"]

    def expect_b(b):
        if ("b", b) not in box:
            fx = expect_fx()["result"]
            t_lut = pte_lut(frame, fx["a_q"], b)["t_q"]
            d = (t_lut - fx["t_q"]).ravel()
            hist = "、".join(f"{v:+d}: {int((d == v).sum())}" for v in range(-3, 4))
            t_lines = [f"   影像 B = {b}：Δt_q 分布 {hist}；|Δt_q| > 3：{int((np.abs(d) > 3).sum())}",
                       f"     t_q = 1（T_MIN）的 pixel：理想 {int((fx['t_q'] == 1).sum())}、查表 {int((t_lut == 1).sum())}；"
                       f"超過 guard（|Δt_q| ≤ {T_GUARD[b]}）：{int((np.abs(d) > T_GUARD[b]).sum())}"]
            out = golden.sd_fixed(frame, fx["a_q"], t_lut, 10, "lut")["out"]
            diff = np.abs(out.astype(np.int64) - fx["out"].astype(np.int64))
            n_diff = int((diff > 0).sum())
            d_line = (f"     8-bit 輸出和 FX 不同：channel 值 {n_diff}／{diff.size}（{(diff > 0).mean():.4%}）；"
                      f"pixel {int((diff > 0).any(axis=2).sum())}／{h * w}；最大差 {int(diff.max())}")
            box[("b", b)] = {"t_lines": t_lines, "d_line": d_line, "n_diff": n_diff, "h_line": f"     查表版輸出陣列 SHA-256：{sha(out)}"}
        return box[("b", b)]

    def fx_check():
        rec = expect_fx()
        want = [f"輸入：合成影像（SHA-256 {hashlib.sha256(raw).hexdigest()}），H×W {h}×{w}",
                f"FX 輸出陣列 SHA-256：{sha(rec['result']['out'])}"]
        used = f"除霧用 A_F {rec['used']['a_far']}、A_N {rec['used']['a_near']}"
        rc, lines = run_compare()
        ok = rc == 0 and all(x in lines for x in want) and any(used in line for line in lines)
        return ok, f"rc {rc}"
    _try(ck, "CP FX：exit code 0、輸入 SHA-256、除霧用 A_F／A_N、FX 輸出 hash 都和另外計算的相同", fx_check)
    for b in B_CHOICES:
        _try(ck, f"CP B={b}：Δt_q 分布與 T_MIN 行和另外計算的相同",
             lambda b=b: (all(x in run_compare()[1] for x in expect_b(b)["t_lines"]), ""))
        _try(ck, f"CP B={b}：8-bit 輸出差（channel、pixel、最大差）和另外計算的相同，而且不是 0",
             lambda b=b: (lambda e: (e["n_diff"] > 0 and e["d_line"] in run_compare()[1], f"預期 {e['n_diff']} 個 channel 值不同"))(
                 expect_b(b)))
        _try(ck, f"CP B={b}：查表版輸出 hash 和另外計算的相同（SD 用候選 A，和 FX 相同）",
             lambda b=b: (expect_b(b)["h_line"] in run_compare()[1], ""))


def selfcheck():
    ck = golden._Checker()
    _selfcheck_tables(ck)
    _selfcheck_hand(ck)
    _selfcheck_equivalence(ck)
    _selfcheck_oracle(ck)
    _selfcheck_bounds(ck)
    _selfcheck_inputs(ck)
    _selfcheck_compare(ck)
    print(f"結果：{ck.total - ck.failed}/{ck.total} PASS")
    return 0 if ck.failed == 0 else 1


# ---------------------------------------------------------------- mutation

MUTATIONS = [
    # (名稱, 函式, 原字串, 改成, 指定檢查的標籤含)；4.13.10 的 12 項
    ("M1 K_r 改成 22", "_ratio_lut", "kr = 23", "kr = 22", "H1 B=8：灰色 pixel 的 r_q"),
    ("M2 ratio 表改用 floor", "ratio_table", "((1 << (kr + 1)) + s) // (2 * np.maximum(s, 1))", "(1 << kr) // np.maximum(s, 1)",
     "T1 E 表：s = 2–510 的 RN 不等式"),
    ("M3 ratio 的 shift 改成 >> 8", "_ratio_lut", ">> (kr - 16)", ">> (kr - 15)", "H1 B=8：灰色 pixel 的 r_q"),
    ("M4 index 的 RS 改成截斷", "inv_a_lookup", "((a + half) >> shr)", "(a >> shr)", "H9 B=8"),
    ("M5 n ≤ B 的 index 沒有對齊到 B bit", "inv_a_lookup", "(a << shl) - (1 << b)", "a - (1 << nn)", "H6 B=8"),
    ("M6 I/A 表改用 floor", "inv_a_table", "((1 << (L_INV + b + 1)) + d) // (2 * d)", "(1 << (L_INV + b)) // d",
     "T2 B=8 R 表：每一項的 RN 不等式"),
    ("M7 最後的 shift 改成 n − 8", "inv_a_lookup", ">> (nn - 7)", ">> (nn - 8)", "H1 B=8：灰色 pixel 的 n、j、m_c、m_q"),
    ("M8 拿掉飽和", "pte_lut", "np.minimum(m_raw, M_SAT)", "m_raw", "H3 B=8"),
    ("M9 A_q ≤ 127 的分支在 I ≥ 1 時回傳 0", "pte_lut", "np.where(i == 0, 0, M_SAT)", "np.where(i == 0, 0, 0)",
     "H5 B=8：A_q = 127 不查表"),
    ("M10 A_q = 0 的 channel 當成 m = 0", "pte_lut", "np.where(valid, m_c, M_SAT)", "np.where(valid, m_c, 0)", "H8 B=8"),
    ("M11 先精確找最小 channel，再查一次表", "pte_lut", "np.where(valid, m_c, M_SAT).min(axis=2)",
     "np.take_along_axis(m_c, np.argmin(np.where(valid, (i << 24) // np.maximum(a, 1), 1 << 62), axis=2)[..., None], axis=2)[..., 0]",
     "H7 B=8"),
    ("M12 pte_lut 偷用精確除法", "inv_a_lookup", "(i * inv_a_table(b)[j]) >> (nn - 7)", "(i << 24) // a",
     "H1 B=8：灰色 pixel 的 n、j、m_c、m_q"),
    # 2026-11-07 Codex 獨立 code review 的兩項 P2 之後補的 mutation（func 是 None：只在第一個 def 之前的常數區替換）
    ("M13 飽和常數改成 131070（M_SAT − 1）", None, "M_SAT = golden.M_SAT ", "M_SAT = golden.M_SAT - 1 ", "H3 B=8"),
    ("M14 compare 的查表輸出接成 FX 輸出", "compare_command",
     'out = golden.sd_fixed(frame, res["a_q"], lut["t_q"], T_FRAC, "lut")["out"]', 'out = res["out"]',
     "CP B=8：8-bit 輸出差"),
    ("M15 compare 的查表端 SD 改成理想整數 SD（混入 SD 模式的差異）", "compare_command",
     'lut["t_q"], T_FRAC, "lut")', 'lut["t_q"], T_FRAC, "ideal")', "CP B=8：查表版輸出 hash"),
    ("M16 compare 的 FX 改成 1/I_max F = 16（不公平的設定）", "compare_command", "inv_frac=20", "inv_frac=16", "CP FX"),
]

CONTROLS = [
    # 對照組：只讓 pte_lut 丟例外的改錯，不能算抓到。(名稱, 函式, 原字串, 改成, 指定檢查的標籤含, 預期判定)
    ("C1 pte_lut 一開始就丟 RuntimeError（只有例外）", "pte_lut", "_check_inputs(frame, a_q, b, t_frac)",
     'raise RuntimeError("對照組：只有例外")', "H1 B=8：灰色 pixel 的 n、j、m_c、m_q", ONLY_EXC),
    ("C2 compare 的讀檔丟 OSError（前置失敗，只有例外）", "compare_command", "raw, frame = (read_rgb or golden._read_rgb)(path)",
     'raise OSError("對照組：讀檔失敗")', "CP FX", ONLY_EXC),
]


def _verdict(rc, fails, error, expect):
    """和 ate_quality._verdict 相同的判定規則（複製規則，不 import ate_quality）：
    抓到 = selfcheck 跑完、exit code 1，而且指定檢查是正常 FAIL（不帶 EXC_MARK）。"""
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


def _run_source(source):
    """沿用 golden._run_selfcheck_source；namespace 沒有 __file__，所以在最前面設定成本檔路徑（只影響 sys.path 的處理）。"""
    return golden._run_selfcheck_source(f"__file__ = {os.path.abspath(__file__)!r}\n" + source)


def mutation_command():
    import inspect

    path = __file__
    source = open(path, encoding="utf-8").read()
    print(f"原始碼：{path}（SHA-256 {hashlib.sha256(source.encode()).hexdigest()}）")
    print(f"ate_golden.py：SHA-256 {_sha256(golden.__file__)}")
    rc, fails, error, result = _run_source(source)
    print(f"baseline（未改）：exit code {rc}，{result}，FAIL {len(fails)}，例外 {error}")
    if rc != 0 or error:
        return 1

    def run(entry):
        name, func, old, new, expect = entry[:5]
        if func is None:                          # 模組常數區：第一個 def 之前
            head_end = source.index("\ndef ")
            head, tail = source[:head_end], source[head_end:]
            if head.count(old) != 1:
                print(f"{name}：替換位置不唯一，無法執行")
                return None
            mutated = head.replace(old, new) + tail
        else:
            body = inspect.getsource(globals()[func])
            if body.count(old) != 1 or source.count(body) != 1:
                print(f"{name}：替換位置不唯一，無法執行")
                return None
            mutated = source.replace(body, body.replace(old, new))
        res = _run_source(mutated)
        verdict, ok = _verdict(res[0], res[1], res[2], expect)
        return verdict, ok, res

    def show(entry, label, res):
        name, func, old, new, expect = entry[:5]
        rc, fails, error, result = res
        print(f"{name}［{label}］")
        print(f"  替換：{func + '()' if func else '模組常數區'}：{old!r} → {new!r}")
        print(f"  預期失敗的檢查含：{expect!r}")
        print(f"  selfcheck exit code {rc}，{result}，例外 {error}")
        for f in fails:
            print(f"  FAIL {f}")

    caught = 0
    for entry in MUTATIONS:
        got = run(entry)
        if got is None:
            return 1
        verdict, ok, res = got
        caught += ok
        show(entry, verdict, res)
    controls = 0
    for entry in CONTROLS:
        got = run(entry)
        if got is None:
            return 1
        verdict, ok, res = got
        good = not ok and verdict == entry[5] and res[2] is None
        controls += good
        show(entry, f"對照組：{verdict}；預期「{entry[5]}」；{'符合預期（不算抓到）' if good else '不符合預期'}", res)
    rc, fails, error, result = _run_source(open(path, encoding="utf-8").read())
    print(f"baseline（結束後重跑）：exit code {rc}，{result}，例外 {error}")
    print(f"結果：{caught}/{len(MUTATIONS)} 個 mutation 在預期項目失敗；對照組 {controls}/{len(CONTROLS)} 依預期不算抓到")
    return 0 if caught == len(MUTATIONS) and controls == len(CONTROLS) and rc == 0 and error is None else 1


# ---------------------------------------------------------------- measure、compare

def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _header(title):
    print(title)
    print(f"程式：ate_pte_lut.py SHA-256 {_sha256(__file__)}；ate_golden.py SHA-256 {_sha256(golden.__file__)}")
    print(f"環境：Python {sys.version.split()[0]}、numpy {np.__version__}")


def _measure_ratio():
    lo, hi = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    keep = (lo <= hi) & ((lo + hi) > 0)
    lo, hi = lo[keep].astype(np.int64), hi[keep].astype(np.int64)
    s = lo + hi
    r_i = (lo << 16) // s
    r_l = _ratio_lut(lo, hi)
    k_i = (1 << 16) - ((r_i * r_i) >> 16)
    k_l = (1 << 16) - ((r_l * r_l) >> 16)
    dr, dk = r_l - r_i, k_l - k_i
    print(f"1. ratio 窮舉（0 ≤ lo ≤ hi ≤ 255、s > 0）：{lo.size} 組")
    print(f"   r：不同 {int((dr != 0).sum())} 組；Δr 的範圍 {int(dr.min())}–{int(dr.max())}；signed 平均 {dr.mean():+.6f}")
    print(f"   k：不同 {int((dk != 0).sum())} 組；Δk 的範圍 {int(dk.min())}–{int(dk.max())}；signed 平均 {dk.mean():+.6f}")
    lt = lo < hi
    print(f"   r_lut 最大值：全部 {int(r_l.max())}；lo < hi {int(r_l[lt].max())}；lo = hi {int(r_l[~lt].max())}")
    ok = int(np.abs(dr).max()) <= 1 and int(np.abs(dk).max()) <= 1 and int(r_l.max()) <= 32768 and int(r_l[lt].max()) <= 32704
    print(f"   4.13.4 的推導（|Δr| ≤ 1、|Δk| ≤ 1、r ≤ 32768、lo < hi 時 ≤ 32704）：{'成立' if ok else '不成立'}")
    return ok


def _measure_inv(b):
    a = np.arange(1, A_Q_MAX + 1, dtype=np.int64)
    small, big = a <= SMALL_A_Q, a > SMALL_A_Q
    n_all = _bit_length(a) - 1
    groups = {n: {"cnt": 0, "diff": 0, "max": 0, "sum": 0, "ratio": 0.0} for n in range(7, 16)}
    small_bad = sat_diff = viol = 0
    lim_num = (1 << 17) + (1 << (b + 1)) + 1          # ε_B = lim_num / 2^(B+18)
    scale = 1 << (b + 18)
    for iv in range(256):
        i = np.full(a.shape, iv, np.int64)
        _, _, m_raw = inv_a_lookup(i, a, b)
        ideal = (i << 24) // a
        lut_small = np.where(i == 0, 0, M_SAT)
        small_bad += int((lut_small[small] != np.minimum(ideal[small], M_SAT)).sum())
        d = (m_raw - ideal)[big]
        ab, nb = a[big], n_all[big]
        sat_diff += int((np.minimum(m_raw[big], M_SAT) != np.minimum(ideal[big], M_SAT)).sum())
        # |d| ≤ m*·ε_B + 1  ⇔  |d|·A_q·2^(B+18) ≤ I·2^24·lim_num + A_q·2^(B+18)（整數比較）
        lhs = np.abs(d) * ab * scale
        rhs = (iv << 24) * lim_num + ab * scale
        viol += int((lhs > rhs).sum())
        bound = (iv * (1 << 24) / ab) * (lim_num / scale) + 1
        for n in range(7, 16):
            sel = nb == n
            if not sel.any():
                continue
            g, dn = groups[n], d[sel]
            g["cnt"] += int(sel.sum())
            g["diff"] += int((dn != 0).sum())
            g["max"] = max(g["max"], int(np.abs(dn).max()))
            g["sum"] += int(dn.sum())
            g["ratio"] = max(g["ratio"], float((np.abs(dn) / bound[sel]).max()))
    total = 256 * A_Q_MAX
    print(f"2. I/A 窮舉，B = {b}：I = 0–255 × A_q = 1–65280，{total} 組")
    print(f"   A_q = 1–127（特殊分支）：飽和後和理想值不同 {small_bad} 組")
    print("   n | 組數 | 飽和前不同 | 最大 |Δm| | 最大 |Δm| ／上界 | signed 平均 Δm")
    for n, g in groups.items():
        print(f"   {n} | {g['cnt']} | {g['diff']} | {g['max']} | {g['ratio']:.4f} | {g['sum'] / g['cnt']:+.6f}")
    print(f"   飽和後（m_c）和理想值不同：{sat_diff} 組；超過 4.13.5 上界：{viol} 組")
    return small_bad == 0 and viol == 0


def _t_report(label, t_lut, t_ref, b):
    d = (t_lut - t_ref).ravel()
    hist = "、".join(f"{v:+d}: {int((d == v).sum())}" for v in range(-3, 4))
    other = int((np.abs(d) > 3).sum())
    viol = int((np.abs(d) > T_GUARD[b]).sum())
    print(f"   {label} B = {b}：Δt_q 分布 {hist}；|Δt_q| > 3：{other}")
    print(f"     t_q = 1（T_MIN）的 pixel：理想 {int((t_ref == 1).sum())}、查表 {int((t_lut == 1).sum())}；"
          f"超過 guard（|Δt_q| ≤ {T_GUARD[b]}）：{viol}")
    return viol == 0


def _measure_resources():
    e = ratio_table()
    rows = [("ratio（K_r = 23）", len(e), int(e.max()).bit_length(), RESOURCE_HAND["ratio"])]
    for b in B_CHOICES:
        r = inv_a_table(b)
        rows.append((f"I/A，B = {b}（L = 17）", len(r), int(r.max()).bit_length(), RESOURCE_HAND[b]))
    print("5. 資源（由表的定義計數；logical bits = 項數 × 位寬，不是面積或 timing）")
    ok = True
    for name, cnt, width, hand in rows:
        bits = cnt * width
        ok &= bits == hand
        print(f"   {name}：項數 {cnt}、entry 位寬 {width}、logical bits {bits}（4.13.7 手算 {hand}，{'相同' if bits == hand else '不同'}）")
    for b, key in ((8, "three8"), (10, "three10")):
        bits = 3 * len(inv_a_table(b)) * int(inv_a_table(b).max()).bit_length()
        ok &= bits == RESOURCE_HAND[key]
        print(f"   三份 I/A 表，B = {b}：{bits}（4.13.7 手算 {RESOURCE_HAND[key]}，{'相同' if bits == RESOURCE_HAND[key] else '不同'}）")
    print("   每個 pixel 的讀取：ratio 表 1 次、I/A 表 3 次（R、G、B）")
    return ok


def measure_command():
    t0 = time.perf_counter()
    _header("ate_pte_lut.py measure（E Notes 4.13.10 量測 1、2、3、5；不讀影像、不寫檔；只記錄，不判畫質）")
    ok = _measure_ratio()
    for b in B_CHOICES:
        ok &= _measure_inv(b)
    print(f"3. t：固定 seed {SEED} 的隨機 pixel {N_RANDOM} 個（I 每色 0–255 均勻；A_q 每色 0–65280 均勻），和 pte_fixed 比")
    rng = np.random.default_rng(SEED)
    px = rng.integers(0, 256, size=(N_RANDOM, 1, 3)).astype(np.uint8)
    aq = rng.integers(0, A_Q_MAX + 1, size=(N_RANDOM, 1, 3))
    t_ref = golden.pte_fixed(px, aq, T_FRAC)["t_q"]
    for b in B_CHOICES:
        ok &= _t_report("隨機", pte_lut(px, aq, b)["t_q"], t_ref, b)
    ok &= _measure_resources()
    print(f"判定：{'推導與手算都符合' if ok else '有不符合的項目（停止點：回報，不調整上界或預期值）'}")
    print(f"耗時 {time.perf_counter() - t0:.1f} 秒")
    return 0 if ok else 1


def compare_command(path, read_rgb=None):
    """read_rgb 只給 selfcheck 換成記憶體替身；CLI 不傳，用 golden._read_rgb 讀檔。"""
    t0 = time.perf_counter()
    _header("ate_pte_lut.py compare（E Notes 4.13.10 量測 3、4 的影像部分；只讀影像、不寫檔；輸出差不是畫質）")
    raw, frame = (read_rgb or golden._read_rgb)(path)
    h, w = frame.shape[:2]
    print(f"輸入：{path}（SHA-256 {hashlib.sha256(raw).hexdigest()}），H×W {h}×{w}")
    active0 = golden.make_active(**golden.ACTIVE0_DEFAULT, inv_frac=20, pala_fixed=True, pte_frac=T_FRAC, sd_mode="lut")
    rec = golden.run_static(frame, active0)
    res = rec["result"]
    used = rec["used"]
    print(f"設定：FX（1/I_max F = 20、定點 PALA、PTE 理想定點 T = 10、SD 候選 A）、靜態排程；"
          f"除霧用 A_F {used['a_far']}、A_N {used['a_near']}")
    print(f"FX 輸出陣列 SHA-256：{hashlib.sha256(res['out'].tobytes()).hexdigest()}")
    ok = True
    n_ch = res["out"].size
    for b in B_CHOICES:
        lut = pte_lut(frame, res["a_q"], b)
        ok &= _t_report("影像", lut["t_q"], res["t_q"], b)
        out = golden.sd_fixed(frame, res["a_q"], lut["t_q"], T_FRAC, "lut")["out"]
        diff = np.abs(out.astype(np.int64) - res["out"].astype(np.int64))
        print(f"     8-bit 輸出和 FX 不同：channel 值 {int((diff > 0).sum())}／{n_ch}（{(diff > 0).mean():.4%}）；"
              f"pixel {int((diff > 0).any(axis=2).sum())}／{h * w}；最大差 {int(diff.max())}")
        print(f"     查表版輸出陣列 SHA-256：{hashlib.sha256(out.tobytes()).hexdigest()}")
    print(f"判定：{'guard 都成立' if ok else '有 pixel 超過 guard（停止點：回報）'}")
    print(f"耗時 {time.perf_counter() - t0:.1f} 秒")
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="AVSD Final E：PTE reciprocal LUT 第一輪（E Notes 4.13）")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selfcheck", help="表內容、手算例、逐欄位等價、Fraction 參考版、誤差上界與輸入檢查")
    sub.add_parser("mutation", help="固定的 12 項改錯與 1 個只丟例外的對照組")
    sub.add_parser("measure", help="窮舉表與單元誤差、100 萬組隨機 pixel 的 t 差、資源計數；不讀影像")
    p = sub.add_parser("compare", help="一張影像：FX 設定下查表版和理想定點 PTE 的 t 差與 8-bit 輸出差；不寫檔")
    p.add_argument("image")
    args = parser.parse_args(argv)
    if args.command == "selfcheck":
        return selfcheck()
    if args.command == "mutation":
        return mutation_command()
    if args.command == "measure":
        return measure_command()
    return compare_command(args.image)


if __name__ == "__main__":
    sys.exit(main())
