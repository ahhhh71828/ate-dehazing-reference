"""AVSD Final E：完整軟體 reference v0 向量包（主路徑、統計路徑、CPU 更新、排程、LUT 內容）。

把 ate_golden.py（軟體 reference v0）與 ate_pte_lut.py（PTE 查表候選 B = 8、10）在小型合成資料上的
整數中間值與預期輸出，寫成固定欄寬的文字 hex，供之後的 RTL／ISR testbench 讀取。
這是軟體 reference 的預期答案：不代表團隊已選定硬體格式、B 或 T，也不代表 RTL 已驗證。

設定固定：F = 20、T = 10、tie = first、th_round = floor、SD LUT K = 10；PTE 查表 K_r = 23、L = 17、B = 8／10。
只重用既有數值函式（golden.apply_pass／pte_fixed／sd_fixed／stats_pass／cpu_update／run_video／run_static、
lut.pte_lut／ratio_table／inv_a_table），不改算法、不加依賴。合成影像最大 21×21，不讀 datasets 或照片。

用法（在 dehaze-e/ 執行，Python 一律 -B）：
    .venv/bin/python -B ate_vectors.py selfcheck   # 記憶體內建兩次、讀回、手算／oracle／接線／guard；不寫檔
    .venv/bin/python -B ate_vectors.py write       # 寫 results/vectors-v0/*.hex，寫後從磁碟讀回並做同一組檢查
    .venv/bin/python -B ate_vectors.py verify      # 不寫檔：重建並和磁碟上的檔案逐 byte 比較，再做同一組檢查
"""

import argparse
import hashlib
import math
import os
import sys
from fractions import Fraction

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import ate_golden as golden  # noqa: E402
import ate_pte_lut as lut  # noqa: E402

OUT_DIR = os.path.join(_HERE, "results", "vectors-v0")
FILES = ("frames", "stats_pixels", "stats", "cpu", "schedule", "pala", "pte", "sd", "lut")
F_INV, T_FRAC, K_SD = 20, 10, 10
FX = {"inv_frac": F_INV, "pala_fixed": True, "pte_frac": T_FRAC, "sd_mode": "lut"}   # tie、th_round 用預設 first、floor
SRC = {"EX": 0, "L8": 1, "L10": 2, "DIRECT": 3}
SRC_B = {"L8": 8, "L10": 10}
MODE = {"ideal": 0, "lut": 1}
GUARD_T = {1: 3, 2: 1}       # 本 prompt 的 guard（字面值，不引用 lut.T_GUARD）：src L8 的 |Δt_q| ≤ 3、L10 ≤ 1
SEED = 20261110
HEX = set("0123456789ABCDEF")


# ---------------------------------------------------------------- 欄位與編碼

NOINV = object()


class Fld:
    """一個欄位：digits 個大寫 hex。signed 欄是 4·digits bit 的二補數；inv 不是 NOINV 時，全 F 代表 inv（None 或 −1），
    而且全 F 不在合法範圍內。"""

    def __init__(self, name, digits, desc, lo=0, hi=None, signed=False, inv=NOINV):
        span = 16 ** digits
        if hi is None:
            hi = span // 2 - 1 if signed else span - (2 if inv is not NOINV else 1)
        if signed:
            assert -(span // 2) <= lo <= hi < span // 2 and inv is NOINV, name
        else:
            assert 0 <= lo <= hi < span - (1 if inv is not NOINV else 0), name
        self.name, self.digits, self.desc, self.lo, self.hi, self.signed, self.inv = name, digits, desc, lo, hi, signed, inv
        self.scale = _scale_of(name)

    def key(self):
        return (self.name, self.digits, self.signed, self.lo, self.hi, "-" if self.inv is NOINV else self.inv, self.scale, self.desc)

    def _is_inv(self, v):
        return self.inv is not NOINV and (v is None if self.inv is None else (v is not None and v == self.inv))

    def enc(self, v):
        if self._is_inv(v):
            return "F" * self.digits
        if type(v) is not int or not self.lo <= v <= self.hi:
            raise ValueError(f"{self.name}：{v!r} 不是 {self.lo}–{self.hi} 的 Python int")
        return format(v % 16 ** self.digits, f"0{self.digits}X")

    def dec(self, s):
        if len(s) != self.digits or not set(s) <= HEX:
            raise ValueError(f"{self.name}：{s!r} 不是 {self.digits} 位大寫 hex")
        if self.inv is not NOINV and s == "F" * self.digits:
            return self.inv
        v = int(s, 16)
        if self.signed and v >= 16 ** self.digits // 2:
            v -= 16 ** self.digits
        if not self.lo <= v <= self.hi:
            raise ValueError(f"{self.name}：{s} = {v} 超出 {self.lo}–{self.hi}")
        return v

    def header(self):
        inv = "-" if self.inv is NOINV else "F" * self.digits + ("/none" if self.inv is None else f"/{self.inv}")
        return (f"# field {self.name} digits={self.digits} bits={4 * self.digits} signed={int(self.signed)} "
                f"min={self.lo} max={self.hi} invalid={inv} scale={self.scale} : {self.desc}")


_IDS = {"vec_id", "case_id", "seq", "frame_id", "stats_id", "upd_id", "row", "col", "i", "j", "src", "mode", "table",
        "index", "vec0", "nvec", "far"}


def _scale_of(name):
    """實際值 = 欄位整數 × scale。"-" 是識別碼、代碼或旗標；"見說明" 依同列其他欄位（num）或表代碼（value）。"""
    if name in _IDS or name.endswith(("_valid", "_act", "_trig", "measured", "_i", "_j")) or name.startswith(("tie_", "j_")):
        return "-"
    if name.endswith("inv_q") or name == "product":
        return "2^-20"
    if name == "eta_q" or name.startswith("A_q_"):
        return "2^-8"
    if name in ("r_q", "r2_q", "k_q", "m_q") or name.startswith("m_c_"):
        return "2^-16"
    if name == "p":
        return "2^-36"
    if name in ("u", "t_q", "inv_t_q"):
        return "2^-10"
    if name.startswith("num_") or name == "value":
        return "見說明"
    return "1"


def _parse_field(line):
    left, desc = line.split(" : ", 1)
    tok = left.split(" ")
    if tok[:2] != ["#", "field"]:
        raise ValueError(line)
    kv = dict(t.split("=", 1) for t in tok[3:])
    inv = NOINV
    if kv["invalid"] != "-":
        hexs, meaning = kv["invalid"].split("/")
        inv = None if meaning == "none" else int(meaning)
        if hexs != "F" * int(kv["digits"]):
            raise ValueError(line)
    if int(kv["bits"]) != 4 * int(kv["digits"]):
        raise ValueError(line)
    f = Fld(tok[2], int(kv["digits"]), desc, int(kv["min"]), int(kv["max"]), kv["signed"] == "1", inv)
    f.scale = kv["scale"]
    return f


def _rgb(prefix, desc, digits=2, inv=NOINV):
    return [Fld(f"{prefix}{c}", digits, f"{desc}{c}" + ("；無效時全 F" if inv is not NOINV else ""), hi=255, inv=inv)
            for c in "RGB"]


C3 = "RGB"
CASE = Fld("case_id", 2, "案例編號（檔頭的 case 行）", hi=0xFE)
SEQ = Fld("seq", 2, "排程中的 frame 順序（0 起；靜態排程只有 0）；單元案例為全 F", hi=0xFE, inv=None)
FRAME = Fld("frame_id", 2, "影像編號（frames.hex 檔頭的 frame 行）", hi=0xFE)
FRAME_REF = Fld("frame_id", 2, "影像編號；全 F 表示單元案例（1×n 的列，RGB 直接寫在本列，不在 frames.hex）", hi=0xFE, inv=None)
VEC = Fld("vec_id", 4, "主路徑向量編號（一個 pixel 一個；pala、pte、sd 以它連接）", hi=0xFFFE)
ROW = Fld("row", 3, "原圖的列（0 起）；單元案例固定 0", hi=1079)
COL = Fld("col", 3, "原圖的行（0 起）；單元案例是案例內的 pixel 序號", hi=1919)
STATS_ID = Fld("stats_id", 2, "統計 pass 編號（stats.hex）", hi=0xFE)
UPD_ID = Fld("upd_id", 2, "CPU 更新編號（cpu.hex）", hi=0xFE)
PIX = _rgb("", "輸入 pixel 的 8-bit ")
AQ = [Fld(f"A_q_{c}", 4, f"A·256（{c}）；A_q = (256 − eta_q)·A_F + eta_q·A_N，保留 8 個小數 bit", hi=65280) for c in C3]
IDS = [VEC, CASE, SEQ, FRAME_REF, ROW, COL]


def _grp(g, what):
    return ([Fld(f"{g}_valid", 1, f"{what}是否有點（0 = 空群組）", hi=1),
             Fld(f"{g}_count", 5, f"{what}的點數", hi=230400),
             Fld(f"{g}_max", 3, f"{what}的最大 dark；空群組全 F", hi=255, inv=None),
             Fld(f"{g}_i", 3, "所選點的降採樣列（並列取 raster 第一個）；空群組全 F", hi=359, inv=None),
             Fld(f"{g}_j", 3, "所選點的降採樣行；空群組全 F", hi=639, inv=None)]
            + _rgb(f"{g}_", f"所選點的 ", digits=3, inv=None))


def _act(p, what):
    return ([Fld(f"{p}measured", 1, f"{what}：1 = 量測值；0 = active0 預設（第一次更新不做 step 4）", hi=1),
             Fld(f"{p}t_h", 2, f"{what}：T_h", hi=255)]
            + _rgb(f"{p}AF_", f"{what}：A_F ") + _rgb(f"{p}AN_", f"{what}：A_N ")
            + [Fld(f"{p}I_max", 3, f"{what}：I_max = A_F 三色和", hi=765),
               Fld(f"{p}inv_q", 6, f"{what}：floor(2^20 / I_max)，I_max = 0 時 0；1/I_max = inv_q·2^-20", hi=1 << 20)])


def _upd(g):
    return ([Fld(f"{g}_act", 1, "處理代碼（檔頭 code 行）", hi=4),
             Fld(f"{g}_diff", 3, "三色和上一個 active 的最大絕對差；沒有做 step 4（代碼 0、1、2）時全 F", hi=255, inv=None),
             Fld(f"{g}_trig", 1, "1 = 差 > 3，三色都取 floor((本張 + 上一個) / 2)", hi=1)]
            + _rgb(f"{g}_used_", "step 4 之後寫進下一個 active 的 "))


SCHEMA = {
    "frames": [FRAME, Fld("row", 3, "原圖的列（0 起）", hi=1079), Fld("col", 3, "原圖的行（0 起）", hi=1919)] + PIX,
    "stats_pixels": [STATS_ID, FRAME, Fld("i", 3, "降採樣列（= row / 3）", hi=359), Fld("j", 3, "降採樣行（= col / 3）", hi=639),
                     Fld("row", 3, "原圖的列 = 3i", hi=1077), Fld("col", 3, "原圖的行 = 3j", hi=1917)]
                    + _rgb("", "降採樣點的 ")
                    + [Fld("min", 2, "min(R, G, B)", hi=255),
                       Fld("dark", 2, "dark channel：降採樣圖上以本點為中心的 5×5 min，影像外補 255", hi=255),
                       Fld("far", 1, "1 = dark ≥ t_h_prev（遠群）；0 = 近群", hi=1)],
    "stats": [STATS_ID, CASE, SEQ, FRAME, Fld("t_h_prev", 2, "分群用的 T_h", hi=255),
              Fld("H", 3, "原圖列數", lo=1, hi=1080), Fld("W", 3, "原圖行數", lo=1, hi=1920),
              Fld("Hd", 3, "降採樣列數 ceil(H/3)", lo=1, hi=360), Fld("Wd", 3, "降採樣行數 ceil(W/3)", lo=1, hi=640),
              Fld("N", 5, "降採樣點數 Hd·Wd", lo=1, hi=230400),
              Fld("gray_sum", 7, "降採樣點 R + G + B 的總和（28 bit 可容納 1080p 的 176256000）", hi=176256000),
              Fld("gray_mean", 2, "floor(gray_sum / (3N))", hi=255)]
             + _grp("far", "遠群") + _grp("near", "近群"),
    "cpu": [UPD_ID, CASE, SEQ, Fld("stats_id", 2, "輸入的統計 pass（stats.hex）；全 F 表示直接給定的合成統計值，不經統計路徑", hi=0xFE, inv=None)]
           + _act("prev_", "上一個 active（統計這張時生效）")[:-2]
           + [Fld("gray_sum", 7, "本張 gray_sum", hi=176256000), Fld("N", 5, "本張 N", lo=1, hi=230400),
              Fld("far_valid", 1, "本張遠群是否有點", hi=1)] + _rgb("far_raw_", "本張遠群的 ", digits=3, inv=None)
           + [Fld("near_valid", 1, "本張近群是否有點", hi=1)] + _rgb("near_raw_", "本張近群的 ", digits=3, inv=None)
           + _upd("far") + _upd("near") + _act("next_", "下一個 active")[1:],
    "schedule": [CASE, SEQ, Fld("mode", 1, "0 = static（第一次讀取只統計，第二次讀取除霧）；1 = video", hi=1), FRAME,
                 STATS_ID, UPD_ID, Fld("vec0", 4, "本張主路徑的第一個 vec_id", hi=0xFFFE),
                 Fld("nvec", 4, "本張主路徑的向量數 = H·W（vec_id 連續）", lo=1, hi=0xFFFE)]
                + _act("stat_", "統計時的 active") + _act("use_", "除霧時的 active") + _act("next_", "下一個 active"),
    "pala": IDS + PIX + _rgb("AF_", "A_F 的 ") + _rgb("AN_", "A_N 的 ")
            + [Fld("I_max", 3, "A_F 三色和", hi=765),
               Fld("inv_q", 6, "floor(2^20 / I_max)，I_max = 0 時 0；1/I_max = inv_q·2^-20", hi=1 << 20),
               Fld("S", 3, "R + G + B", hi=765),
               Fld("delta", 3, "S < I_max 時 I_max − S，否則 0", hi=765),
               Fld("product", 6, "delta·inv_q（≤ 2^20）", hi=1 << 20),
               Fld("eta_q", 3, "product >> 12（floor）；η = eta_q / 256，1.0 = 256", hi=256)] + AQ,
    "pte": IDS + [Fld("src", 1, "PTE 版本代碼（檔頭 code 行）", hi=2)] + PIX + AQ
           + [Fld("r_q", 4, "ratio·2^16（ratio = lo / (lo + hi)）", hi=32768),
              Fld("r2_q", 4, "(r_q·r_q) >> 16", hi=16384),
              Fld("k_q", 5, "k·2^16 = 2^16 − r2_q；1.0 = 65536", lo=49152, hi=65536)]
           + [Fld(f"n_{c}", 2, "查表的 n = bit_length(A_q) − 1（只有 L8、L10）；EX 列與 A_q = 0 為全 F（−1）", hi=15, inv=-1) for c in C3]
           + [Fld(f"j_{c}", 3, "I/A 表的 index（只有 L8、L10 且 A_q ≥ 128）；其餘全 F（−1）", hi=1024, inv=-1) for c in C3]
           + [Fld(f"m_c_{c}", 8, "I/A·2^16。EX：floor(I·2^24 / A_q)，未飽和；L8、L10：查表後已飽和到 131071。"
                                  "A_q = 0 的 channel 不參加：全 F（−1）", hi=0xFF000000, inv=-1) for c in C3]
           + [Fld("m_q", 5, "min(m_c, 131071)，A_q = 0 的 channel 不參加；三色 A_q 都是 0 時 0", hi=131071),
              Fld("p", 10, "15·m_q·k_q（ω = 15/16 的精確乘積，scale 2^-36）", hi=(1 << 37) - 1),
              Fld("u", 3, "p >> 26（floor）", hi=1919),
              Fld("t_q", 3, "t·2^10，1.0 = 1024；u ≥ 1023 時為 1（下限 1/1024），否則 1024 − u", lo=1, hi=1024)],
    "sd": IDS + [Fld("src", 1, "t_q 的來源代碼（檔頭 code 行）", hi=3), Fld("mode", 1, "SD 版本代碼（檔頭 code 行）", hi=1)] + PIX + AQ
          + [Fld("t_q", 3, "t·2^10（src 0–2 等於同 vec_id、同 src 的 pte.hex 列）", lo=1, hi=1024),
             Fld("inv_t_q", 6, "LUT：RN(2^20 / t_q)（lut.hex 表 3）；IDEAL 為全 F", lo=1024, hi=1 << 20, inv=None)]
          + [Fld(f"num_{c}", 10, "IDEAL：N = (256·I − A_q)·2^10 + A_q·t_q；LUT：P = (256·I − A_q)·inv_t_q + A_q·2^10。"
                                  "J = N / D 或 P / 2^18", lo=-(1 << 36), hi=1 << 36, signed=True) for c in C3]
          + [Fld("den", 5, "IDEAL：D = 256·t_q；LUT 為全 F（分母固定 2^18）", lo=256, hi=1 << 18, inv=None)]
          + [Fld(f"rnd_{c}", 5, "floor(J + 1/2)，clip 前，可為負", lo=-(1 << 19), hi=(1 << 19) - 1, signed=True) for c in C3]
          + [Fld(f"tie_{c}", 1, "1 = J 剛好是整數 + 1/2", hi=1) for c in C3]
          + [Fld(f"out_{c}", 2, "clip(rnd, 0, 255)", hi=255) for c in C3],
    "lut": [Fld("table", 1, "表代碼（檔頭 code 行）", hi=3),
            Fld("index", 3, "表 0：s = lo + hi（0–510）；表 1、2：j（0–2^B）；表 3：t_q（1–1024）", hi=1024),
            Fld("value", 6, "表 0：RN(2^23 / s)，s < 2 為 0；表 1、2：RN(2^(17+B) / (2^B + j))；表 3：RN(2^20 / t_q)",
                hi=1 << 22)],
}

TITLES = {
    "frames": "合成輸入影像（RGB 分欄，一個 pixel 一列，raster 順序；不是 DRAM packed 格式）",
    "stats_pixels": "統計路徑：逐降採樣點的 RGB、min、dark 與分群",
    "stats": "統計路徑：每次統計 pass 的摘要（gray_sum、遠近群代表點）",
    "cpu": "CPU 更新：上一個 active、本張統計、step 4 與下一個 active",
    "schedule": "排程：每張 frame 統計時、除霧時與下一張的參數，以及主路徑向量的範圍",
    "pala": "主路徑 PALA：reciprocal（inv_q）、eta_q、A_q",
    "pte": "主路徑 PTE：精確整數除法（EX）與查表候選 B = 8、10（L8、L10），同一份 RGB 與 A_q",
    "sd": "主路徑 SD：理想整數（IDEAL）與直接 reciprocal LUT 候選（LUT），各 t_q 來源分開",
    "lut": "LUT 內容：PTE ratio 表、I/A 表（B = 8、10）、SD reciprocal 表",
}

CODES = {
    "pte": ["# code src 0=EX 精確整數除法（ate_golden.pte_fixed，理想定點 reference；m_c 未飽和）",
            "# code src 1=L8 查表候選 B = 8（ate_pte_lut.pte_lut；m_c 已飽和）",
            "# code src 2=L10 查表候選 B = 10（ate_pte_lut.pte_lut；m_c 已飽和）"],
    "sd": ["# code src 0=EX、1=L8、2=L10：t_q 取自同 vec_id、同 src 的 pte.hex 列；3=DIRECT：A_q、t_q 直接給定（本檔的 case 行註明來源）",
           "# code mode 0=IDEAL 理想整數 SD（ate_golden.sd_fixed ideal）",
           "# code mode 1=LUT 直接 reciprocal LUT 候選 A（ate_golden.sd_fixed lut，K = 10、T = 10）"],
    "cpu": ["# code act 0=第一次更新（上一個是 active0），直接採用本張",
            "# code act 1=第一次更新且本群為空：用另一群本張的值",
            "# code act 2=本群為空：沿用上一個 active 的值",
            "# code act 3=step 4：最大差 ≤ 3，採用本張",
            "# code act 4=step 4：最大差 > 3，三色都取 floor((本張 + 上一個) / 2)"],
    "schedule": ["# code mode 0=static：stat_ 是 active0，use_ 等於 next_（CPU 更新後第二次讀取除霧）",
                 "# code mode 1=video：frame n 的 stat_ 等於 use_；use_(n) 等於 next_(n−1)"],
    "lut": ["# code table 0=PTE ratio 倒數表 E[s]（ate_pte_lut.ratio_table，K_r = 23）",
            "# code table 1=PTE I/A 倒數表 R[j]，B = 8（ate_pte_lut.inv_a_table(8)，L = 17）",
            "# code table 2=PTE I/A 倒數表 R[j]，B = 10（ate_pte_lut.inv_a_table(10)，L = 17）",
            "# code table 3=SD reciprocal inv[t_q]（ate_golden.sd_fixed lut 的 sd_inv_q，K = 10、T = 10）"],
}

STATUS = "# status 軟體 reference v0 的預期答案；不是團隊選定的硬體格式，B、T 未選定，RTL 未驗證"
CONFIG = ("# config F=20 T=10 tie=first th_round=floor sd_lut_K=10 pte_lut_Kr=23 pte_lut_L=17 pte_lut_B=8,10 "
          "omega=15/16 gamma=2 active0=T_h:128,A_F:FF.FF.FF,A_N:FF.FF.FF,measured:0")
ENCODING = ("# encoding 文字 hex（大寫）、單一空白分隔、每欄固定位數、UTF-8、LF；signed 欄是 4×digits bit 的二補數；"
            "invalid 欄寫成全 F（全 F 不在合法範圍內）；以 # 開頭的是檔頭")


# ---------------------------------------------------------------- 案例

_rgb_sum = golden._rgb_sum
AQ9 = (39520, 40000, 40480)
W3 = (255, 255, 255)

CHAIN = [   # (case, 名稱, 說明, 來源, [(pixels, A_F, A_N)])：PALA → PTE → SD 一路接下去
    (0x01, "M9", "m9 pixel（《技術說明》第 10.1 節）", "ate_golden selfcheck P1（F = 20）、X1（T = 10）、SD D1",
     [([(60, 90, 120)], (220, 200, 180), (100, 120, 140))]),
    (0x02, "P2-BR", "S 小於／等於／大於 I_max 的分支（S = 0、599、600、700）", "ate_golden selfcheck P2",
     [([(0, 0, 0), _rgb_sum(599), (200, 200, 200), _rgb_sum(700)], (200, 200, 200), (100, 50, 0))]),
    (0x03, "P2-I0", "I_max = 0：全部走 A_F 分支、inv_q = 0（零分母）", "ate_golden selfcheck P2",
     [([(0, 0, 0), (5, 5, 5)], (0, 0, 0), (10, 20, 30))]),
    (0x04, "P2-I1", "I_max = 1：純黑 pixel 的 eta_q = 256（1.0）", "ate_golden selfcheck P2",
     [([(0, 0, 0), (1, 0, 0)], (1, 0, 0), (10, 20, 30))]),
    (0x05, "P2-I765", "I_max = 765、純黑：eta_q = 255", "ate_golden selfcheck P2", [([(0, 0, 0)], W3, (0, 0, 0))]),
    (0x06, "P2-A255", "A_F = A_N = 255 → A_q = 65280（最大合法值）", "ate_golden selfcheck P2",
     [([(0, 0, 0), (100, 50, 25)], W3, W3)]),
    (0x07, "P3-F20", "η 截位（floor）：product 798710／800080 → eta_q 194／195", "ate_golden selfcheck P3",
     [([_rgb_sum(182), _rgb_sum(181)], W3, (0, 0, 0))]),
    (0x08, "P3-I256", "I_max = 256：eta_q = delta（256、255、156）", "ate_golden selfcheck P3",
     [([(0, 0, 0), _rgb_sum(1), _rgb_sum(100)], (86, 85, 85), (0, 0, 0))]),
    (0x09, "RCP", "reciprocal 端點 I_max = 2、3、7、375、390、600（純黑 pixel，η 最大的分支；一個 pixel 一組參數）",
     "ate_golden selfcheck R1（F = 20 的手算 q）", [([(0, 0, 0)], _rgb_sum(v), (0, 0, 0)) for v in (2, 3, 7, 375, 390, 600)]),
]

PTE_DIRECT = [   # (case, 名稱, 說明, 來源, pixels, A_q)：A_q 直接給定，PTE → SD
    (0x10, "X2-H1", "灰色 pixel、A_q = 51200", "ate_golden X2；ate_pte_lut H1", [(100, 100, 100)], [(51200,) * 3]),
    (0x11, "X3-H2", "純黑與 lo = 0 的 pixel（ratio 分母 s = 0、k = 1、m = 0）", "ate_golden X3；ate_pte_lut H2、Q2",
     [(0, 0, 0), (0, 100, 200)], [AQ9, AQ9]),
    (0x12, "X4-H3", "m 飽和：I = 255、A_q = 256；t 下限", "ate_golden X4；ate_pte_lut H3", [(255,) * 3], [(256,) * 3]),
    (0x13, "X5", "A_q[R] = 0 的 channel 不參加（零分母）", "ate_golden X5；ate_pte_lut X5", [(10, 20, 30)], [(0, 51200, 51200)]),
    (0x14, "X6", "三色 A_q 都是 0 → m_q = 0、t = 1", "ate_golden X6；ate_pte_lut X6", [(10, 20, 30)], [(0, 0, 0)]),
    (0x15, "T2", "t 下限邊界：u = 1022、1023、1024", "ate_golden selfcheck T2（T = 10）", [(255,) * 3] * 3,
     [(45945,) * 3, (45900,) * 3, (45856,) * 3]),
    (0x16, "T2M", "最大合法輸入：A_q = 1（精確 m_c = 255·2^24）、A_q = 65280（m_c = 65536）", "ate_golden T2；ate_pte_lut H5 的小 A_q 分支",
     [(255,) * 3] * 2, [(1,) * 3, (65280,) * 3]),
    (0x17, "H5", "A_q = 127 不查表、128 查表；I = 0 的小 A_q", "ate_pte_lut H5；ate_pte_lut R1（精確版）",
     [(1, 1, 1), (0, 5, 5)], [(127, 128, 0), (127, 0, 0)]),
    (0x18, "H6", "n = 7 ≤ B 的正規化（左移對齊到 B bit）", "ate_pte_lut H6", [(1, 1, 1)], [(200, 0, 0)]),
    (0x19, "H7", "三色各自查表後再取 min（和精確版選到不同 channel）", "ate_pte_lut H7；ate_pte_lut R1",
     [(100, 101, 199)], [(51200, 51713, 0)]),
    (0x1A, "H8", "lo = 0 的反例（查表 m_q 32767、精確 32768）", "ate_pte_lut H8；ate_pte_lut R1", [(0, 100, 100)], [(0, 51200, 51200)]),
    (0x1B, "H9", "LUT index 的四捨五入（A_q = 51327）", "ate_pte_lut H9", [(100, 100, 100)], [(51327, 0, 0)]),
]

SD_DIRECT = [   # (case, 名稱, 說明, I, A_q, t_q)：三色相同；A_q、t_q 直接給定
    (0x20, "D2", "J = 99.5：.5 進位", 100, 25728, 512),
    (0x21, "D3", "J = −100：clip 到 0", 0, 25600, 512),
    (0x22, "D4", "J = 510：clip 到 255", 255, 0, 512),
    (0x23, "D5", "t = 1：J = I", 37, 51200, 1024),
    (0x24, "D6", "t = 1/1024（t_q = 1）", 100, 25601, 1),
    (0x25, "D7", "LUT 改變輸出（理想 165、LUT 166）", 16, 0, 99),
    (0x26, "D8", "J = 100.5：tie 向上（不是 half-even）", 100, 25472, 512),
]

CPU_UNIT = [   # (case, 名稱, 說明, prev, 遠群 raw, 近群 raw, gray_sum, N, gray_mean 手算)
    (0x40, "C1", "差 3 不平均、差 4 平均；A_F、A_N 分開判斷", "prev", (203, 200, 200), (104, 100, 100), 17249, 49, 117),
    (0x41, "C2", "負方向的差 4、差 3", "prev", (196, 200, 200), (97, 100, 100), 17249, 49, 117),
    (0x42, "C3", "平均時三色都取 floor（總和是奇數）", "prev", (205, 199, 200), (100, 100, 100), 17249, 49, 117),
    (0x43, "C4", "floor((255 + 250) / 2) = 252（8-bit 相加不得溢位）", "prev_hi", (255, 255, 255), (100, 100, 100), 17249, 49, 117),
    (0x44, "C5", "空群組：沿用上一個 active", "prev", None, (101, 100, 100), 13300, 49, 90),
    (0x45, "C6", "第一次更新：不做 step 4；近群為空時用遠群本張的值", "a0", (240, 240, 240), None, 19200, 49, 130),
    (0x46, "C7", "I_max = 0：inv_q 寫 0", "a0", (0, 0, 0), (0, 0, 0), 0, 49, 0),
]

STATS_UNIT = [   # (case, 名稱, 說明, 來源, frame 名稱, t_h；None 表示用 oracle dark 的中位數)
    (0x30, "ST-A", "21×21 隨機（降採樣 7×7，5×5 視窗內部；T_h = oracle dark 的中位數）", f"ate_golden 案例 a 的作法（本檔 seed {SEED}）", "SA21", None),
    (0x31, "ST-B", "20×17 隨機（尾端：降採樣 7×6；角落視窗補 255）", f"ate_golden 案例 b 的作法（縮成 21 以內，本檔 seed {SEED}）", "SB2017", 128),
    (0x32, "ST-C", "dark 恰好等於 T_h → 歸遠群；近群為空", "ate_golden 案例 c", "C120", 120),
    (0x33, "ST-D", "遠群兩點並列最大，取 raster 第一個", "ate_golden 案例 d", "D13", 150),
    (0x34, "ST-E", "近群最大 dark 為 0（仍 valid）；遠群為空", "ate_golden 案例 e", "E15", 1),
    (0x35, "ST-F", "遠群為空（dark 120 < 121）", "ate_golden 案例 f", "C120", 121),
]

SCHEDULES = [   # (case, 名稱, 說明, 來源, mode, frame 名稱)
    (0x50, "VS2", "video：近群差 3／差 4（9×9 單色 × 4）", "ate_golden 排程 S2", "video", ["C120", "C123", "C127", "C127"]),
    (0x51, "VS3", "video：遠群變空、近群逐張收斂（9×9 單色 × 6）", "ate_golden 排程 S3", "video", ["G130"] * 2 + ["B100"] * 4),
    (0x52, "VX15", "video：兩區 15×9 × 7（兩群都有、近群 step 4 收斂）", "ate_golden 排程 S6 的 A 序列（影像縮成 15×9，T_h 手算 103）",
     "video", ["X15"] * 7),
    (0x53, "SS4", "static：9×9 單色（A = I → 輸出 = 輸入）", "ate_golden 排程 S4", "static", ["B100"]),
    (0x54, "SX15", "static：兩區 15×9（第一次讀取用 active0 的 T_h 128 分群）", "ate_golden 排程 S5 的觀念（影像縮成 15×9）",
     "static", ["X15"]),
    (0x55, "SRND", "static：21×21 上方亮區＋下方隨機（兩群都有，PALA 混合）", f"本檔合成（seed {SEED}）", "static", ["RND21"]),
]


def _filled(h, w, rgb):
    return golden._filled(h, w, rgb)


def make_frames():
    """所有合成影像，依固定順序產生（同一個 rng，順序固定）。回傳 [(名稱, 說明, 陣列)]。"""
    rng = np.random.default_rng(SEED)
    sa = rng.integers(0, 256, size=(21, 21, 3), dtype=np.uint8)
    sb = rng.integers(1, 256, size=(20, 17, 3), dtype=np.uint8)      # 1–255：左上角 3×3 的最小值不是 0
    sb[18, 15] = (1, 2, 3)                                            # 最後一個抽樣位置（降採樣 (6, 5)）
    sb[19, 15] = sb[18, 16] = (250, 250, 250)                         # 不是抽樣位置
    d13 = np.zeros((13, 13, 3), np.uint8)                             # ate_golden 案例 d
    for i in range(5):
        for j in range(5):
            high = (i <= 2 and j <= 2) or (i >= 2 and j >= 2)
            golden._put_down(d13, i, j, (200, 200, 200) if high else (10, 10, 10))
    golden._put_down(d13, 0, 0, (200, 210, 220))
    golden._put_down(d13, 4, 4, (220, 200, 210))
    e15 = _filled(15, 15, (50, 60, 70))
    golden._put_down(e15, 2, 2, (0, 90, 90))
    x15 = np.concatenate([_filled(6, 9, (60, 70, 80)), _filled(9, 9, (120, 125, 130))])
    rnd = np.concatenate([rng.integers(140, 256, size=(12, 21, 3), dtype=np.uint8),
                          rng.integers(0, 256, size=(9, 21, 3), dtype=np.uint8)])
    return [("SA21", "21×21 隨機 0–255", sa),
            ("SB2017", "20×17 隨機 1–255；(18,15) = (1,2,3)；(19,15)、(18,16) = 250", sb),
            ("C120", "9×9 單色 (120,130,140)", _filled(9, 9, (120, 130, 140))),
            ("D13", "13×13 案例 d（並列）", d13),
            ("E15", "15×15 單色 (50,60,70)，降採樣 (2,2) = (0,90,90)", e15),
            ("C123", "9×9 單色 (123,130,140)", _filled(9, 9, (123, 130, 140))),
            ("C127", "9×9 單色 (127,130,140)", _filled(9, 9, (127, 130, 140))),
            ("G130", "9×9 單色 (130,130,130)", _filled(9, 9, (130, 130, 130))),
            ("B100", "9×9 單色 (100,110,120)", _filled(9, 9, (100, 110, 120))),
            ("X15", "15×9 兩區：列 0–5 = Q (60,70,80)、列 6–14 = P (120,125,130)", x15),
            ("RND21", "21×21：列 0–11 隨機 140–255、列 12–20 隨機 0–255", rnd)]


# ---------------------------------------------------------------- 產生

PTE_KEYS = ("r_q", "r2_q", "k_q", "m_c", "m_q", "p", "u", "t_q")


class Model:
    def __init__(self):
        self.rows = {f: [] for f in FILES}
        self.frames = {}         # 名稱 → (id, 說明, 陣列)
        self.cases = {}          # id → (名稱, 說明, 來源)
        self.case_files = {}     # id → 出現的檔案
        self.n = {"vec": 0, "stats": 0, "upd": 0}

    def new(self, kind):
        v = self.n[kind]
        self.n[kind] += 1
        return v

    def add(self, f, case, row):
        if len(row) != len(SCHEMA[f]):
            raise AssertionError(f"{f}：欄數 {len(row)} ≠ {len(SCHEMA[f])}")
        self.rows[f].append(tuple(row))
        if case is not None:
            self.case_files.setdefault(case, set()).add(f)


def _i3(a):
    return tuple(int(v) for v in a)


def _emit_main(m, case, seq, frame_id, frame, col0=0, active=None, res=None, a_q=None, t_q=None):
    """一批 pixel 的主路徑。res（golden.apply_pass 的結果）給定時用它的 PALA、A_q 與 EX 的 PTE；
    否則 a_q 直接給定（t_q 也給定時只做 SD）。L8、L10 與兩種 SD 都用同一份 RGB 與 A_q 另外計算。"""
    if res is None and active is not None:
        res = golden.apply_pass(frame, active)
    if res is not None:
        a_q = res["a_q"]
    a_q = np.asarray(a_q, np.int64)
    if t_q is None:
        pte = {"EX": {k: res[k] for k in PTE_KEYS} if res is not None else golden.pte_fixed(frame, a_q, T_FRAC)}
        for s, b in SRC_B.items():
            pte[s] = lut.pte_lut(frame, a_q, b, T_FRAC)
        srcs = ("EX", "L8", "L10")
        tq = {s: pte[s]["t_q"] for s in srcs}
    else:
        srcs, tq = ("DIRECT",), {"DIRECT": np.asarray(t_q, np.int64)}
    sdr = {(s, md): golden.sd_fixed(frame, a_q, tq[s], T_FRAC, md) for s in srcs for md in MODE}
    h, w = frame.shape[:2]
    for r in range(h):
        for c in range(w):
            ids = (m.new("vec"), case, seq, frame_id, r, col0 + c)
            px, aq = _i3(frame[r, c]), _i3(a_q[r, c])
            if res is not None:
                s_sum, i_max = sum(px), active["i_max"]
                m.add("pala", case, ids + px + active["a_far"] + active["a_near"]
                      + (i_max, active["inv_i_max_q"], s_sum, i_max - s_sum if s_sum < i_max else 0,
                         int(res["product"][r, c]), int(res["eta_q"][r, c])) + aq)
            if t_q is None:
                for s in srcs:
                    d = pte[s]
                    nj = ((-1,) * 6 if s == "EX" else _i3(d["n"][r, c]) + _i3(d["j"][r, c]))
                    m.add("pte", case, ids + (SRC[s],) + px + aq + (int(d["r_q"][r, c]), int(d["r2_q"][r, c]), int(d["k_q"][r, c]))
                          + nj + _i3(d["m_c"][r, c]) + (int(d["m_q"][r, c]), int(d["p"][r, c]), int(d["u"][r, c]), int(d["t_q"][r, c])))
            for s in srcs:
                for md in MODE:
                    o = sdr[(s, md)]
                    ideal = md == "ideal"
                    m.add("sd", case, ids + (SRC[s], MODE[md]) + px + aq
                          + (int(tq[s][r, c]), None if ideal else int(o["sd_inv_q"][r, c]))
                          + _i3((o["sd_n"] if ideal else o["sd_p"])[r, c]) + (int(o["sd_d"][r, c]) if ideal else None,)
                          + _i3(o["sd_rounded"][r, c]) + tuple(int(bool(v)) for v in o["sd_tie"][r, c]) + _i3(o["out"][r, c]))


def _grp_row(g):
    return (1, g["count"], g["max_dark"], *g["pos"], *g["rgb"]) if g["valid"] else (0, 0) + (None,) * 6


def _emit_stats(m, case, seq, frame_id, frame, t_h, s=None):
    if s is None:
        s = golden.stats_pass(frame, t_h, "first")
    sid = m.new("stats")
    down, dark = s["frame_down"], s["dark"]
    hd, wd = down.shape[:2]
    for i in range(hd):
        for j in range(wd):
            px, dk = _i3(down[i, j]), int(dark[i, j])
            m.add("stats_pixels", None, (sid, frame_id, i, j, golden.STEP * i, golden.STEP * j) + px + (min(px), dk, int(dk >= t_h)))
    h, w = frame.shape[:2]
    m.add("stats", case, (sid, case, seq, frame_id, t_h, h, w, hd, wd, s["n"], s["gray_sum"], s["gray_mean"])
          + _grp_row(s["far"]) + _grp_row(s["near"]))
    return sid


def _act_code(raw, measured, triggered):
    if raw is None:
        return 2 if measured else 1
    if not measured:
        return 0
    return 4 if triggered else 3


def _emit_cpu(m, case, seq, sid, stats, prev, nxt, info):
    uid = m.new("upd")

    def raw(g):
        return (1, *_i3(stats[g]["rgb"])) if stats[g]["valid"] else (0, None, None, None)

    def upd(g):
        x = info[g]
        return (_act_code(x["raw"], prev["measured"], x["triggered"]), x["diff"], int(x["triggered"])) + _i3(x["used"])

    m.add("cpu", case, (uid, case, seq, sid, int(prev["measured"]), prev["t_h"], *prev["a_far"], *prev["a_near"],
                        stats["gray_sum"], stats["n"], *raw("far"), *raw("near"), *upd("far"), *upd("near"),
                        nxt["t_h"], *nxt["a_far"], *nxt["a_near"], nxt["i_max"], nxt["inv_i_max_q"]))
    return uid


def _act_row(a):
    return (int(a["measured"]), a["t_h"], *a["a_far"], *a["a_near"], a["i_max"], a["inv_i_max_q"])


def active0():
    return golden.make_active(**golden.ACTIVE0_DEFAULT, **FX)


def build():
    m = Model()
    for name, desc, arr in make_frames():
        fid = len(m.frames)
        m.frames[name] = (fid, desc, arr)
        h, w = arr.shape[:2]
        for r in range(h):
            for c in range(w):
                m.add("frames", None, (fid, r, c) + _i3(arr[r, c]))
    # A. 主路徑單元案例
    for case, name, desc, src, groups in CHAIN:
        m.cases[case] = (name, desc, src)
        col = 0
        for pixels, af, an in groups:
            frame = np.array([pixels], np.uint8)
            act = golden.make_active(0, af, an, measured=True, **FX)
            _emit_main(m, case, None, None, frame, col0=col, active=act)
            col += len(pixels)
    for case, name, desc, src, pixels, aqs in PTE_DIRECT:
        m.cases[case] = (name, desc, src + "（A_q 直接給定）")
        _emit_main(m, case, None, None, np.array([pixels], np.uint8), a_q=np.array([aqs], np.int64))
    for case, name, desc, i, aq, tq in SD_DIRECT:
        m.cases[case] = (name, desc, f"ate_golden selfcheck SD {name}（A_q = {aq}、t_q = {tq} 直接給定）")
        _emit_main(m, case, None, None, np.array([[(i,) * 3]], np.uint8), a_q=np.array([[(aq,) * 3]], np.int64),
                   t_q=np.array([[tq]], np.int64))
    # B. 統計路徑單元案例
    for case, name, desc, src, fname, t_h in STATS_UNIT:
        m.cases[case] = (name, desc, src)
        fid, _, arr = m.frames[fname]
        if t_h is None:
            dk = sorted(v for row in golden._ref_stats(arr, 0)["dark"] for v in row)
            t_h = dk[len(dk) // 2]
        _emit_stats(m, case, None, fid, arr, t_h)
    # C. CPU 更新單元案例
    prevs = {"prev": golden.make_active(120, (200,) * 3, (100,) * 3, measured=True, **FX),
             "prev_hi": golden.make_active(120, (250,) * 3, (100,) * 3, measured=True, **FX), "a0": active0()}
    for case, name, desc, pk, far, near, gs, n, gm in CPU_UNIT:
        m.cases[case] = (name, desc, f"ate_golden selfcheck cpu_update {name.lower()}（統計值直接給定；gray_sum、N 依 gray_mean 手算選定）")
        stats = {"far": {"valid": far is not None, "rgb": far}, "near": {"valid": near is not None, "rgb": near},
                 "gray_sum": gs, "n": n, "gray_mean": gm}
        nxt, info = golden.cpu_update(stats, prevs[pk])
        _emit_cpu(m, case, None, None, stats, prevs[pk], nxt, info)
    # D. 排程：用既有 golden 排程（FX 設定）取得 active 與 A_q
    for case, name, desc, src, mode, fnames in SCHEDULES:
        m.cases[case] = (name, desc, src)
        arrs = [m.frames[f][2] for f in fnames]
        recs = [golden.run_static(arrs[0], active0())] if mode == "static" else golden.run_video(arrs, active0())
        for seq, (rec, fname) in enumerate(zip(recs, fnames)):
            fid, _, arr = m.frames[fname]
            sid = _emit_stats(m, case, seq, fid, arr, rec["stats_active"]["t_h"], rec["stats"])
            uid = _emit_cpu(m, case, seq, sid, rec["stats"], rec["stats_active"], rec["next"], rec["update"])
            vec0 = m.n["vec"]
            _emit_main(m, case, seq, fid, arr, active=rec["used"], res=rec["result"])
            m.add("schedule", case, (case, seq, int(mode == "video"), fid, sid, uid, vec0, m.n["vec"] - vec0)
                  + _act_row(rec["stats_active"]) + _act_row(rec["used"]) + _act_row(rec["next"]))
    # E. LUT 內容（既有函式產生）
    for s, v in enumerate(lut.ratio_table()):
        m.add("lut", None, (0, s, int(v)))
    for t, b in ((1, 8), (2, 10)):
        for j, v in enumerate(lut.inv_a_table(b)):
            m.add("lut", None, (t, j, int(v)))
    tq = np.arange(1, 1025, dtype=np.int64)[None, :]
    inv = golden.sd_fixed(np.zeros((1, 1024, 3), np.uint8), np.zeros((1, 1024, 3), np.int64), tq, T_FRAC, "lut")["sd_inv_q"][0]
    for t, v in zip(range(1, 1025), inv):
        m.add("lut", None, (3, t, int(v)))
    return m


def _sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def render(m):
    gen, gol, lu = _sha256_file(os.path.abspath(__file__)), _sha256_file(golden.__file__), _sha256_file(lut.__file__)
    out = {}
    for f in FILES:
        lines = [f"# ate_vectors v0｜{f}.hex｜{TITLES[f]}", f"# schema {f}.v0", STATUS,
                 f"# generator ate_vectors.py sha256={gen}", f"# source ate_golden.py sha256={gol}",
                 f"# source ate_pte_lut.py sha256={lu}", CONFIG, ENCODING]
        lines += CODES.get(f, [])
        if any(fl.name == "frame_id" for fl in SCHEMA[f]):
            lines += [f"# frame {fid:02X} {name} {arr.shape[0]}x{arr.shape[1]}：{desc}" for name, (fid, desc, arr) in m.frames.items()]
        lines += [f"# case {cid:02X} {n}：{d}（來源：{s}）" for cid, (n, d, s) in sorted(m.cases.items())
                  if f in m.case_files.get(cid, ())]
        lines += [fl.header() for fl in SCHEMA[f]]
        lines.append("# columns " + " ".join(fl.name for fl in SCHEMA[f]))
        lines.append(f"# rows {len(m.rows[f])}")
        lines += [" ".join(fl.enc(v) for fl, v in zip(SCHEMA[f], row)) for row in m.rows[f]]
        out[f] = ("\n".join(lines) + "\n").encode("utf-8")
    return out


# ---------------------------------------------------------------- 讀回

def parse(data):
    """只依檔頭宣告的欄位解析。回傳 (欄位, rows, 檔頭行)。格式不符時丟 ValueError。"""
    text = data.decode("utf-8")
    if "\r" in text or not text.endswith("\n"):
        raise ValueError("必須是 LF、以換行結尾")
    header, body = [], []
    for ln in text[:-1].split("\n"):
        if ln.startswith("#"):
            if body:
                raise ValueError("資料列之後又出現檔頭")
            header.append(ln)
        else:
            body.append(ln)
    fields = [_parse_field(ln) for ln in header if ln.startswith("# field ")]
    cols = next(ln for ln in header if ln.startswith("# columns ")).split(" ")[2:]
    n_rows = int(next(ln for ln in header if ln.startswith("# rows ")).split(" ")[2])
    if cols != [f.name for f in fields] or n_rows != len(body):
        raise ValueError(f"columns／rows 不符：{len(cols)} 欄、宣告 {n_rows} 列、實際 {len(body)} 列")
    rows = []
    for ln in body:
        tok = ln.split(" ")
        if len(tok) != len(fields):
            raise ValueError(f"欄數 {len(tok)} ≠ {len(fields)}：{ln[:60]}")
        rows.append(tuple(f.dec(t) for f, t in zip(fields, tok)))
    return fields, rows, header


def readback(ck, texts, m, where):
    """逐檔讀回：檔頭 schema 和程式相同、列數與順序、每欄解碼值和記憶體逐列相同。回傳 {檔名: [dict]}。"""
    tables = {}
    for f in FILES:
        try:
            fields, rows, header = parse(texts[f])
            same_schema = [x.key() for x in fields] == [x.key() for x in SCHEMA[f]]
            src_ok = (f"# source ate_golden.py sha256={_sha256_file(golden.__file__)}" in header
                      and f"# source ate_pte_lut.py sha256={_sha256_file(lut.__file__)}" in header)
            ok = same_schema and src_ok and rows == m.rows[f]
            ck.check(f"R {where} {f}.hex：檔頭 schema、source SHA-256 和程式相同；{len(rows)} 列每欄位數、大寫 hex、範圍、"
                     f"signed／invalid 解碼後和記憶體逐列相同（順序相同）", ok,
                     f"schema {same_schema}、source {src_ok}、列 {len(rows)}／{len(m.rows[f])}")
            tables[f] = [dict(zip([x.name for x in fields], r)) for r in rows]
        except Exception as e:
            ck.check(f"R {where} {f}.hex：讀回", False, f"{type(e).__name__}：{e}")
            tables[f] = []
    return tables


# ---------------------------------------------------------------- 檢查（只用檔案內容、手算常數、既有獨立 oracle、重新呼叫 golden 排程）

def _try(ck, label, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ck.check(label, False, f"{type(e).__name__}：{e}")
        return
    ck.check(label, bool(ok), detail)


def _v3(r, p):
    return tuple(r[f"{p}{c}"] for c in C3)


def _frames_of(T):
    by = {}
    for r in T["frames"]:
        by.setdefault(r["frame_id"], []).append(r)
    out, ok = {}, True
    for fid, rows in by.items():
        h, w = max(r["row"] for r in rows) + 1, max(r["col"] for r in rows) + 1
        ok &= [(r["row"], r["col"]) for r in rows] == [(i, j) for i in range(h) for j in range(w)]
        out[fid] = np.array([[r["R"], r["G"], r["B"]] for r in rows], np.uint8).reshape(h, w, 3)
    return out, ok


# 手算常數（出處見各 case 行；沒有寫的欄位由 oracle 核對）
H_PALA = {   # (case, col): (I_max, inv_q, product, eta_q, A_q)；None = 不指定
    (1, 0): (600, 1747, 576510, 140, (39520, 40000, 40480)),
    (2, 0): (600, 1747, 1048200, 255, (25700, 12950, 200)), (2, 1): (600, 1747, 1747, 0, (51200,) * 3),
    (2, 2): (600, 1747, 0, 0, (51200,) * 3), (2, 3): (600, 1747, 0, 0, (51200,) * 3),
    (3, 0): (0, 0, 0, 0, (0, 0, 0)), (3, 1): (0, 0, 0, 0, (0, 0, 0)),
    (4, 0): (1, 1 << 20, 1 << 20, 256, (2560, 5120, 7680)), (4, 1): (1, 1 << 20, 0, 0, (256, 0, 0)),
    (5, 0): (765, 1370, 1048050, 255, (255, 255, 255)),
    (6, 0): (765, 1370, 1048050, 255, (65280,) * 3), (6, 1): (765, 1370, None, None, (65280,) * 3),
    (7, 0): (765, 1370, 798710, 194, (15810,) * 3), (7, 1): (765, 1370, 800080, 195, (15555,) * 3),
    (8, 0): (256, 4096, 1 << 20, 256, (0, 0, 0)), (8, 1): (256, 4096, 255 * 4096, 255, (86, 85, 85)),
    (8, 2): (256, 4096, 156 * 4096, 156, (8600, 8500, 8500)),
    (9, 0): (2, 524288, 1048576, 256, (0, 0, 0)), (9, 1): (3, 349525, 1048575, 255, (3, 0, 0)),
    (9, 2): (7, 149796, 1048572, 255, (7, 0, 0)), (9, 3): (375, 2796, 1048500, 255, (255, 120, 0)),
    (9, 4): (390, 2688, 1048320, 255, (255, 135, 0)), (9, 5): (600, 1747, 1048200, 255, (255, 255, 90)),
}

_N = -1
H_PTE = {}   # (case, col, src): {欄位: 值}；m_c、n、j 是三色 tuple


def _hp(case, col, srcs, **kv):
    for s in srcs:
        H_PTE.setdefault((case, col, s), {}).update(kv)


EX, L8, L10, LL = (0,), (1,), (2,), (1, 2)
_hp(1, 0, EX, r_q=21845, r2_q=7281, k_q=58255, m_c=(25471, 37748, 49734), m_q=25471, p=22257196575, u=331, t_q=693)
_hp(0x10, 0, EX, r_q=32768, k_q=49152, m_q=32768, t_q=664)
_hp(0x10, 0, LL, r_q=32767, r2_q=16383, k_q=49153, n=(15,) * 3, m_c=(32767,) * 3, m_q=32767, p=24158945265, u=359, t_q=665)
_hp(0x10, 0, L8, j=(144,) * 3)
_hp(0x10, 0, L10, j=(576,) * 3)
for _c in (0, 1):
    _hp(0x11, _c, (0, 1, 2), r_q=0, k_q=65536, m_q=0, t_q=1024)
_hp(0x12, 0, EX, m_c=(16711680,) * 3, m_q=131071, u=1439, t_q=1)
_hp(0x12, 0, LL, n=(8,) * 3, j=(0,) * 3, m_c=(131071,) * 3, m_q=131071, u=1440, t_q=1)
_hp(0x13, 0, EX, m_c=(_N, 6553, 9830), k_q=61440, m_q=6553, t_q=935)
_hp(0x13, 0, LL, m_c=(_N, 6553, 9830), k_q=61441, m_q=6553, t_q=935)
_hp(0x14, 0, (0, 1, 2), m_c=(_N,) * 3, m_q=0, t_q=1024)
for _c, (_u, _t) in enumerate(((1022, 2), (1023, 1), (1024, 1))):
    _hp(0x15, _c, EX, u=_u, t_q=_t)
_hp(0x16, 0, EX, m_c=(4278190080,) * 3, m_q=131071)
_hp(0x16, 0, LL, n=(0,) * 3, j=(_N,) * 3, m_c=(131071,) * 3, m_q=131071)
_hp(0x16, 1, EX, m_c=(65536,) * 3)
_hp(0x17, 0, LL, n=(6, 7, _N), j=(_N, 0, _N), m_c=(131071, 131071, _N), t_q=1)
_hp(0x17, 0, EX, m_c=(132104, 131072, _N))
_hp(0x17, 1, (0, 1, 2), m_c=(0, _N, _N), m_q=0, t_q=1024)
_hp(0x18, 0, LL, n=(7, _N, _N), m_c=(83886, _N, _N), t_q=103)
_hp(0x18, 0, L8, j=(144, _N, _N))
_hp(0x18, 0, L10, j=(576, _N, _N))
_hp(0x19, 0, LL, m_c=(32767, 32768, _N), m_q=32767, t_q=598)
_hp(0x19, 0, L8, j=(144, 148, _N))
_hp(0x19, 0, L10, j=(576, 592, _N))
_hp(0x19, 0, EX, m_c=(32768, 32767, _N))
_hp(0x1A, 0, LL, r_q=0, k_q=65536, m_q=32767, t_q=545)
_hp(0x1A, 0, EX, m_q=32768, t_q=544)
_hp(0x1B, 0, LL, m_c=(32686, _N, _N))
_hp(0x1B, 0, L8, j=(145, _N, _N))
_hp(0x1B, 0, L10, j=(580, _N, _N))

H_SD = {   # (case, col, src, mode): {欄位: 值}
    (1, 0, 0, 0): {"num": (2647520, 10352960, 18058400), "den": 177408, "out": (15, 58, 102)},
    (1, 0, 0, 1): {"inv_t_q": 1513, "num": (3914400, 15299520, 26684640), "out": (15, 58, 102)},
}
for _case, (_ri, _oi, _ol) in zip((0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26),
                                  ((100, 100, 100), (-100, 0, 0), (510, 255, 255), (37, 37, 37), (96, 96, 96), (165, 165, 166),
                                   (101, 101, 101))):
    H_SD[(_case, 0, 3, 0)] = {"rnd": (_ri,) * 3, "out": (_oi,) * 3}
    H_SD[(_case, 0, 3, 1)] = {"out": (_ol,) * 3}
H_SD[(0x20, 0, 3, 0)]["tie"] = (1,) * 3
H_SD[(0x25, 0, 3, 0)]["tie"] = (0,) * 3
H_SD[(0x25, 0, 3, 1)].update(inv_t_q=10592, tie=(1,) * 3)
H_SD[(0x26, 0, 3, 1)].update(rnd=(101,) * 3, tie=(1,) * 3)
H_SD[(0x26, 0, 3, 0)]["tie"] = (1,) * 3

H_STATS = {   # case: (t_h, H, W, Hd, Wd, N, gray_sum, gray_mean, far, near)；group = (valid, count, max, i, j, rgb)，None = 不指定
    0x30: (None, 21, 21, 7, 7, 49, None, None, None, None),
    0x31: (128, 20, 17, 7, 6, 42, None, None, None, None),
    0x32: (120, 9, 9, 3, 3, 9, 3510, 130, (1, 9, 120, 0, 0, (120, 130, 140)), (0, 0, None, None, None, (None,) * 3)),
    0x33: (150, 13, 13, 5, 5, 25, 10500, 140, (1, 2, 200, 0, 0, (200, 210, 220)), (1, 23, 10, 0, 1, (200, 200, 200))),
    0x34: (1, 15, 15, 5, 5, 25, 4500, 60, (0, 0, None, None, None, (None,) * 3), (1, 25, 0, 0, 0, (50, 60, 70))),
    0x35: (121, 9, 9, 3, 3, 9, 3510, 130, (0, 0, None, None, None, (None,) * 3), (1, 9, 120, 0, 0, (120, 130, 140))),
}

H_CPU = {   # case: (next_t_h, next A_F, next A_N, I_max, inv_q, (far act, diff, trig), (near act, diff, trig))
    0x40: (117, (203, 200, 200), (102, 100, 100), 603, 1738, (3, 3, 0), (4, 4, 1)),
    0x41: (117, (198, 200, 200), (97, 100, 100), 598, 1753, (4, 4, 1), (3, 3, 0)),
    0x42: (117, (202, 199, 200), (100, 100, 100), 601, 1744, (4, 5, 1), (3, 0, 0)),
    0x43: (117, (252, 252, 252), (100, 100, 100), 756, 1387, (4, 5, 1), (3, 0, 0)),
    0x44: (90, (200, 200, 200), (101, 100, 100), 600, 1747, (2, None, 0), (3, 1, 0)),
    0x45: (130, (240, 240, 240), (240, 240, 240), 720, 1456, (0, None, 0), (1, None, 0)),
    0x46: (0, (0, 0, 0), (0, 0, 0), 0, 0, (0, None, 0), (0, None, 0)),
}

_c, _c3, _c5, _c7 = (120, 130, 140), (123, 130, 140), (125, 130, 140), (127, 130, 140)
_g, _b = (130, 130, 130), (100, 110, 120)
_P, _Q = (120, 125, 130), (60, 70, 80)
F1, F2 = (1, None, 0), (2, None, 0)
H_SCHED = {   # case: [(stat T_h, use A_F, use A_N, use I_max, use inv_q, far(act,diff,trig), near(...), next T_h, next A_F, next A_N,
              #         next I_max, next inv_q, gray_mean)]
    0x50: [(128, W3, W3, 765, 1370, F1, (0, None, 0), 130, _c, _c, 390, 2688, 130),
           (130, _c, _c, 390, 2688, F2, (3, 3, 0), 131, _c, _c3, 390, 2688, 131),
           (131, _c, _c3, 390, 2688, F2, (4, 4, 1), 132, _c, _c5, 390, 2688, 132),
           (132, _c, _c5, 390, 2688, F2, (3, 2, 0), 132, _c, _c7, 390, 2688, 132)],
    0x51: [(128, W3, W3, 765, 1370, (0, None, 0), F1, 130, _g, _g, 390, 2688, 130),
           (130, _g, _g, 390, 2688, (3, 0, 0), F2, 130, _g, _g, 390, 2688, 130),
           (130, _g, _g, 390, 2688, F2, (4, 30, 1), 110, _g, (115, 120, 125), 390, 2688, 110),
           (110, _g, (115, 120, 125), 390, 2688, F2, (4, 15, 1), 110, _g, (107, 115, 122), 390, 2688, 110),
           (110, _g, (107, 115, 122), 390, 2688, F2, (4, 7, 1), 110, _g, (103, 112, 121), 390, 2688, 110),
           (110, _g, (103, 112, 121), 390, 2688, F2, (3, 3, 0), 110, _g, _b, 390, 2688, 110)],
    0x52: [(128, W3, W3, 765, 1370, F1, (0, None, 0), 103, _P, _P, 375, 2796, 103),
           (103, _P, _P, 375, 2796, (3, 0, 0), (4, 60, 1), 103, _P, (90, 97, 105), 375, 2796, 103),
           (103, _P, (90, 97, 105), 375, 2796, (3, 0, 0), (4, 30, 1), 103, _P, (75, 83, 92), 375, 2796, 103),
           (103, _P, (75, 83, 92), 375, 2796, (3, 0, 0), (4, 15, 1), 103, _P, (67, 76, 86), 375, 2796, 103),
           (103, _P, (67, 76, 86), 375, 2796, (3, 0, 0), (4, 7, 1), 103, _P, (63, 73, 83), 375, 2796, 103),
           (103, _P, (63, 73, 83), 375, 2796, (3, 0, 0), (3, 3, 0), 103, _P, _Q, 375, 2796, 103),
           (103, _P, _Q, 375, 2796, (3, 0, 0), (3, 0, 0), 103, _P, _Q, 375, 2796, 103)],
    0x53: [(128, _b, _b, 330, 3177, F1, (0, None, 0), 110, _b, _b, 330, 3177, 110)],
    0x54: [(128, _P, _P, 375, 2796, F1, (0, None, 0), 103, _P, _P, 375, 2796, 103)],
}


def _cmp_fields(row, want, fields3=()):
    bad = []
    for k, v in want.items():
        got = _v3(row, f"{k}_") if k in fields3 else row[k]
        if v is not None and got != v:
            bad.append(f"{k} {got} ≠ {v}")
    return bad


def check_tables(ck, T):
    frames, raster_ok = _frames_of(T)
    ck.check(f"W frames.hex：{len(frames)} 張影像，每張的 row、col 是完整的 raster 順序，最大 21×21",
             raster_ok and all(a.shape[0] <= 21 and a.shape[1] <= 21 for a in frames.values()))
    pala = {r["vec_id"]: r for r in T["pala"]}
    pte = {(r["vec_id"], r["src"]): r for r in T["pte"]}
    sd = {(r["vec_id"], r["src"], r["mode"]): r for r in T["sd"]}
    stats = {r["stats_id"]: r for r in T["stats"]}
    cpu = {r["upd_id"]: r for r in T["cpu"]}
    spx = {}
    for r in T["stats_pixels"]:
        spx.setdefault(r["stats_id"], []).append(r)
    unit = {}
    for f, key in (("pala", lambda r: (r["case_id"], r["col"])), ("pte", lambda r: (r["case_id"], r["col"], r["src"])),
                   ("sd", lambda r: (r["case_id"], r["col"], r["src"], r["mode"]))):
        unit[f] = {key(r): r for r in T[f] if r["seq"] is None}

    # ---- A. 主路徑：手算
    print("A 主路徑（PALA／PTE／SD）")
    bad = []
    for (case, col), (im, q, prod, eta, aq) in H_PALA.items():
        r = unit["pala"].get((case, col))
        bad += [f"{case:02X}/{col} 缺"] if r is None else [f"{case:02X}/{col} {b}" for b in _cmp_fields(
            r, {"I_max": im, "inv_q": q, "product": prod, "eta_q": eta, "A_q": aq}, ("A_q",))]
    ck.check(f"H PALA 手算 {len(H_PALA)} 個 pixel（P1、P2、P3、R1 的 F = 20 數字：inv_q、product、eta_q 截位、A_q；含 I_max 0、1、2、3、7、256、765）",
             not bad, "；".join(bad[:5]))
    bad = []
    for (case, col, src), want in H_PTE.items():
        r = unit["pte"].get((case, col, src))
        bad += [f"{case:02X}/{col}/{src} 缺"] if r is None else [f"{case:02X}/{col}/{src} {b}" for b in _cmp_fields(
            r, want, ("m_c", "n", "j"))]
    ck.check(f"H PTE 手算 {len(H_PTE)} 組（ate_golden X1–X6、T2；ate_pte_lut H1–H9、R1：EX／L8／L10 分開）", not bad, "；".join(bad[:5]))
    bad = []
    for (case, col, src, mode), want in H_SD.items():
        r = unit["sd"].get((case, col, src, mode))
        bad += [f"{case:02X}/{src}/{mode} 缺"] if r is None else [f"{case:02X}/{src}/{mode} {b}" for b in _cmp_fields(
            r, want, ("num", "rnd", "tie", "out"))]
    ck.check(f"H SD 手算 {len(H_SD)} 組（ate_golden SD D1–D8：IDEAL／LUT 分開；.5 進位、負 J、clip、t = 1、t = 1/1024、tie 向上）",
             not bad, "；".join(bad[:5]))
    ck.check("H 欄位表示：k_q = 65536、eta_q = 256 都出現且讀回正確；EX 的 m_c 有未飽和值（> 131071），L8／L10 的 m_c 都 ≤ 131071",
             any(r["k_q"] == 65536 for r in T["pte"]) and any(r["eta_q"] == 256 for r in T["pala"])
             and any(r["src"] == 0 and max(_v3(r, "m_c_")) > 131071 for r in T["pte"])
             and all(max(_v3(r, "m_c_")) <= 131071 for r in T["pte"] if r["src"] in (1, 2)))

    # ---- A. 主路徑：oracle（每一列）
    def pala_oracle():
        bad = 0
        for r in T["pala"]:
            px, af, an, aq = _v3(r, ""), _v3(r, "AF_"), _v3(r, "AN_"), _v3(r, "A_q_")
            im, q, s = r["I_max"], r["inv_q"], sum(px)
            q_ok = q == 0 if im == 0 else (im * q <= 1 << 20 < im * (q + 1) and q == math.floor(Fraction(1 << 20, im)))
            d = im - s if s < im else 0
            eta = r["eta_q"]
            bad += not (q_ok and im == sum(af) and r["S"] == s and r["delta"] == d and r["product"] == d * q
                        and eta * 4096 <= r["product"] < (eta + 1) * 4096 and 0 <= eta <= 256
                        and all(a == 256 * f - eta * (f - n) for a, f, n in zip(aq, af, an)))
        return bad == 0, f"不符 {bad} 列"
    _try(ck, f"O PALA 全部 {len(T['pala'])} 列：inv_q = floor(2^20/I_max)（Fraction 與 I_max·q ≤ 2^20 < I_max·(q+1)）、"
             f"product = delta·q、eta_q·2^12 ≤ product < (eta_q+1)·2^12、A_q = 256·A_F − eta_q·(A_F − A_N)（另一種寫法）", pala_oracle)

    def pte_ex_oracle():
        bad = 0
        for r in T["pte"]:
            if r["src"] != 0:
                continue
            px, aq = _v3(r, ""), _v3(r, "A_q_")
            rq, kq, mq, tq = golden._pte_oracle(px, aq, T_FRAC)
            mc = tuple(math.floor(Fraction(256 * i, a) * 2 ** 16) if a > 0 else -1 for i, a in zip(px, aq))
            p = r["p"]
            bad += not ((r["r_q"], r["k_q"], r["m_q"], r["t_q"]) == (rq, kq, mq, tq) and _v3(r, "m_c_") == mc
                        and r["r2_q"] << 16 <= r["r_q"] ** 2 < (r["r2_q"] + 1) << 16 and p == 15 * r["m_q"] * r["k_q"]
                        and r["u"] << 26 <= p < (r["u"] + 1) << 26 and _v3(r, "n_") == (-1,) * 3 and _v3(r, "j_") == (-1,) * 3)
        return bad == 0, f"不符 {bad} 列"
    _try(ck, f"O PTE EX 全部 {sum(r['src'] == 0 for r in T['pte'])} 列：r_q、k_q、m_q、t_q 和 ate_golden._pte_oracle（Fraction）相同；"
             f"m_c = floor(256·I/A_q·2^16)（未飽和）；r2_q、p、u 的整數關係；n、j 為 −1", pte_ex_oracle)

    def pte_lut_oracle():
        bad = 0
        keys = ("r_q", "r2_q", "k_q", "m_q", "p", "u", "t_q")
        for r in T["pte"]:
            if r["src"] == 0:
                continue
            o = lut._oracle_pixel(_v3(r, ""), _v3(r, "A_q_"), 8 if r["src"] == 1 else 10)
            bad += not (all(r[k] == o[k] for k in keys) and list(_v3(r, "m_c_")) == o["m_c"]
                        and list(_v3(r, "n_")) == o["n"] and list(_v3(r, "j_")) == o["j"])
        return bad == 0, f"不符 {bad} 列"
    _try(ck, f"O PTE L8／L10 全部 {sum(r['src'] != 0 for r in T['pte'])} 列：10 個欄位和 ate_pte_lut._oracle_pixel（Fraction、逐步 Python int）相同",
         pte_lut_oracle)

    def sd_oracle():
        bad = 0
        for r in T["sd"]:
            px, aq, tq = _v3(r, ""), _v3(r, "A_q_"), r["t_q"]
            md = "ideal" if r["mode"] == 0 else "lut"
            js, outs = golden._sd_oracle(px, aq, tq, T_FRAC, md)
            if md == "ideal":
                got = [Fraction(n, r["den"]) for n in _v3(r, "num_")]
                extra = r["den"] == 256 * tq and r["inv_t_q"] is None
            else:
                got = [Fraction(n, 1 << 18) for n in _v3(r, "num_")]
                extra = r["den"] is None and r["inv_t_q"] == math.floor(Fraction(1 << 20, tq) + Fraction(1, 2))
            bad += not (extra and got == js and list(_v3(r, "out_")) == outs
                        and list(_v3(r, "rnd_")) == [math.floor(j + Fraction(1, 2)) for j in js]
                        and list(_v3(r, "tie_")) == [int(j.denominator == 2) for j in js])
        return bad == 0, f"不符 {bad} 列"
    _try(ck, f"O SD 全部 {len(T['sd'])} 列：J（N/D 或 P/2^18 的精確分數）、out 和 ate_golden._sd_oracle（Fraction）相同；"
             f"rnd = floor(J + 1/2)、tie、D = 256·t_q、inv_t_q = RN(2^20/t_q)", sd_oracle)

    # ---- A. 接線與 guard
    def links_main():
        bad = []
        for (vec, src), r in pte.items():
            base = pte.get((vec, 0))
            if base is None or any(r[k] != base[k] for k in ("case_id", "seq", "frame_id", "row", "col", "R", "G", "B",
                                                              "A_q_R", "A_q_G", "A_q_B")):
                bad.append(f"pte {vec}/{src} 和 EX 列的 RGB、A_q 或識別不同")
            p = pala.get(vec)
            if p is not None and any(r[k] != p[k] for k in ("case_id", "seq", "frame_id", "row", "col", "R", "G", "B",
                                                             "A_q_R", "A_q_G", "A_q_B")):
                bad.append(f"pte {vec} 和 pala 不同")
        cnt = {}
        for v, s in pte:
            cnt[v] = cnt.get(v, set()) | {s}
        bad += [f"pte {v} 版本 {sorted(ss)}" for v, ss in cnt.items() if ss != {0, 1, 2}]
        bad += [f"pala {v} 沒有 pte" for v in pala if v not in cnt]
        for (vec, src, mode), r in sd.items():
            if src == 3:
                if vec in cnt or vec in pala:
                    bad.append(f"sd {vec} DIRECT 卻有 pte／pala 列")
                continue
            p = pte.get((vec, src))
            if p is None or r["t_q"] != p["t_q"] or any(r[k] != p[k] for k in ("case_id", "seq", "frame_id", "row", "col", "R", "G",
                                                                                 "B", "A_q_R", "A_q_G", "A_q_B")):
                bad.append(f"sd {vec}/{src}/{mode} 和 pte 不同")
        sdk = {}
        for v, s, md in sd:
            sdk.setdefault(v, set()).add((s, md))
        bad += [f"sd {v} 組合 {sorted(k)}" for v, k in sdk.items()
                if k != ({(s, md) for s in (0, 1, 2) for md in (0, 1)} if v in cnt else {(3, 0), (3, 1)})]
        bad += [f"pte {v} 沒有 sd" for v in cnt if v not in sdk]
        return not bad, "；".join(bad[:5])
    _try(ck, "W 主路徑接線：同一 vec_id 的 EX、L8、L10 共用同一份 RGB 與 A_q（和 pala 相同）；每個 src 都有 IDEAL、LUT 兩列，"
             "t_q 等於該 src 的 pte 列；DIRECT 只出現在 sd", links_main)

    def lut_used():
        tb = {}
        for r in T["lut"]:
            tb.setdefault(r["table"], {})[r["index"]] = r["value"]
        bad = 0
        for r in T["pte"]:
            if r["src"] == 0:
                continue
            px, aq = _v3(r, ""), _v3(r, "A_q_")
            lo, hi = min(px), max(px)
            bad += r["r_q"] != (lo * tb[0][lo + hi]) >> 7
            for i, a, n, j, mc in zip(px, aq, _v3(r, "n_"), _v3(r, "j_"), _v3(r, "m_c_")):
                if j >= 0:
                    bad += mc != min((i * tb[r["src"]][j]) >> (n - 7), 131071)
        for r in T["sd"]:
            if r["mode"] == 1:
                bad += r["inv_t_q"] != tb[3][r["t_q"]]
        return bad == 0, f"不符 {bad} 處"
    _try(ck, "W LUT 使用：L8／L10 列的 r_q = (lo·E[lo+hi]) >> 7、查表 channel 的 m_c = min((I·R_B[j]) >> (n − 7), 131071)；"
             "SD LUT 列的 inv_t_q = 表 3[t_q]（都用 lut.hex 的值）", lut_used)

    def guards():
        bad, dist = 0, {1: {}, 2: {}}
        for (vec, src), r in pte.items():
            if src in (1, 2):
                d = r["t_q"] - pte[(vec, 0)]["t_q"]
                dist[src][d] = dist[src].get(d, 0) + 1
                bad += abs(d) > GUARD_T[src]
        sd_bad, sd_n = 0, 0
        for (vec, src, mode), r in sd.items():
            if mode == 0:
                o = sd[(vec, src, 1)]
                diff = [abs(a - b) for a, b in zip(_v3(r, "out_"), _v3(o, "out_"))]
                sd_bad += max(diff) > 1
                sd_n += sum(x > 0 for x in diff)
        print(f"  紀錄：Δt_q（L8 − EX）分布 {dict(sorted(dist[1].items()))}；（L10 − EX）{dict(sorted(dist[2].items()))}；"
              f"SD LUT 與 IDEAL 不同的 channel 值 {sd_n} 個")
        return bad == 0 and sd_bad == 0 and any(dist[1].get(d) for d in dist[1] if d) and sd_n > 0, \
            f"t 超過 guard {bad}、SD 差 > 1 {sd_bad}"
    _try(ck, "G guard：每個 vec 的 |t_q(L8) − t_q(EX)| ≤ 3、|t_q(L10) − t_q(EX)| ≤ 1；同一 t_q 來源下 |out(LUT) − out(IDEAL)| ≤ 1"
             "（另確認各版本確實有不同的值，不是同一份答案）", guards)

    # ---- B. 統計路徑
    print("B 統計路徑")
    bad = []
    for case, want in H_STATS.items():
        r = next((x for x in T["stats"] if x["case_id"] == case), None)
        if r is None:
            bad.append(f"{case:02X} 缺")
            continue
        names = ("t_h_prev", "H", "W", "Hd", "Wd", "N", "gray_sum", "gray_mean")
        bad += [f"{case:02X} {b}" for b in _cmp_fields(r, dict(zip(names, want[:8])))]
        for g, gw in (("far", want[8]), ("near", want[9])):
            if gw is not None:
                got = (r[f"{g}_valid"], r[f"{g}_count"], r[f"{g}_max"], r[f"{g}_i"], r[f"{g}_j"], _v3(r, f"{g}_"))
                if got != gw:
                    bad.append(f"{case:02X} {g} {got} ≠ {gw}")
    ck.check("H 統計手算（ate_golden 案例 c、d、e、f：dark = T_h 歸遠群、並列取第一個、空群組、近群最大 dark 0；a、b 的尺寸與 N）",
             not bad, "；".join(bad[:5]))

    def stats_b_hand():
        r = next(x for x in T["stats"] if x["case_id"] == 0x31)
        px = {(p["i"], p["j"]): p for p in spx[r["stats_id"]]}
        down = np.array([[_v3(px[(i, j)], "") for j in range(6)] for i in range(7)])
        tl, br = int(down[:3, :3].min()), int(down[4:7, 3:6].min())
        a = next(x for x in T["stats"] if x["case_id"] == 0x30)
        dk = sorted(p["dark"] for p in spx[a["stats_id"]])
        return (_v3(px[(6, 5)], "") == (1, 2, 3) and px[(0, 0)]["dark"] == tl and px[(6, 5)]["dark"] == br and br <= 1 and tl > 0
                and a["t_h_prev"] == dk[len(dk) // 2] and a["far_valid"] == a["near_valid"] == 1), f"tl {tl}、br {br}"
    _try(ck, "H 統計 ST-B 尾端：原圖 (18,15) 落在降採樣 (6,5)；左上、右下角 dark = 影像內 3×3 的最小值（外圍補 255 才成立），右下 ≤ 1、"
             "左上 > 0；ST-A 的 T_h = dark 中位數、兩群都非空", stats_b_hand)

    def stats_oracle():
        bad = 0
        for sid, r in stats.items():
            arr = frames[r["frame_id"]]
            ref = golden._ref_stats(arr, r["t_h_prev"])
            pts = spx.get(sid, [])
            exp_pts = [(i, j) for i in range(len(ref["down"])) for j in range(len(ref["down"][0]))]
            ok = [(p["i"], p["j"]) for p in pts] == exp_pts and len(pts) == r["N"]
            for p in pts:
                rgb = _v3(p, "")
                ok &= (p["frame_id"] == r["frame_id"] and (p["row"], p["col"]) == (3 * p["i"], 3 * p["j"])
                       and rgb == ref["down"][p["i"]][p["j"]] == _i3(arr[p["row"], p["col"]]) and p["min"] == min(rgb)
                       and p["dark"] == ref["dark"][p["i"]][p["j"]] and p["far"] == int(p["dark"] >= r["t_h_prev"]))
            ok &= ((r["H"], r["W"]) == arr.shape[:2] and (r["Hd"], r["Wd"]) == ref["shape_down"] and r["N"] == ref["n"]
                   and r["gray_sum"] == ref["gray_sum"] and r["gray_mean"] == ref["gray_mean"])
            for g in ("far", "near"):
                e = ref[g]
                got = {"valid": bool(r[f"{g}_valid"]), "count": r[f"{g}_count"], "max_dark": r[f"{g}_max"],
                       "pos": None if r[f"{g}_i"] is None else (r[f"{g}_i"], r[f"{g}_j"]),
                       "rgb": None if r[f"{g}_R"] is None else _v3(r, f"{g}_")}
                ok &= got == e
            bad += not ok
        return bad == 0, f"不符 {bad} 次"
    _try(ck, f"O 統計 {len(stats)} 次 pass（{len(T['stats_pixels'])} 個降採樣點）：降採樣位置與 RGB、5×5 dark（補 255）、"
             f"gray_sum、gray_mean、遠近群的 count／max／位置／RGB 都和 ate_golden._ref_stats（逐點迴圈）相同；RGB 也等於 frames.hex",
         stats_oracle)

    # ---- C. CPU 更新
    print("C CPU 更新")
    bad = []
    for case, (th, af, an, im, q, fw, nw) in H_CPU.items():
        r = next((x for x in T["cpu"] if x["case_id"] == case), None)
        if r is None:
            bad.append(f"{case:02X} 缺")
            continue
        got = (r["next_t_h"], _v3(r, "next_AF_"), _v3(r, "next_AN_"), r["next_I_max"], r["next_inv_q"],
               (r["far_act"], r["far_diff"], r["far_trig"]), (r["near_act"], r["near_diff"], r["near_trig"]))
        if got != (th, af, an, im, q, fw, nw):
            bad.append(f"{case:02X} {got}")
    ck.check("H CPU 手算 c1–c7（差 3 不平均、差 4 平均、負方向、三色 floor、255+250、空群組沿用、第一次更新與空群組、I_max = 0；"
             "下一個 T_h、A_F、A_N、I_max、inv_q 與旗標）", not bad, "；".join(bad[:3]))

    def cpu_rules():
        bad = 0
        for r in T["cpu"]:
            ok = r["next_t_h"] == r["gray_sum"] // (3 * r["N"]) and r["next_I_max"] == sum(_v3(r, "next_AF_"))
            im, q = r["next_I_max"], r["next_inv_q"]
            ok &= q == 0 if im == 0 else im * q <= 1 << 20 < im * (q + 1)
            for g, other, nx in (("far", "near", "next_AF_"), ("near", "far", "next_AN_")):
                raw = _v3(r, f"{g}_raw_") if r[f"{g}_valid"] else None
                oth = _v3(r, f"{other}_raw_") if r[f"{other}_valid"] else None
                prev = _v3(r, "prev_AF_" if g == "far" else "prev_AN_")
                used, act, diff, trig = _v3(r, f"{g}_used_"), r[f"{g}_act"], r[f"{g}_diff"], r[f"{g}_trig"]
                ok &= used == _v3(r, nx)
                if not r["prev_measured"]:
                    ok &= (act == 0 and used == raw) if raw is not None else (act == 1 and used == oth)
                    ok &= diff is None and trig == 0
                elif raw is None:
                    ok &= act == 2 and used == prev and diff is None and trig == 0
                else:
                    dmax = max(abs(a - b) for a, b in zip(raw, prev))
                    avg = tuple((a + b) // 2 for a, b in zip(raw, prev))
                    ok &= diff == dmax and ((act, trig, used) == (4, 1, avg) if dmax > 3 else (act, trig, used) == (3, 0, raw))
            bad += not ok
        return bad == 0, f"不符 {bad} 列"
    _try(ck, f"O CPU 全部 {len(T['cpu'])} 列：依規則另寫的 Python 判斷（T_h = floor(gray_sum/3N)、第一次更新、空群組、"
             f"step 4 差 > 3 時三色 floor 平均、和上一個 active 比）；I_max 與 inv_q", cpu_rules)

    # ---- D. 排程
    print("D 排程")
    sched = {}
    for r in T["schedule"]:
        sched.setdefault(r["case_id"], []).append(r)

    def act_of(r, p):
        return (r[f"{p}measured"], r[f"{p}t_h"], _v3(r, f"{p}AF_"), _v3(r, f"{p}AN_"), r[f"{p}I_max"], r[f"{p}inv_q"])

    bad = []
    for case, want in H_SCHED.items():
        rows = sched.get(case, [])
        if len(rows) != len(want):
            bad.append(f"{case:02X} frame 數 {len(rows)}")
            continue
        for r, w in zip(rows, want):
            c = cpu[r["upd_id"]]
            got = (r["stat_t_h"], _v3(r, "use_AF_"), _v3(r, "use_AN_"), r["use_I_max"], r["use_inv_q"],
                   (c["far_act"], c["far_diff"], c["far_trig"]), (c["near_act"], c["near_diff"], c["near_trig"]),
                   r["next_t_h"], _v3(r, "next_AF_"), _v3(r, "next_AN_"), r["next_I_max"], r["next_inv_q"],
                   stats[r["stats_id"]]["gray_mean"])
            if got != w:
                bad.append(f"{case:02X}/{r['seq']} {got}")
    ck.check("H 排程手算：video S2（差 3／4）、S3（遠群變空、收斂）、兩區 15×9 × 7（S6 的 A 序列、T_h 103）、static S4、static 兩區"
             "（每張統計時、除霧時、下一張的 T_h、A、I_max、inv_q 與 step 4）", not bad, "；".join(bad[:3]))

    def sched_links():
        bad = []
        a0 = (0, 128, W3, W3, 765, 1370)
        for case, rows in sched.items():
            mode = rows[0]["mode"]
            bad += [f"{case:02X} seq"] if [r["seq"] for r in rows] != list(range(len(rows))) else []
            bad += [f"{case:02X} static 多張"] if mode == 0 and len(rows) != 1 else []
            for n, r in enumerate(rows):
                st, c = stats[r["stats_id"]], cpu[r["upd_id"]]
                if mode == 0:
                    ok = act_of(r, "stat_") == a0 and act_of(r, "use_") == act_of(r, "next_")
                else:
                    ok = act_of(r, "stat_") == act_of(r, "use_") and (act_of(r, "stat_") == a0 if n == 0
                                                                      else act_of(r, "use_") == act_of(rows[n - 1], "next_"))
                ok &= ((st["case_id"], st["seq"], st["frame_id"], st["t_h_prev"]) == (case, r["seq"], r["frame_id"], r["stat_t_h"])
                       and (c["case_id"], c["seq"], c["stats_id"]) == (case, r["seq"], r["stats_id"])
                       and (c["prev_measured"], c["prev_t_h"], _v3(c, "prev_AF_"), _v3(c, "prev_AN_")) == act_of(r, "stat_")[:4]
                       and (c["gray_sum"], c["N"]) == (st["gray_sum"], st["N"])
                       and (c["far_valid"], c["near_valid"]) == (st["far_valid"], st["near_valid"])
                       and _v3(c, "far_raw_") == _v3(st, "far_") and _v3(c, "near_raw_") == _v3(st, "near_")
                       and (1, c["next_t_h"], _v3(c, "next_AF_"), _v3(c, "next_AN_"), c["next_I_max"], c["next_inv_q"])
                       == act_of(r, "next_"))
                arr = frames[r["frame_id"]]
                h, w = arr.shape[:2]
                vecs = list(range(r["vec0"], r["vec0"] + r["nvec"]))
                ok &= r["nvec"] == h * w
                for k, v in enumerate(vecs):
                    p = pala.get(v)
                    ok &= (p is not None and (p["case_id"], p["seq"], p["frame_id"], p["row"], p["col"]) == (case, r["seq"], r["frame_id"], k // w, k % w)
                           and _v3(p, "") == _i3(arr[k // w, k % w]) and _v3(p, "AF_") == act_of(r, "use_")[2]
                           and _v3(p, "AN_") == act_of(r, "use_")[3] and (p["I_max"], p["inv_q"]) == act_of(r, "use_")[4:])
                if not ok:
                    bad.append(f"{case:02X}/{n}")
        return not bad, "；".join(bad[:5])
    _try(ck, "W 排程接線：static 的統計用 active0、除霧用 CPU 更新後的 active；video 的統計與除霧用同一個 active、frame n 用 frame n−1 "
             "算出的下一個；stats、cpu、pala 的識別與參數和 schedule 列相同；每張的向量覆蓋整張 raster、RGB 等於 frames.hex", sched_links)

    def golden_rerun():
        """從 frames.hex 重建影像，重新呼叫 golden 排程（SD lut、ideal 各一次），和檔案比較。"""
        bad = []
        for case, rows in sched.items():
            arrs = [frames[r["frame_id"]] for r in rows]
            runs = {}
            for md in MODE:
                a0 = golden.make_active(**golden.ACTIVE0_DEFAULT, **dict(FX, sd_mode=md))
                runs[md] = [golden.run_static(arrs[0], a0)] if rows[0]["mode"] == 0 else golden.run_video(arrs, a0)
            for r, ri, rl in zip(rows, runs["ideal"], runs["lut"]):
                ok = golden._trace_key(ri) == golden._trace_key(rl)
                ok &= (golden._trace_key(rl)[:3] == (r["stat_t_h"], _v3(r, "use_AF_"), _v3(r, "use_AN_"))
                       and golden._trace_key(rl)[5:] == (r["next_t_h"], _v3(r, "next_AF_"), _v3(r, "next_AN_")))
                ok &= np.array_equal(ri["result"]["a_q"], rl["result"]["a_q"]) and np.array_equal(ri["result"]["t_q"], rl["result"]["t_q"])
                w = frames[r["frame_id"]].shape[1]
                for k, v in enumerate(range(r["vec0"], r["vec0"] + r["nvec"])):
                    y, x = divmod(k, w)
                    ok &= (_v3(pala[v], "A_q_") == _i3(rl["result"]["a_q"][y, x]) and pala[v]["eta_q"] == int(rl["result"]["eta_q"][y, x])
                           and pte[(v, 0)]["t_q"] == int(rl["result"]["t_q"][y, x])
                           and _v3(sd[(v, 0, 1)], "out_") == _i3(rl["result"]["out"][y, x])
                           and _v3(sd[(v, 0, 0)], "out_") == _i3(ri["result"]["out"][y, x]))
                if not ok:
                    bad.append(f"{case:02X}/{r['seq']}")
        return not bad, "；".join(bad[:5])
    _try(ck, "W golden 重跑：由 frames.hex 重建影像，用 ate_golden 的 run_video／run_static（FX；SD lut 與 ideal 各一次）重算，"
             "trace、A_q、eta_q、EX 的 t_q 和檔案相同；sd.hex 的 EX＋LUT、EX＋IDEAL 輸出分別等於兩次 golden 的輸出；"
             "SD 模式不改統計與更新鏈", golden_rerun)

    def sched_static_hand():
        bad = 0
        for case, rgbs in ((0x53, None), (0x54, _P)):
            r = sched[case][0]
            for v in range(r["vec0"], r["vec0"] + r["nvec"]):
                px = _v3(pala[v], "")
                if rgbs is None or px == rgbs:
                    bad += any(_v3(sd[(v, s, md)], "out_") != px for s in (0, 1, 2) for md in (0, 1))
        r = sched[0x55][0]
        c = cpu[r["upd_id"]]
        ok = (c["far_valid"] == c["near_valid"] == 1 and (c["far_act"], c["near_act"]) == (0, 0)
              and _v3(r, "use_AF_") == _v3(c, "far_raw_") and _v3(r, "use_AN_") == _v3(c, "near_raw_")
              and _v3(r, "use_AF_") != _v3(r, "use_AN_") and r["stat_t_h"] == 128)
        etas = {pala[v]["eta_q"] for v in range(r["vec0"], r["vec0"] + r["nvec"])}
        return bad == 0 and ok and len(etas) > 2, f"A = I 卻輸出不同 {bad}；SRND {ok}；eta 種類 {len(etas)}"
    _try(ck, "H 排程輸出：static S4 全部、static 兩區的 P 區 pixel（A = I）在 EX／L8／L10 × IDEAL／LUT 都輸出 = 輸入；"
             "SRND 第一次更新兩群都有、A_F ≠ A_N、eta_q 有多種值", sched_static_hand)

    # ---- E. LUT
    print("E LUT 內容")

    def lut_tables():
        tb = {}
        for r in T["lut"]:
            tb.setdefault(r["table"], []).append((r["index"], r["value"]))
        e, r8, r10, sdt = (dict(tb.get(k, [])) for k in range(4))
        ok = ([i for i, _ in tb[0]] == list(range(511)) and [i for i, _ in tb[1]] == list(range(257))
              and [i for i, _ in tb[2]] == list(range(1025)) and [i for i, _ in tb[3]] == list(range(1, 1025)))
        ok &= e[0] == e[1] == 0 and e[2] == 1 << 22 and all(abs(2 * e[s] * s - (1 << 24)) < s for s in range(2, 511))
        for b, rb in ((8, r8), (10, r10)):
            ok &= rb[0] == 1 << 17 and rb[1 << b] == 1 << 16
            ok &= all(abs(2 * rb[j] * ((1 << b) + j) - (1 << (17 + b + 1))) < (1 << b) + j for j in rb)
        ok &= sdt[1] == 1 << 20 and sdt[1024] == 1 << 10
        ok &= all((2 * v - 1) * t <= 1 << 21 < (2 * v + 1) * t for t, v in sdt.items())
        return ok, ""
    _try(ck, "E LUT 內容：E[s]（s = 0–510）、R_8[j]（0–256）、R_10[j]（0–1024）、SD inv[t_q]（1–1024）的 index 完整；"
             "端點 E[2] = 2^22、R[0] = 2^17、R[2^B] = 2^16、inv[1] = 2^20、inv[1024] = 2^10；每一項的 RN 整數不等式", lut_tables)


# ---------------------------------------------------------------- 指令

def _summary(m, texts):
    for f in FILES:
        print(f"  {f}.hex：{len(m.rows[f])} 列，{len(texts[f])} bytes，SHA-256 {hashlib.sha256(texts[f]).hexdigest()}")


def _counts(m):
    print(f"  案例 {len(m.cases)} 個；影像 {len(m.frames)} 張；主路徑向量 {m.n['vec']} 個；統計 pass {m.n['stats']} 次；CPU 更新 {m.n['upd']} 次")


def _header():
    print(f"程式：ate_vectors.py SHA-256 {_sha256_file(os.path.abspath(__file__))}")
    print(f"      ate_golden.py SHA-256 {_sha256_file(golden.__file__)}")
    print(f"      ate_pte_lut.py SHA-256 {_sha256_file(lut.__file__)}")
    print(f"環境：Python {sys.version.split()[0]}、numpy {np.__version__}")


def selfcheck():
    _header()
    ck = golden._Checker()
    print("D 記憶體內建兩次")
    m1, m2 = build(), build()
    t1, t2 = render(m1), render(m2)
    _counts(m1)
    ck.check("D 兩次建出的 9 個檔，bytes 逐一相同", all(t1[f] == t2[f] for f in FILES))
    _summary(m1, t1)
    print("R 讀回（記憶體內的文字）")
    tables = readback(ck, t1, m1, "記憶體")
    check_tables(ck, tables)
    print(f"結果：{ck.total - ck.failed}/{ck.total} PASS")
    return 0 if ck.failed == 0 else 1


def _read_disk():
    out = {}
    for f in FILES:
        with open(os.path.join(OUT_DIR, f + ".hex"), "rb") as fh:
            out[f] = fh.read()
    return out


def write_command():
    _header()
    ck = golden._Checker()
    m1, m2 = build(), build()
    t1, t2 = render(m1), render(m2)
    _counts(m1)
    ck.check("D 寫檔前在記憶體建兩次，9 個檔的 bytes 逐一相同", all(t1[f] == t2[f] for f in FILES))
    if ck.failed:
        print("停止：記憶體內兩次不同，不寫檔")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in FILES:
        with open(os.path.join(OUT_DIR, f + ".hex"), "wb") as fh:
            fh.write(t1[f])
    print(f"已寫入 {OUT_DIR}")
    disk = _read_disk()
    ck.check("D 磁碟讀回的 bytes 和記憶體逐一相同", all(disk[f] == t1[f] for f in FILES))
    _summary(m1, disk)
    tables = readback(ck, disk, m1, "磁碟")
    check_tables(ck, tables)
    print(f"結果：{ck.total - ck.failed}/{ck.total} PASS")
    return 0 if ck.failed == 0 else 1


def verify_command():
    _header()
    ck = golden._Checker()
    m = build()
    t = render(m)
    _counts(m)
    disk = _read_disk()
    ck.check("D 重新建的 9 個檔和磁碟上的 bytes 逐一相同", all(disk[f] == t[f] for f in FILES))
    _summary(m, disk)
    tables = readback(ck, disk, m, "磁碟")
    check_tables(ck, tables)
    print(f"結果：{ck.total - ck.failed}/{ck.total} PASS")
    return 0 if ck.failed == 0 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="AVSD Final E：軟體 reference v0 向量包")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selfcheck", help="記憶體內建兩次、讀回、手算／oracle／接線／guard；不寫檔")
    sub.add_parser("write", help="寫 results/vectors-v0/*.hex，寫後從磁碟讀回並做同一組檢查")
    sub.add_parser("verify", help="不寫檔：重建並和磁碟上的檔案逐 byte 比較，再做同一組檢查")
    args = parser.parse_args(argv)
    if args.command == "selfcheck":
        return selfcheck()
    if args.command == "write":
        return write_command()
    return verify_command()


if __name__ == "__main__":
    sys.exit(main())
