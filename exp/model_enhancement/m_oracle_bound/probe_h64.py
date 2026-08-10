# -*- coding: utf-8 -*-
"""诊断：H=64 base 第1天 CRPS vs 公开基线（H=8, 0.86）——检验 horizon 稀释程度。

注意：导入 run_m 会 wrap sys.stdout，本脚本不再自 wrap。
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from exp.model_enhancement.m_oracle_bound.run_m import (
    load_wide, build_window, train_model, predict_quantiles, crps_quantiles,
    T, H8, H64, TRAIN_DAYS, TEST_DAYS, METEO_COLS)

wide = load_wide("data/processed/standard.parquet")
n = len(wide)
days = TRAIN_DAYS + TEST_DAYS
feat = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
feat = [c for c in feat if c in wide.columns] + ["conc_0.5"]

i0, i1 = 0, days * 8
print("window1:", wide.index[i0], "->", wide.index[i1-1])

# base8 (H=8) 锚定
X8, y8, cur8, s8, w8 = build_window(wide, i0, i1, feat, 0, H8)
n8 = len(X8); ntr8 = int(n8 * TRAIN_DAYS/(TRAIN_DAYS+TEST_DAYS))
d8 = y8 - cur8[:, None]; sd8 = float(np.std(d8[:ntr8])) + 1e-8
m8 = train_model(X8, (d8/sd8).astype(np.float32), s8, w8, ntr8, H8, 0, 30, "cpu", 0)
q8 = predict_quantiles(m8, X8[ntr8:], "cpu")
c8 = cur8[ntr8:, None, None] + q8 * sd8
crps8 = np.mean([crps_quantiles(c8[:,0,h], c8[:,1,h], c8[:,2,h], y8[ntr8:,h]) for h in range(H8)])
print(f"base8 H=8:  CRPS={crps8:.4f}  (公开基线≈0.86)")

# base64 (H=64)
X6, y6, cur6, s6, w6 = build_window(wide, i0, i1, feat, 0, H64)
n6 = len(X6); ntr6 = int(n6 * TRAIN_DAYS/(TRAIN_DAYS+TEST_DAYS))
d6 = y6 - cur6[:, None]; sd6 = float(np.std(d6[:ntr6])) + 1e-8
m6 = train_model(X6, (d6/sd6).astype(np.float32), s6, w6, ntr6, H64, 0, 30, "cpu", 0)
q6 = predict_quantiles(m6, X6[ntr6:], "cpu")
c6 = cur6[ntr6:, None, None] + q6 * sd6
def day_crps(q, y, day):
    h0, h1 = day*8, day*8+8
    return np.mean([crps_quantiles(q[:,0,h], q[:,1,h], q[:,2,h], y[:,h]) for h in range(h0,h1)])
obs6 = y6[ntr6:]
print(f"base64 H=64: CRPS_d1={day_crps(c6,obs6,0):.4f}  CRPS_d8={day_crps(c6,obs6,7):.4f}  "
      f"CRPS_all={np.mean([day_crps(c6,obs6,d) for d in range(8)]):.4f}")
print("稀释判定: base64_d1 若 <1.2 → 可接受；若 >1.5 → H=64 稀释严重，需改设计")

# oracle 收敛验证：oracle_7 是否学会复制已知轨迹（30 epoch，头注入设计）
Xo, yo, curo, so, wo = build_window(wide, i0, i1, feat, 56, H64)
no = len(Xo); ntro = int(no * TRAIN_DAYS/(TRAIN_DAYS+TEST_DAYS))
do = yo - curo[:, None]; sdo = float(np.std(do[:ntro])) + 1e-8
mo = train_model(Xo, (do/sdo).astype(np.float32), so, wo, ntro, H64, 56, 30, "cpu", 0)
qo = predict_quantiles(mo, Xo[ntro:], "cpu")
co = curo[ntro:, None, None] + qo * sdo
obso = yo[ntro:]
op = np.empty_like(obso); op[:, :56] = obso[:, :56]; op[:, 56:] = obso[:, 55:56]
crps_op = np.mean([np.mean(crps_quantiles(op[:,h], op[:,h], op[:,h], obso[:,h])) for h in range(H64)])
def day_crps_o(q, y, day):
    h0, h1 = day*8, day*8+8
    return np.mean([crps_quantiles(q[:,0,h], q[:,1,h], q[:,2,h], y[:,h]) for h in range(h0,h1)])
print(f"oracle_7 H=64: CRPS_all={np.mean([day_crps_o(co,obso,d) for d in range(8)]):.4f}  "
      f"orac_persist={crps_op:.4f}  CRPS_d1={day_crps_o(co,obso,0):.4f}  CRPS_d8={day_crps_o(co,obso,7):.4f}")
print("收敛判定: oracle_7 CRPS_all 若 << orac_persist（复制已知段→近0 + 尾部预测）→ 探针有效")

# 特征贡献 30 epoch：ar_only（仅 conc 历史）
Xa, ya, cura, sa, wa = build_window(wide, i0, i1, ["conc_0.5"], 0, H64)
na = len(Xa); ntra = int(na * TRAIN_DAYS/(TRAIN_DAYS+TEST_DAYS))
da = ya - cura[:, None]; sda = float(np.std(da[:ntra])) + 1e-8
ma = train_model(Xa, (da/sda).astype(np.float32), sa, wa, ntra, H64, 0, 30, "cpu", 0)
qa = predict_quantiles(ma, Xa[ntra:], "cpu")
ca = cura[ntra:, None, None] + qa * sda
obsa = ya[ntra:]
print(f"ar_only H=64: CRPS_d1={day_crps_o(ca,obsa,0):.4f}  CRPS_d8={day_crps_o(ca,obsa,7):.4f}  "
      f"CRPS_all={np.mean([day_crps_o(ca,obsa,d) for d in range(8)]):.4f}")
print("特征贡献判定: base_d1<ar_d1 → temp/气象有用；≈ → 纯自回归")


