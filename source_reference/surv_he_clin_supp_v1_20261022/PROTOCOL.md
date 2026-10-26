# HE + 临床预测复发：预先登记的次要分析（surv_he_clin_supp_v1_20261022）

本计划补充 `../surv_he_clin_v1_20261022`（下称 surv1，已于 2026-10-22 16:45:38 锁定，LOCK.json sha256 `3ba39ee0…`）。**在 surv1 任何成绩出来之前锁定**：`supp.py lock-write` 会在 surv1 的 RESULTS.json 已经存在时拒绝锁定。本计划不改变 surv1 的主要指标和判定，这里的结果都只作为次要分析报告。

## 1. 输入（不训练任何模型）

- surv1 的 50 次正式运行，经 surv1 自己的检查读取（`evaluate.collect`：每次运行的收据经过哈希校验、来自锁定的代码签名、同一平台；技术检查是 PASS）。surv1 的代码或 LOCK.json 与锁定时不一致就拒绝运行。
- 病例、终点、84 个月截尾、DFS = 0 的 5 例排除、Leibovich 评分（Fuhrman 版）、HE_z（`evaluate.standardise`，在队列内标准化）都与 surv1 完全相同。
- 两个模型：**Leibovich 单独**：Cox(Leibovich)；**融合**：Cox(Leibovich + HE_z)。
  - 韩国用交叉拟合：系数和 Breslow 基线风险都来自另外 4 折。
  - 苏格兰用韩国 295 例拟合的系数和基线风险，原样冻结。
  - 预测复发概率 = 1 − exp(−H₀(t)·exp(xβ))。

## 2. 主要时间点

**36 个月。** 苏格兰未复发者的中位随访只有 53 个月：到 60 个月时，95 个未复发者中 57 人已截尾，只剩 38 人仍在随访，60 个月的绝对风险估计不稳定。36 个月时还有 88 人在随访，事件数 21 / 27。时间依赖 AUC 同时报告 36 和 60 个月。

## 3. 分析（两个编码器都做，Phikon-v2 为主）

| 分析 | 方法 |
|---|---|
| 时间依赖 AUC | 累积/动态 AUC（Uno 2007，用本队列的截尾 Kaplan–Meier 做逆概率加权）：病例 = t 前复发，对照 = t 后仍无复发。分别计算 Leibovich、融合模型线性预测值、HE_z 单独在 36 和 60 个月的 AUC，以及 ΔAUC（融合 − Leibovich）的 95% 区间（病例 bootstrap 2,000 次，模型固定，每次重抽重新估计截尾分布） |
| Uno C 指数 | 截断在 60 个月，ΔC（融合 − Leibovich）及 bootstrap 区间；作为 surv1 Harrell C 的稳健性分析 |
| 校准（36 个月） | 校准斜率（在该队列上对 log(−log(1 − 预测复发率)) 做 Cox 回归，即被校准的预测值本身的线性预测值，韩国包含各折自己的基线风险；理想值为 1）及 95% 区间；观察/预期比（Kaplan–Meier 36 个月复发率 ÷ 平均预测复发率）；按预测风险等分组（韩国 5 组、苏格兰 3 组）比较预测值与 Kaplan–Meier 观察值（Greenwood 区间）。两个模型都报告 |
| 决策曲线（36 个月） | 净获益（Vickers 2008 的 Kaplan–Meier 版本）：阈值概率 0.02–0.50，步长 0.01；比较融合模型、Leibovich 单独、全部治疗、全部不治疗。报告阈值 0.10、0.20、0.30 处的净获益，以及融合模型同时优于 Leibovich 和"全部治疗"且净获益 > 0 的阈值 |
| 分层 Kaplan–Meier | Leibovich 风险组：低危 0–2、中危 3–5、高危 ≥ 6（Leibovich 2003）。每组内按 HE_z 高低分组，切点 = 韩国 HE_z 中位数，与 surv1 的 log-rank 相同。报告每格例数、事件数、36 和 60 个月无复发率（该时间点仍有人随访才报告，并附仍随访人数）、组内 log-rank；另报告全队列 HE 高低的 KM 曲线和 log-rank |
| 重分类 | 36 个月的净重分类改善（NRI，Pencina 2011 的 Kaplan–Meier 版本），融合模型对比 Leibovich：无分类 NRI（任何上调或下调，95% bootstrap 区间）；分类 NRI 的切点为 36 个月复发风险 < 10%、10–30%、≥ 30%，复发者与未复发者分开报告 |

图（每个编码器 × 队列一张，共 4 个面板）：HE 高低 KM 曲线、Leibovich 组内 KM 曲线、36 个月校准图、决策曲线。

## 4. 运行与报告

- surv1 的 evaluate.py 写出 RESULTS.json 后，A6000 上的 `run_when_ready.sh` 自动运行 `supp.py`（只用 CPU，几分钟）。
- 输出：`SUPP_RESULTS.json`、`SUPP_RESULTS.md`（中文）、`figures/*.png`。
- 不设判定，所有结果原样报告；不得据此修改 surv1 的标签、病例、截尾、阈值或模型。
