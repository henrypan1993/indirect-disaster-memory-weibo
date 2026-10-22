# 更新后的假设、模型与估计结果（H1 / H2）

> 文档日期：2026-08-11  
> 2026-10-22 修订稿以 Q1（时期对比，仓库 `h2_*`，调整后 OR = 1.46）和 Q2（更宽边缘样本的参与，仓库 `h1_broader_sample`）为准。下文是仓库内部编号备忘，不替代附录 D。  
> 依据仓库当时冻结的样本标志、主模型规格与 `outputs/public/` 已估计系数整理。

---

## 0. 核实结论

| 主张 | 仓库现状 | 判定 |
|------|----------|------|
| **H1 与 H2 采样范围对齐** | `sample_h1_eligible` 与 `sample_h2_eligible` 均为 `peripheral ∩ disaster_impact_related ∩ 有效表达`（`scripts/03_prepare_model_data.py`） | **成立** |
| **D1 已从主分析移除** | `main_model_specs()` 仍含 D1-M1；`outputs/public/table_main_focal_coefficients.csv` 仍报告 D1；README / Appendix D 仍保留 Boundary test | **尚未在代码与公开表中移除** |

因此：下文按**更新后的确认性框架（仅 H1、H2）**整理假设与结果；并注明仓库中 D1 代码路径仍在，但不再作为本框架的确认性假设。

---

## 1. 核心分析总体（H1 = H2）

**Disaster-impact-related posts from peripheral accounts**

```text
include_main == 1
& peripheral == 1          # verified == 0 & followers < p90
& disaster_impact_related == 1
    # label_narrative_clean ∈ {Trauma-Help-Loss, Memory-Reactivation}
& indirect_clean 非缺失     # Direct=0；Indirect/Mixed=1；Unclear 排除
```

| 样本 | N（eligible / 主规格估计） | 账户数（H1-M1） |
|------|---------------------------|----------------|
| 核心样本（H1 与 H2 共用） | **5,846** | 5,476 |
| H1-M2 / H2-M2（composition，complete-case） | **5,804** | — |
| 附录：宽边缘稳健性（任意叙事，仅 H1） | **10,806** | 9,897 |

**H1 与 H2 的区别仅在于问题与因变量，不在于样本：**

| | H1 | H2 |
|--|----|----|
| 样本 | 核心样本（上） | **同左** |
| 问题 | 表达形式 ↔ 互动 | 风险语境（T2）↔ 表达形式 |
| 设计 | T1+T2 合并，控制 `t2` | T1 vs T2，焦点为 `t2` |

---

## 2. 更新后的假设

### H1（表达形式与互动）

在灾害影响相关的边缘账号帖文中，**Indirect/Mixed** 表达与更高的（对数）互动量相关。

### H2（风险语境与表达形式）

在同一总体中，**再激活期 T2（2025）** 相对于灾害期 T1（2021），Indirect/Mixed 表达的发生几率更高。

### 不再作为确认性假设

- **原 D1（边界检验）**：全 Core 上 Peripheral → Indirect/Mixed。代码与旧公开表仍存，**不纳入本更新框架的主假设**。

---

## 3. 模型规格

共同控制（边缘子样本）：`log_followers`，`post_hour_cst`，`hashtag_count`，`text_length`，`C(topic_id)`。  
估计：H1 为 OLS；H2 为 Logit（Binomial GLM）；均使用 **按账号聚类的稳健标准误**。不含 `verified`（边缘子样本内恒为 0）。

### 3.1 H1-M1（主模型，`spec_id=main`）

$$
\log(1+\text{Total Engagement})_i
=
\beta_0
+
\beta_1 \text{IndirectMixed}_i
+
\beta_2 \text{T2}_i
+
\gamma X_i
+
\text{Topic FE}
+
\varepsilon_i
$$

分量稳健性（同协变量、同核心样本）：

$$
Y \in \{\log(1+\text{Likes}),\ \log(1+\text{Comments}),\ \log(1+\text{Reposts})\}
$$

### 3.2 H1-M2（内容构成调整，非替代主模型）

在 H1-M1 上增加：`relative_window_day`，`C(label_narrative_clean)`，`C(label_emotion_clean)`。

### 3.3 H2-M1（主模型，`spec_id=h2_m1_period`）

$$
\text{logit}\,P(\text{IndirectMixed}_i=1)
=
\beta_0
+
\beta_1 \text{T2}_i
+
\gamma X_i
+
\text{Topic FE}
$$

### 3.4 H2-M2（内容构成调整）

在 H2-M1 上增加：`relative_window_day`，叙事 FE，情绪 FE（与 H1-M2 对称）。

### 3.5 附录稳健性（非主结果）

| 规格 | 作用 |
|------|------|
| `h1_broader_sample` | 全部边缘帖文（不限灾害叙事），同 H1-M1 公式 |
| Peripheral p80 / p90 / p95 | 核心样本上重定义边缘阈值后重估 H1 |

---

## 4. 估计结果（当前公开表）

显著性：\* *p* < .05，\*\* *p* < .01，\*\*\* *p* < .001。百分比变化 = \(\exp(\beta)-1\)。

### 4.1 H1 焦点系数

| 规格 | 焦点项 | Coef. (SE) | 约 % 变化 | N |
|------|--------|------------:|----------:|--:|
| **H1-M1（主）** | Indirect/Mixed | **0.123\*\*\* (0.035)** | **+13.1%** | 5,846 |
| H1-M2 | Indirect/Mixed | 0.099\*\* (0.036) | +10.5% | 5,804 |
| 宽边缘附录 | Indirect/Mixed | 0.132\*\*\* (0.029) | +14.1% | 10,806 |

**H1-M1 分量（核心样本）：**

| 因变量 | Indirect/Mixed | N |
|--------|---------------:|--:|
| log(likes) | 0.106\*\*\* (0.031) | 5,846 |
| log(comments) | 0.061\* (0.024) | 5,846 |
| log(reposts) | 0.003 (0.013) | 5,846 |

分量显著性模式本身不等于系数差异；正式检验见 `scripts/13_test_engagement_asymmetry.py`（stacked OLS + Wald）与 `outputs/public/table_engagement_asymmetry_wald.csv`。

| 对比 | F | df | *p* |
|------|--:|:---|----:|
| 联合：β_likes = β_comments = β_reposts | 7.15 | (2, 5475) | .0008\*\*\* |
| likes vs reposts | 14.07 | (1, 5475) | .0002\*\*\* |
| comments vs reposts | 6.23 | (1, 5475) | .013\* |
| likes vs comments | 3.46 | (1, 5475) | .063 |

自由度：分子 df = 约束个数；分母 df = n_accounts − 1 = 5476 − 1 = 5475（账户聚类 F 检验）。

**Peripheral 阈值敏感性（核心样本 H1）：** p80 → 0.114\*\* (N=5,786)；p90 → 0.123\*\*\* (N=5,846)；p95 → 0.123\*\*\* (N=5,863)。

### 4.2 H2 焦点系数

| 规格 | 焦点项 | Coef. (SE) | OR | N |
|------|--------|------------:|---:|--:|
| **H2-M1（主）** | T2 | **0.651\*\*\* (0.095)** | **1.92** | 5,846 |
| H2-M2 | T2 | 0.381\*\* (0.116) | 1.46 | 5,804 |

H2-M1 OR 95% CI ≈ [1.59, 2.31]；H2-M2 OR 95% CI ≈ [1.17, 1.84]（Appendix D）。

### 4.3 结果解读（确认性层面）

1. **H1**：在灾害影响 × 边缘账号核心样本中，Indirect/Mixed 与更高总互动（及点赞、评论）相关；转发生效不显著。stacked Wald 正式拒绝三分量系数相等（尤其 likes/comments 显著大于 reposts）。构成调整后关联仍显著但略减弱。  
2. **H2**：同一核心样本中，T2 与更高的 Indirect/Mixed 几率相关；构成调整后 OR 由约 1.92 降至约 1.46，仍显著。  
3. **样本对齐的含义**：H1/H2 回答同一总体上的不同问题（表达→互动 vs 时期→表达），避免“H1 用更宽边缘话语、H2 用灾害子样本”的错位比较。

---

## 5. 与探索性分析的边界（非确认性）

下列路径**不改变** H1/H2 主结果，仅作延伸：

| 路径 | 样本 | 角色 |
|------|------|------|
| `10_export_engagement_deviance_cases.py` | 核心样本期望残差 | 正/负 deviance 个案（负侧要求 engagement>0） |
| `11_explore_indirect_composition.py` | Peripheral × Indirect/Mixed（**不**强制灾害叙事） | 叙事/情绪构成与互动差异的探索性相关解释 |

---

## 6. 主要数据来源

- 样本标志：`scripts/03_prepare_model_data.py`  
- 规格：`scripts/common.py`（`_h1_spec` / `_h1_m2_spec` / `_h2_m1_spec` / `_h2_m2_spec`）  
- 系数：`outputs/public/table_main_focal_coefficients.csv`，`table_h1_m1_m2_comparison.csv`，`table_h2_m1_m2_comparison.csv`，`table_h1_engagement_components.csv`，`table_engagement_asymmetry_wald.csv`，`table_h1_broader_sample.csv`，`table_peripheral_threshold_sensitivity.csv`  
- 叙述性完整表：`appendices/Appendix_D.md`

---

## 7. 后续若正式“移除 D1”

需另行改动（本文件未执行）：从 `main_model_specs` / `composition_or_conditional_specs`、公开表构建、README、Appendix C–D 中撤下 D1；并更新 `outputs/public/`。在完成前，仓库仍同时保留 D1 旧结果。
