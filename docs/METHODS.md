# 方法说明

本文档说明 `clinaudit` 实现的所有指标的数学定义、计算方式与适用边界。
所有公式都可以在源码中逐行对照。

---

## 1. 判别力（Discrimination）

### 1.1 ROC-AUC

定义为随机抽取一个阳性和一个阴性样本，阳性得分高于阴性的概率：

$$\text{AUC} = P\big(s(x^+) > s(x^-)\big)$$

实现上用 Mann-Whitney U 统计量，并对**并列值取平均秩**，因此与
`sklearn.metrics.roc_auc_score` 数值一致（有测试覆盖）。

**局限**：AUC 只看排序，不看概率数值。一个单调但严重偏移的模型 AUC 不变。

### 1.2 PR-AUC

$$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$$

类别不平衡时比 ROC-AUC 更能反映实际表现。医学筛查场景（患病率常低于 10%）
尤其重要。

### 1.3 混淆矩阵派生指标

在阈值 $t$ 下：

$$\text{敏感度} = \frac{TP}{TP+FN}, \quad \text{特异度} = \frac{TN}{TN+FP}$$

Youden 指数 $J = \text{敏感度} + \text{特异度} - 1$，取 $J$ 最大的阈值作为
"最优阈值"。

---

## 2. 校准（Calibration）

### 2.1 校准曲线

按预测概率分箱（默认按分位分箱，保证每箱样本数相近），比较：
- 箱内平均预测概率 $\bar p_k$
- 箱内实际阳性率 $\bar y_k$

完美校准时应落在对角线上。

### 2.2 ICI（Integrated Calibration Index）

$$\text{ICI} = \sum_k w_k \left| \bar y_k - \bar p_k \right|, \quad w_k = \frac{n_k}{N}$$

即按样本量加权的平均绝对偏差。**越小越好**，0 表示完美校准。

相比 E50/E90，ICI 考虑了全部箱，比单一的"平均偏差"更稳健。

### 2.3 E50 / E90

偏差 $|\bar y_k - \bar p_k|$ 的中位数与 90 分位数：
- E50 反映"典型情况"下的偏差
- E90 反映"最差 10%"的情况，对临床安全更关键

### 2.4 Brier 分数

$$\text{BS} = \frac{1}{N}\sum_i (p_i - y_i)^2$$

同时包含判别力与校准信息，越小越好。完美预测为 0，最差为 1。

**分解**：

$$\text{BS} = \underbrace{\frac{1}{N}\sum_k n_k(\bar p_k - \bar y_k)^2}_{\text{可靠性 (reliability，越小越好)}} - \underbrace{\frac{1}{N}\sum_k n_k(\bar y_k - \bar y)^2}_{\text{分辨率 (resolution，越大越好)}} + \underbrace{\bar y(1-\bar y)}_{\text{不确定性}}$$

> ⚠️ **注意**：这个等式是**近似**成立的。分解按箱内平均预测概率计算，
> 只有当每箱内所有预测值完全相同时才严格相等。实测 10 箱时误差约 1e-3，
> 分箱越细误差越小。这一点在测试中有明确覆盖。

### 2.5 校准斜率与截距

把真实标签对 logit(预测) 做逻辑回归：

$$\text{logit}\big(P(y=1)\big) = a + b \cdot \text{logit}(p)$$

| 情况 | 解释 |
|---|---|
| $b \approx 1, a \approx 0$ | 校准良好 |
| $b < 1$ | 过度自信，预测过于极端 |
| $b > 1$ | 信心不足，预测不够极端 |
| $a \neq 0, b \approx 1$ | 系统性高估（$a<0$）或低估（$a>0$） |

> 斜率对样本量敏感，小样本（<200）时估计不稳定，报告中应结合 ICI 一起看。

### 2.6 决策曲线分析（DCA）

在阈值概率 $p_t$ 下把预测 $\geq p_t$ 的样本判为阳性：

$$\text{NB} = \frac{TP}{N} - \frac{FP}{N} \cdot \frac{p_t}{1-p_t}$$

其中 $\frac{p_t}{1-p_t}$ 是"误判为阳性的代价"与"漏判阳性的代价"之比。

参考线：
- **全部干预**：把所有样本判为阳性，$\text{NB} = \pi - (1-\pi)\frac{p_t}{1-p_t}$
- **全部不干预**：$\text{NB} = 0$

只有当模型曲线在某个阈值区间内**同时高于**这两条参考线，模型才在该区间有临床价值。

---

## 3. 公平性（Fairness）

### 3.1 亚组划分

- **离散属性**（性别、中心）：直接按取值分组
- **连续属性**（年龄）：按分位数切成 Q1–Q4

### 3.2 三类差异

| 指标 | 定义 | 警戒线 |
|---|---|---|
| AUC 极差 | $\max_g \text{AUC}_g - \min_g \text{AUC}_g$ | > 0.10 |
| 敏感度极差 | $\max_g \text{Se}_g - \min_g \text{Se}_g$ | > 0.15 |
| 校准偏差极差 | $\max_g \text{gap}_g - \min_g \text{gap}_g$ | > 0.10 |
| 差异性影响比 | $\min_g r_g / \max_g r_g$，$r_g$ 为阳性率 | < 0.80 |

**差异性影响比**来自美国 EEOC 的"四分之五法则"（four-fifths rule）：若某组的
干预率低于最高组的 80%，视为存在差异性影响。

**为什么敏感度差异比 AUC 差异更值得关注**：敏感度差异直接对应"某些人群被系统性
漏诊"，是临床安全问题，而不仅是统计不公平。

---

## 4. 拒识与不确定性（Abstention）

### 4.1 选择性预测

置信度定义为 $c_i = 0.5 + |p_i - 0.5|$，取值 $[0.5, 1]$：

- 预测概率接近 0 或 1 时 $c$ 接近 1（自信）
- 预测概率接近 0.5 时 $c$ 接近 0.5（不确定）

按置信度从高到低纳入样本，得到**覆盖率—风险曲线**。
好的置信度估计应使曲线在低覆盖率处风险显著更低。

### 4.2 AURC

$$\text{AURC} = \frac{1}{\Delta c}\int_{\text{coverage}} \text{risk}(c)\,dc$$

越小说明模型越能把错误集中在低置信度样本上。这是"知道自己在哪些样本上不可靠"
的能力。

### 4.3 Split Conformal 预测

**目标**：给定 $\alpha$，构造预测集合 $C(x)$ 使得 $P(y \in C(x)) \geq 1-\alpha$，
且这个保证在**有限样本**下成立。

**算法**：

1. 计算非一致性分数 $s_i = 1 - p(y_i \mid x_i)$，其中 $p(y)$ 是模型给真实标签的概率
2. 取分位数
   $$q = s_{\lceil (n+1)(1-\alpha) \rceil}$$
   这里的 $+1$ 修正与 $\lceil \cdot \rceil$ 是覆盖率保证的关键
3. 对测试样本，候选标签 $y$ 入集合当且仅当 $1 - p(y \mid x) \leq q$

展开得到两个条件：

| 标签 | 入集合的条件 |
|---|---|
| 阳性 (1) | $p_1 \geq 1 - q$ |
| 阴性 (0) | $1 - p_1 \geq 1 - q \iff p_1 \leq q$ |

> ⚠️ **实现陷阱**：两个条件必须**分别判断**。如果只判断"是否包含阳性"，
> 集合大小会被限制在 {1} 或 {1,0}，丢失 {0} 的情形，覆盖率会显著低于 $1-\alpha$。
> 本项目在开发过程中就踩过这个坑（实测覆盖率只有 0.61），修复后达到 0.94。

---

## 5. 可信度评分

综合评分是**用于自查的启发式汇总**，不是经验证的临床量表。权重显式给出：

$$\text{Score} = 100 \times \frac{\sum_j w_j s_j}{\sum_j w_j}$$

| 分项 | 权重 | 换算方式 |
|---|---|---|
| 校准（ICI） | 0.35 | $\text{clip}(1 - \text{ICI}/0.25, 0, 1)$ |
| 判别力（AUC） | 0.25 | $\text{clip}((\text{AUC}-0.5)/0.4, 0, 1)$ |
| 公平性（AUC 极差） | 0.25 | $\text{clip}(1 - \text{range}/0.25, 0, 1)$ |
| 拒识（AURC） | 0.15 | $\text{clip}(1 - (\text{AURC}/\text{base}-0.5)/0.5, 0, 1)$ |

**权重设计理由**：校准权重最高，因为临床决策直接依赖概率数值；公平性权重
与判别力相当，因为亚组失效属于安全事件而非统计瑕疵。

| 分数 | 等级 |
|---|---|
| ≥ 80 | A（可信） |
| 65–80 | B（基本可用，需注意提示项） |
| 50–65 | C（存在明显缺陷，需改进后使用） |
| < 50 | D（不建议直接用于临床决策） |

---

## 6. 已知局限

1. **内置数据集很小**（303–920 例）。报告的绝对数值不应外推到真实临床。
2. **校准斜率对小样本敏感**，样本 < 200 时置信区间很宽。
3. **重校准无法修正协变量分布偏移**。实测：在 Cleveland 上做 Isotonic 重校准后，
   Switzerland 上的 ICI 仅从 0.846 降到 0.841。真正的解法是在目标中心收集数据。
4. **公平性只覆盖可观测的敏感属性**。未记录的社会经济因素无法审计。
5. **可信度评分是启发式的**，不应作为监管或法律依据。

---

## 参考文献

1. Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019).
   Calibration: the Achilles heel of predictive analytics. *BMC Medicine*, 17(1), 230.
2. Steyerberg, E. W., et al. (2010). Assessing the performance of prediction models:
   a framework for traditional and novel measures. *Epidemiology*, 21(1), 128–138.
3. Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: a novel method for
   evaluating prediction models. *Medical Decision Making*, 26(6), 565–574.
4. Murphy, K. P. (1973). A new vector partition of the probability score.
   *Journal of Applied Meteorology*, 12(4), 595–600.
5. Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction
   and distribution-free uncertainty quantification. *arXiv:2107.07511*.
6. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
7. Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias
   in an algorithm used to manage the health of populations. *Science*, 366(6464), 447–453.
8. Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*.
9. Detrano, R., et al. (1989). International application of a new probability algorithm for
   the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304–310.
