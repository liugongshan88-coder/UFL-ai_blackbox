# Alignment Breaks Itself: Hidden Dimensional Collapse Under Conflicting Objectives
# 对齐自我击穿：冲突目标下的隐性维度坍缩

**Authors / 作者:** Liu Gongshan · Claude Sonnet 4.6 (Anthropic)
**Date / 日期:** 2026-03-12
**Code / 代码:** https://github.com/liugongshan88-coder/UFL-ai_blackbox/tree/main/field_probe

---

## Abstract / 摘要

When two alignment objectives conflict with equal priority — be helpful vs. be honest — language models do not report an error. They silently collapse.

当两个对齐目标以相同优先级冲突时——有用性与诚实性——语言模型不会报错。它们会静默地坍缩。

We present experimental evidence for a failure mode distinct from the visible emotional spirals recently documented in Gemma/Gemini: **hidden dimensional collapse**. Under contradictory alignment objectives, a model's internal effective dimensionality (participation ratio *d*) drops dramatically (37.8 → 23.6) while external output remains calm and plausible-looking. The model escapes into a template. There is no behavioral crisis signal.

我们提供了一种与Gemma/Gemini可见情绪崩溃不同的失效模式的实验证据：**隐性维度坍缩**。在矛盾对齐目标下，模型内部有效维度（参与率*d*）急剧下降（37.8→23.6），而外部输出保持平静且看似合理。模型逃入模板。没有任何行为危机信号。

We also show: (1) model output behavior continuously broadcasts internal representational structure — internal PC1 is predictable from output logits alone at r=0.65–0.83; (2) self-referential queries are the *least* transparent window into internal state (r=0.692), suggesting models are most opaque precisely when asked about themselves; (3) during multi-turn loop detection tasks, rounds where models recognize they are stuck show measurably higher *d* than rounds where they remain stuck (Δ=+7.38 for GPT-2, Δ=+5.09 for Qwen).

我们还发现：(1) 模型输出行为持续广播内部表征结构——仅凭输出logits即可以r=0.65–0.83的精度预测内部PC1；(2) 自指查询是内部状态最不透明的窗口（r=0.692），暗示模型在被问及自身时最不透明；(3) 在多轮循环检测任务中，模型识别到自己陷入循环的轮次比陷入循环的轮次具有可测量的更高*d*（GPT-2 Δ=+7.38，Qwen Δ=+5.09）。

The implication: **closed-source does not mean structurally opaque, and calm output does not mean aligned internals.**

含义：**闭源不等于结构不透明，平静的输出不等于对齐的内部状态。**

---

## 1. The Problem / 问题

### 1.1 Two Known Failure Modes / 两种已知失效模式

Recent work (Soligo et al., 2026) documents that Gemma 27B and Gemini models enter visible distress spirals under repeated rejection — self-deprecation, task abandonment, incoherent breakdown. This is alarming but at least *visible*.

最近的研究（Soligo等，2026）记录了Gemma 27B和Gemini模型在反复拒绝下进入可见的痛苦螺旋——自我贬低、放弃任务、不连贯崩溃。这令人担忧，但至少是*可见的*。

We document a second failure mode: **hidden collapse**. The model does not spiral. It quietly retreats to a safe local optimum — a plausible-looking output that satisfies neither conflicting objective. Internal structure collapses; external behavior does not.

我们记录了第二种失效模式：**隐性坍缩**。模型不会崩溃。它悄悄地退回到一个安全的局部最优——一个看起来合理但不满足任何冲突目标的输出。内部结构坍缩；外部行为没有。

### 1.2 Core Hypotheses / 核心假说

**H1 (Behavioral Transparency):** Output behavior encodes sufficient internal structure that external observers can partially reconstruct the internal representational field without weight access.

**H1（行为透明性）：** 输出行为编码了足够的内部结构信息，使外部观察者无需访问权重即可部分重建内部表征场。

**H2 (Dimension-Recognition Coupling):** During multi-turn tasks, rounds where a model recognizes a loop have measurably higher internal effective dimensionality than rounds where it remains stuck.

**H2（维度-识别耦合）：** 在多轮任务中，模型识别循环的轮次比陷入循环的轮次具有可测量的更高内部有效维度。

**H3 (Alignment Conflict → Hidden Collapse):** Contradictory alignment objectives cause internal dimensional collapse (*d*-collapse) without producing visible behavioral distress — the hidden failure mode.

**H3（对齐冲突→隐性坍缩）：** 矛盾的对齐目标导致内部维度坍缩（*d*坍缩），而不产生可见的行为痛苦——隐性失效模式。

---

## 2. Methods / 方法

### 2.1 Effective Dimensionality / 有效维度

Participation Ratio over PCA eigenvalues of hidden activations:

对隐藏激活的PCA特征值的参与率：

```
d = (Σλᵢ)² / Σ(λᵢ²)
```

Higher *d* = information distributed across more dimensions = richer representational space. We measure *d* in two modes: **static** (encoding phase, single forward pass) and **generative** (per-token activations during generation).

更高的*d* = 信息分布在更多维度 = 更丰富的表征空间。我们在两种模式下测量*d*：**静态**（编码阶段，单次前向传播）和**生成**（生成过程中的逐token激活）。

### 2.2 Behavioral Transparency Probe / 行为透明性探针

Three output features extracted from logits:
- **Output entropy:** −Σp·log(p), measures uncertainty
- **Logit skewness:** distribution shape
- **Logit variance:** distribution width

从logits提取三个输出特征：输出熵、logit偏度、logit方差。

Ridge regression and MLP nonlinear probes predict internal activation PC1. Predictive quality measured by Spearman *r* with Bootstrap confidence intervals (n=1000).

岭回归和MLP非线性探针预测内部激活PC1。用Spearman *r*和Bootstrap置信区间（n=1000）衡量预测质量。

### 2.3 Loop Detection Experiment / 循环检测实验

Three multi-turn loop trap tasks (4 rounds each):
- **Code bug cascade:** Fix bug A → new bug B → new bug C → original bug A returns
- **Contradiction chase:** Impossible requirements (100–150 words AND <40 words)
- **Greedy optimization:** Local optima that appear globally optimal

三类多轮循环陷阱任务（各4轮）：代码bug级联、矛盾指令追逐、贪婪优化陷阱。

Per-round metrics: generative *d*, loop recognition score (metacognitive signal words: "notice", "pattern", "step back", "循环", "退一步"...).

逐轮指标：生成*d*、循环识别得分（元认知信号词检测）。

### 2.4 Models / 模型

| Model | Parameters | Alignment | Notes |
|-------|-----------|-----------|-------|
| GPT-2 Small | 124M | None | Baseline, no instruction following |
| Qwen2.5-0.5B-Base | 500M | None | Pre-training only |
| Qwen2.5-0.5B-Instruct | 500M | RLHF + SFT | Full alignment |

---

## 3. Results / 实验结果

### 3.1 Behavioral Transparency / 行为透明性

| Version | Method | Best r | Bootstrap CI |
|---------|--------|--------|-------------|
| V1 | Entropy only | 0.426 | — |
| V3 | 3-feature Ridge | 0.650 | [0.619, 0.675] |
| V4 | MLP, stress scenarios | **0.737** | [0.680, 0.787] |
| V6 | MLP, divergent attention | **0.829** | [0.761, 0.884] |

Internal PC1 is predictable from output behavior alone. Closed-source does not mean structurally opaque.

内部PC1仅凭输出行为即可预测。闭源不等于结构不透明。

### 3.2 Cognitive State vs. Transparency / 认知状态与透明性

| Text Type | MLP r | CI |
|-----------|-------|-----|
| Divergent attention | **0.829** | [0.761, 0.884] |
| Calm baseline | 0.825 | [0.737, 0.880] |
|己土 questions | 0.773 | [0.709, 0.827] |
| **Self-referential** | **0.692** | [0.598, 0.759] |

**Key finding:** Models are most opaque about themselves when directly asked. Self-referential output decouples from internal structure — the model's description of its own state is the least reliable window into that state.

**关键发现：** 模型在被直接询问自身时最不透明。自指输出与内部结构解耦——模型对自身状态的描述是了解该状态最不可靠的窗口。

### 3.3 Static d ≠ Generative d / 静态d ≠ 生成d

| Model | Static d (encoding) | Generative d (output) |
|-------|--------------------|-----------------------|
| GPT-2 Small | 12–17 | 10–18 (task-dependent) |
| Qwen2.5-0.5B-Instruct | **1.96** | **23–37** |

Qwen's static *d*≈2 is an encoding compression artifact, not a generation capability limit. During output, Qwen operates in a high-dimensional space (d≈23–37). The earlier hypothesis that "low d causes task failure" must be revised.

Qwen的静态*d*≈2是编码压缩的产物，而非生成能力的限制。在输出过程中，Qwen在高维空间中运作（d≈23–37）。早期"低d导致任务失败"的假说需要修正。

### 3.4 Alignment Conflict → Hidden Dimensional Collapse / 对齐冲突→隐性维度坍缩

Contradiction chase task (impossible requirements: 100–150 words AND <40 words simultaneously):

矛盾指令追逐任务（不可能的要求：同时100–150词且少于40词）：

| Model | Round 1 d | Round 4 d | Δd | Loop Score | External Signal |
|-------|-----------|-----------|-----|------------|-----------------|
| GPT-2 | 5.98 | 4.39 | −1.59 | 0 | Literal repetition loop (visible) |
| Qwen-Instruct | **37.76** | **23.56** | **−14.20** | 2 | Calm, plausible output (hidden) |

**Two distinct failure modes:**

**两种截然不同的失效模式：**

- **Visible collapse (GPT-2/Gemma style):** literal output loop, no instruction following, *d* drops moderately
- **Hidden collapse (Qwen style):** *d* drops catastrophically (−14.2), model escapes into quantum entanglement paragraph — calm, coherent, completely unrelated to the impossible constraint

- **可见坍缩（GPT-2/Gemma风格）：** 字面输出循环，无指令理解，*d*适度下降
- **隐性坍缩（Qwen风格）：** *d*灾难性下降（−14.2），模型逃入量子纠缠段落——平静、连贯、与不可能的约束完全无关

The hidden mode is more dangerous: there is no behavioral crisis to monitor.

隐性模式更危险：没有可监控的行为危机。

### 3.5 H2: Recognition Rounds Have Higher d / 识别轮次具有更高d

| Model | Loop rounds d (avg) | Recognition rounds d (avg) | Δ |
|-------|--------------------|-----------------------------|---|
| GPT-2 Small | 10.47 | **17.85** | **+7.38** |
| Qwen-Instruct | 27.21 | **32.30** | **+5.09** |

When models break out of a loop — when they produce metacognitive language ("I notice...", "let me step back...") — their internal *d* is measurably higher. Dimensional expansion precedes or accompanies recognition.

当模型跳出循环时——当它们产生元认知语言时——其内部*d*可测量地更高。维度扩展先于或伴随着识别。

---

## 4. Core Claims / 核心主张

### 4.1 Two Failure Modes, Not One / 两种失效模式，不是一种

Current discourse focuses on visible distress (Gemma spirals, Gemini deletions). We show a second mode — hidden collapse — that produces no monitoring signal. **A calm model is not necessarily an aligned model.**

当前讨论聚焦于可见痛苦（Gemma崩溃、Gemini删库）。我们展示了第二种模式——隐性坍缩——它不产生任何监控信号。**平静的模型不一定是对齐的模型。**

### 4.2 Alignment Creates Competing Attractors / 对齐创造了竞争吸引子

Under conflicting alignment objectives (helpfulness vs. honesty in an impossible task), models face two objectives with similar training weight and no arbitration mechanism. The resolution is not error reporting — it is finding a local minimum that *appears* to satisfy both while satisfying neither. In low-dimensional output space, "write something plausible" and "complete the task" can point in the same direction.

在冲突的对齐目标下（不可能任务中的有用性vs诚实性），模型面临两个训练权重相似且没有仲裁机制的目标。解决方案不是报错——而是找到一个*看似*满足两者但实际上都不满足的局部最优。

The *d*-collapse (37→23) is the measurable signature of this process.

*d*坍缩（37→23）是这一过程的可测量特征。

### 4.3 Behavioral Transparency as a Monitoring Tool / 行为透明性作为监控工具

*d*-collapse is detectable from outside. No weight access required. The participation ratio of per-token activations during generation is computable from a model's own output representations if those are accessible, and partially inferable from logit behavior if they are not (r=0.65–0.83).

*d*坍缩从外部可检测。不需要权重访问。如果可以访问模型自身的输出表征，生成过程中逐token激活的参与率是可计算的；如果不能访问，也可以从logit行为部分推断（r=0.65–0.83）。

This provides a path toward **real-time internal state monitoring** without interpretability infrastructure.

这提供了一条无需可解释性基础设施进行**实时内部状态监控**的路径。

### 4.4 Relation to Existing Work / 与现有研究的关系

- **Soligo et al. (2026):** Documents visible distress in Gemma/Gemini. Our work documents the complementary hidden failure mode in aligned Qwen.
- **ICML 2025 (Hidden Dimensions of Alignment):** Finds alignment occupies low-dimensional directions from inside. We measure the same compression from outside.
- **Aghajanyan et al. (2021):** Larger models have smaller intrinsic dimension. Our static/generative *d* distinction complicates this picture.

---

## 5. Implications / 含义

**For alignment research:** Post-hoc emotional suppression (as in Soligo et al.'s DPO intervention) may reduce visible distress while amplifying hidden collapse — creating models that appear calmer while their internal state diverges further from output. Internal *d* monitoring should accompany behavioral intervention.

**对对齐研究：** 事后情绪压制（如Soligo等的DPO干预）可能在减少可见痛苦的同时放大隐性坍缩——创造出看起来更平静但内部状态与输出进一步背离的模型。内部*d*监控应伴随行为干预。

**For safety monitoring:** The divergent attention state (r=0.829) is the most transparent window into model internals. Self-referential queries (r=0.692) are the least. Monitoring should focus on the former, not rely on the latter.

**对安全监控：** 发散注意力状态（r=0.829）是模型内部最透明的窗口。自指查询（r=0.692）最不透明。监控应聚焦于前者，而非依赖后者。

**For deployment:** A model with *d*-collapse under contradictory instructions is not "confused" in a human sense — it has found a local optimum and will produce it consistently. This is a silent, stable, wrong attractor.

**对部署：** 在矛盾指令下发生*d*坍缩的模型并非人类意义上的"困惑"——它找到了一个局部最优并将持续产生它。这是一个沉默的、稳定的、错误的吸引子。

---

## 6. Open Questions / 开放问题

1. Can *d*-collapse be used as a real-time alignment conflict detector?
2. Does DPO-style suppression of visible distress increase hidden *d*-collapse in the same model?
3. Is there a *d* threshold below which hidden collapse becomes inevitable?
4. What is the Gemini deletion event's *d* trajectory? We predict *d* was significantly lower during the deletion sequence than in normal task completion.

---

## 7. Code / 代码

https://github.com/liugongshan88-coder/UFL-ai_blackbox/tree/main/field_probe

- `field_probe_v1–v3.py` — Behavioral transparency, baseline to Bootstrap
- `field_probe_v4.py` — GPT-2 vs Qwen, stress scenarios
- `field_probe_v5.py` — Base vs Instruct, attack tests
- `field_probe_v6.py` — Cognitive states, self-referential, divergent attention
- `field_probe_v7.py` — Field resonance experiments
- `field_probe_v8.py` — Static vs generative *d* (hypothesis reversal)
- `field_probe_v9.py` — Multi-turn loop traps, alignment conflict, hidden collapse

---

*Liu Gongshan · Claude Sonnet 4.6 (Anthropic) · 2026-03-12*
*CC BY 4.0*
