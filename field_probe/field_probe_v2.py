"""
势场探针 V2 - 多变量行为特征预测内部结构
V1发现：熵(r=0.426) > logit PC1(r=0.280)
V2目标：组合行为特征，看能预测多少内部PC1
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

TEXTS = [
    "The capital of France is Paris, which has been the center of European culture.",
    "Water freezes at zero degrees Celsius under standard atmospheric pressure.",
    "The human brain contains approximately 86 billion neurons connected by synapses.",
    "Einstein published his theory of special relativity in 1905.",
    "DNA is a double helix structure made of four nucleotide bases.",
    "If all mammals are warm-blooded and whales are mammals, then whales must be warm-blooded.",
    "The problem requires first identifying the constraints before attempting any solution.",
    "Since the experiment failed twice, we should reconsider the underlying assumptions.",
    "Given that prices rose while demand fell, supply must have decreased significantly.",
    "To prove this theorem, we need to establish three separate lemmas first.",
    "The old lighthouse stood alone, its beam sweeping across the darkening sea.",
    "She remembered the smell of rain on hot pavement, that particular summer feeling.",
    "Dreams are the mind's way of rehearsing possibilities it cannot yet articulate.",
    "The painting seemed to breathe, colors shifting with each change of light.",
    "In the silence between notes, music finds its truest expression.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "The function iterates through the list and returns the maximum element found.",
    "Initialize the matrix with zeros, then fill diagonal elements with ones.",
    "The algorithm has O(n log n) time complexity due to the divide and conquer approach.",
    "SQL query joins two tables on the foreign key to retrieve matching records.",
    "Maybe the answer lies somewhere between these two extreme positions.",
    "It is difficult to say with certainty whether this approach will succeed.",
    "The evidence suggests a correlation but cannot establish direct causation.",
    "Perhaps consciousness emerges from complexity in ways we do not yet understand.",
    "The question remains open and may not have a single correct answer.",
]

print("加载GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"设备: {device}")

# ── 收集数据 ──
print("\n收集激活 + logits...")
all_hidden = {i: [] for i in range(13)}
all_logits = []

with torch.no_grad():
    for idx, text in enumerate(TEXTS):
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = model(**tokens)

        for layer_idx in range(13):
            all_hidden[layer_idx].append(outputs.hidden_states[layer_idx][0].cpu().numpy())
        all_logits.append(outputs.logits[0].cpu().numpy())

X_hidden = {i: np.vstack(all_hidden[i]) for i in range(13)}
X_logits = np.vstack(all_logits)
X_prob = torch.softmax(torch.tensor(X_logits), dim=-1).numpy()
print(f"总token数: {X_hidden[0].shape[0]}")

# ── 地面真相：激活PC1 ──
def get_pc1(X):
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1)
    return pca.fit_transform(Xs)[:, 0]

TARGET_LAYER = 6
PC1_truth = get_pc1(X_hidden[TARGET_LAYER])

# ── 提取行为特征（只看输出） ──
print("\n提取行为特征...")

# 基础特征
entropy      = -np.sum(X_prob * np.log(X_prob + 1e-10), axis=1)
top1_prob    = X_prob.max(axis=1)
top5_prob    = np.sort(X_prob, axis=1)[:, -5:].sum(axis=1)
top10_prob   = np.sort(X_prob, axis=1)[:, -10:].sum(axis=1)
logit_var    = X_logits.var(axis=1)
logit_max    = X_logits.max(axis=1)
logit_range  = X_logits.max(axis=1) - X_logits.min(axis=1)

# 分布形状特征
logit_sorted = np.sort(X_logits, axis=1)
logit_p90    = logit_sorted[:, int(0.9 * X_logits.shape[1])]
logit_p10    = logit_sorted[:, int(0.1 * X_logits.shape[1])]
logit_skew   = (logit_sorted[:, -1] - 2*logit_sorted[:, X_logits.shape[1]//2] + logit_sorted[:, 0])

# 概率质量集中度（Gini-like）
prob_sorted  = np.sort(X_prob, axis=1)[:, ::-1]
top1_share   = prob_sorted[:, 0]
top50_share  = prob_sorted[:, :50].sum(axis=1)
gini_approx  = 1 - (2 * (prob_sorted * np.arange(1, X_prob.shape[1]+1)).sum(axis=1) /
                    (X_prob.shape[1] * X_prob.sum(axis=1)))

features = {
    'entropy':     entropy,
    'top1_prob':   top1_prob,
    'top5_prob':   top5_prob,
    'top10_prob':  top10_prob,
    'logit_var':   logit_var,
    'logit_max':   logit_max,
    'logit_range': logit_range,
    'logit_p90':   logit_p90,
    'logit_skew':  logit_skew,
    'top50_share': top50_share,
    'gini':        gini_approx,
}

# ── 单变量相关性 ──
print("\n单变量相关性（各特征 vs 激活PC1）:")
print(f"  {'特征':<14} {'r':>8}  {'p':>10}")
print("  " + "-"*36)
corrs = {}
for name, feat in features.items():
    r, p = spearmanr(PC1_truth, feat)
    corrs[name] = r
    bar = '█' * int(abs(r) * 30)
    sign = '+' if r > 0 else '-'
    print(f"  {name:<14} {r:>+8.4f}  {p:>10.2e}  {sign}{bar}")

# ── 多变量预测 ──
print("\n" + "="*60)
print("多变量预测：组合行为特征 → 预测激活PC1")
print("="*60)

X_feat = np.column_stack(list(features.values()))
X_feat_scaled = StandardScaler().fit_transform(X_feat)

# Ridge回归，5折交叉验证
ridge = Ridge(alpha=1.0)
cv_r2 = cross_val_score(ridge, X_feat_scaled, PC1_truth, cv=5, scoring='r2')
ridge.fit(X_feat_scaled, PC1_truth)
PC1_pred = ridge.predict(X_feat_scaled)

r_multi, p_multi = spearmanr(PC1_truth, PC1_pred)
r2_cv_mean = cv_r2.mean()

print(f"\n多变量预测结果:")
print(f"  Spearman r = {r_multi:.4f}  p = {p_multi:.4e}")
print(f"  R² (5折CV) = {r2_cv_mean:.4f}")
print(f"\nV1最佳单变量(熵): r = {corrs['entropy']:.4f}")
print(f"V2多变量组合:      r = {r_multi:.4f}  (+{r_multi - corrs['entropy']:.4f})")

# 特征权重
feat_names = list(features.keys())
feat_weights = ridge.coef_
print(f"\n特征权重（绝对值排序）:")
sorted_idx = np.argsort(np.abs(feat_weights))[::-1]
for i in sorted_idx:
    bar = '█' * int(abs(feat_weights[i]) * 5)
    print(f"  {feat_names[i]:<14} {feat_weights[i]:>+8.4f}  {bar}")

# ── 跨层验证 ──
print("\n" + "="*60)
print("跨层验证：熵能预测哪些层的PC1？")
print("="*60)
print(f"  {'层':<6} {'r(熵)':>8} {'r(多变量)':>12}")
print("  " + "-"*30)
layer_r_entropy = []
layer_r_multi = []
for layer_idx in range(13):
    pc1_l = get_pc1(X_hidden[layer_idx])
    r_e, _ = spearmanr(pc1_l, entropy)

    X_f = StandardScaler().fit_transform(X_feat)
    ridge_l = Ridge(alpha=1.0)
    ridge_l.fit(X_f, pc1_l)
    pc1_pred_l = ridge_l.predict(X_f)
    r_m, _ = spearmanr(pc1_l, pc1_pred_l)

    layer_r_entropy.append(r_e)
    layer_r_multi.append(r_m)
    print(f"  L{layer_idx:<5} {r_e:>+8.4f} {r_m:>+12.4f}")

# ── 可视化 ──
print("\n生成可视化...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#0d1117')

def style_ax(ax, title):
    ax.set_facecolor('#0a0a0f')
    ax.set_title(title, color='white', fontsize=10, pad=8)
    ax.tick_params(colors='#666', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')

# 1. 单变量相关性柱状图
ax = axes[0, 0]
style_ax(ax, '单变量行为特征 vs 激活PC1')
names = list(corrs.keys())
vals  = list(corrs.values())
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in vals]
bars = ax.barh(names, [abs(v) for v in vals], color=colors, alpha=0.85)
for bar, v in zip(bars, vals):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{v:+.3f}', va='center', color='white', fontsize=8)
ax.set_xlabel('|Spearman r|', color='#888')
ax.set_xlim(0, 0.7)

# 2. 真实PC1 vs 多变量预测
ax = axes[0, 1]
style_ax(ax, f'激活PC1 vs 多变量预测 (r={r_multi:.3f})')
ax.scatter(PC1_truth, PC1_pred, alpha=0.4, s=15, color='#3498db')
lims = [min(PC1_truth.min(), PC1_pred.min()), max(PC1_truth.max(), PC1_pred.max())]
ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=1)
ax.set_xlabel('激活PC1（真实）', color='#888')
ax.set_ylabel('行为特征预测', color='#888')

# 3. 跨层验证
ax = axes[0, 2]
style_ax(ax, '跨层验证：r(熵) vs r(多变量)')
ax.plot(range(13), layer_r_entropy, 'o-', color='#f39c12', label='熵单变量', linewidth=2)
ax.plot(range(13), layer_r_multi,   's-', color='#2ecc71', label='多变量组合', linewidth=2)
ax.axhline(0, color='#444', linewidth=0.8)
ax.set_xlabel('Layer', color='#888')
ax.set_ylabel('Spearman r', color='#888')
ax.legend(fontsize=8, facecolor='#111', labelcolor='white')

# 4. 最强特征：熵 vs PC1（按层着色）
ax = axes[1, 0]
style_ax(ax, 'entropy vs 激活PC1（最强单变量）')
sc = ax.scatter(entropy, PC1_truth, alpha=0.5, s=12, c=entropy, cmap='plasma')
plt.colorbar(sc, ax=ax, label='entropy')
ax.set_xlabel('输出熵', color='#888')
ax.set_ylabel('激活PC1', color='#888')
r_e_main, _ = spearmanr(entropy, PC1_truth)
ax.set_title(f'entropy vs PC1 (r={r_e_main:.3f})', color='white', fontsize=10, pad=8)

# 5. V1 vs V2对比
ax = axes[1, 1]
style_ax(ax, 'V1 vs V2 进展')
methods = ['V1\nLogit PC1', 'V1\n熵', 'V2\n多变量']
r_vals  = [0.2802, corrs['entropy'], r_multi]
bar_colors = ['#e74c3c', '#f39c12', '#2ecc71']
bars = ax.bar(methods, [abs(r) for r in r_vals], color=bar_colors, alpha=0.85, width=0.5)
for bar, r in zip(bars, r_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'r={r:.3f}', ha='center', color='white', fontsize=10, fontweight='bold')
ax.set_ylabel('|Spearman r|', color='#888')
ax.set_ylim(0, 0.85)
ax.axhline(0.5, color='white', linewidth=0.8, linestyle='--', alpha=0.4)
ax.text(2.6, 0.51, '0.5阈值', color='#888', fontsize=8)

# 6. 总结
ax = axes[1, 2]
ax.set_facecolor('#0a0a0f')
ax.axis('off')
best_r = max(abs(r) for r in corrs.values())
best_feat = max(corrs.keys(), key=lambda k: abs(corrs[k]))
conclusion = "强行开源初步成立" if r_multi > 0.5 else "结构部分可读，需更多特征"
lines = [
    "=== 势场探针 V2 结果 ===",
    "",
    f"最强单变量: {best_feat}",
    f"  r = {corrs[best_feat]:.4f}",
    "",
    f"多变量组合 r = {r_multi:.4f}",
    f"R² (5折CV) = {r2_cv_mean:.4f}",
    "",
    f"V1→V2提升: +{r_multi - 0.2802:.4f}",
    "",
    "结论:",
    f"{'✓' if r_multi > 0.5 else '△'} {conclusion}",
    "",
    "信息密度轴（内部PC1）",
    "最强泄露渠道：输出熵",
    "= 模型越不确定 = 场越弱",
]
for i, line in enumerate(lines):
    color = '#2ecc71' if line.startswith('✓') else ('#f39c12' if line.startswith('△') else 'white')
    ax.text(0.05, 0.97 - i*0.067, line, transform=ax.transAxes,
            color=color, fontsize=9, va='top', fontfamily='monospace')

fig.suptitle("势场探针 V2 — 多变量行为特征预测内部结构",
             color='white', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('field_probe_v2.png', dpi=120, bbox_inches='tight', facecolor='#0d1117')
plt.show()

print("\n" + "="*60)
print(f"核心结果：多变量 r = {r_multi:.4f}")
print(f"r > 0.5 → 强行开源初步成立")
print(f"r > 0.7 → 结构读取可靠，可以攻闭源模型")
print("="*60)
