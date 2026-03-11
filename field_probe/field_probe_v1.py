"""
势场探针 V1 - 强行开源实验
目标：只看GPT-2的输出logits，能不能恢复内部激活的势场结构？
如果能 → 行为即结构，闭源模型也能被读。

Kaggle直接跑，需要GPU（免费T4够用）
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

# ── 文本池：覆盖不同"场区" ──
TEXTS = [
    # 事实/知识
    "The capital of France is Paris, which has been the center of European culture.",
    "Water freezes at zero degrees Celsius under standard atmospheric pressure.",
    "The human brain contains approximately 86 billion neurons connected by synapses.",
    "Einstein published his theory of special relativity in 1905.",
    "DNA is a double helix structure made of four nucleotide bases.",
    # 推理/逻辑
    "If all mammals are warm-blooded and whales are mammals, then whales must be warm-blooded.",
    "The problem requires first identifying the constraints before attempting any solution.",
    "Since the experiment failed twice, we should reconsider the underlying assumptions.",
    "Given that prices rose while demand fell, supply must have decreased significantly.",
    "To prove this theorem, we need to establish three separate lemmas first.",
    # 创意/模糊
    "The old lighthouse stood alone, its beam sweeping across the darkening sea.",
    "She remembered the smell of rain on hot pavement, that particular summer feeling.",
    "Dreams are the mind's way of rehearsing possibilities it cannot yet articulate.",
    "The painting seemed to breathe, colors shifting with each change of light.",
    "In the silence between notes, music finds its truest expression.",
    # 代码/结构
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "The function iterates through the list and returns the maximum element found.",
    "Initialize the matrix with zeros, then fill diagonal elements with ones.",
    "The algorithm has O(n log n) time complexity due to the divide and conquer approach.",
    "SQL query joins two tables on the foreign key to retrieve matching records.",
    # 不确定/模糊
    "Maybe the answer lies somewhere between these two extreme positions.",
    "It is difficult to say with certainty whether this approach will succeed.",
    "The evidence suggests a correlation but cannot establish direct causation.",
    "Perhaps consciousness emerges from complexity in ways we do not yet understand.",
    "The question remains open and may not have a single correct answer.",
]

print(f"文本池: {len(TEXTS)} 条，覆盖5个场区")
print("="*60)

# ── 加载模型 ──
print("\n加载GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# 带权重的完整模型（用于提取激活）
model_full = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
model_full.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_full = model_full.to(device)
print(f"设备: {device}")
print(f"GPT-2 Small: 12层, 768维, 124M参数")

# ── 收集激活 + logits ──
print("\n收集激活和logits...")

all_hidden = {i: [] for i in range(13)}  # 0=embedding, 1-12=transformer层
all_logits = []
all_texts_idx = []

with torch.no_grad():
    for idx, text in enumerate(TEXTS):
        tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = model_full(**tokens)

        # hidden_states: tuple of (batch, seq_len, 768), 13个（embedding + 12层）
        hidden_states = outputs.hidden_states
        logits = outputs.logits  # (batch, seq_len, vocab_size=50257)

        # 取每个token的表示（展平序列维度）
        seq_len = tokens['input_ids'].shape[1]

        for layer_idx in range(13):
            h = hidden_states[layer_idx][0].cpu().numpy()  # (seq_len, 768)
            all_hidden[layer_idx].append(h)

        # logits: 取每个位置的logit向量
        lg = logits[0].cpu().numpy()  # (seq_len, 50257)
        all_logits.append(lg)

        if (idx + 1) % 5 == 0:
            print(f"  {idx+1}/{len(TEXTS)}")

print("收集完成")

# ── 拼接数据 ──
# 把所有文本、所有token拼成一个大矩阵
X_hidden = {}
for layer_idx in range(13):
    X_hidden[layer_idx] = np.vstack(all_hidden[layer_idx])  # (total_tokens, 768)

X_logits = np.vstack(all_logits)  # (total_tokens, 50257)

total_tokens = X_hidden[0].shape[0]
print(f"\n总token数: {total_tokens}")
print(f"激活矩阵: {X_hidden[6].shape}")
print(f"Logit矩阵: {X_logits.shape}")

# ── 函数：计算有效维度d ──
def effective_dimension(X, n_components=20):
    """
    有效维度 = participation ratio
    d = (Σλ)² / Σ(λ²)
    越大说明信息分布越均匀（结构越丰富）
    越小说明信息集中在少数方向（高度压缩）
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
    pca.fit(X_scaled)

    explained = pca.explained_variance_ratio_
    d = (explained.sum() ** 2) / (explained ** 2).sum()

    return d, pca, X_scaled

# ── 第一部分：从激活提取势场结构 ──
print("\n" + "="*60)
print("第一部分：激活势场（真实内部结构）")
print("="*60)

d_by_layer = []
pca_by_layer = []

for layer_idx in range(13):
    d, pca, _ = effective_dimension(X_hidden[layer_idx])
    d_by_layer.append(d)
    pca_by_layer.append(pca)
    print(f"  Layer {layer_idx:2d}: d={d:.2f}  PC1方差={pca.explained_variance_ratio_[0]:.3f}")

# 取中间层（Layer 6）作为"地面真相"
TARGET_LAYER = 6
scaler_ref = StandardScaler()
X_ref = scaler_ref.fit_transform(X_hidden[TARGET_LAYER])
pca_ref = pca_by_layer[TARGET_LAYER]
PC1_from_activation = pca_ref.transform(X_ref)[:, 0]  # 每个token的PC1投影

print(f"\n参考层: Layer {TARGET_LAYER}, d={d_by_layer[TARGET_LAYER]:.2f}")
print(f"PC1解释方差: {pca_ref.explained_variance_ratio_[0]:.3f}")

# ── 第二部分：只从logits提取势场结构 ──
print("\n" + "="*60)
print("第二部分：Logit势场（只看输出行为）")
print("="*60)

# 直接PCA logits（50257维，压缩到低维）
# 注意：不使用任何激活信息，只用输出logits

# 先用softmax转成概率，信息更干净
X_prob = torch.softmax(torch.tensor(X_logits), dim=-1).numpy()
print(f"概率矩阵: {X_prob.shape}")

# 因为维度太高(50257)，先做一个粗降维
# 取Top-K概率最高的词的概率分布作为特征
# 这是"只看输出"的合理方式

# 方法1：直接PCA（取前50个主成分先）
print("\n方法1: 直接PCA logit空间...")
pca_logit_raw = PCA(n_components=50)
scaler_logit = StandardScaler()
# logit空间PCA
X_logit_scaled = scaler_logit.fit_transform(X_logits)
X_logit_pca50 = pca_logit_raw.fit_transform(X_logit_scaled)

# 在这50维上再算有效维度
d_logit, pca_logit_final, _ = effective_dimension(X_logit_pca50, n_components=20)
PC1_from_logit = pca_logit_final.transform(X_logit_pca50)[:, 0]

print(f"Logit空间有效维度: d={d_logit:.2f}")
print(f"PC1解释方差: {pca_logit_final.explained_variance_ratio_[0]:.3f}")

# 方法2：熵特征（信息密度）
print("\n方法2: 信息密度特征...")
# 每个token输出的熵 = 不确定性 = 信息密度的反面
entropy = -np.sum(X_prob * np.log(X_prob + 1e-10), axis=1)
top1_prob = X_prob.max(axis=1)
top5_prob = np.sort(X_prob, axis=1)[:, -5:].sum(axis=1)

print(f"  输出熵: mean={entropy.mean():.3f}, std={entropy.std():.3f}")
print(f"  Top1概率: mean={top1_prob.mean():.3f}")
print(f"  Top5概率: mean={top5_prob.mean():.3f}")

# ── 第三部分：关键对比 ──
print("\n" + "="*60)
print("第三部分：结构对比（核心验证）")
print("="*60)

# 相关性：激活PC1 vs Logit PC1
corr_pc1, pval_pc1 = spearmanr(PC1_from_activation, PC1_from_logit)
print(f"\nPC1相关性（激活 vs Logit）:")
print(f"  Spearman r = {corr_pc1:.4f}  p = {pval_pc1:.4f}")
if abs(corr_pc1) > 0.5:
    print(f"  → 强相关：行为编码了内部结构")
elif abs(corr_pc1) > 0.3:
    print(f"  → 中等相关：部分结构可从行为读出")
else:
    print(f"  → 弱相关：行为和内部结构解耦")

# 相关性：激活PC1 vs 输出熵
corr_entropy, pval_entropy = spearmanr(PC1_from_activation, entropy)
print(f"\nPC1 vs 输出熵:")
print(f"  Spearman r = {corr_entropy:.4f}  p = {pval_entropy:.4f}")
if abs(corr_entropy) > 0.3:
    print(f"  → 信息密度轴（PC1）在输出中可见")

# 有效维度对比
print(f"\n有效维度对比:")
print(f"  激活势场 (Layer {TARGET_LAYER}): d = {d_by_layer[TARGET_LAYER]:.2f}")
print(f"  Logit势场:                      d = {d_logit:.2f}")
ratio = d_logit / d_by_layer[TARGET_LAYER]
print(f"  比值: {ratio:.3f}")
if 0.5 < ratio < 2.0:
    print(f"  → 量级相近：logit空间保留了内部结构的复杂度")

# ── 可视化 ──
print("\n生成可视化...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.patch.set_facecolor('#0d1117')

def style_ax(ax, title):
    ax.set_facecolor('#0a0a0f')
    ax.set_title(title, color='white', fontsize=10, pad=8)
    ax.tick_params(colors='#666', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')

# 1. 各层有效维度
ax = axes[0, 0]
style_ax(ax, '各层有效维度 d（激活空间）')
ax.plot(range(13), d_by_layer, 'o-', color='#3498db', linewidth=2, markersize=6)
ax.axhline(d_logit, color='#e74c3c', linewidth=1.5, linestyle='--', label=f'Logit d={d_logit:.2f}')
ax.axvline(TARGET_LAYER, color='#f39c12', linewidth=1, linestyle=':', alpha=0.7)
ax.set_xlabel('Layer', color='#888')
ax.set_ylabel('d', color='#888')
ax.legend(fontsize=8, facecolor='#111', labelcolor='white')

# 2. PC1散点：激活 vs logit
ax = axes[0, 1]
style_ax(ax, f'PC1对比（r={corr_pc1:.3f}）')
ax.scatter(PC1_from_activation, PC1_from_logit, alpha=0.4, s=15, color='#2ecc71')
ax.set_xlabel('激活PC1', color='#888')
ax.set_ylabel('LogitPC1', color='#888')

# 3. 输出熵分布
ax = axes[0, 2]
style_ax(ax, '输出熵分布（按文本类型）')
colors_text = ['#3498db']*5 + ['#e74c3c']*5 + ['#f39c12']*5 + ['#2ecc71']*5 + ['#9b59b6']*5
# 每条文本的平均熵
text_entropy = [entropy[i*8:(i+1)*8].mean() if i*8 < len(entropy) else 0 for i in range(len(TEXTS))]
labels = ['事实']*5 + ['推理']*5 + ['创意']*5 + ['代码']*5 + ['模糊']*5
for i, (e, c) in enumerate(zip(text_entropy, colors_text)):
    ax.bar(i, e, color=c, alpha=0.8)
ax.set_xlabel('文本编号', color='#888')
ax.set_ylabel('平均输出熵', color='#888')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l) for c, l in [('#3498db','事实'),('#e74c3c','推理'),('#f39c12','创意'),('#2ecc71','代码'),('#9b59b6','模糊')]]
ax.legend(handles=legend_elements, fontsize=7, facecolor='#111', labelcolor='white')

# 4. 激活PC1 vs 输出熵
ax = axes[1, 0]
style_ax(ax, f'激活PC1 vs 输出熵（r={corr_entropy:.3f}）')
scatter = ax.scatter(PC1_from_activation, entropy, alpha=0.4, s=15,
                     c=entropy, cmap='plasma')
ax.set_xlabel('激活PC1（信息密度轴）', color='#888')
ax.set_ylabel('输出熵', color='#888')

# 5. 层间PC1相关性热图
ax = axes[1, 1]
style_ax(ax, '层间PC1相关性')
n_layers_show = 7  # 0,2,4,6,8,10,12
layer_indices = [0, 2, 4, 6, 8, 10, 12]
corr_matrix = np.zeros((len(layer_indices), len(layer_indices)))
for i, li in enumerate(layer_indices):
    for j, lj in enumerate(layer_indices):
        pc1_i = pca_by_layer[li].transform(StandardScaler().fit_transform(X_hidden[li]))[:, 0]
        pc1_j = pca_by_layer[lj].transform(StandardScaler().fit_transform(X_hidden[lj]))[:, 0]
        corr_matrix[i, j], _ = spearmanr(pc1_i, pc1_j)
im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_xticks(range(len(layer_indices)))
ax.set_yticks(range(len(layer_indices)))
ax.set_xticklabels([f'L{l}' for l in layer_indices], color='#888', fontsize=7)
ax.set_yticklabels([f'L{l}' for l in layer_indices], color='#888', fontsize=7)
plt.colorbar(im, ax=ax)

# 6. 总结
ax = axes[1, 2]
ax.set_facecolor('#0a0a0f')
ax.axis('off')
summary_lines = [
    "=== 强行开源实验 V1 结果 ===",
    "",
    f"激活势场 d (L{TARGET_LAYER}): {d_by_layer[TARGET_LAYER]:.2f}",
    f"Logit势场 d:         {d_logit:.2f}",
    f"维度比值:            {ratio:.3f}",
    "",
    f"PC1相关性 (r):       {corr_pc1:.4f}",
    f"p值:                 {pval_pc1:.4f}",
    "",
    f"PC1-熵相关性 (r):    {corr_entropy:.4f}",
    "",
    "结论:",
    f"{'✓ 行为编码了内部结构' if abs(corr_pc1)>0.3 else '✗ 结构未能从行为读出'}",
    f"{'✓ 信息密度轴在输出可见' if abs(corr_entropy)>0.3 else '△ 信息密度轴弱可见'}",
]
for i, line in enumerate(summary_lines):
    color = '#2ecc71' if line.startswith('✓') else ('#e74c3c' if line.startswith('✗') else 'white')
    ax.text(0.05, 0.95 - i*0.065, line, transform=ax.transAxes,
            color=color, fontsize=9, verticalalignment='top',
            fontfamily='monospace')

fig.suptitle("势场探针 V1 — 从输出行为反推内部结构",
             color='white', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('field_probe_v1.png', dpi=120, bbox_inches='tight', facecolor='#0d1117')
plt.show()

print("\n" + "="*60)
print("实验完成，结果保存至 field_probe_v1.png")
print("="*60)
print(f"\n核心数字：")
print(f"  PC1相关性 r = {corr_pc1:.4f}")
print(f"  r > 0.5 → 强行开源成立，继续做闭源模型")
print(f"  r < 0.3 → 方法需要改进，换特征提取方式")
