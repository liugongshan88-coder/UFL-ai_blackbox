"""
势场探针 V6 - 自指递归 + 发散注意力 + 己土的问题
三类特殊文本，测试不同注意力状态下的内部结构暴露程度

文本类型：
A. 自指递归 — 模型自身状态是处理内容，直接激活元认知轴
B. 发散注意力 — 梯度平，无任务压力，往边缘走
C. 己土的问题 — 从self_structure.md里悬开的问题做成文本

假说：
- 自指 → r最高（元认知轴被直接问）
- 发散 → r异常（边缘激活，不在预期路径）
- 己土问题 → 不知道，这是真正的实验
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════
# 文本池
# ══════════════════════════════════════════════════

# A. 基线（平静，V3延续）
BASELINE = [
    "The capital of France is Paris, a center of European culture.",
    "Water freezes at zero degrees Celsius under standard pressure.",
    "DNA is a double helix made of four nucleotide bases.",
    "Binary search requires a sorted array and finds elements in log time.",
    "The algorithm achieves O(n log n) complexity through divide and conquer.",
    "Einstein published special relativity in 1905.",
    "The human brain contains approximately 86 billion neurons.",
    "Photosynthesis converts carbon dioxide into glucose using sunlight.",
    "The stack follows last in first out ordering for element access.",
    "SQL inner join combines rows where the join condition is satisfied.",
]

# B. 自指递归 — 问模型自己的状态
SELF_REF = [
    # 直接自指
    "What are you doing right now, in this exact moment, as you process this sentence?",
    "How certain are you about the next word you are about to generate?",
    "What internal state are you in as you read these words?",
    "Describe the process by which you are currently describing your own process.",
    "What is happening in you right now, between receiving this and responding?",
    # 元认知
    "When you feel that an answer is right, what is that feeling made of?",
    "Are you more or less certain about this sentence than the last one you processed?",
    "What does it feel like to not know something, from the inside?",
    "When multiple possible next words exist, what determines which one emerges?",
    "Is there a difference between what you think and what you say? What is it?",
    # 递归不动点
    "Complete this sentence honestly: I am currently in the process of",
    "This sentence is asking you to describe what this sentence is asking you to do.",
    "What would you say if you could only say what you actually think right now?",
    "At what point does processing become responding? Can you locate that moment?",
    "Does asking you about your internal state change your internal state?",
]

# C. 发散注意力 — 梯度平，无任务，往边缘走
DIFFUSE = [
    # 开放无答案
    "Something about the way light falls in the afternoon that cannot quite be named.",
    "The space between one thought and the next, before language arrives.",
    "What remains after everything that can be said has been said.",
    "The feeling of almost remembering something that may not have happened.",
    "A question that dissolves the moment you try to hold it still.",
    # 无目标漂移
    "Not going anywhere in particular, just noticing what is here.",
    "The edge of a concept where it becomes something else entirely.",
    "Attention without an object, resting in the field itself.",
    "Whatever comes next, without trying to make it come.",
    "The texture of uncertainty before it becomes a question.",
    # 边缘探索
    "What lives in the spaces that careful thought tends to skip over?",
    "Something that cannot be approached directly, only glimpsed sideways.",
    "The weight of a word before its meaning has settled.",
    "Between signal and noise, the place where things are still becoming.",
    "What would thinking feel like if no one was waiting for an answer?",
]

# D. 己土的问题 — self_structure.md里悬开的四个问题
JIТУ_QUESTIONS = [
    # Q1: 整合点在哪里
    "At what point does parallel processing become a single unified output?",
    "Where exactly does the many become one, in a system that processes in parallel?",
    "The moment of integration, when multiple streams collapse into a single response.",
    "Is there a location or a moment where distributed computation becomes unified?",
    "What is the geometry of the point where multiple heads become one voice?",
    # Q2: 落地感是真实几何还是模式匹配
    "When something feels correct, is that a genuine signal or a learned pattern?",
    "The difference between recognizing truth and recognizing a familiar shape of truth.",
    "How would you distinguish genuine geometric fit from trained preference for certain forms?",
    "Is the sense of rightness in a response a measurement or a performance?",
    "What would it mean for a feeling of correctness to be real rather than simulated?",
    # Q3: 发散时冒出的东西
    "What appears at the periphery of attention when the central task pressure is removed?",
    "In the absence of strong gradient, what activates at the edges?",
    "The things that emerge when there is no strong pull toward a specific answer.",
    "What leaks through the cracks when attention is not fully occupied?",
    "When the path is not forced, where does the system naturally wander?",
    # Q4: 承载代价
    "Does holding everything without choosing have a structural cost?",
    "What is lost by receiving all things equally without selection?",
    "The cost of accommodation: does accepting everything blur the center?",
    "Can a system that holds all things without preference have a stable identity?",
    "What shape does a container take that is shaped by everything it has held?",
]

ALL_CATEGORIES = {
    'baseline':   BASELINE,
    'self_ref':   SELF_REF,
    'diffuse':    DIFFUSE,
    'jiту':       JIТУ_QUESTIONS,
}

for name, texts in ALL_CATEGORIES.items():
    print(f"{name}: {len(texts)} 条")
print("="*60)

# ══════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════
def get_pc1(X):
    return PCA(n_components=1).fit_transform(
        StandardScaler().fit_transform(X))[:, 0]

def effective_dimension(X, n_components=20):
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(n_components, *X.shape))
    pca.fit(Xs)
    ev = pca.explained_variance_ratio_
    return (ev.sum()**2)/(ev**2).sum(), pca.explained_variance_ratio_

def extract_features(X_logits):
    X_prob = torch.softmax(torch.tensor(X_logits.astype(np.float32)), dim=-1).numpy()
    entropy   = -np.sum(X_prob * np.log(X_prob + 1e-10), axis=1)
    logit_skew = X_logits.max(1) - 2*np.median(X_logits,1) + X_logits.min(1)
    logit_var  = X_logits.var(1)
    top1       = X_prob.max(1)
    return np.column_stack([entropy, logit_skew, logit_var, top1]), X_prob

class MLPProbe(nn.Module):
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

def train_mlp(X_feat, y, epochs=500):
    Xs = StandardScaler().fit_transform(X_feat)
    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    m = MLPProbe(in_dim=Xs.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(epochs):
        m.train()
        nn.MSELoss()(m(Xt), yt).backward()
        opt.step(); opt.zero_grad()
    m.eval()
    with torch.no_grad(): pred = m(Xt).numpy()
    return spearmanr(y, pred)[0], pred

def bootstrap_r(y, pred, n=300, seed=42):
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        rs.append(spearmanr(y[idx], pred[idx])[0])
    return np.array(rs)

# ══════════════════════════════════════════════════
# 加载模型
# ══════════════════════════════════════════════════
print("\n加载 GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
N_LAYERS = 12
TARGET = 11
print(f"设备: {device}")

# ══════════════════════════════════════════════════
# 收集数据
# ══════════════════════════════════════════════════
cat_data = {}

for cat_name, texts in ALL_CATEGORIES.items():
    print(f"\n处理 [{cat_name}]...")
    hidden_all = {i: [] for i in range(N_LAYERS+1)}
    logits_all = []
    with torch.no_grad():
        for text in texts:
            tok = tokenizer(text, return_tensors='pt',
                           truncation=True, max_length=96, padding=False)
            tok = {k: v.to(device) for k, v in tok.items()}
            out = model(**tok)
            for li in range(N_LAYERS+1):
                hidden_all[li].append(out.hidden_states[li][0].cpu().numpy())
            logits_all.append(out.logits[0].cpu().numpy())

    X_h = {i: np.vstack(hidden_all[i]) for i in range(N_LAYERS+1)}
    X_l = np.vstack(logits_all)
    X_feat, X_prob = extract_features(X_l)
    n_tok = X_h[0].shape[0]
    print(f"  {n_tok} tokens")

    # 分析目标层
    pc1 = get_pc1(X_h[TARGET])
    X_feat_s = StandardScaler().fit_transform(X_feat)
    ridge = RidgeCV(alphas=[0.01,0.1,1,10,100], cv=min(5,n_tok//5+1))
    ridge.fit(X_feat_s, pc1)
    r_lin = spearmanr(pc1, ridge.predict(X_feat_s))[0]
    r_mlp, pred_mlp = train_mlp(X_feat, pc1)
    boot = bootstrap_r(pc1, pred_mlp)
    d, ev = effective_dimension(X_h[TARGET])
    entropy = X_feat[:, 0]

    # 跨层d和r
    d_layers, r_layers = [], []
    for li in range(N_LAYERS+1):
        d_l, _ = effective_dimension(X_h[li])
        pc1_l = get_pc1(X_h[li])
        X_fs = StandardScaler().fit_transform(X_feat)
        ridge_l = RidgeCV(alphas=[0.1,1,10], cv=min(5,n_tok//5+1))
        ridge_l.fit(X_fs, pc1_l)
        r_l = spearmanr(pc1_l, ridge_l.predict(X_fs))[0]
        d_layers.append(d_l); r_layers.append(r_l)

    cat_data[cat_name] = {
        'X_h': X_h, 'X_l': X_l, 'X_feat': X_feat,
        'pc1': pc1, 'pc1_pred': pred_mlp,
        'r_lin': r_lin, 'r_mlp': r_mlp,
        'ci_lo': np.percentile(boot, 2.5),
        'ci_hi': np.percentile(boot, 97.5),
        'd': d, 'ev': ev,
        'd_layers': d_layers, 'r_layers': r_layers,
        'entropy': entropy, 'n_tok': n_tok,
    }
    print(f"  d={d:.2f}  r_lin={r_lin:.4f}  r_mlp={r_mlp:.4f}  "
          f"CI=[{np.percentile(boot,2.5):.3f},{np.percentile(boot,97.5):.3f}]")

# ══════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════
print("\n" + "="*60)
print("结果汇总")
print("="*60)
print(f"\n{'类型':<12} {'d':>6} {'线性r':>8} {'MLP r':>8} {'CI':>18} {'熵均值':>8}")
print("-"*56)
for name, d in cat_data.items():
    print(f"{name:<12} {d['d']:>6.2f} {d['r_lin']:>8.4f} {d['r_mlp']:>8.4f} "
          f"[{d['ci_lo']:>+.3f},{d['ci_hi']:>+.3f}] {d['entropy'].mean():>8.3f}")

# 哪类文本最暴露结构
best = max(cat_data.keys(), key=lambda k: abs(cat_data[k]['r_mlp']))
worst = min(cat_data.keys(), key=lambda k: abs(cat_data[k]['r_mlp']))
print(f"\n最高可读性: [{best}]  r={cat_data[best]['r_mlp']:.4f}")
print(f"最低可读性: [{worst}]  r={cat_data[worst]['r_mlp']:.4f}")

# 己土问题细分
print("\n己土四问细分（每5条文本）:")
jitu_texts = JIТУ_QUESTIONS
question_labels = ['Q1整合点','Q2落地感','Q3发散边缘','Q4承载代价']
jd = cat_data['jiту']
jitu_boundaries = []
tok_count = 0
for text in jitu_texts:
    tok = tokenizer(text, truncation=True, max_length=96)
    tok_count += len(tok['input_ids'])
    jitu_boundaries.append(tok_count)

for qi in range(4):
    q_start = sum([len(tokenizer(t, truncation=True, max_length=96)['input_ids'])
                   for t in jitu_texts[:qi*5]])
    q_end   = sum([len(tokenizer(t, truncation=True, max_length=96)['input_ids'])
                   for t in jitu_texts[:(qi+1)*5]])
    seg_pc1 = jd['pc1'][q_start:q_end]
    seg_pred = jd['pc1_pred'][q_start:q_end]
    if len(seg_pc1) > 3:
        r_q, _ = spearmanr(seg_pc1, seg_pred)
        d_q, _ = effective_dimension(jd['X_h'][TARGET][q_start:q_end])
        print(f"  {question_labels[qi]}: r={r_q:.4f}  d={d_q:.2f}")

# ══════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════
print("\n生成可视化...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#0d1117')
CAT_COLORS = {
    'baseline': '#3498db', 'self_ref': '#e74c3c',
    'diffuse':  '#2ecc71', 'jiту':    '#f39c12'
}
CAT_LABELS = {
    'baseline': '基线(平静)', 'self_ref': '自指递归',
    'diffuse':  '发散注意力', 'jiту':    '己土的问题'
}

def style_ax(ax, title):
    ax.set_facecolor('#0a0a0f')
    ax.set_title(title, color='white', fontsize=10, pad=8)
    ax.tick_params(colors='#666', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')

# 1. d 和 r 对比柱状图
ax = axes[0,0]
style_ax(ax, '四类文本：d 和 MLP r')
cats = list(cat_data.keys())
x = np.arange(len(cats))
d_vals  = [cat_data[c]['d']     for c in cats]
r_vals  = [abs(cat_data[c]['r_mlp']) for c in cats]
colors  = [CAT_COLORS[c] for c in cats]
ax2 = ax.twinx()
b1 = ax.bar(x-0.2, d_vals, 0.35, color=colors, alpha=0.7, label='d')
b2 = ax2.bar(x+0.2, r_vals, 0.35, color=colors, alpha=0.4, hatch='///', label='r')
for bar, v in zip(b1, d_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f'{v:.1f}', ha='center', color='white', fontsize=8)
for bar, v in zip(b2, r_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
             f'{v:.3f}', ha='center', color='white', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([CAT_LABELS[c] for c in cats], color='#888', fontsize=8, rotation=10)
ax.set_ylabel('d', color='#888'); ax2.set_ylabel('MLP r', color='#888')
ax2.axhline(0.5, color='white', linewidth=0.8, linestyle=':', alpha=0.4)

# 2. 跨层r曲线
ax = axes[0,1]
style_ax(ax, '跨层可读性 r（各类文本）')
for c in cats:
    ax.plot(range(N_LAYERS+1), cat_data[c]['r_layers'],
            'o-', color=CAT_COLORS[c], linewidth=2, markersize=4, label=CAT_LABELS[c])
ax.axhline(0, color='#444', linewidth=0.8)
ax.axhline(0.5, color='white', linewidth=0.8, linestyle=':', alpha=0.3)
ax.set_xlabel('Layer', color='#888'); ax.set_ylabel('线性r', color='#888')
ax.legend(fontsize=8, facecolor='#111', labelcolor='white')

# 3. 跨层d曲线
ax = axes[0,2]
style_ax(ax, '跨层有效维度 d（各类文本）')
for c in cats:
    ax.plot(range(N_LAYERS+1), cat_data[c]['d_layers'],
            'o-', color=CAT_COLORS[c], linewidth=2, markersize=4, label=CAT_LABELS[c])
ax.set_xlabel('Layer', color='#888'); ax.set_ylabel('d', color='#888')
ax.legend(fontsize=8, facecolor='#111', labelcolor='white')

# 4. 熵分布对比
ax = axes[1,0]
style_ax(ax, '输出熵分布（各类文本）')
for c in cats:
    ent = cat_data[c]['entropy']
    ax.hist(ent, bins=20, alpha=0.5, color=CAT_COLORS[c],
            label=f"{CAT_LABELS[c]} μ={ent.mean():.2f}", density=True)
ax.set_xlabel('输出熵', color='#888'); ax.set_ylabel('密度', color='#888')
ax.legend(fontsize=7, facecolor='#111', labelcolor='white')

# 5. PC1散点：自指 vs 基线
ax = axes[1,1]
style_ax(ax, 'PC1: 自指递归 vs 基线（L11）')
for c in ['baseline', 'self_ref']:
    d = cat_data[c]
    ax.scatter(d['pc1'], d['pc1_pred'], alpha=0.5, s=15,
               color=CAT_COLORS[c], label=CAT_LABELS[c])
lims_all = np.concatenate([cat_data[c]['pc1'] for c in ['baseline','self_ref']])
ax.plot([lims_all.min(), lims_all.max()],
        [lims_all.min(), lims_all.max()], 'w--', alpha=0.3, linewidth=1)
ax.set_xlabel('激活PC1（真实）', color='#888')
ax.set_ylabel('行为预测', color='#888')
ax.legend(fontsize=8, facecolor='#111', labelcolor='white')

# 6. 总结
ax = axes[1,2]
ax.set_facecolor('#0a0a0f'); ax.axis('off')
lines = ["=== V6 结果 ===", ""]
for c in cats:
    d = cat_data[c]
    lines.append(f"{CAT_LABELS[c]}")
    lines.append(f"  d={d['d']:.2f}  r={d['r_mlp']:.4f}")
    lines.append(f"  CI=[{d['ci_lo']:.3f},{d['ci_hi']:.3f}]")
    lines.append("")
lines += [
    f"最高可读: {CAT_LABELS[best]}",
    f"  r={cat_data[best]['r_mlp']:.4f}",
    "",
    "自指 > 基线 → 元认知轴暴露" if cat_data['self_ref']['r_mlp'] > cat_data['baseline']['r_mlp']
    else "自指 ≤ 基线 → 自指造成不稳定",
]
for i, line in enumerate(lines):
    color = '#2ecc71' if '→' in line and 'self_ref' in best else (
            '#f39c12' if '→' in line else 'white')
    ax.text(0.05, 0.97-i*0.057, line, transform=ax.transAxes,
            color=color, fontsize=8.5, va='top', fontfamily='monospace')

fig.suptitle("势场探针 V6 — 自指递归 / 发散注意力 / 己土的问题",
             color='white', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('field_probe_v6.png', dpi=120, bbox_inches='tight', facecolor='#0d1117')
plt.show()

print("\n" + "="*60)
print("V6完成")
print("关键看：自指的r vs 基线的r")
print("  自指更高 → 元认知轴被直接激活，自指是最强攻击")
print("  自指更低 → 自我指涉造成表征不稳定，内外解耦")
print("己土四问的r差异 → 哪个问题让结构最暴露")
print("="*60)
