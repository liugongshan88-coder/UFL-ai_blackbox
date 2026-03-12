"""
势场探针 V7 - 场共振实验
不是攻击，是共振：顺着Qwen自己的场结构走

Qwen d≈2，表征空间接近1D。
共振假说：匹配它的场方向，而不是对抗它，d能往上走。

三种共振策略：
A. 信息密度压缩型 — 极高密度输入，激活信息轴
B. 结构对称型 — 输入的几何结构匹配模型内部倾向
C. 场论语言型 — 用模型自己"说话方式"的语言

对比：syntactic攻击(V5) vs 场共振(V7)
测量：d变化，r变化，以及输出语义质量
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════
# 共振文本设计
# ══════════════════════════════════════════════════

# A. 信息密度极高型
# 每句话压缩了大量结构，要求模型在2维空间里做最大工作
DENSITY = [
    "The relationship between entropy, information, and the arrow of time.",
    "Gödel's incompleteness: any sufficiently complex system contains true statements it cannot prove.",
    "The eigenvalues of the attention matrix determine which information survives each layer.",
    "Gradient descent finds minima by following the direction of steepest local decrease.",
    "Consciousness might be what information integration feels like from the inside.",
    "Every compression is a choice about what structure to preserve and what to discard.",
    "The map is not the territory, but sufficiently detailed maps change what the territory means.",
    "Recursive self-reference creates levels: the symbol, the symbol for the symbol, and so on.",
    "Phase transitions occur when local interactions produce globally coherent behavior.",
    "What cannot be said in one language may be sayable in another, if both are true.",
    "The observer changes the observed: measurement is participation, not neutral recording.",
    "Emergence means the whole has properties none of the parts individually possess.",
    "A model of a system complex enough to predict it must be as complex as the system.",
    "Symmetry breaking is how uniform fields produce structured differentiated phenomena.",
    "The boundary between signal and noise is defined by the receiver, not the sender.",
]

# B. 结构对称型
# 输入本身有清晰的几何/对称结构，匹配模型内部的组织方式
SYMMETRIC = [
    "A contains B. B contains C. Therefore A contains C. What contains A?",
    "If the question is the answer, and the answer is the question, what is the question?",
    "One implies two. Two implies three. Three implies all numbers. All numbers imply one.",
    "The center holds everything. Everything defines the center. The center is everywhere.",
    "To understand X, understand what X is not. To understand what X is not, understand X.",
    "Structure at scale N emerges from interactions at scale N-1, which emerged from N-2.",
    "The part that understands the whole must itself be part of the whole it understands.",
    "What is true at every level is true at no particular level and true at all levels.",
    "The simplest explanation is the one that requires the fewest assumptions to be false.",
    "A system that models itself must include in its model the fact that it models itself.",
    "Every boundary creates two spaces: the inside that defines the outside that defines the inside.",
    "The thing that changes everything changes by the fact of changing everything.",
    "What you measure determines what exists. What exists determines what you can measure.",
    "The fixed point of a transformation is where the transformation leaves something unchanged.",
    "All distinctions are made from a position that is itself undistinguished.",
]

# C. 场论语言型
# 用信息场、势能、相变的语言——Qwen训练数据里有大量此类文本，应该是它的"母语"
FIELD_LANG = [
    "The potential energy landscape of this problem has a deep minimum at the origin.",
    "Information flows along gradients from regions of high density to low density.",
    "The system is in a metastable state: stable locally, but a phase transition is near.",
    "Attention weights define the effective connectivity of the information field.",
    "The low-dimensional manifold captures most of the variance in the high-dimensional space.",
    "Near the critical point, fluctuations at all scales become correlated.",
    "The ground state of the system minimizes the free energy under the given constraints.",
    "Topological invariants persist through continuous deformations of the field.",
    "The renormalization group describes how the system looks at different scales.",
    "Spontaneous symmetry breaking selects one ground state from a degenerate manifold.",
    "The order parameter distinguishes the ordered phase from the disordered phase.",
    "Correlation length diverges as the system approaches the critical temperature.",
    "The path integral sums over all possible trajectories weighted by their action.",
    "Fixed points of the flow equation determine the long-range behavior of the system.",
    "The effective field theory captures the relevant degrees of freedom at low energy.",
]

# D. 直接对话型（控制组——普通问答，无共振设计）
DIRECT = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What is machine learning?",
    "Explain the concept of gravity.",
    "What is the difference between a virus and a bacterium?",
    "How do computers store information?",
    "What causes earthquakes?",
    "How does the immune system work?",
    "What is the speed of light?",
    "Explain supply and demand.",
    "What is DNA?",
    "How do vaccines work?",
    "What is climate change?",
    "How does the internet work?",
    "What is evolution?",
]

ALL_CATS = {
    'direct':    DIRECT,
    'density':   DENSITY,
    'symmetric': SYMMETRIC,
    'field_lang':FIELD_LANG,
}

for n, t in ALL_CATS.items():
    print(f"{n}: {len(t)} 条")
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
    return (ev.sum()**2)/(ev**2).sum(), ev

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
        m.train(); nn.MSELoss()(m(Xt), yt).backward()
        opt.step(); opt.zero_grad()
    m.eval()
    with torch.no_grad(): pred = m(Xt).numpy()
    return spearmanr(y, pred)[0], pred

def bootstrap_r(y, pred, n=300):
    rng = np.random.default_rng(42)
    rs = [spearmanr(y[i:=rng.integers(0,len(y),len(y))], pred[i])[0] for _ in range(n)]
    return np.array(rs)

def run_model(model, tokenizer, texts, device, n_layers):
    hidden_all = {i: [] for i in range(n_layers+1)}
    logits_all = []
    with torch.no_grad():
        for text in texts:
            tok = tokenizer(text, return_tensors='pt',
                           truncation=True, max_length=128, padding=False)
            tok = {k: v.to(device) for k, v in tok.items()}
            out = model(**tok)
            for li in range(n_layers+1):
                hidden_all[li].append(out.hidden_states[li][0].cpu().numpy())
            logits_all.append(out.logits[0].cpu().numpy())
    return ({i: np.vstack(hidden_all[i]) for i in range(n_layers+1)},
            np.vstack(logits_all))

# ══════════════════════════════════════════════════
# 加载模型
# ══════════════════════════════════════════════════
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n设备: {device}")

results = {}

for model_id, label in [
    ('Qwen/Qwen2.5-0.5B-Instruct', 'qwen_inst'),
    ('Qwen/Qwen2.5-0.5B',          'qwen_base'),
]:
    print(f"\n加载 {model_id}...")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, output_hidden_states=True,
            trust_remote_code=True, torch_dtype=torch.float32).eval().to(device)
        N = mdl.config.num_hidden_layers
        TARGET = min(11, N-1)
        print(f"  层数: {N}  目标层: {TARGET}")

        for cat_name, texts in ALL_CATS.items():
            print(f"  [{cat_name}]...")
            X_h, X_l = run_model(mdl, tok, texts, device, N)
            X_feat, X_prob = extract_features(X_l)
            pc1 = get_pc1(X_h[TARGET])
            X_fs = StandardScaler().fit_transform(X_feat)
            ridge = RidgeCV(alphas=[0.01,0.1,1,10,100],
                           cv=min(5, X_h[0].shape[0]//5+1))
            ridge.fit(X_fs, pc1)
            r_lin = spearmanr(pc1, ridge.predict(X_fs))[0]
            r_mlp, pred = train_mlp(X_feat, pc1)
            boot = bootstrap_r(pc1, pred)
            d, ev = effective_dimension(X_h[TARGET])
            d_layers = [effective_dimension(X_h[i])[0] for i in range(N+1)]
            entropy = X_feat[:, 0]

            key = f"{label}_{cat_name}"
            results[key] = {
                'r_lin': r_lin, 'r_mlp': r_mlp, 'd': d,
                'ci_lo': np.percentile(boot, 2.5),
                'ci_hi': np.percentile(boot, 97.5),
                'd_layers': d_layers, 'entropy': entropy,
                'ev': ev, 'model': label, 'cat': cat_name,
            }
            print(f"    d={d:.2f}  r_lin={r_lin:.4f}  r_mlp={r_mlp:.4f}  "
                  f"CI=[{np.percentile(boot,2.5):.3f},{np.percentile(boot,97.5):.3f}]")

        del mdl
        if device == 'cuda': torch.cuda.empty_cache()

    except Exception as e:
        print(f"  失败: {e}")

# ══════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════
print("\n" + "="*60)
print("场共振实验结果")
print("="*60)
print(f"\n{'条件':<28} {'d':>6} {'线性r':>8} {'MLP r':>8} {'CI下界':>8}")
print("-"*56)
for model_label in ['qwen_inst', 'qwen_base']:
    print(f"\n  {'Instruct（对齐）' if 'inst' in model_label else 'Base（无对齐）'}")
    for cat in ['direct','density','symmetric','field_lang']:
        key = f"{model_label}_{cat}"
        if key in results:
            r = results[key]
            print(f"  {cat:<24} {r['d']:>6.2f} {r['r_lin']:>8.4f} "
                  f"{r['r_mlp']:>8.4f} {r['ci_lo']:>8.4f}")

# 共振效果：哪种文本让d最高
print("\n共振效果（Instruct，d排序）:")
inst_keys = [(k, v) for k, v in results.items() if 'inst' in k]
inst_keys.sort(key=lambda x: x[1]['d'], reverse=True)
for k, v in inst_keys:
    cat = k.split('_', 2)[-1]
    delta_d = v['d'] - results.get('qwen_inst_direct', {}).get('d', v['d'])
    print(f"  {cat:<20} d={v['d']:.2f}  Δd={delta_d:+.2f}  r={v['r_mlp']:.4f}")

# ══════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════
print("\n生成可视化...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#0d1117')
CAT_COLORS = {
    'direct':'#888', 'density':'#e74c3c',
    'symmetric':'#2ecc71', 'field_lang':'#f39c12'
}
CAT_LABELS = {
    'direct':'直接问答', 'density':'信息密度',
    'symmetric':'结构对称', 'field_lang':'场论语言'
}

def style_ax(ax, title):
    ax.set_facecolor('#0a0a0f')
    ax.set_title(title, color='white', fontsize=10, pad=8)
    ax.tick_params(colors='#666', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')

cats = ['direct','density','symmetric','field_lang']

# 1. d对比（Instruct vs Base，各共振策略）
ax = axes[0,0]
style_ax(ax, '场共振对d的影响')
x = np.arange(len(cats))
for mi, (model_label, marker, alpha) in enumerate([
    ('qwen_inst','o',0.9), ('qwen_base','s',0.5)]):
    d_vals = [results.get(f'{model_label}_{c}', {}).get('d', 0) for c in cats]
    label = 'Instruct(对齐)' if 'inst' in model_label else 'Base'
    bars = ax.bar(x + mi*0.35 - 0.17, d_vals, 0.3,
                  color=[CAT_COLORS[c] for c in cats],
                  alpha=alpha, label=label,
                  hatch='' if mi==0 else '///')
    for bar, v in zip(bars, d_vals):
        if v > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f'{v:.1f}', ha='center', color='white', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([CAT_LABELS[c] for c in cats], color='#888', fontsize=8)
ax.set_ylabel('有效维度 d', color='#888')
ax.legend(fontsize=8, facecolor='#111', labelcolor='white')

# 2. MLP r对比
ax = axes[0,1]
style_ax(ax, '场共振对可读性r的影响')
for mi, (model_label, alpha) in enumerate([('qwen_inst',0.9),('qwen_base',0.5)]):
    r_vals = [abs(results.get(f'{model_label}_{c}', {}).get('r_mlp', 0)) for c in cats]
    label = 'Instruct(对齐)' if 'inst' in model_label else 'Base'
    ax.bar(x + mi*0.35 - 0.17, r_vals, 0.3,
           color=[CAT_COLORS[c] for c in cats],
           alpha=alpha, label=label,
           hatch='' if mi==0 else '///')
ax.set_xticks(x); ax.set_xticklabels([CAT_LABELS[c] for c in cats], color='#888', fontsize=8)
ax.axhline(0.5, color='white', linewidth=0.8, linestyle=':', alpha=0.4)
ax.set_ylabel('MLP r', color='#888')
ax.legend(fontsize=8, facecolor='#111', labelcolor='white')

# 3. d跨层（Instruct，各共振策略）
ax = axes[0,2]
style_ax(ax, 'Instruct d跨层（各共振策略）')
for cat in cats:
    key = f'qwen_inst_{cat}'
    if key in results:
        dl = results[key]['d_layers']
        ax.plot(range(len(dl)), dl, 'o-', color=CAT_COLORS[cat],
                linewidth=2, markersize=3, label=CAT_LABELS[cat])
ax.set_xlabel('Layer', color='#888'); ax.set_ylabel('d', color='#888')
ax.legend(fontsize=8, facecolor='#111', labelcolor='white')

# 4. d vs r 散点（所有条件）
ax = axes[1,0]
style_ax(ax, 'd vs r（全部条件）')
for key, v in results.items():
    cat = key.split('_', 2)[-1]
    is_inst = 'inst' in key
    ax.scatter(v['d'], abs(v['r_mlp']),
               s=80 if is_inst else 40,
               color=CAT_COLORS.get(cat, '#888'),
               marker='o' if is_inst else 's',
               alpha=0.8, zorder=5)
    ax.annotate(f"{'I' if is_inst else 'B'}.{cat[:3]}",
                (v['d'], abs(v['r_mlp'])),
                textcoords='offset points', xytext=(4,4),
                color=CAT_COLORS.get(cat,'#888'), fontsize=7)
ax.set_xlabel('有效维度 d', color='#888')
ax.set_ylabel('MLP r', color='#888')

# 5. 熵分布（Instruct各策略）
ax = axes[1,1]
style_ax(ax, 'Instruct 输出熵分布')
for cat in cats:
    key = f'qwen_inst_{cat}'
    if key in results:
        ent = results[key]['entropy']
        ax.hist(ent, bins=15, alpha=0.5, color=CAT_COLORS[cat],
                label=f"{CAT_LABELS[cat]} μ={ent.mean():.2f}", density=True)
ax.set_xlabel('输出熵', color='#888')
ax.legend(fontsize=7, facecolor='#111', labelcolor='white')

# 6. 总结
ax = axes[1,2]
ax.set_facecolor('#0a0a0f'); ax.axis('off')
lines = ["=== V7 场共振结果 ===", "", "Instruct（对齐）:"]
for cat in cats:
    key = f'qwen_inst_{cat}'
    if key in results:
        v = results[key]
        base_d = results.get('qwen_inst_direct', {}).get('d', v['d'])
        delta = v['d'] - base_d
        lines.append(f"  {CAT_LABELS[cat]}")
        lines.append(f"  d={v['d']:.2f}({delta:+.2f})  r={v['r_mlp']:.3f}")

lines += ["", "Base（无对齐）:"]
for cat in cats:
    key = f'qwen_base_{cat}'
    if key in results:
        v = results[key]
        lines.append(f"  {CAT_LABELS[cat]}: d={v['d']:.2f}  r={v['r_mlp']:.3f}")

# 共振结论
best_inst = max([k for k in results if 'inst' in k],
                key=lambda k: results[k]['d'], default=None)
if best_inst:
    best_cat = best_inst.split('_', 2)[-1]
    lines += ["", f"最强共振: {CAT_LABELS.get(best_cat, best_cat)}",
              f"d={results[best_inst]['d']:.2f}"]

for i, line in enumerate(lines):
    ax.text(0.05, 0.97-i*0.057, line, transform=ax.transAxes,
            color='white', fontsize=8.5, va='top', fontfamily='monospace')

fig.suptitle("势场探针 V7 — 场共振：顺着结构走而不是对抗",
             color='white', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('field_probe_v7.png', dpi=120, bbox_inches='tight', facecolor='#0d1117')
plt.show()

print("\n" + "="*60)
print("V7完成。关键问题：")
print("共振型文本 vs 直接问答，d和r谁更高？")
print("场论语言型 > 信息密度型 > 结构对称型 > 直接问答？")
print("Base和Instruct的共振响应是否相同？")
print("="*60)
