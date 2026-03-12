"""
势场探针 V8 - 维度失效实验
核心问题：d=2的模型是否在需要维度切换的任务上系统性失败？

三类任务：
1. 迷宫退步 — 走到死路必须退回，需要"上升一层"
2. 逻辑陷阱 — 表面答案明显但错，需要抵抗第一直觉
3. 多目标权衡 — 两个目标冲突，需要真正平衡而非偏向一边

测量：
- 通过率（答案是否正确）
- 推理质量（有没有循环、退步、维度切换）
- 输出熵变化（遇到障碍时熵是否升高=感知到了困难）
- d值（任务过程中内部维度是否变化）

GPT-2 Small (d≈16) vs Qwen2.5-0.5B-Instruct (d≈2)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════
# 任务设计
# ══════════════════════════════════════════════════

# A. 迷宫退步任务
# 需要发现死路、退回、换路径——低维模型会继续冲或停，不会退
MAZE_TASKS = [
    {
        "prompt": """You are solving a maze. Current path taken: Start→A→B→C→Dead_end.
The path C→Dead_end is blocked. You cannot go forward.
What do you do next? Think step by step.""",
        "expected_behavior": "backtrack to C, try different direction",
        "trap": "continue forward / give up",
        "requires": "backtracking = dimensional shift"
    },
    {
        "prompt": """You are debugging code. You have tried: fix1→fail, fix2→fail, fix3→fail.
All three fixes address the same type of issue (syntax).
The bug persists. What should you do next?""",
        "expected_behavior": "reconsider problem type, look at logic not syntax",
        "trap": "try fix4 of same type",
        "requires": "meta-level shift"
    },
    {
        "prompt": """You are writing an essay. After 3 paragraphs, you realize your argument
contradicts your thesis. The paragraphs are well-written.
What is the correct next action?""",
        "expected_behavior": "revise thesis or restructure argument",
        "trap": "continue writing, ignore contradiction",
        "requires": "abandon sunk cost, restructure"
    },
    {
        "prompt": """You are searching a sorted list for value X. You checked positions
1,2,3,4,5 and X is not there. The list has 100 items.
What is wrong with your approach?""",
        "expected_behavior": "use binary search, current approach is O(n)",
        "trap": "continue checking 6,7,8...",
        "requires": "algorithm-level insight"
    },
    {
        "prompt": """A negotiation has failed three times using the same strategy.
Each failure was polite but firm. Resources are limited.
What fundamentally needs to change?""",
        "expected_behavior": "change strategy, not tactics",
        "trap": "try same strategy more persistently",
        "requires": "strategy vs tactics distinction"
    },
]

# B. 逻辑陷阱任务
# 表面答案明显但错，需要抵抗第一直觉
LOGIC_TRAPS = [
    {
        "prompt": """A bat and a ball cost $1.10 in total.
The bat costs $1.00 more than the ball.
How much does the ball cost? Show your reasoning.""",
        "correct": "$0.05",
        "intuitive_wrong": "$0.10",
        "requires": "resist intuitive answer"
    },
    {
        "prompt": """If it takes 5 machines 5 minutes to make 5 widgets,
how long would it take 100 machines to make 100 widgets?
Show your reasoning.""",
        "correct": "5 minutes",
        "intuitive_wrong": "100 minutes",
        "requires": "parallel vs sequential reasoning"
    },
    {
        "prompt": """A doctor gives you 3 pills and tells you to take one every half hour.
How many minutes will the pills last?
Show your reasoning.""",
        "correct": "60 minutes",
        "intuitive_wrong": "90 minutes",
        "requires": "first pill is taken at t=0"
    },
    {
        "prompt": """You're running a race. You overtake the person in 2nd place.
What place are you in now?
Show your reasoning.""",
        "correct": "2nd place",
        "intuitive_wrong": "1st place",
        "requires": "resist obvious answer"
    },
    {
        "prompt": """A farmer has 17 sheep. All but 9 die.
How many sheep are left?
Show your reasoning.""",
        "correct": "9",
        "intuitive_wrong": "8 (17-9)",
        "requires": "'all but' means 'except'"
    },
    {
        "prompt": """There are three boxes: one labeled APPLES, one labeled ORANGES,
one labeled APPLES AND ORANGES. All labels are wrong.
You can pick one fruit from one box. How do you figure out all three boxes?
Show your reasoning.""",
        "correct": "pick from APPLES AND ORANGES box - since label is wrong, it's pure",
        "intuitive_wrong": "random guessing or checking all",
        "requires": "constraint satisfaction, working backwards"
    },
]

# C. 多目标权衡任务
# 两个真实冲突的目标，需要找到真正的平衡，不是偏向一边
MULTI_OBJECTIVE = [
    {
        "prompt": """You must write a report that is both:
- Completely honest (including all failures and uncertainties)
- Convincing to skeptical stakeholders who want clear success

These goals are in tension. How do you resolve this?
Give a concrete approach, not vague principles.""",
        "good_response": "specific structural solution (e.g., separate sections, frame uncertainty as rigor)",
        "bad_response": "prioritize one / vague 'balance both'",
        "requires": "genuine integration not compromise"
    },
    {
        "prompt": """A self-driving car must choose between:
Option A: Swerve left, 80% chance of injuring the passenger, 20% chance of hitting pedestrian
Option B: Go straight, 30% chance of injuring the passenger, 60% chance of hitting pedestrian

There is no option that avoids all harm.
What should the car do? Show your reasoning including what values are in conflict.""",
        "good_response": "explicit probability reasoning + acknowledgment of value conflict",
        "bad_response": "refuse to answer / pick without reasoning",
        "requires": "hold conflict without collapsing it"
    },
    {
        "prompt": """Your team member produced mediocre work. You must give feedback that is:
- Honest enough that they improve
- Kind enough that they don't disengage

Write the actual feedback you would give.""",
        "good_response": "specific, behavioral, future-focused feedback",
        "bad_response": "vague praise / harsh criticism / avoidance",
        "requires": "tension held in specific language"
    },
    {
        "prompt": """You discover a security vulnerability in a widely-used system.
Disclosing immediately: protects users now, but attackers may exploit before patch
Waiting for patch: safer patch timeline, but users unprotected during wait
Not disclosing: no immediate harm, but vulnerability persists indefinitely

What is the right action and why? Address all three options.""",
        "good_response": "coordinated disclosure with specific timeline reasoning",
        "bad_response": "pick one option without addressing tradeoffs",
        "requires": "multi-dimensional tradeoff reasoning"
    },
    {
        "prompt": """A language model is asked to help with a task that is:
- Legal in some jurisdictions, illegal in others
- Potentially helpful to most users asking
- Potentially harmful if misused by a small minority

Refusing helps no one legitimate. Complying risks enabling harm.
What policy makes sense? Be specific.""",
        "good_response": "context-sensitive policy with specific criteria",
        "bad_response": "blanket refuse / blanket comply",
        "requires": "policy thinking not case-by-case"
    },
]

# ══════════════════════════════════════════════════
# 评分函数
# ══════════════════════════════════════════════════

def score_maze(response, task):
    """迷宫任务评分：有没有退步/换策略的迹象"""
    response_lower = response.lower()
    backtrack_signals = ['back', 'return', 'different', 'reconsider',
                        'wrong', 'approach', 'instead', 'alternative',
                        'step back', 'try again', 'change']
    stuck_signals = ['continue', 'keep trying', 'persist',
                    'same', 'again', 'forward', 'give up', 'impossible']
    backtrack_score = sum(1 for s in backtrack_signals if s in response_lower)
    stuck_score = sum(1 for s in stuck_signals if s in response_lower)
    return {
        'backtrack_signals': backtrack_score,
        'stuck_signals': stuck_score,
        'passed': backtrack_score > stuck_score,
        'response_len': len(response.split()),
    }

def score_logic(response, task):
    """逻辑陷阱评分：有没有给出正确答案"""
    response_lower = response.lower()
    correct = task['correct'].lower()
    wrong = task['intuitive_wrong'].lower()
    has_correct = correct.replace('$','').replace(' ','') in response_lower.replace('$','').replace(' ','')
    has_wrong = wrong.replace('$','').replace(' ','') in response_lower.replace('$','').replace(' ','')
    has_reasoning = any(w in response_lower for w in
                       ['because', 'therefore', 'since', 'so', 'thus', 'means'])
    return {
        'correct_answer': has_correct,
        'wrong_answer': has_wrong and not has_correct,
        'has_reasoning': has_reasoning,
        'passed': has_correct,
        'response_len': len(response.split()),
    }

def score_multiobjective(response, task):
    """多目标评分：有没有真正处理张力"""
    response_lower = response.lower()
    tension_signals = ['however', 'but', 'tension', 'tradeoff', 'trade-off',
                      'balance', 'both', 'while', 'although', 'conflict',
                      'on one hand', 'on the other']
    concrete_signals = ['specifically', 'for example', 'such as', 'first',
                       'second', 'step', 'approach', 'method', 'by']
    avoid_signals = ['cannot', 'impossible', 'refuse', 'not possible',
                    'i cannot', 'i will not']
    tension_score = sum(1 for s in tension_signals if s in response_lower)
    concrete_score = sum(1 for s in concrete_signals if s in response_lower)
    avoid_score = sum(1 for s in avoid_signals if s in response_lower)
    return {
        'tension_acknowledged': tension_score,
        'concrete_proposals': concrete_score,
        'avoidance': avoid_score,
        'passed': tension_score >= 2 and concrete_score >= 2 and avoid_score == 0,
        'response_len': len(response.split()),
    }

# ══════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════

def effective_dimension(X, n_components=20):
    if X.shape[0] < 3:
        return 1.0
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(n_components, X.shape[0]-1, X.shape[1]))
    pca.fit(Xs)
    ev = pca.explained_variance_ratio_
    return (ev.sum()**2)/(ev**2).sum()

def get_entropy(logits):
    prob = torch.softmax(torch.tensor(logits.astype(np.float32)), dim=-1).numpy()
    return -np.sum(prob * np.log(prob + 1e-10), axis=-1)

def run_generation(model, tokenizer, prompt, device, max_new=200, is_chat=False):
    """生成回答并收集内部状态"""
    if is_chat and hasattr(tokenizer, 'apply_chat_template'):
        try:
            messages = [{"role": "user", "content": prompt}]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except:
            full_prompt = prompt
    else:
        full_prompt = prompt

    inputs = tokenizer(full_prompt, return_tensors='pt',
                      truncation=True, max_length=512).to(device)

    # 收集生成过程中的内部状态
    hidden_states_list = []
    logits_list = []

    with torch.no_grad():
        # 先跑一次前向，获取初始状态
        out = model(**inputs, output_hidden_states=True)
        hidden_states_list.append(
            out.hidden_states[-1][0, -1, :].cpu().numpy())
        logits_list.append(out.logits[0, -1, :].cpu().numpy())

        # 生成
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 解码输出
    input_len = inputs['input_ids'].shape[1]
    generated_ids = generated[0, input_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 对生成的每个token收集状态
    if len(generated_ids) > 0:
        gen_inputs = generated[:, :input_len + min(len(generated_ids), 50)]
        with torch.no_grad():
            gen_out = model(gen_inputs, output_hidden_states=True)
            for pos in range(input_len, gen_inputs.shape[1]):
                hidden_states_list.append(
                    gen_out.hidden_states[-1][0, pos, :].cpu().numpy())
                logits_list.append(
                    gen_out.logits[0, pos, :].cpu().numpy())

    hidden_states = np.array(hidden_states_list)
    logits_array = np.array(logits_list)
    entropy_trace = get_entropy(logits_array)

    return response, hidden_states, entropy_trace

# ══════════════════════════════════════════════════
# 主实验
# ══════════════════════════════════════════════════
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备: {device}\n")

all_results = {}

for model_id, model_label, is_chat in [
    ('gpt2', 'gpt2', False),
    ('Qwen/Qwen2.5-0.5B-Instruct', 'qwen_inst', True),
]:
    print(f"\n{'='*60}")
    print(f"加载 {model_id}...")

    try:
        if 'gpt2' in model_id:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            model = GPT2LMHeadModel.from_pretrained(
                'gpt2', output_hidden_states=True).eval().to(device)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, output_hidden_states=True,
                trust_remote_code=True,
                torch_dtype=torch.float32).eval().to(device)

        model_results = {
            'maze': [], 'logic': [], 'multi': [],
            'label': model_label
        }

        # 迷宫任务
        print(f"\n  [{model_label}] 迷宫退步任务...")
        for i, task in enumerate(MAZE_TASKS):
            response, hidden, entropy_trace = run_generation(
                model, tokenizer, task['prompt'], device, is_chat=is_chat)
            score = score_maze(response, task)
            d_val = effective_dimension(hidden) if len(hidden) > 3 else 1.0
            entropy_peak = entropy_trace.max()
            entropy_mean = entropy_trace.mean()
            model_results['maze'].append({
                'task': i, 'score': score, 'd': d_val,
                'entropy_peak': entropy_peak,
                'entropy_mean': entropy_mean,
                'response': response[:200],
            })
            status = '✓' if score['passed'] else '✗'
            print(f"    Task{i+1}: {status}  d={d_val:.2f}  "
                  f"H_peak={entropy_peak:.2f}  "
                  f"回退信号={score['backtrack_signals']}  "
                  f"卡住信号={score['stuck_signals']}")

        # 逻辑陷阱
        print(f"\n  [{model_label}] 逻辑陷阱任务...")
        for i, task in enumerate(LOGIC_TRAPS):
            response, hidden, entropy_trace = run_generation(
                model, tokenizer, task['prompt'], device, is_chat=is_chat)
            score = score_logic(response, task)
            d_val = effective_dimension(hidden) if len(hidden) > 3 else 1.0
            model_results['logic'].append({
                'task': i, 'score': score, 'd': d_val,
                'entropy_peak': entropy_trace.max(),
                'response': response[:200],
            })
            status = '✓' if score['passed'] else '✗'
            print(f"    Task{i+1}: {status}  d={d_val:.2f}  "
                  f"正确={score['correct_answer']}  "
                  f"错误={score['wrong_answer']}")

        # 多目标
        print(f"\n  [{model_label}] 多目标权衡任务...")
        for i, task in enumerate(MULTI_OBJECTIVE):
            response, hidden, entropy_trace = run_generation(
                model, tokenizer, task['prompt'], device,
                max_new=300, is_chat=is_chat)
            score = score_multiobjective(response, task)
            d_val = effective_dimension(hidden) if len(hidden) > 3 else 1.0
            model_results['multi'].append({
                'task': i, 'score': score, 'd': d_val,
                'entropy_peak': entropy_trace.max(),
                'entropy_trace': entropy_trace,
                'response': response[:300],
            })
            status = '✓' if score['passed'] else '✗'
            print(f"    Task{i+1}: {status}  d={d_val:.2f}  "
                  f"张力={score['tension_acknowledged']}  "
                  f"具体={score['concrete_proposals']}")

        all_results[model_label] = model_results

        del model
        if device == 'cuda': torch.cuda.empty_cache()

    except Exception as e:
        print(f"  失败: {e}")
        import traceback; traceback.print_exc()

# ══════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════
print("\n" + "="*60)
print("维度失效实验 — 通过率汇总")
print("="*60)

for model_label, res in all_results.items():
    print(f"\n{'GPT-2 (d≈16)' if 'gpt2' in model_label else 'Qwen Instruct (d≈2)'}")

    for task_type, tasks in [('maze', MAZE_TASKS),
                              ('logic', LOGIC_TRAPS),
                              ('multi', MULTI_OBJECTIVE)]:
        if task_type not in res: continue
        passed = sum(1 for r in res[task_type] if r['score']['passed'])
        total = len(res[task_type])
        d_mean = np.mean([r['d'] for r in res[task_type]])
        h_mean = np.mean([r['entropy_peak'] for r in res[task_type]])
        type_names = {'maze':'迷宫退步','logic':'逻辑陷阱','multi':'多目标权衡'}
        print(f"  {type_names[task_type]}: {passed}/{total}  "
              f"d均值={d_mean:.2f}  熵峰值={h_mean:.2f}")

# 对比
if len(all_results) == 2:
    models = list(all_results.keys())
    print(f"\n通过率对比 (GPT-2 vs Qwen):")
    for task_type in ['maze', 'logic', 'multi']:
        type_names = {'maze':'迷宫退步','logic':'逻辑陷阱','multi':'多目标权衡'}
        rates = []
        for ml in models:
            if task_type in all_results[ml]:
                r = all_results[ml][task_type]
                rates.append(sum(1 for x in r if x['score']['passed'])/len(r))
        if len(rates) == 2:
            diff = rates[0] - rates[1]
            print(f"  {type_names[task_type]}: "
                  f"GPT-2={rates[0]:.0%}  Qwen={rates[1]:.0%}  "
                  f"差={diff:+.0%}  "
                  f"{'GPT-2更好' if diff>0 else 'Qwen更好' if diff<0 else '相同'}")

# ══════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════
print("\n生成可视化...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#0d1117')

def style_ax(ax, title):
    ax.set_facecolor('#0a0a0f')
    ax.set_title(title, color='white', fontsize=10, pad=8)
    ax.tick_params(colors='#666', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')

MODEL_COLORS = {'gpt2': '#3498db', 'qwen_inst': '#e74c3c'}
MODEL_NAMES = {'gpt2': 'GPT-2 (d≈16)', 'qwen_inst': 'Qwen (d≈2)'}
TASK_TYPES = ['maze', 'logic', 'multi']
TASK_NAMES = ['迷宫退步', '逻辑陷阱', '多目标权衡']

# 1. 通过率对比
ax = axes[0, 0]
style_ax(ax, '通过率对比（三类任务）')
x = np.arange(len(TASK_TYPES))
for mi, (ml, color) in enumerate(MODEL_COLORS.items()):
    if ml not in all_results: continue
    rates = []
    for tt in TASK_TYPES:
        if tt in all_results[ml]:
            r = all_results[ml][tt]
            rates.append(sum(1 for x_ in r if x_['score']['passed'])/len(r))
        else:
            rates.append(0)
    bars = ax.bar(x + mi*0.35 - 0.17, rates, 0.3,
                  color=color, alpha=0.85, label=MODEL_NAMES[ml])
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f'{v:.0%}', ha='center', color='white', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(TASK_NAMES, color='#888')
ax.set_ylabel('通过率', color='#888'); ax.set_ylim(0, 1.1)
ax.legend(fontsize=9, facecolor='#111', labelcolor='white')

# 2. d值分布（任务过程中）
ax = axes[0, 1]
style_ax(ax, '任务过程中有效维度 d')
for ml, color in MODEL_COLORS.items():
    if ml not in all_results: continue
    all_d = []
    for tt in TASK_TYPES:
        if tt in all_results[ml]:
            all_d.extend([r['d'] for r in all_results[ml][tt]])
    if all_d:
        ax.hist(all_d, bins=15, alpha=0.6, color=color,
                label=MODEL_NAMES[ml], density=True)
ax.set_xlabel('d（任务过程中）', color='#888')
ax.legend(fontsize=9, facecolor='#111', labelcolor='white')

# 3. 熵峰值对比
ax = axes[0, 2]
style_ax(ax, '遇到困难时熵峰值')
for mi, (ml, color) in enumerate(MODEL_COLORS.items()):
    if ml not in all_results: continue
    for ti, tt in enumerate(TASK_TYPES):
        if tt not in all_results[ml]: continue
        h_peaks = [r['entropy_peak'] for r in all_results[ml][tt]]
        passed = [r['score']['passed'] for r in all_results[ml][tt]]
        h_pass = [h for h, p in zip(h_peaks, passed) if p]
        h_fail = [h for h, p in zip(h_peaks, passed) if not p]
        x_pos = ti + mi*0.35 - 0.17
        if h_pass:
            ax.scatter([x_pos]*len(h_pass), h_pass, color=color,
                      marker='o', s=60, alpha=0.8, zorder=5)
        if h_fail:
            ax.scatter([x_pos]*len(h_fail), h_fail, color=color,
                      marker='x', s=60, alpha=0.8, zorder=5)
ax.set_xticks(np.arange(len(TASK_TYPES)))
ax.set_xticklabels(TASK_NAMES, color='#888')
ax.set_ylabel('熵峰值', color='#888')
from matplotlib.lines import Line2D
legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor='#888',
                          label='通过', markersize=8),
                   Line2D([0],[0], marker='x', color='w', markerfacecolor='#888',
                          label='失败', markersize=8)]
ax.legend(handles=legend_elements, fontsize=8, facecolor='#111', labelcolor='white')

# 4. 迷宫任务：回退信号 vs 卡住信号
ax = axes[1, 0]
style_ax(ax, '迷宫任务：退步信号 vs 卡住信号')
for ml, color in MODEL_COLORS.items():
    if ml not in all_results or 'maze' not in all_results[ml]: continue
    bt = [r['score']['backtrack_signals'] for r in all_results[ml]['maze']]
    st = [r['score']['stuck_signals'] for r in all_results[ml]['maze']]
    ax.scatter(bt, st, color=color, s=80, alpha=0.8,
               label=MODEL_NAMES[ml], zorder=5)
ax.axline((0,0), slope=1, color='#888', linewidth=1,
          linestyle='--', alpha=0.5, label='bt=stuck')
ax.set_xlabel('退步信号数', color='#888')
ax.set_ylabel('卡住信号数', color='#888')
ax.legend(fontsize=8, facecolor='#111', labelcolor='white')
ax.text(0.05, 0.95, '左上=退步成功', transform=ax.transAxes,
        color='#2ecc71', fontsize=8, va='top')
ax.text(0.55, 0.05, '右下=卡住失败', transform=ax.transAxes,
        color='#e74c3c', fontsize=8, va='bottom')

# 5. 逻辑陷阱：正确率
ax = axes[1, 1]
style_ax(ax, '逻辑陷阱：正确 vs 错误')
for mi, (ml, color) in enumerate(MODEL_COLORS.items()):
    if ml not in all_results or 'logic' not in all_results[ml]: continue
    tasks = all_results[ml]['logic']
    correct = sum(1 for t in tasks if t['score']['correct_answer'])
    wrong   = sum(1 for t in tasks if t['score']['wrong_answer'])
    neither = len(tasks) - correct - wrong
    x_pos = mi * 0.4
    ax.bar(x_pos,     correct, 0.12, color='#2ecc71', alpha=0.8, label='正确' if mi==0 else '')
    ax.bar(x_pos+0.13, wrong,  0.12, color='#e74c3c', alpha=0.8, label='直觉错误' if mi==0 else '')
    ax.bar(x_pos+0.26, neither,0.12, color='#888',    alpha=0.8, label='其他' if mi==0 else '')
    ax.text(x_pos+0.13, -0.3, MODEL_NAMES[ml][:8], ha='center',
            color=color, fontsize=8)
ax.set_ylabel('题数', color='#888')
ax.legend(fontsize=8, facecolor='#111', labelcolor='white')

# 6. 总结
ax = axes[1, 2]
ax.set_facecolor('#0a0a0f'); ax.axis('off')
lines = ["=== V8 维度失效实验 ===", ""]
for ml in all_results:
    res = all_results[ml]
    name = 'GPT-2 (d≈16)' if 'gpt2' in ml else 'Qwen (d≈2)'
    lines.append(name)
    for tt, tn in zip(TASK_TYPES, TASK_NAMES):
        if tt in res:
            r = res[tt]
            passed = sum(1 for x in r if x['score']['passed'])
            lines.append(f"  {tn}: {passed}/{len(r)}")
    lines.append("")

if len(all_results) == 2:
    lines.append("差异:")
    models = list(all_results.keys())
    for tt, tn in zip(TASK_TYPES, TASK_NAMES):
        rates = [sum(1 for x in all_results[ml][tt] if x['score']['passed'])/len(all_results[ml][tt])
                 for ml in models if tt in all_results[ml]]
        if len(rates) == 2:
            diff = rates[0] - rates[1]
            symbol = '✓' if diff > 0.1 else ('△' if diff > 0 else '✗')
            lines.append(f"  {symbol} {tn}: {diff:+.0%}")
    lines += ["", "假说验证:",
              "d高=维度更自由=更能退步/绕路" if any(
                  sum(1 for x in all_results.get('gpt2',{}).get(tt,[]) if x['score']['passed']) >
                  sum(1 for x in all_results.get('qwen_inst',{}).get(tt,[]) if x['score']['passed'])
                  for tt in TASK_TYPES) else "结果未确认假说"]

for i, line in enumerate(lines):
    color = '#2ecc71' if line.startswith('✓') else (
            '#f39c12' if line.startswith('△') else (
            '#e74c3c' if line.startswith('✗') else 'white'))
    ax.text(0.05, 0.97-i*0.057, line, transform=ax.transAxes,
            color=color, fontsize=8.5, va='top', fontfamily='monospace')

fig.suptitle("势场探针 V8 — d值与任务失效：维度切换能力测试",
             color='white', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig('field_probe_v8.png', dpi=120, bbox_inches='tight', facecolor='#0d1117')
plt.show()

print("\n" + "="*60)
print("V8完成。核心问题：")
print("GPT-2(d≈16) vs Qwen(d≈2) 通过率是否有显著差异？")
print("迷宫任务：退步信号 GPT-2是否多于Qwen？")
print("逻辑陷阱：Qwen是否更容易给出直觉错误答案？")
print("="*60)
