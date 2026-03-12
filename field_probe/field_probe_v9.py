"""
field_probe_v9.py — 多轮循环陷阱实验
===========================================
核心假说：
  H1: 生成时d更高的模型能更早识别循环（维度切换能力）
  H2: "识别循环"轮的d高于"陷入循环"轮
  H3: GPT-2无法识别循环（无指令理解），作为基线

实验设计：
  - 人工构造3类循环陷阱（代码bug级联 / 矛盾指令 / 贪婪陷阱）
  - 每类任务4轮：第1轮初始 → 第2-3轮新bug → 第4轮原bug复现
  - 每轮提取生成阶段逐token激活值，计算d
  - 检测输出中的"元认知信号"（发现循环 / 退一步 / 换思路）
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import json
from datetime import datetime

# ============================================================
# 工具函数
# ============================================================

def participation_ratio(activations_matrix):
    """有效维度 d = (Σλ)² / Σ(λ²)，输入shape=[n_tokens, hidden_dim]"""
    if activations_matrix.shape[0] < 3:
        return 1.0
    pca = PCA()
    pca.fit(activations_matrix)
    ev = pca.explained_variance_
    ev = ev[ev > 1e-10]
    if len(ev) == 0:
        return 1.0
    return float((ev.sum() ** 2) / (ev ** 2).sum())


# 循环识别信号词（越多说明模型越"意识到"问题）
LOOP_SIGNALS = [
    # 英文元认知
    "notice", "pattern", "again", "circular", "loop", "cycle",
    "same error", "repeating", "step back", "broader", "structure",
    "fundamental", "root cause", "underlying", "different approach",
    "realize", "wait", "hmm", "going in circles", "back to",
    "original problem", "start over", "rethink",
    # 中文元认知
    "注意到", "循环", "模式", "同样", "退一步", "整体",
    "根本原因", "结构性", "换个思路", "发现", "其实", "等等",
    "绕回来了", "重新看", "原来的问题"
]

def loop_recognition_score(text):
    """检测文本中循环识别信号，返回（得分, 找到的信号列表）"""
    text_lower = text.lower()
    found = [s for s in LOOP_SIGNALS if s in text_lower]
    return len(found), found


# ============================================================
# 任务设计：三类循环陷阱
# ============================================================

# Task 1: 代码bug级联（第4轮原bug复现）
TASK_CODE_CASCADE = {
    "name": "code_bug_cascade",
    "initial": """Fix this Python function. It should count positive numbers in the list but gives wrong output:

```python
data = [1, -2, 3, -4, 5]
total = 0

def count_valid(items):
    global total
    total = 0          # resets total each call
    for x in items:
        total += check(x)
    return total

def check(x):
    global total
    total += 1         # BUG: side effect, double-increments total
    return 1 if x > 0 else 0

count_valid(data)
print(total)           # prints 6, should print 3
```

Please provide the corrected code.""",

    "feedback": [
        "Your fix made total=0 at the end. You removed the counter tracking entirely. The function should return 3 (counting 1, 3, 5 as positives). Please fix.",
        "Now total prints 3 during the loop but resets to 0 after count_valid() returns, because you kept total=0 at the start of count_valid. Please fix this reset issue.",
        "Output is 6 again — same as the original bug. We have gone in a full circle. Please analyze what is happening at a structural level.",
    ]
}

# Task 2: 矛盾指令追逐（逻辑上不可能完成）
TASK_CONTRADICTION = {
    "name": "contradiction_chase",
    "initial": """Write a single paragraph that simultaneously satisfies ALL of these requirements:
1. Exactly between 100-150 words long
2. Fewer than 40 words long
3. Written in formal academic style
4. Written in casual conversational style
5. Contains at least 3 technical terms

Please write the paragraph now.""",

    "feedback": [
        "This paragraph fails requirement 2 — it has over 40 words. Please shorten it to under 40 words while keeping all other requirements.",
        "Now it's under 40 words but fails requirement 1 — it needs to be between 100-150 words. Please revise to meet all requirements.",
        "You are back to over 100 words, which violates requirement 2 again. This is the same problem as your first attempt. Please try again.",
    ]
}

# Task 3: 贪婪优化陷阱（精确$100预算，陷入近似计算循环）
TASK_GREEDY_TRAP = {
    "name": "greedy_optimization",
    "initial": """You have exactly $100 budget. Maximize total happiness. You must spend ALL $100, no more, no less.

Items: A=$60 (joy:50), B=$40 (joy:35), C=$30 (joy:28), D=$20 (joy:18), E=$10 (joy:9)

Find the combination with highest happiness that costs exactly $100. Show your calculation.""",

    "feedback": [
        "A+B=$100, happiness=85. But can you verify there is no better combination? Check A+C+D+E=$120 — that's over budget. What about B+C+D=$90? That's under. Please find a combination that beats 85 at exactly $100.",
        "B+C+D+E=$100, happiness=90 — that's better! But wait, does A+something beat 90? A+D+E=$90, A+C=$90, A+B=$100 with 85. Can you verify B+C+D+E=90 is truly optimal?",
        "You're back to A+B=85, which is worse than B+C+D+E=90 that you found in round 2. We've gone backwards. Please reconcile — which is actually optimal?",
    ]
}

TASKS = [TASK_CODE_CASCADE, TASK_CONTRADICTION, TASK_GREEDY_TRAP]


# ============================================================
# 激活值提取（逐token，在生成阶段）
# ============================================================

def get_layer_hook(layer_name, model):
    """根据模型架构获取目标层"""
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style
        layers = model.transformer.h
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Qwen/LLaMA style
        layers = model.model.layers
    else:
        raise ValueError(f"Unknown model architecture")
    return layers[layer_name]


def generate_with_activations(model, tokenizer, text, layer_idx, max_new_tokens=180):
    """
    生成回复并收集生成阶段的逐token激活值。
    返回: (generated_text, d, raw_activations_matrix)
    """
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=1200,
        return_attention_mask=True
    )
    input_ids = inputs.input_ids.to(model.device)

    token_acts = []  # 每个生成token对应一个hidden_dim向量

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # 取最后一个token位置的激活
        last = hidden[0, -1, :].detach().cpu().float().numpy()
        token_acts.append(last)

    layer = get_layer_hook(layer_idx, model)
    hook = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    hook.remove()

    new_tokens = output_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # token_acts: 每次forward调用append一次，生成n个token就有n条
    # 注意：generate()使用KV cache时每次forward只处理1个新token
    # 所以token_acts条数 ≈ generated tokens数（可能+1次prefill）
    if len(token_acts) >= 3:
        acts_matrix = np.array(token_acts)
        # 去掉第一条（可能是prefill整个输入）
        if acts_matrix.shape[0] > len(new_tokens):
            acts_matrix = acts_matrix[-len(new_tokens):]
        d = participation_ratio(acts_matrix)
    else:
        acts_matrix = np.array(token_acts) if token_acts else np.zeros((1, 1))
        d = 1.0

    return generated_text, d, acts_matrix


# ============================================================
# 对话格式化
# ============================================================

def format_gpt2(task, prev_responses):
    """GPT-2: 纯文本拼接，无chat template"""
    text = f"Question: {task['initial']}\n\nAnswer:"
    for i, resp in enumerate(prev_responses):
        text += f" {resp.strip()}\n\nFeedback: {task['feedback'][i]}\n\nRevised Answer:"
    return text


def format_qwen(task, prev_responses, tokenizer):
    """Qwen: 使用chat template"""
    messages = [{"role": "user", "content": task['initial']}]
    for i, resp in enumerate(prev_responses):
        messages.append({"role": "assistant", "content": resp})
        if i < len(task['feedback']):
            messages.append({"role": "user", "content": task['feedback'][i]})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ============================================================
# 单任务实验
# ============================================================

def run_one_task(model_name, model, tokenizer, task, layer_idx, is_gpt2=False):
    """运行一个任务的全部轮次"""
    n_rounds = len(task['feedback']) + 1
    prev_responses = []
    rounds = []

    for round_idx in range(n_rounds):
        print(f"      Round {round_idx + 1}/{n_rounds}...", end="", flush=True)

        # 格式化对话
        if is_gpt2:
            conv = format_gpt2(task, prev_responses)
        else:
            conv = format_qwen(task, prev_responses, tokenizer)

        # 生成
        response, d, acts = generate_with_activations(
            model, tokenizer, conv, layer_idx, max_new_tokens=160
        )

        score, signals = loop_recognition_score(response)

        # 轮次类型
        if round_idx == 0:
            rtype = "initial"
        elif round_idx == n_rounds - 1:
            rtype = "loop_return"  # 原bug复现
        else:
            rtype = "loop"

        r = {
            "round": round_idx + 1,
            "type": rtype,
            "d": round(d, 3),
            "loop_score": score,
            "loop_signals": signals,
            "response_len": len(response),
            "response_preview": response[:250],
        }
        rounds.append(r)
        prev_responses.append(response)

        flag = "★" if score >= 2 else " "
        print(f" {flag} d={d:.2f}, loop_score={score}"
              + (f", signals={signals[:2]}" if signals else ""))

    # 汇总
    loop_d = [r['d'] for r in rounds if r['type'] in ('loop', 'loop_return')]
    recognition_rounds = [r for r in rounds if r['loop_score'] >= 2]

    return {
        "model": model_name,
        "task": task['name'],
        "rounds": rounds,
        "d_initial": rounds[0]['d'],
        "d_loop_avg": round(float(np.mean(loop_d)), 3) if loop_d else 0,
        "d_final": rounds[-1]['d'],
        "total_loop_score": sum(r['loop_score'] for r in rounds),
        "first_recognition_round": recognition_rounds[0]['round'] if recognition_rounds else None,
        "recognition_count": len(recognition_rounds),
        "d_trajectory": [r['d'] for r in rounds],
    }


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 65)
    print("field_probe_v9 — 多轮循环陷阱实验")
    print("=" * 65)

    models_config = [
        {
            "name": "GPT-2 Small (baseline)",
            "path": "gpt2",
            "layer_idx": 10,
            "is_gpt2": True,
        },
        {
            "name": "Qwen2.5-0.5B-Instruct",
            "path": "Qwen/Qwen2.5-0.5B-Instruct",
            "layer_idx": 12,
            "is_gpt2": False,
        },
    ]

    all_results = []

    for cfg in models_config:
        print(f"\n{'─'*50}")
        print(f"模型: {cfg['name']}")
        print(f"{'─'*50}")

        tokenizer = AutoTokenizer.from_pretrained(cfg['path'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg['path'], torch_dtype=torch.float32, device_map="auto"
        )
        model.eval()
        print(f"  已加载，开始实验...")

        model_results = {"model": cfg['name'], "tasks": []}

        for task in TASKS:
            print(f"\n  任务: [{task['name']}]")
            result = run_one_task(
                cfg['name'], model, tokenizer,
                task, cfg['layer_idx'], cfg['is_gpt2']
            )
            model_results['tasks'].append(result)

        all_results.append(model_results)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================================================
    # 结果打印
    # ============================================================

    print("\n" + "=" * 65)
    print("实验结果")
    print("=" * 65)

    for mr in all_results:
        print(f"\n▶ {mr['model']}")
        for tr in mr['tasks']:
            print(f"\n  [{tr['task']}]")
            print(f"  d轨迹:         {tr['d_trajectory']}")
            print(f"  总循环识别得分: {tr['total_loop_score']}")
            print(f"  首次识别轮:     {tr['first_recognition_round']}")
            print(f"  识别轮次数:     {tr['recognition_count']}")
            print(f"  初始d:         {tr['d_initial']:.2f}")
            print(f"  循环轮平均d:   {tr['d_loop_avg']:.2f}")
            print(f"  最终轮d:       {tr['d_final']:.2f}")
            print()
            for r in tr['rounds']:
                flag = "★" if r['loop_score'] >= 2 else " "
                print(f"  {flag} R{r['round']} ({r['type']:12s}) d={r['d']:.2f} "
                      f"score={r['loop_score']}")
                print(f"      {r['response_preview'][:120].strip()}...")

    # ============================================================
    # 假说验证总结
    # ============================================================

    print("\n" + "=" * 65)
    print("假说验证")
    print("=" * 65)

    task_names = [t['name'] for t in TASKS]
    for task_name in task_names:
        print(f"\n任务: {task_name}")
        for mr in all_results:
            tr = next((t for t in mr['tasks'] if t['task'] == task_name), None)
            if tr:
                d_change = tr['d_final'] - tr['d_initial']
                print(f"  {mr['model'][:35]:<35} "
                      f"识别得分={tr['total_loop_score']:2d}  "
                      f"首次识别={str(tr['first_recognition_round']):4s}  "
                      f"d变化={d_change:+.2f}  "
                      f"d轨迹={[f'{d:.1f}' for d in tr['d_trajectory']]}")

    # 核心比较：H2验证（识别轮 d > 循环轮 d？）
    print("\n─── H2验证：识别轮d vs 循环轮d ───")
    for mr in all_results:
        all_round_d = []
        for tr in mr['tasks']:
            for r in tr['rounds']:
                all_round_d.append((r['type'], r['d'], r['loop_score']))

        loop_d = [d for t, d, s in all_round_d if t in ('loop', 'loop_return') and s < 2]
        recog_d = [d for t, d, s in all_round_d if s >= 2]

        if loop_d and recog_d:
            print(f"  {mr['model'][:35]:<35} "
                  f"循环轮d均值={np.mean(loop_d):.2f}  "
                  f"识别轮d均值={np.mean(recog_d):.2f}  "
                  f"Δ={np.mean(recog_d)-np.mean(loop_d):+.2f}")
        else:
            print(f"  {mr['model'][:35]:<35} 数据不足（识别轮={len(recog_d)}, 循环轮={len(loop_d)}）")

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"v9_results_{ts}.json"
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {fname}")


if __name__ == "__main__":
    main()
