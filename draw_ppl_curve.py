import argparse
import os

import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from eval_ppl import auto_parse_path

# ==============================================================================
# 1. 配置部分 (请根据你的情况修改)
# ==============================================================================

parser = argparse.ArgumentParser(description='Evaluate model perplexity on benchmark datasets')
parser.add_argument('--model_paths', type=str, required=True, 
                    help='Path to model checkpoint or pretrained model name, split by ";" if multiple')
parser.add_argument('--model_name', type=str, default='gpt2',
                    help='Model architecture to use, choose from: gpt2, Qwen/Qwen2-1.5B, etc.')
parser.add_argument('--metric_name', type=str, default='arc_challenge_gpt/bits_per_byte,none',
                    help='dataset name to use, split by ";" if multiple')
parser.add_argument('--wandb_id', type=str, default=None,
                    help='WandB run id for logging')
args = parser.parse_args()

args.wandb_ids = []
args.model_paths = args.model_paths.split(";")
for model_path in args.model_paths:
    args.model_path = model_path
    args.wandb_id = "auto"
    args = auto_parse_path(args)
    args.wandb_ids.append(args.wandb_id)

# 你的W&B用户名或团队名
WANDB_ENTITY = "fengmingquan-sjtu"
# 你的W&B项目名
WANDB_PROJECT = "owm"
# 你要分析的实验ID (可以从实验页面的URL中找到)
WANDB_RUN_ID = args.wandb_id

# W&B中记录的指标名称
# 从你的附图二看，指标名是 'arc_challenge_gpt/bits_per_byte,none'
METRIC_NAME = args.metric_name.split(";")
STEP_NAME = 'train_step'

# 关键参数：每个训练step包含的token数量
# 这个值通常是 global_batch_size * sequence_length
# 例如: 如果你的全局批大小是512，序列长度是2048，那么这个值就是 512 * 2048 = 1,048,576
#TOKENS_PER_STEP = 500000
TOKENS_PER_STEP = 1

# 输出图像文件名
OUTPUT_FILENAME = os.path.join(args.model_path, "scaling_law_plot.png")


# ==============================================================================
# 2. 定义Scaling Law函数和数据获取函数
# ==============================================================================

def scaling_law_func(S, A, alpha, C):
    """
    定义要拟合的幂律函数 (Chinchilla-style scaling law for loss/BPB)
    BPB(S) = A / S^alpha + C
    S: Total training steps
    A, alpha, C: 拟合参数
    C: BPB的理论最小值 (无穷多token时)
    """
    return A * (S**(-alpha)) + C

def fetch_and_process_wandb_data(entity, project, run_id, metric_name, step_name, tokens_per_step):
    """从W&B获取数据并进行处理"""
    print(f"正在连接W&B API并获取实验: {entity}/{project}/{run_id}...")
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        #wandb.init(id=run_id, resume='must', project=project)
        #run = wandb.run
    except Exception as e:
        print(f"错误：无法获取W&B实验。请检查你的ENTITY, PROJECT, RUN_ID是否正确，并确保已登录W&B。")
        print(f"W&B API 错误信息: {e}")
        return None, None

    # 获取历史记录，指定我们需要的指标

    history = run.history(keys=[step_name,metric_name[0]])
    history = history.dropna(subset=[step_name, metric_name[0]])
    print("history.columns", history.columns)
    for m in metric_name[1:]:
        history_m = run.history(keys=[step_name, m])
        print("history_m.columns", history_m.columns)
        history_m = history_m.dropna(subset=[step_name, m])
        history = history.merge(history_m, on=step_name, how='inner')

    if history.empty:
        print(f"错误：在实验中没有找到名为 '{step_name}' 和 '{metric_name}' 的有效数据。")
        return None, None

    print(f"成功获取 {len(history)} 个数据点。")


    total_steps = history[step_name].to_numpy() 
    
    bpb_values = [history[m] for m in metric_name if m in history.columns]
    bpb_values = np.mean(np.vstack(bpb_values), axis=0)  # 如果有多个指标，取平均值
    
    mask = total_steps >= 10000
    total_steps = total_steps[mask]
    bpb_values = bpb_values[mask]
    return total_steps, bpb_values

# 用于格式化x轴刻度的辅助函数
def format_ticks(value, pos):
    if value >= 1e9:
        return f'{value/1e9:.1f}B'
    if value >= 1e6:
        return f'{value/1e6:.0f}M'
    if value >= 1e3:
        return f'{value/1e3:.0f}K'
    return str(int(value))


# ==============================================================================
# 3. 主程序
# ==============================================================================
if __name__ == "__main__":
    
    print("正在绘制图像...")
    plt.style.use('seaborn-v0_8-whitegrid') # 使用一个好看的样式
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    colors = sns.color_palette('deep', n_colors=len(args.model_paths)) 
    for i in range(len(args.model_paths)):
        model_path = args.model_paths[i]
        wandb_id = args.wandb_ids[i]
        print(f"处理模型路径: {model_path}，W&B ID: {wandb_id}")
        total_steps, bpb_values = fetch_and_process_wandb_data(
            WANDB_ENTITY, WANDB_PROJECT, wandb_id, METRIC_NAME, STEP_NAME, TOKENS_PER_STEP
        )

        if total_steps is not None and bpb_values is not None:
            # 步骤二：拟合BPB关于训练token数的曲线
            print("正在拟合scaling law曲线...")
            try:
                # 提供一个初始猜测值 [A, alpha, C]，这有助于拟合
                initial_guess = [max(bpb_values), 0.1, min(bpb_values)]
                params, covariance = curve_fit(scaling_law_func, total_steps, bpb_values, p0=initial_guess)
                A, alpha, C = params
                print("拟合成功！")
                print(f"  拟合方程: BPB(S) = {A:.4f} / S^{alpha:.4f} + {C:.4f}")
            except RuntimeError:
                print("错误：曲线拟合失败。请检查数据或调整初始猜测值。")
                A, alpha, C = None, None, None



        # 绘制原始数据点
        ax.scatter(total_steps, bpb_values, label=f'{i+1}. {model_path.split("/")[-1]}', color=colors[i], zorder=5)

        # 如果拟合成功，绘制拟合曲线
        if A is not None:
            # 创建更平滑的x轴用于绘制曲线
            x_fit = np.linspace(min(total_steps), max(total_steps), 400)
            y_fit = scaling_law_func(x_fit, A, alpha, C)
            
            fit_label = f'{i+1}. BPB(S) = {A:.2f} / S^{alpha:.2f} + {C:.2f}'
            ax.plot(x_fit, y_fit, color=colors[i], linewidth=2, label=fit_label)

    # 设置图像标题和坐标轴标签
    ax.set_title(f"Scaling Law", fontsize=16)
    ax.set_xlabel("Total Training Steps", fontsize=12)
    ax.set_ylabel("Bits Per Byte (BPB)", fontsize=12)

    # 格式化x轴，使其更易读
    ax.xaxis.set_major_formatter(FuncFormatter(format_ticks))
    plt.xticks(rotation=0)

    # 设置对数坐标轴（通常scaling law在log-log坐标下更清晰）
    # 如果你希望是线性坐标轴，注释掉下面两行
    ax.set_xscale('log')
    ax.set_yscale('log')
    # 如果使用对数坐标，需要重新格式化刻度，matplotlib会自动处理
    # 但如果数据范围很大，手动格式化可能更好
    # 如果使用对数坐标，FuncFormatter需要稍作调整或直接使用默认的科学计数法

    # 显示图例
    ax.legend(fontsize=12)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(OUTPUT_FILENAME)
    print(f"图像已保存为 '{OUTPUT_FILENAME}'")
    
    # 显示图像
    plt.show()



