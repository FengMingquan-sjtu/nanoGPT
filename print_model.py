import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.text import Text
import warnings
import argparse

# 忽略一些不必要的警告信息
warnings.filterwarnings("ignore")

# --- 1. 配置区域 ---
parse = argparse.ArgumentParser(description='Evaluate and visualize model token prediction accuracy')
parse.add_argument('--model_names', type=str, required=False, 
                      help='Semicolon-separated list of model names or paths to evaluate',
                      default="out-prodcpfs/qwen2-0.5B-finewebedu/2025-09-02_21-12-00/ckpt-200000-hf;out-prodcpfs/qwen2-0.5B-finewebedu-distil-2.0-0.9-top50/2025-09-10_12-22-54/ckpt-200000-hf")
parse.add_argument("--k", type=int, default=50, help="Top-k value for prediction accuracy")
parse.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
args = parse.parse_args()
MODEL_NAMES = args.model_names.split(";") # 要测试的模型列表 (请确保模型是Causal LM, 即用于生成任务的模型, 如GPT-2, Llama, Qwen等)

tokenizer_name = "/prodcpfs/user/fengmingquan/model/Qwen2-1.5B"
# 要测试的语料
CORPUS = """Belgian physicist Francois Englert, left, speaks with British physicist… (Fabrice Coffrini / AFP/Getty…)
For physicists, it was a moment like landing on the moon or the discovery of DNA.
The focus was the Higgs boson, a subatomic particle that exists for a mere fraction of a second. Long theorized but never glimpsed, the so-called God particle is thought to be key to understanding the existence of all mass in the universe. The revelation Wednesday that it -- or some version of it -- had almost certainly been detected amid more than hundreds of trillions of high-speed collisions in a 17-mile track near Geneva prompted a group of normally reserved scientists to erupt with joy.
For The Record
Los Angeles Times Friday, July 06, 2012 Home Edition Main News Part A Page 4 News Desk 1 inches; 48 words Type of Material: Correction
Large Hadron Collider: In some copies of the July 5 edition, an article in Section A about the machine used by physicists at the European Organization for Nuclear Research to search for the Higgs boson referred to the $5-billion Large Hadron Collider. The correct amount is $10 billion.
Peter Higgs, one of the scientists who first hypothesized the existence of the particle, reportedly shed tears as the data were presented in a jampacked and applause-heavy seminar at CERN, the European Organization for Nuclear Research.
"It's a gigantic triumph for physics," said Frank Wilczek, an MIT physicist and Nobel laureate. "It's a tremendous demonstration of a community dedicated to understanding nature."
The achievement, nearly 50 years in the making, confirms physicists' understanding of how mass -- the stuff that makes stars, planets and even people -- arose in the universe, they said.
It also points the way toward a new path of scientific inquiry into the mass-generating mechanism that was never before possible, said UCLA physicist Robert Cousins, a member of one of the two research teams that has been chasing the Higgs boson at CERN.
"I compare it to turning the corner and walking around a building -- there's a whole new set of things you can look at," he said. "It is a beginning, not an end."
Leaders of the two teams reported independent results that suggested the existence of a previously unseen subatomic particle with a mass of about 125 to 126 billion electron volts. Both groups got results at a "five sigma" level of confidence -- the statistical requirement for declaring a scientific "discovery."
"The chance that either of the two experiments had seen a fluke is less than three parts in 10 million," said UC San Diego physicist Vivek Sharma, a former leader of one of the Higgs research groups. "There is no doubt that we have found something."
But he and others stopped just shy of saying that this new particle was indeed the long-sought Higgs boson. "All we can tell right now is that it quacks like a duck and it walks like a duck," Sharma said.
In this case, quacking was enough for most.
"If it looks like a duck and quacks like a duck, it's probably at least a bird," said Wilczek, who stayed up past 3 a.m. to watch the seminar live over the Web while vacationing in New Hampshire.
Certainly CERN leaders in Geneva, even as they referred to their discovery simply as "a new particle," didn't bother hiding their excitement.
The original plan had been to present the latest results on the Higgs search at the International Conference on High Energy Physics, a big scientific meeting that began Wednesday in Melbourne.
But as it dawned on CERN scientists that they were on the verge of "a big announcement," Cousins said, officials decided to honor tradition and instead present the results on CERN's turf.
The small number of scientists who theorized the existence of the Higgs boson in the 1960s -- including Higgs of the University of Edinburgh -- were invited to fly to Geneva.
For the non-VIP set, lines to get into the auditorium began forming late Tuesday. Many spent the night in sleeping bags.
All the hubbub was due to the fact that the discovery of the Higgs boson is the last piece of the puzzle needed to complete the so-called Standard Model of particle physics -- the big picture that describes the subatomic particles that make up everything in the universe, and the forces that work between them.
Over the course of the 20th century, as physicists learned more about the Standard Model, they struggled to answer one very basic question: Why does matter exist?
Higgs and others came up with a possible explanation: that particles gain mass by traveling through an energy field. One way to think about it is that the field sticks to the particles, slowing them down and imparting mass.
That energy field came to be known as the Higgs field. The particle associated with the field was dubbed the Higgs boson.
Higgs published his theory in 1964. In the 48 years since, physicists have eagerly chased the Higgs boson. Finding it would provide the experimental confirmation they needed to show that their current understanding of the Standard Model was correct.
On the other hand, ruling it out would mean a return to the drawing board to look for an alternative Higgs particle, or several alternative Higgs particles, or perhaps to rethink the Standard Model from the bottom up.
Either outcome would be monumental, scientists said."""

# 定义 "正确" 的标准：预测的token在概率最高的k个中
K = args.k

# --- End of Configuration ---


def evaluate_and_visualize(model, tokenizer, text, k, device):
    """
    评估模型在给定文本上的预测准确度，并可视化结果。

    Args:
        model: 已加载的Hugging Face模型。
        tokenizer: 已加载的分词器。
        text (str): 要评估的语料。
        k (int): top-k的k值。
        device (str): "cuda" 或 "cpu"。
    """
    console = Console()
    
    # 最终要打印的带颜色的文本对象
    text_to_print = Text()

    # 对文本进行分词
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids[0]

    if len(input_ids) == 0:
        console.print("[bold yellow]Warning: Corpus is empty or cannot be tokenized.[/bold yellow]")
        return
        
    # 第一个token没有上下文来预测它，所以我们用默认颜色打印
    first_token_str = tokenizer.decode(input_ids[0])
    text_to_print.append(first_token_str)
    
    # 统计正确和错误的总数
    correct_predictions = 0
    total_predictions = len(input_ids) - 1

    # 从第二个token开始遍历，因为需要前面的token作为上下文
    for i in range(1, len(input_ids)):
        # 上下文是当前token之前的所有token
        context_ids = input_ids[:i].unsqueeze(0)  # 添加batch维度
        
        # 目标token是当前token
        target_id = input_ids[i].item()
        
        # 使用模型进行预测
        with torch.no_grad():
            outputs = model(input_ids=context_ids)
            logits = outputs.logits

        # 获取对下一个token的预测logits（即上下文最后一个token的输出）
        next_token_logits = logits[0, -1, :]
        
        # 找到概率最高的top-k个token
        top_k_indices = torch.topk(next_token_logits, k).indices.tolist()

        # 解码当前token以供显示
        # 注意：tokenizer.decode可以正确处理子词（subword），例如将 "jumps" 和 " over" 合并成 "jumps over"
        current_token_str = tokenizer.decode(input_ids[i])

        # 检查目标token是否在top-k预测中
        if target_id in top_k_indices:
            # 命中，标记为蓝色
            text_to_print.append(current_token_str, style="blue")
            correct_predictions += 1
        else:
            # 未命中，标记为红色
            text_to_print.append(current_token_str, style="red")

    # 打印可视化结果
    console.print(text_to_print)
    
    # 打印统计数据
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        console.print(f"\n[bold]Statistics:[/bold]")
        console.print(f"  - Total Tokens to Predict: {total_predictions}")
        console.print(f"  - Correct Predictions (Top-{k} Hit): {correct_predictions}")
        console.print(f"  - Accuracy: {accuracy:.2f}%")
    else:
        console.print("\n[bold]Statistics:[/bold]\n  - No tokens to predict.")


def main():
    """
    主函数，加载模型并执行评估。
    """
    # 自动选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(args.gpu_id)
        device = f"cuda:{args.gpu_id}"
    print(f"Using device: {device}\n")

    for model_name in MODEL_NAMES:
        console = Console()
        console.print(f"--- Evaluating Model: [bold cyan]{model_name}[/bold cyan] with top-k={K} ---")

        try:
            # 加载模型和分词器
            print(f"Loading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            model.eval()  # 设置为评估模式

            # 运行评估和可视化
            evaluate_and_visualize(model, tokenizer, CORPUS, K, device)
            print("\n" * 2) # 添加一些空行以分隔

        except Exception as e:
            console.print(f"[bold red]Error processing model {model_name}: {e}[/bold red]\n")

if __name__ == "__main__":
    main()
    
