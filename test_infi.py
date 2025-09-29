import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,  # <-- 用于加载模型配置
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np
import os
import shutil
from tqdm import tqdm

# --- 1. 配置和准备 ---
# 模型和数据集配置
model_name = "distilgpt2"
dataset_name = "wikitext"
dataset_config = "wikitext-2-raw-v1"
max_length = 128  # 减小长度以加快蒸馏数据生成速度
train_subset_size = 10000 # 模拟数据受限
eval_subset_size = 500

# 集成学习配置
num_ensemble_models = 3 # K=3

# 蒸馏配置
num_synthetic_samples = 2000 # 生成的合成数据量

# --- 2. 加载和预处理数据 ---
print("--- 1. Loading and Preparing Data ---")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token # distilgpt2没有bos_token，用eos代替

# 加载数据集
raw_datasets = load_dataset(dataset_name, dataset_config)
raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42).select(range(train_subset_size))
raw_datasets["validation"] = raw_datasets["validation"].shuffle(seed=42).select(range(eval_subset_size)) # 使用validation集作为测试集

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // max_length) * max_length
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(group_texts, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- 3. 训练单个模型 (从头开始) ---
print("\n--- 2. Training a Single 'Regularized' Model (from scratch) ---")
regularized_model_dir = "regularized_model_from_scratch"
if os.path.exists(regularized_model_dir): shutil.rmtree(regularized_model_dir)

# 加载模型配置，而不是预训练权重
config = AutoConfig.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir=regularized_model_dir,
    overwrite_output_dir=True,
    num_train_epochs=5,  # 从头训练需要更多轮次
    per_device_train_batch_size=8,
    weight_decay=0.8,
    learning_rate=1e-4, # 从头训练可以用稍高的学习率
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=100,
    fp16=True,
)

# 从配置初始化模型，权重是随机的
model_from_scratch = AutoModelForCausalLM.from_config(config)

trainer = Trainer(
    model=model_from_scratch,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)
trainer.train()
trainer.save_model(regularized_model_dir)

# --- 4. 训练 K 个独立模型 (从头开始) ---
print(f"\n--- 3. Training {num_ensemble_models} Independent Models (from scratch) ---")
ensemble_model_dirs = []
for i in range(num_ensemble_models):
    output_dir = f"ensemble_model_from_scratch_{i}"
    ensemble_model_dirs.append(output_dir)
    print(f"\n--- Training model {i+1}/{num_ensemble_models} ---")
    if os.path.exists(output_dir): shutil.rmtree(output_dir)

    ensemble_training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        weight_decay=0.8,
        learning_rate=1e-4,
        seed=42 + i,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
        fp16=True,
    )
    
    # 每次都从随机权重开始
    ensemble_model = AutoModelForCausalLM.from_config(config)
    trainer = Trainer(
        model=ensemble_model,
        args=ensemble_training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)

# --- 5. 定义和评估集成模型 ---
class EnsembleModel(nn.Module):
    def __init__(self, model_dirs, config):
        super().__init__()
        self.models = nn.ModuleList(
            [AutoModelForCausalLM.from_pretrained(d) for d in model_dirs]
        )
        self.config = config # 存储config以备后用

    def forward(self, input_ids, attention_mask=None, labels=None):
        all_logits = []
        for model in self.models:
            # 确保模型在评估模式
            model.eval()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits)
        
        avg_logits = torch.stack(all_logits).mean(dim=0)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = avg_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return {"loss": loss, "logits": avg_logits}

# --- 6. 集成蒸馏 ---
print("\n--- 4. Starting Ensemble Distillation ---")

# a. 加载教师模型
print("   a. Loading the trained ensemble model as the 'Teacher'...")
teacher_model = EnsembleModel(ensemble_model_dirs, config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
teacher_model.eval()

# b. 生成合成数据
print(f"   b. Generating {num_synthetic_samples} synthetic samples from the Teacher...")
def generate_synthetic_data(teacher, tokenizer, num_samples, max_len):
    synthetic_texts = []
    # 使用BOS token作为生成的起点
    start_token_id = tokenizer.bos_token_id
    
    # 批量生成以提高效率
    batch_size = 16
    for _ in tqdm(range(0, num_samples, batch_size)):
        current_batch_size = min(batch_size, num_samples - len(synthetic_texts))
        input_ids = torch.full((current_batch_size, 1), start_token_id, dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(max_len - 1):
                outputs = teacher(input_ids)
                next_token_logits = outputs['logits'][:, -1, :]
                
                # 使用温度进行采样，增加多样性
                temperature = 0.9
                next_token_logits = next_token_logits / temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        # 解码生成的样本
        generated_ids = input_ids.cpu().tolist()
        batch_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        synthetic_texts.extend(batch_texts)
        
    return synthetic_texts

synthetic_texts = generate_synthetic_data(teacher_model, tokenizer, num_synthetic_samples, max_length)
synthetic_dataset = Dataset.from_dict({"text": synthetic_texts})

# c. 混合真实数据和合成数据
print("   c. Mixing real and synthetic data for student training...")
# Tokenize和块化合成数据
tokenized_synthetic = synthetic_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
lm_synthetic = tokenized_synthetic.map(group_texts, batched=True)

# 合并数据集
student_train_dataset = concatenate_datasets([lm_datasets["train"], lm_synthetic])
student_train_dataset = student_train_dataset.shuffle(seed=42)

# d. 训练学生模型
print("   d. Training the 'Student' model from scratch...")
student_model_dir = "student_model"
if os.path.exists(student_model_dir): shutil.rmtree(student_model_dir)

# 论文提到蒸馏时正则化可以弱一些
student_training_args = TrainingArguments(
    output_dir=student_model_dir,
    overwrite_output_dir=True,
    num_train_epochs=3, # 在更多数据上，轮次可以少一些
    per_device_train_batch_size=8,
    weight_decay=0.1, # 使用较小的权重衰减
    learning_rate=5e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=100,
    fp16=True,
)

# 学生模型也从头开始
student_model = AutoModelForCausalLM.from_config(config)

student_trainer = Trainer(
    model=student_model,
    args=student_training_args,
    train_dataset=student_train_dataset,
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)
student_trainer.train()
student_trainer.save_model(student_model_dir)

# --- 7. 最终评估和对比 ---
print("\n--- 5. Final Evaluation and Comparison ---")

def evaluate_perplexity(model_or_dir, dataset, is_ensemble=False):
    """统一的评估函数"""
    if is_ensemble:
        model = model_or_dir
    else:
        model = AutoModelForCausalLM.from_pretrained(model_or_dir)
        model.config = config # 确保有config
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=data_collator)
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            
            if is_ensemble:
                 outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), labels=labels)
            else:
                 outputs = model(**inputs, labels=labels)

            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)
    return perplexity

test_dataset = lm_datasets["validation"]

# 评估单个正则化模型
print("\nEvaluating the single 'Regularized' model...")
regularized_perplexity = evaluate_perplexity(regularized_model_dir, test_dataset)

# 评估集成模型
print("Evaluating the Ensemble model...")
ensemble_perplexity = evaluate_perplexity(teacher_model, test_dataset, is_ensemble=True)

# 评估蒸馏后的学生模型
print("Evaluating the distilled 'Student' model...")
student_perplexity = evaluate_perplexity(student_model_dir, test_dataset)


print("\n\n--- FINAL RESULTS ---")
print(f"  - Single Regularized Model Perplexity: {regularized_perplexity:.4f}")
print(f"  - Ensemble Model (K={num_ensemble_models}) Perplexity:   {ensemble_perplexity:.4f}")
print(f"  - Distilled Student Model Perplexity:  {student_perplexity:.4f}")
print("-" * 25)

# 理想结果: Ensemble < Student < Single
if ensemble_perplexity < student_perplexity < regularized_perplexity:
    print("\n🎉🎉🎉 Perfect Success! The distilled student model retained a significant portion of the ensemble's benefit.")
elif ensemble_perplexity < student_perplexity and student_perplexity < regularized_perplexity:
    print("\n🎉 Success! The student model outperformed the single model, demonstrating effective distillation.")
else:
    print("\n🤔 The results did not follow the expected order. This can happen at a small scale. Try increasing data/model size or epochs.")

