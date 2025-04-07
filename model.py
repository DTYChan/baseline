import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# 1. 配置量化，并加载模型和分词器（模型加载时已经过量化）
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
config = AutoConfig.from_pretrained(r"C:\Users\陈大才子\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    config=config,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True
)
model = prepare_model_for_kbit_training(model)

# 2. 配置 lora，并应用到模型上
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
)

peft_model = get_peft_model(model, lora_config)

# 3. 打印模型参数
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
print_trainable_parameters(peft_model)

# 4. 加载数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_dataset = dataset["train"].select(range(20))
eval_dataset = dataset["validation"].select(range(5)) 

# 5. 数据预处理
def tokenize_function(examples):
    result = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    result["labels"] = result["input_ids"].copy()  # 确保 labels 与 input_ids 相同
    result["attention_mask"] = result["attention_mask"].copy()  # 显式生成 attention_mask
    return result

tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_datasets = eval_dataset.map(tokenize_function, batched=True)

# 6. 配置训练参数
training_args = TrainingArguments(
    output_dir="./Qwen_QLoRA",
    eval_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # 增加批量大小
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",  # 设置保存策略为每个 epoch 结束时保存
    save_total_limit=1,  # 只保存最新的一个模型
    logging_dir="./logs",  # 添加日志目录
    logging_steps=10,  # 每 10 步记录一次日志
)

# 6. 训练模型
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_eval_datasets,
)

trainer.train()
peft_model.save_pretrained("./Qwen_QLoRA")
tokenizer.save_pretrained("./Qwen_QLoRA")

# 推理示例
input_text = "I am a student at SJTU"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(peft_model.device)
# 显式传递 attention_mask
attention_mask = torch.ones_like(input_ids)

# 生成文本
output = peft_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)

# 输出结果
print(tokenizer.decode(output[0], skip_special_tokens=True))