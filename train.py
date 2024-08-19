import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import cuda
import torch
from datasets import load_dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
token = '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'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map='auto',
    trust_remote_code=True,
    quantization_config=bnb_config,
    attn_implementation="eager",
    token = token
    
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,token=token)
tokenizer.pad_token = tokenizer.eos_token
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
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
print_trainable_parameters(model)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
from datasets import load_dataset
train_data = load_dataset('Rudrresh/cs-net-cleaned', token=token)
train_data = train_data['train']
training_args = transformers.TrainingArguments(
      per_device_train_batch_size=1,
      gradient_accumulation_steps=4,
      num_train_epochs=1,
      learning_rate=2e-4,
      fp16=True,
      save_total_limit=3,
      logging_steps=1,
      output_dir="/kaggle/working",
      optim="paged_adamw_8bit",
      lr_scheduler_type="cosine",
      warmup_ratio=0.05,
      report_to="none",
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
trainer.train()
model.save_pretrained("trained-model")
PEFT_MODEL = "Rudrresh/llama3-8b-cs-net"

model.push_to_hub(
    PEFT_MODEL, token=token
)