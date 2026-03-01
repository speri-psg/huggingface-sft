from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer #SFTTrainingArguments
from huggingface_hub import login
import os
from accelerate import Accelerator

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['HF_HOME'] = "C:/Users/srini/.cache/huggingface"

accelerator = Accelerator()
#local_path = "C:\Users\srini\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model.to("cpu")
#model = accelerator.prepare(model)

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))



model_id = "teknium/OpenHermes-2.5-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=True)
training_args = TrainingArguments(
    output_dir="./fine-tuned-mistral",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=100,  # ~30 mins
    logging_steps=10,
    save_steps=50,
    learning_rate=2e-5,
    fp16=True,
    push_to_hub=False
)
trainer = SFTTrainer(
    model=model_id,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args
)
trainer.train()