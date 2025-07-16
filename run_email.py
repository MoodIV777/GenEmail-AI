from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载 base 模型 + LoRA 权重
base_model_name = "Qwen/Qwen3-8B"
adapter_path = "./model/adapters"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(model, adapter_path)

# 推理示例
inputs = tokenizer("如何撰写一封 SEO 邮件给客户类型 A？", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
