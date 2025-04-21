from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Path to the base model (if you saved it locally or want to download it)
base_model_path = 'mistralai/Mistral-7B-v0.1'  # This is the base model; change it if needed
adapter_model_path = './models/fine_tuned/adapter_model'  # Path to your adapter model

# Load the base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True)

# Load the LoRA adapter
adapter_model = PeftModel.from_pretrained(model, adapter_model_path)

# Test input text
input_text = "What is the importance of football tactics?"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")

# Generate a response from the fine-tuned model with LoRA adapter
outputs = adapter_model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)

# Decode the output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Model Response:", response)
