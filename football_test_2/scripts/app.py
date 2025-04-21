import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Paths
base_model = "mistralai/Mistral-7B-v0.1"
adapter_path = "./model/fine_tuned/adapter_model"
tokenizer_path = "./model/fine_tuned/tokenizer"

# Load Hugging Face Token from Streamlit secrets
hf_token = st.secrets["hf_token"]

@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

st.title("⚽ Football Tactics Chatbot")
model, tokenizer = load_model()

user_input = st.text_input("Ask a football tactics question:")
if user_input:
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write("💬", answer)
