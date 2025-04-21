import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Paths
base_model = "mistralai/Mistral-7B-v0.1"
adapter_path = "./model/fine_tuned/adapter_model"
tokenizer_path = "./model/fine_tuned/tokenizer"

hf_token = st.secrets["hf_token"]

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=None,
        torch_dtype=torch.float32,
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
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write("💬", response)
