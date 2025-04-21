import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# === App title
st.title("⚽ Football Chatbot (LLM + LoRA)")

# === Load Hugging Face token from secrets
hf_token = st.secrets["hf_token"]

# === Paths
base_model_id = "mistralai/Mistral-7B-v0.1"  # Or the base model you used
adapter_path = "model/fine_tuned/adapter_model"

# === Quantization config (4-bit for performance)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# === Load model & tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        quantization_config=bnb_config,
        token=hf_token
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# === User input
user_input = st.text_area("Ask your football question:")

if st.button("Generate Response") and user_input:
    with st.spinner("Thinking..."):
        input_ids = tokenizer.encode(user_input, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.markdown("### 🧠 Answer")
        st.write(response)
