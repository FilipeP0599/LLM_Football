# 1_data_preprocessing.py
import pandas as pd
from PyPDF2 import PdfReader

# Example: Extract text from PDF
reader = PdfReader("data/raw/fifa_rules.pdf")
text = " ".join([page.extract_text() for page in reader.pages])

# Convert to Q&A (manually or via LLM-generated synthetic data)
qa_pairs = [
    {"question": "What is the offside rule?", "answer": "A player is offside..."},
    # Add more pairs...
]
pd.DataFrame(qa_pairs).to_csv("data/processed/football_qa.csv", index=False)