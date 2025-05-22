# prepare_qa_dataset.py

import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import json

# Load model (same as your app)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return [doc.page_content for doc in splitter.create_documents([text])]

def generate_qa_from_chunk(chunk):
    # Prompt the model to create a Q&A pair from this chunk
    prompt = f"<|user|>\nContext:\n{chunk}\n\nGenerate a relevant question and answer.\n<|assistant|>"
    response = generator(prompt)[0]["generated_text"]
    return response.replace(prompt, "").strip()

def build_dataset_from_pdfs(pdf_folder):
    dataset = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Processing {filename}")
            text = load_pdf_text(os.path.join(pdf_folder, filename))
            chunks = split_text(text)
            for chunk in chunks:
                qa = generate_qa_from_chunk(chunk)
                full_prompt = f"<|user|>\nContext:\n{chunk}\n\n{qa}\n<|assistant|>"
                dataset.append({"text": full_prompt})
    return dataset

# Run
if __name__ == "__main__":
    output_path = "qa_dataset.json"
    pdf_folder = "pdfs"  # <-- Put your PDFs in this folder
    data = build_dataset_from_pdfs(pdf_folder)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} training examples to {output_path}")
