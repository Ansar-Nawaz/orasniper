import os
import json
import requests
from threading import Thread
from typing import Iterator
import gradio as gr
import torch
import re
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from PyPDF2 import PdfReader
from textblob import TextBlob  # Sentiment analysis

# Suppress FutureWarnings
warnings.simplefilter("ignore", category=FutureWarning)

# Ensure the correct version of huggingface_hub and transformers
os.system("pip install --upgrade huggingface_hub==0.24.0 transformers==4.38.2")

# GitHub Raw URL for Oracle Documentation PDFs
GITHUB_REPO_URL = "https://raw.githubusercontent.com/Ansar-Nawaz/OraDocuments/"
GITHUB_API_URL = "https://api.github.com/repos/Ansar-Nawaz/OraDocuments/contents/"

# Model Configuration
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

DESCRIPTION = """
# Oracle Sniper Chatbot
This chatbot assists in resolving Oracle database issues using AI and Oracle documentation.
"""

# Load AI Model based on hardware availability
model_id = "mistralai/Mistral-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", force_download=True)
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, force_download=True)
    DESCRIPTION += "\n<p>Running on CPU \U0001F976 This demo does not work well on CPU.</p>"

tokenizer.use_default_system_prompt = False

def get_pdf_list() -> list:
    """Fetches a list of all PDFs from the GitHub repository."""
    try:
        response = requests.get(GITHUB_API_URL)
        response.raise_for_status()
        files = response.json()
        return [file["name"] for file in files if file["name"].endswith(".pdf")]
    except Exception:
        return []

def fetch_pdf_text(pdf_name: str) -> str:
    """Fetches and extracts text from a PDF stored in the GitHub repository."""
    pdf_url = f"{GITHUB_REPO_URL}{pdf_name}"
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        
        reader = PdfReader("temp.pdf")
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        os.remove("temp.pdf")
        return text
    except Exception:
        return ""

def search_oracle_docs(query: str) -> dict:
    """Searches all available Oracle documentation PDFs for relevant information."""
    pdfs = get_pdf_list()
    for pdf in pdfs:
        text = fetch_pdf_text(pdf)
        if text and query.lower() in text.lower():
            return {"response": f"Relevant information found in {pdf}", "content": text[:1000] + "..."}
    return {"response": "No relevant Oracle documentation found in PDFs."}

def detect_error_code(message: str) -> str:
    """Extracts Oracle error codes from the user message, handling various formats."""
    match = re.search(r'\b[A-Za-z]+-\d{5}\b', message, re.IGNORECASE)
    return match.group(0) if match else ""

def analyze_sentiment(message: str) -> str:
    """Analyzes the sentiment of the user's message."""
    sentiment = TextBlob(message).sentiment.polarity
    if sentiment < -0.3:
        return "frustrated"
    elif sentiment > 0.3:
        return "positive"
    return "neutral"

def generate(
    message: str,
    chat_history: list,
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1,
) -> Iterator[str]:
    """Handles user queries, first searching Oracle PDFs before using AI model."""
    try:
        error_code = detect_error_code(message)
        sentiment = analyze_sentiment(message)
        
        oracle_json = search_oracle_docs(error_code if error_code else message)
        yield json.dumps({"response": oracle_json.get("response", ""), "content": oracle_json.get("content", ""), "sentiment": sentiment})
        
        if "Relevant information found" in oracle_json.get("response", ""):
            return

        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        for user, assistant in chat_history:
            conversation.extend([
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ])
        conversation.append({"role": "user", "content": message})

        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True)
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        input_ids = input_ids.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": tokenizer.eos_token_id
        }
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
            yield json.dumps({"response": "".join(outputs).replace("<|EOT|>", ""), "sentiment": sentiment})
    except Exception as e:
        yield json.dumps({"error": str(e)})

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    chatbot = gr.ChatInterface(generate)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=True)
