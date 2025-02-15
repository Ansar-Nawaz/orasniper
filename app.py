import os
import json
import requests
from threading import Thread
from typing import Iterator
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from PyPDF2 import PdfReader

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
if torch.cuda.is_available():
    model_id = "deepseek-ai/deepseek-coder-6.7b-instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False
else:
    DESCRIPTION += "\n<p>Running on CPU \U0001F976 This demo does not work well on CPU.</p>"


def get_pdf_list() -> list:
    """Fetches a list of all PDFs from the GitHub repository."""
    try:
        response = requests.get(GITHUB_API_URL)
        response.raise_for_status()
        files = response.json()
        return [file["name"] for file in files if file["name"].endswith(".pdf")]
    except Exception as e:
        return []


def fetch_pdf_text(pdf_name: str) -> str:
    """Fetches and extracts text from a PDF stored in the GitHub repository."""
    pdf_url = f"{GITHUB_REPO_URL}{pdf_name}"
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        # Save the PDF temporarily
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        
        # Extract text from the PDF
        reader = PdfReader("temp.pdf")
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        # Remove temporary file
        os.remove("temp.pdf")
        return text
    except Exception as e:
        return ""


def search_oracle_docs(query: str) -> str:
    """Searches all available Oracle documentation PDFs for relevant information."""
    pdfs = get_pdf_list()
    for pdf in pdfs:
        text = fetch_pdf_text(pdf)
        if text and query.lower() in text.lower():
            return json.dumps({"response": f"Relevant information found in {pdf}", "content": text[:1000] + "..."})
    return json.dumps({"response": "No relevant Oracle documentation found in PDFs."})


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
        # Search Oracle Documentation for relevant information
        oracle_response = search_oracle_docs(message)
        oracle_json = json.loads(oracle_response)
        yield json.dumps(oracle_json)  # Ensure valid JSON response
        
        # If relevant documentation is found, return early
        if "Relevant information found" in oracle_json.get("response", ""):
            return

        # AI Model Processing
        conversation = []
        if system_prompt:
            conversation.append({"role": "system", "content": system_prompt})
        for user, assistant in chat_history:
            conversation.extend([
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ])
        conversation.append({"role": "user", "content": message})

        # Tokenize input
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True)
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        input_ids = input_ids.to(model.device)

        # Streaming output using AI model
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
            yield json.dumps({"response": "".join(outputs).replace("<|EOT|>", "")})

    except Exception as e:
        yield json.dumps({"error": str(e)})


# Gradio Chat Interface
chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label="System prompt", lines=6),
        gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS),
        gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9),
        gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50),
        gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1),
    ],
    stop_btn=None,
    examples=[
        ["ORA-16198 encountered during Data Guard setup"],
        ["How to configure RMAN backup with FRA?"],
        ["What is the difference between ASM and traditional file systems?"],
    ],
)

# Launch Gradio App
with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    chat_interface.render()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=True)
