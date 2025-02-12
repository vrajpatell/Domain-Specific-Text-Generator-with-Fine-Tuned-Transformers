# app/app.py

import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Specify the path to the fine-tuned model
MODEL_PATH = "fine_tuned_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the fine-tuned model and tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.to(device)

def generate_text(prompt, max_length=100, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95):
    """
    Generate text based on a prompt using the fine-tuned GPT-2 model.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]
    # Return a single string if only one sequence is generated
    return generated_texts[0] if num_return_sequences == 1 else generated_texts

# Build the Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter prompt here...", label="Prompt"),
        gr.Slider(minimum=50, maximum=300, step=10, value=100, label="Max Length"),
        gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Number of Sequences"),
        gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Domain-Specific Text Generator",
    description="Generate text using the fine-tuned GPT-2 model on your domain-specific data.",
)

if __name__ == "__main__":
    iface.launch()
