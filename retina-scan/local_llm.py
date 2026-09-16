"""
Offline Local LLM Assistant.
Uses a tiny instruction-tuned language model to answer patient questions based
on their scan results. Runs entirely locally on the CPU.
"""

import threading
import gc
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_pipeline = None
_lock = threading.Lock()

def get_pipeline():
    """
    Lazy-loads the model only when requested.
    Uses a lock to prevent concurrent loading in Streamlit.
    """
    global _pipeline
    with _lock:
        if _pipeline is None:
            # We use CPU by default for broadest compatibility, but if a GPU is available, it will use it.
            device = 0 if torch.cuda.is_available() else -1
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, 
                torch_dtype=torch.float32 if device == -1 else torch.float16,
                low_cpu_mem_usage=True
            )
            
            _pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
    return _pipeline

def format_prompt(messages):
    """
    Formats the chat messages for Qwen.
    """
    # Qwen uses chatml format, pipeline's tokenizer.apply_chat_template handles this
    tokenizer = get_pipeline().tokenizer
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def generate_response(messages):
    """
    Generates a response using the local pipeline.
    """
    pipe = get_pipeline()
    prompt = format_prompt(messages)
    
    # Generate text
    outputs = pipe(
        prompt, 
        max_new_tokens=200, 
        do_sample=True,
        temperature=0.3,
        return_full_text=False # Only return the generated part
    )
    
    return outputs[0]['generated_text'].strip()

def build_system_prompt(grade: int, class_name: str, vcdr: float, avr: float):
    return (
        f"You are a helpful, empathetic medical AI assistant inside the RetinaScan app. "
        f"You work offline on the patient's device. "
        f"The user has just received their retinal scan results. "
        f"Current findings: "
        f"- Diabetic Retinopathy: Grade {grade} ({class_name}) "
        f"- Glaucoma Risk (VCDR): {vcdr} "
        f"- Hypertensive Risk (AVR): {avr} "
        f"Answer the user's questions clearly and simply based on these results. "
        f"Keep answers concise (under 4 sentences). Remind them you are an AI, not a doctor, if they ask for a formal diagnosis."
    )
