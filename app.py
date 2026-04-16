import os
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login

# ── Credentials from HF Secrets ───────────────────────────────────────────────
HF_TOKEN     = os.environ.get("FAQ")
BASE_MODEL   = os.environ.get("BASE_MODEL",   "unsloth/Phi-3-mini-4k-instruct-bnb-4bit")
LORA_ADAPTER = os.environ.get("LORA_ADAPTER", "TechAvenger/MyFaqSolver")

MAX_NEW_TOKENS = 512

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("Logged in to HuggingFace Hub ✅")
else:
    print("No HF_TOKEN found — assuming public model")

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"Loading tokenizer from: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=HF_TOKEN)

print(f"Loading base model: {BASE_MODEL}")

# llm_int8_enable_fp32_cpu_offload lets layers spill to CPU RAM when VRAM is full
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
)

# Give GPU as much VRAM as possible, spill the rest to CPU RAM
max_memory = {}
if torch.cuda.is_available():
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    usable_mib = int((vram_bytes - 500 * 1024 ** 2) / 1024 ** 2)  # reserve 500 MB
    max_memory[0] = f"{usable_mib}MiB"
    print(f"GPU detected — allocating {usable_mib} MiB")
else:
    print("No GPU — running on CPU (slow)")
max_memory["cpu"] = "12GiB"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory=max_memory,
    trust_remote_code=True,
    quantization_config=bnb_config,
    token=HF_TOKEN,
)

print(f"Loading LoRA adapter: {LORA_ADAPTER}")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER, token=HF_TOKEN)
model.eval()
print("Model ready ✅")


# ── Inference ──────────────────────────────────────────────────────────────────
def answer_question(question: str, history: list):
    if not question.strip():
        return history, ""

    prompt = f"### Question:\n{question.strip()}\n\n### Answer:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer  = decoded[len(prompt):].strip()
    history.append((question, answer))
    return history, ""


def clear_chat():
    return [], ""


# ── CSS ────────────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    font-family: 'Sora', sans-serif !important;
    background: #080a10 !important;
    color: #e2e8f0 !important;
}
.gradio-container {
    max-width: 800px !important;
    margin: 0 auto !important;
    padding: 2rem 1.25rem 3rem !important;
}
#header {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
    border-bottom: 1px solid #16192a;
    margin-bottom: 1.75rem;
}
#header .badge {
    display: inline-block;
    background: #0f1220;
    border: 1px solid #252d4a;
    border-radius: 100px;
    padding: 0.3rem 1rem;
    font-size: 0.72rem;
    color: #6b82c0;
    margin-bottom: 1rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}
#header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    color: #fff;
    margin-bottom: 0.4rem;
}
#header p { font-size: 0.88rem; color: #4b5675; }

#chatbot {
    background: #0b0d15 !important;
    border: 1px solid #16192a !important;
    border-radius: 16px !important;
    min-height: 400px !important;
    padding: 1rem !important;
}
#chatbot .message.user > div,
#chatbot [data-testid="user"] {
    background: #172044 !important;
    border: 1px solid #233060 !important;
    border-radius: 14px 14px 4px 14px !important;
    color: #c5d3f0 !important;
    font-size: 0.91rem !important;
    line-height: 1.55 !important;
    padding: 0.75rem 1rem !important;
}
#chatbot .message.bot > div,
#chatbot [data-testid="bot"] {
    background: #0f1118 !important;
    border: 1px solid #1a1e2e !important;
    border-radius: 14px 14px 14px 4px !important;
    color: #dde4f5 !important;
    font-size: 0.91rem !important;
    line-height: 1.6 !important;
    padding: 0.75rem 1rem !important;
}

#input-row {
    display: flex !important;
    flex-direction: row !important;
    align-items: flex-end !important;
    gap: 0.6rem !important;
    margin-top: 1rem !important;
    width: 100% !important;
}
#question-input { flex: 1 1 auto !important; min-width: 0 !important; }
#question-input textarea {
    background: #0f1118 !important;
    border: 1px solid #1e2338 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.91rem !important;
    padding: 0.8rem 1rem !important;
    resize: none !important;
    width: 100% !important;
    transition: border-color 0.2s !important;
}
#question-input textarea:focus {
    border-color: #3b5bdb !important;
    box-shadow: 0 0 0 3px rgba(59,91,219,0.12) !important;
    outline: none !important;
}
#question-input label { display: none !important; }

#btn-ask, #btn-clear { flex: 0 0 auto !important; align-self: flex-end !important; }
#btn-ask button {
    background: #3b5bdb !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.8rem 1.4rem !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    height: 44px !important;
    transition: background 0.2s, transform 0.1s !important;
}
#btn-ask button:hover  { background: #2f4ac4 !important; }
#btn-ask button:active { transform: scale(0.97) !important; }

#btn-clear button {
    background: #0f1118 !important;
    border: 1px solid #1e2338 !important;
    border-radius: 12px !important;
    color: #5a6a9a !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.85rem !important;
    padding: 0.8rem 1.1rem !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    height: 44px !important;
    transition: background 0.2s, color 0.2s !important;
}
#btn-clear button:hover { background: #151a28 !important; color: #8fa0d0 !important; }

#footer {
    text-align: center;
    color: #252d45;
    font-size: 0.74rem;
    margin-top: 2rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.03em;
}
"""

# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="FAQ Agent") as demo:

    gr.HTML("""
    <div id="header">
        <div class="badge">Phi-3-mini · QLoRA · FAQ Agent</div>
        <h1>FAQ Answer Agent</h1>
        <p>Ask any question — powered by your fine-tuned model</p>
    </div>
    """)

    chatbot = gr.Chatbot(elem_id="chatbot", show_label=False, height=420)

    with gr.Row(elem_id="input-row"):
        question_input = gr.Textbox(
            elem_id="question-input",
            placeholder="Type your question here...",
            lines=1, max_lines=4,
            show_label=False, scale=8,
        )
        ask_btn   = gr.Button("Ask →", elem_id="btn-ask",   scale=1, min_width=80)
        clear_btn = gr.Button("Clear",  elem_id="btn-clear", scale=1, min_width=70)

    gr.HTML('<div id="footer">TechAvenger/MyFaqSolver · Phi-3-mini-4k base · LoRA fine-tuned FAQ dataset</div>')

    ask_btn.click(fn=answer_question, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
    question_input.submit(fn=answer_question, inputs=[question_input, chatbot], outputs=[chatbot, question_input])
    clear_btn.click(fn=clear_chat, outputs=[chatbot, question_input])

if __name__ == "__main__":
    demo.launch()