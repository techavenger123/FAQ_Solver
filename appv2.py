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

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
)

max_memory = {}
if torch.cuda.is_available():
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    usable_mib = int((vram_bytes - 500 * 1024 ** 2) / 1024 ** 2)
    max_memory[0] = f"{usable_mib}MiB"
    print(f"GPU detected — allocating {usable_mib} MiB")
else:
    print("No GPU — running on CPU (slow)")
max_memory["cpu"] = "12GiB"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
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


# ── CSS — no DOM structure selectors, no theme conflicts ──────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ── */
body, .gradio-container, .svelte-container {
    font-family: 'Sora', sans-serif !important;
    background: #080a10 !important;
    color: #e2e8f0 !important;
}
.gradio-container {
    max-width: 820px !important;
    margin: 0 auto !important;
}

/* ── Header ── */
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

/* ── Chatbot ── */
#chatbot {
    background: #0b0d15 !important;
    border: 1px solid #16192a !important;
    border-radius: 16px !important;
}

/* ── Textbox: target the actual textarea element only ── */
#question-input textarea {
    background: #0f1118 !important;
    border: 1px solid #1e2338 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.91rem !important;
    padding: 0.8rem 1rem !important;
    min-height: 44px !important;
    pointer-events: auto !important;
    cursor: text !important;
    user-select: text !important;
    -webkit-user-select: text !important;
}
#question-input textarea:focus {
    border-color: #3b5bdb !important;
    box-shadow: 0 0 0 3px rgba(59,91,219,0.12) !important;
    outline: none !important;
}
/* Hide the label text but keep the element so Gradio doesn't break */
#question-input label span { display: none !important; }

/* ── Ask button ── */
#btn-ask button {
    background: #3b5bdb !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.1s !important;
}
#btn-ask button:hover  { background: #2f4ac4 !important; }
#btn-ask button:active { transform: scale(0.97) !important; }

/* ── Clear button ── */
#btn-clear button {
    background: #0f1118 !important;
    border: 1px solid #1e2338 !important;
    border-radius: 12px !important;
    color: #5a6a9a !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.85rem !important;
    cursor: pointer !important;
    transition: background 0.2s, color 0.2s !important;
}
#btn-clear button:hover { background: #151a28 !important; color: #8fa0d0 !important; }

/* ── Footer ── */
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
with gr.Blocks(title="FAQ Agent") as demo:

    gr.HTML("""
    <div id="header">
        <div class="badge">Phi-3-mini · QLoRA · FAQ Agent</div>
        <h1>FAQ Answer Agent</h1>
        <p>Ask any question — powered by your fine-tuned model</p>
    </div>
    """)

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_label=False,
        height=420,
        # removed type="tuples" — not supported in this Gradio version
    )

    with gr.Row():
        question_input = gr.Textbox(
            elem_id="question-input",
            placeholder="Type your question here...",
            lines=1,
            max_lines=4,
            show_label=False,
            scale=8,
            interactive=True,
            autofocus=True,
        )
        ask_btn   = gr.Button("Ask →", elem_id="btn-ask",   variant="primary",   scale=1, min_width=80)
        clear_btn = gr.Button("Clear",  elem_id="btn-clear", variant="secondary", scale=1, min_width=70)

    gr.HTML('<div id="footer">TechAvenger/MyFaqSolver · Phi-3-mini-4k base · LoRA fine-tuned FAQ dataset</div>')

    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input],
    )
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input],
    )
    clear_btn.click(fn=clear_chat, outputs=[chatbot, question_input])

if __name__ == "__main__":
    demo.launch(css=css)