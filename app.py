import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import spaces

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
LORA_ADAPTER = "TechAvenger/MyFaqSolver"   # ← replace with your repo
MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.2   # low = more factual/consistent for FAQ

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("Loading base model...")
# NEW - correct
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,        # ← fixes the torch_dtype deprecation
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,   # ← correct way to pass 4bit
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model.eval()
print("Model ready ✅")

# ── Inference ─────────────────────────────────────────────────────────────────
@spaces.GPU
def answer_question(question: str, history: list):
    if not question.strip():
        return history, ""

    prompt = f"### Question:\n{question.strip()}\n\n### Answer:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,          # greedy for FAQ = more deterministic
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    # Strip the prompt from the output
    answer = decoded[len(prompt):].strip()

    history.append((question, answer))
    return history, ""

def clear_chat():
    return [], ""

# ── UI ────────────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Sora', sans-serif !important;
    background: #0d0f14 !important;
    color: #e8eaf0 !important;
}

.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 2rem 1.5rem !important;
}

/* Header */
.header-block {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #1e2230;
    margin-bottom: 2rem;
}
.header-block h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin: 0 0 0.4rem;
}
.header-block p {
    font-size: 0.92rem;
    color: #6b7280;
    margin: 0;
}
.badge {
    display: inline-block;
    background: #1a1f2e;
    border: 1px solid #2a3045;
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.75rem;
    color: #7c8db5;
    margin-bottom: 1rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Chatbot */
#chatbot {
    background: #0d0f14 !important;
    border: 1px solid #1e2230 !important;
    border-radius: 16px !important;
    min-height: 420px !important;
}
#chatbot .message.user {
    background: #1a2035 !important;
    border: 1px solid #2a3550 !important;
    border-radius: 12px 12px 4px 12px !important;
    color: #c8d6f0 !important;
    font-size: 0.93rem !important;
}
#chatbot .message.bot {
    background: #111318 !important;
    border: 1px solid #1e2230 !important;
    border-radius: 12px 12px 12px 4px !important;
    color: #e2e8f0 !important;
    font-size: 0.93rem !important;
    font-family: 'Sora', sans-serif !important;
}

/* Input row */
.input-row {
    display: flex;
    gap: 0.6rem;
    margin-top: 1rem;
    align-items: flex-end;
}
#question-input textarea {
    background: #111318 !important;
    border: 1px solid #1e2230 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.93rem !important;
    padding: 0.85rem 1rem !important;
    resize: none !important;
}
#question-input textarea:focus {
    border-color: #3b5bdb !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(59,91,219,0.15) !important;
}
#question-input label { display: none !important; }

#ask-btn {
    background: #3b5bdb !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.85rem 1.6rem !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
    min-width: 90px !important;
}
#ask-btn:hover { background: #2f4ac4 !important; }

#clear-btn {
    background: #1a1f2e !important;
    border: 1px solid #2a3045 !important;
    border-radius: 12px !important;
    color: #7c8db5 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.85rem !important;
    padding: 0.85rem 1.2rem !important;
    cursor: pointer !important;
}
#clear-btn:hover {
    background: #1e2535 !important;
    color: #a0aec0 !important;
}

/* Footer */
.footer-note {
    text-align: center;
    color: #374151;
    font-size: 0.78rem;
    margin-top: 2rem;
    font-family: 'JetBrains Mono', monospace;
}
"""

with gr.Blocks(css=css, title="FAQ Agent") as demo:

    gr.HTML("""
    <div class="header-block">
        <div class="badge">Phi-3-mini · QLoRA · FAQ Agent</div>
        <h1>FAQ Answer Agent</h1>
        <p>Ask any question — powered by your fine-tuned model</p>
    </div>
    """)

    chatbot = gr.Chatbot(
        elem_id="chatbot",
        label="",
        bubble_full_width=False,
        show_label=False,
    )

    with gr.Row(elem_classes="input-row"):
        question_input = gr.Textbox(
            elem_id="question-input",
            placeholder="Type your question here...",
            lines=1,
            max_lines=4,
            show_label=False,
            scale=8,
        )
        ask_btn = gr.Button("Ask →", elem_id="ask-btn", scale=1)
        clear_btn = gr.Button("Clear", elem_id="clear-btn", scale=1)

    gr.HTML('<div class="footer-note">unsloth/Phi-3-mini-4k-instruct · LoRA Rank 16 · Fine-tuned on custom FAQ dataset</div>')

    # Event handlers
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
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, question_input],
    )

if __name__ == "__main__":
    demo.launch()