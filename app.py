import os
import json
import traceback

# Try to import optional dependencies. We don't fail if unavailable.
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    import gradio as gr
except Exception:
    raise RuntimeError("The Gradio package is required to run this app. Please install gradio.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

MODEL_PRIMARY = "VijayChandra/english-to-telugu-translator-nllb"
MODEL_FALLBACK = "aryaumesh/english-to-telugu"

LOCAL_PHRASE_TABLE = {
    "hello": "హలో",
    "hi": "హాయ్",
    "how are you": "మీరు ఎలా ఉన్నారు",
    "thank you": "ధన్యవాదాలు",
    "good morning": "శుభోదయం",
    "good night": "శుభ రాత్రి",
}

translator = None
loaded_model_name = None

def _load_transformers_translator(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer)

if TRANSFORMERS_AVAILABLE:
    try:
        translator = _load_transformers_translator(MODEL_PRIMARY)
        loaded_model_name = MODEL_PRIMARY
    except Exception:
        try:
            translator = _load_transformers_translator(MODEL_FALLBACK)
            loaded_model_name = MODEL_FALLBACK
        except Exception as e:
            print("Warning: transformers present but failed to load models:\n", e)
            translator = None
            loaded_model_name = None


def hf_inference_api_translate(model_id, token, text, max_length=256):
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("requests library not available in this environment")
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    payload = {"inputs": text, "parameters": {"max_length": max_length}}
    resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HF Inference API error {resp.status_code}: {resp.text}")
    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"HF Inference API error: {data.get('error')}")
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        return first.get("generated_text") or first.get("translation_text") or str(first)
    return str(data)


def translate_en_to_te(text, max_length=256, use_hf_token=None, hf_model_id=MODEL_PRIMARY):
    text = (text or "").strip()
    if not text:
        return ""

    if translator is not None:
        try:
            out = translator(text, max_length=max_length)
            if isinstance(out, list) and len(out) > 0:
                tok = out[0]
                return tok.get('translation_text') or tok.get('generated_text') or str(tok)
            return str(out)
        except Exception as e:
            debug = f"\n[Warning: transformers pipeline error: {e}]"
    else:
        debug = ""

    if use_hf_token:
        try:
            return hf_inference_api_translate(hf_model_id, use_hf_token, text, max_length=max_length)
        except Exception as e:
            debug += f"\n[Warning: HF Inference API error: {e}]"

    import re
    key = re.sub(r"[^a-z0-9 ]+", "", text.lower().strip())
    if key in LOCAL_PHRASE_TABLE:
        return LOCAL_PHRASE_TABLE[key] + debug

    return (
        "[No model available to produce a high-quality translation in this environment.]\n"
        "Options:\n"
        "  1) Install transformers locally.\n"
        "  2) Provide a Hugging Face token in the UI.\n"
        "  3) Use the tiny offline demo table.\n\n"
        f"Input echoed: {text}\n{debug}"
    )


def build_ui():
    with gr.Blocks(title="English → Telugu Translator") as demo:
        gr.Markdown("<h1 style='text-align:center;'>English → Telugu Translator</h1>")

        with gr.Row():
            inp = gr.Textbox(label="English", placeholder="Enter text...", lines=4)
            out = gr.Textbox(label="Telugu", lines=4)

        btn = gr.Button("Translate", variant="primary")

        def _wrapped_translate(text):
            try:
                return translate_en_to_te(text)
            except Exception as e:
                return f"Error: {e}\n" + traceback.format_exc()

        btn.click(_wrapped_translate, inp, out)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 8080))
    )
