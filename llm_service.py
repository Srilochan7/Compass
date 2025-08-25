import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_llm():
    model_id = "lochandoesaiml/tinyllama-rakshak"
    adapter_id = "lochandoesaiml/tinyllama-rakshak"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            use_auth_token=True  # pulls your HF login creds
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)

        model = PeftModel.from_pretrained(
            base_model,
            adapter_id,
            use_auth_token=True
        )

        print("✅ Rakshak model loaded successfully")

    except Exception as e:
        print(f"⚠️ Failed to load {model_id}, falling back to TinyLlama base. Error: {e}")
        fallback_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        base_model = AutoModelForCausalLM.from_pretrained(
            fallback_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(fallback_id)
        model = base_model

    return model, tokenizer


def get_llm_advice(model, tokenizer, prompt):
    """
    Act as an AI assistant and give me answers.
    """
    input_text = f"<s>[INST] {prompt} [/INST]"

    inputs = tokenizer(
        input_text,
        return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model.generate(**inputs, max_new_tokens=256)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    cleaned_response = response_text[response_text.find("[/INST]") + len("[/INST]"):].strip()
    return cleaned_response
