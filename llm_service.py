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
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )
    
    model = PeftModel.from_pretrained(
        model_id,
        adapter_name=adapter_id
    )
    
    print("Rakshak model loaded suceesfully")
    
    return model, tokenizer
    
    
    
# Add this function to the end of llm_service.py

def get_llm_advice(model, tokenizer, prompt):
    """
    Act as an ai assistant and give me answers.
    """
    # 1. Format the prompt
    input_text = f"<s>[INST] {prompt} [/INST]"

    # 2. Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Generate a response and decode it
    outputs = model.generate(**inputs, max_new_tokens=256)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean the response to remove the prompt part
    cleaned_response = response_text[response_text.find("[/INST]") + len("[/INST]"):].strip()

    return cleaned_response