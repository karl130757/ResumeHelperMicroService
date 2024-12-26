from transformers import pipeline

_gpt_model = None

def get_gpt_model():
    global _gpt_model
    if _gpt_model is None:
        print("Loading GPT-J model...")
        _gpt_model = pipeline(
            "text-generation",
            model="EleutherAI/gpt-neo-2.7B",  # GPT-J equivalent
            device=0,  # Use GPU if available,
            max_new_tokens=300,
            truncation=True
        )
    return _gpt_model
