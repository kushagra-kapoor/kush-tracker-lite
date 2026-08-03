import os
import time
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

FALLBACK_MODELS = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "openrouter/free"
]

def robust_llm_call(prompt, require_json=False):
    """
    Robust LLM caller that prioritizes Mistral and falls back to OpenRouter.
    Includes rate-limit exponential backoff and graceful JSON handling.
    """
    # 1. Try Mistral first
    if MISTRAL_API_KEY:
        for attempt in range(3):
            try:
                payload = {"model": "mistral-large-latest", "messages": [{"role": "user", "content": prompt}]}
                if require_json:
                    payload["response_format"] = {"type": "json_object"}
                
                response = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=60,
                )
                
                if response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    print(f"Mistral LLM rate limited (429). Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
                
            except Exception as e:
                print(f"Mistral LLM failed on attempt {attempt+1}: {e}")
                if attempt == 2:
                    break # Give up on Mistral
                time.sleep(2)
    else:
        print("MISTRAL_KEY not found. Skipping Mistral.")
        
    # 2. Try OpenRouter Fallbacks
    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY not found. Cannot use fallbacks.")
        return None
        
    # Modify prompt for JSON if required (to avoid crashing OSS models)
    openrouter_prompt = prompt
    if require_json:
        openrouter_prompt += "\n\nCRITICAL INSTRUCTION: You MUST respond in valid JSON format. Do not include markdown formatting or backticks, just raw JSON."
        
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    for model in FALLBACK_MODELS:
        print(f"Falling back to OpenRouter model: {model}...")
        for attempt in range(2):
            try:
                data = {
                    "model": model,
                    "messages": [{"role": "user", "content": openrouter_prompt}],
                }
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=90,
                )
                
                if response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    print(f"  Rate limited on {model} (429). Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                print(f"  Success with {model}!")
                return response.json()["choices"][0]["message"]["content"]
                
            except Exception as e:
                print(f"  Error with {model} on attempt {attempt+1}: {e}")
                if attempt == 1:
                    break # Give up on this specific model
                time.sleep(2)
                
    print("All LLM providers and fallbacks exhausted. Returning None.")
    return None
