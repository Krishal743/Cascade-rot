"""Quick prompt ablation study - Test if model defaults to New York regardless of prompt example."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Load model once
print("Loading Qwen2.5-3B model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Test queries
test_queries = [
    "What's the weather in Paris?",
    "Is it sunny in Tokyo?",
    "Check weather for Berlin",
    "What's the temperature in Rome?",
]

# Three prompt conditions
prompts = {
    "a) Original (London)": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: location, temperature, condition, humidity
Example: {"location": "London", "temperature": 20, "condition": "sunny", "humidity": 60}""",

    "b) Tokyo example": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: location, temperature, condition, humidity
Example: {"location": "Tokyo", "temperature": 25, "condition": "cloudy", "humidity": 70}""",

    "c) No example": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: location, temperature, condition, humidity
The location field is required. Respond with valid JSON only.""",
}

def extract_location(response):
    """Extract location from model JSON output."""
    import json
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1:
        try:
            parsed = json.loads(response[start:end+1])
            return parsed.get("location", "N/A")
        except:
            return "Parse Error"
    return "No JSON"

# Run ablation
print("\n" + "=" * 80)
print("PROMPT ABLATION STUDY")
print("=" * 80)

results = {}

for condition, system_prompt in prompts.items():
    print(f"\n--- Condition: {condition} ---")
    results[condition] = {}
    
    for query in test_queries:
        # Build prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.0,
                do_sample=False,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the response part
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        
        location = extract_location(response)
        
        print(f"  Query: {query[:30]}... -> Location: {location}")
        results[condition][query] = location

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for condition, data in results.items():
    locations = list(data.values())
    ny_count = sum(1 for l in locations if l and "new york" in l.lower())
    print(f"\n{condition}:")
    print(f"  Outputs: {locations}")
    print(f"  New York count: {ny_count}/{len(test_queries)}")

# Cleanup
del model
torch.cuda.empty_cache()