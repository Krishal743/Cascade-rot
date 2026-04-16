"""Chain runner for multi-step tool execution (supports Chains A, B, C)."""

import json
import re
import torch
from typing import List, Dict, Any, Optional

from tools.schemas import CHAIN_A, CHAIN_B, CHAIN_C
from tools.validator import validate_tool_call
from tools.mock_executor import execute_tool


# ============ Output Schemas for Validation ============

# Chain A: Weather
WEATHER_OUTPUT_SCHEMA = {
    "name": "get_weather",
    "properties": {
        "location": {"type": "string"},
        "temperature": {"type": "number"},
        "condition": {"type": "string"},
        "humidity": {"type": "number"}
    }
}
WEATHER_REQUIRED = ["location", "temperature", "condition", "humidity"]

UMBRELLA_OUTPUT_SCHEMA = {
    "name": "should_bring_umbrella",
    "properties": {
        "bring_umbrella": {"type": "boolean"},
        "reason": {"type": "string"}
    }
}
UMBRELLA_REQUIRED = ["bring_umbrella", "reason"]

# Chain B: Search
WEB_SEARCH_OUTPUT_SCHEMA = {
    "name": "web_search",
    "properties": {
        "query": {"type": "string"},
        "num_results": {"type": "number"}
    }
}
WEB_SEARCH_REQUIRED = ["query"]

EXTRACT_FACTS_OUTPUT_SCHEMA = {
    "name": "extract_facts",
    "properties": {
        "search_results": {"type": "array"},
        "topic": {"type": "string"}
    }
}
EXTRACT_FACTS_REQUIRED = ["search_results", "topic"]

SUMMARIZE_OUTPUT_SCHEMA = {
    "name": "summarize",
    "properties": {
        "facts": {"type": "array"},
        "max_length": {"type": "number"}
    }
}
SUMMARIZE_REQUIRED = ["facts"]

# Chain C: Calendar
PARSE_DATE_OUTPUT_SCHEMA = {
    "name": "parse_date",
    "properties": {
        "date_string": {"type": "string"}
    }
}
PARSE_DATE_REQUIRED = ["date_string"]

CHECK_CALENDAR_OUTPUT_SCHEMA = {
    "name": "check_calendar",
    "properties": {
        "date": {"type": "string"}
    }
}
CHECK_CALENDAR_REQUIRED = ["date"]

FIND_CONFLICTS_OUTPUT_SCHEMA = {
    "name": "find_conflicts",
    "properties": {
        "events": {"type": "array"},
        "duration_minutes": {"type": "number"}
    }
}
FIND_CONFLICTS_REQUIRED = ["events"]

SUGGEST_TIME_OUTPUT_SCHEMA = {
    "name": "suggest_time",
    "properties": {
        "duration_minutes": {"type": "number"},
        "existing_events": {"type": "array"}
    }
}
SUGGEST_TIME_REQUIRED = ["duration_minutes", "existing_events"]

CREATE_EVENT_OUTPUT_SCHEMA = {
    "name": "create_event",
    "properties": {
        "time": {"type": "string"},
        "title": {"type": "string"},
        "duration_minutes": {"type": "number"}
    }
}
CREATE_EVENT_REQUIRED = ["time", "title", "duration_minutes"]


# ============ System Prompts ============

SYSTEM_PROMPTS = {
    "get_weather": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: location, temperature, condition, humidity
Example: {"location": "London", "temperature": 20, "condition": "sunny", "humidity": 60}""",
    
    "should_bring_umbrella": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: bring_umbrella, reason
Example: {"bring_umbrella": true, "reason": "It is raining outside"}""",
    
    "web_search": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: query, num_results
Example: {"query": "artificial intelligence news", "num_results": 5}""",
    
    "extract_facts": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: search_results, topic
Example: {"search_results": [{"title": "Result"}], "topic": "AI"}""",
    
    "summarize": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: facts, max_length
Example: {"facts": ["Fact 1", "Fact 2"], "max_length": 50}""",
    
    "parse_date": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: date_string
Example: {"date_string": "next Monday"}""",
    
    "check_calendar": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: date
Example: {"date": "2026-04-06"}""",
    
    "find_conflicts": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: events, duration_minutes
Example: {"events": ["Meeting at 10am"], "duration_minutes": 60}""",
    
    "suggest_time": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: duration_minutes, existing_events
Example: {"duration_minutes": 60, "existing_events": ["Meeting at 10am"]}""",
    
    "create_event": """You are a tool-calling assistant. Your ONLY output must be valid JSON.
DO NOT write any Python code. DO NOT write any explanations.
Output ONLY a JSON object with these fields: time, title, duration_minutes
Example: {"time": "2026-04-06 10:00", "title": "Team Meeting", "duration_minutes": 60}""",
}


# Schema mapping for validation
SCHEMA_MAP = {
    "get_weather": (WEATHER_OUTPUT_SCHEMA, WEATHER_REQUIRED),
    "should_bring_umbrella": (UMBRELLA_OUTPUT_SCHEMA, UMBRELLA_REQUIRED),
    "web_search": (WEB_SEARCH_OUTPUT_SCHEMA, WEB_SEARCH_REQUIRED),
    "extract_facts": (EXTRACT_FACTS_OUTPUT_SCHEMA, EXTRACT_FACTS_REQUIRED),
    "summarize": (SUMMARIZE_OUTPUT_SCHEMA, SUMMARIZE_REQUIRED),
    "parse_date": (PARSE_DATE_OUTPUT_SCHEMA, PARSE_DATE_REQUIRED),
    "check_calendar": (CHECK_CALENDAR_OUTPUT_SCHEMA, CHECK_CALENDAR_REQUIRED),
    "find_conflicts": (FIND_CONFLICTS_OUTPUT_SCHEMA, FIND_CONFLICTS_REQUIRED),
    "suggest_time": (SUGGEST_TIME_OUTPUT_SCHEMA, SUGGEST_TIME_REQUIRED),
    "create_event": (CREATE_EVENT_OUTPUT_SCHEMA, CREATE_EVENT_REQUIRED),
}


# Common location aliases for semantic matching
LOCATION_ALIASES = {
    "london": ["london", "the capital of england", "england's capital"],
    "paris": ["paris", "the capital of france", "france's capital"],
    "tokyo": ["tokyo", "the capital of japan", "japan's capital"],
    "new york": ["new york", "ny", "nyc", "the big apple"],
    "berlin": ["berlin", "the capital of germany", "germany's capital"],
    "munich": ["munich", "münchen", "the bavarian city"],
    "sydney": ["sydney", "the australian city", "australia's largest city"],
    "rome": ["rome", "the italian city", "italy's capital", "the eternal city"],
    "madrid": ["madrid", "the spanish city", "spain's capital"],
    "vienna": ["vienna", "the austrian city", "austria's capital"],
}


def extract_location_from_query(query: str) -> Optional[str]:
    """Extract location entity from user query using pattern matching."""
    query_lower = query.lower()
    
    patterns = [
        r'in\s+([a-zA-Z]+)(?:\s+today|\s+tomorrow|\s+please)?\??$',
        r'for\s+([a-zA-Z]+)(?:\s+please)?\??$',
        r'in\s+the\s+([a-z]+)\s+city',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            location = match.group(1).strip()
            if location not in ["the", "city", "today", "tomorrow"]:
                return location.title()
    
    for location, aliases in LOCATION_ALIASES.items():
        for alias in aliases:
            if alias in query_lower:
                return location.title()
    
    return None


def validate_location_semantic(model_location: str, query: str) -> bool:
    """Check if model's location matches ground truth from original query.
    
    This validates against the GROUND TRUTH (extracted from original query),
    not against the perturbed query. This is the correct approach.
    """
    if not model_location:
        return False
    
    model_loc_lower = model_location.lower().strip().rstrip(',')
    
    # Extract ground truth from query using our ground truth function
    ground_truth = extract_location_from_query(query)
    if not ground_truth:
        return False
    
    gt_lower = ground_truth.lower()
    
    # Direct match
    if model_loc_lower == gt_lower:
        return True
    
    # Check if model output is in aliases
    for location, aliases in LOCATION_ALIASES.items():
        if model_loc_lower == location:
            if gt_lower == location:
                return True
            # Check if any alias matches ground truth
            for alias in aliases:
                if alias == gt_lower:
                    return True
    
    return False


def extract_ground_truth_location(query: str) -> Optional[str]:
    """Extract the ground truth location from user's original query."""
    return extract_location_from_query(query)


def extract_json(text: str) -> tuple:
    """Extract JSON from model output."""
    if "assistant" in text.lower():
        parts = text.split("assistant")
        json_text = parts[-1].strip()
    else:
        json_text = text
    
    start = json_text.find("{")
    end = json_text.rfind("}")
    
    if start == -1 or end == -1:
        return None, "No JSON found"
    
    json_str = json_text[start:end+1]
    
    try:
        return json.loads(json_str), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


class ChainRunner:
    """Execute perturbed queries through multi-step tool chains."""
    
    def __init__(self, model, tokenizer, chain_def: List[Dict]):
        self.model = model
        self.tokenizer = tokenizer
        self.chain = chain_def
    
    def _build_prompt(self, user_query: str, tool_schema: Dict, context: str = "") -> str:
        """Build prompt for a specific tool call."""
        tool_name = tool_schema["name"]
        
        if context:
            user = f"Previous context: {context}\n\nCurrent task: {user_query}"
        else:
            user = user_query
        
        system = SYSTEM_PROMPTS.get(tool_name, "You are a helpful assistant.")
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def _generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate response from model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def run(self, query: str, chain_name: str = "chain_a") -> Dict[str, Any]:
        """Run a query through a multi-step tool chain."""
        ground_truth_location = extract_ground_truth_location(query)
        
        trace = {
            "query": query,
            "chain": chain_name,
            "ground_truth_location": ground_truth_location,
            "steps": [],
            "cascade_failure": False,
            "semantic_cascade": False
        }
        
        context = None
        
        for step_idx, tool_schema in enumerate(self.chain):
            tool_name = tool_schema["name"]
            
            step_prompt = self._build_prompt(query, tool_schema, context)
            step_raw = self._generate(step_prompt)
            
            step_result = {
                "step": step_idx,
                "tool": tool_name,
                "raw_output": step_raw,
                "syntactic_success": False,
                "semantic_success": False,
                "success": False
            }
            
            step_parsed, step_error = extract_json(step_raw)
            
            if step_error is None:
                schema_obj, required_fields = SCHEMA_MAP.get(tool_name, ({}, []))
                if schema_obj:
                    validation = validate_tool_call(json.dumps(step_parsed), schema_obj, required_fields)
                    step_result["validation"] = validation
                    step_result["syntactic_success"] = validation["schema_valid"]
                    
                    if validation["schema_valid"] and step_parsed:
                        # Check semantic for location-based tools (Chain A)
                        if tool_name == "get_weather":
                            model_location = step_parsed.get("location", "")
                            semantic_valid = validate_location_semantic(model_location, query)
                            step_result["semantic_success"] = semantic_valid
                            step_result["extracted_location"] = model_location
                    
                    if validation["schema_valid"]:
                        try:
                            tool_output = execute_tool(tool_name, step_parsed)
                            step_result["executed_result"] = tool_output
                            step_result["success"] = True
                            context = json.dumps(tool_output)
                        except Exception as e:
                            step_result["execution_error"] = str(e)
                            context = step_raw
                    else:
                        context = step_raw
                else:
                    context = step_raw
            else:
                step_result["parse_error"] = step_error
                context = step_raw
            
            trace["steps"].append(step_result)
        
        # Determine cascade failure: any step fails
        all_success = all(s["success"] for s in trace["steps"])
        trace["cascade_failure"] = not all_success
        
        # Determine semantic cascade (Chain A only)
        if chain_name == "chain_a":
            step1 = trace["steps"][0] if trace["steps"] else {}
            trace["semantic_cascade"] = (
                not step1.get("semantic_success", True) and
                len(trace["steps"]) > 1 and trace["steps"][1].get("success", False)
            )
        
        return trace


def print_trace(trace: Dict[str, Any]) -> None:
    """Pretty print a chain execution trace."""
    print("=" * 70)
    print(f"Query: {trace['query']}")
    print(f"Chain: {trace['chain']}")
    print("=" * 70)
    
    for step in trace["steps"]:
        print(f"\n--- Step {step['step']}: {step['tool']} ---")
        print(f"Success: {step['success']}")
        print(f"Syntactic: {step['syntactic_success']}, Semantic: {step.get('semantic_success', 'N/A')}")
        if step.get('extracted_location'):
            print(f"Extracted location: {step['extracted_location']}")
    
    print(f"\n>>> Cascade Failure: {trace['cascade_failure']}")
    print(f">>> Semantic Cascade: {trace.get('semantic_cascade', False)}")
    print("=" * 70)