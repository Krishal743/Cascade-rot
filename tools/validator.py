"""Validator for tool call outputs."""

import json
import jsonschema
from typing import Any, Dict, List, Optional


def validate_tool_call(
    output: str, 
    expected_schema: dict,
    required_fields: Optional[List[str]] = None
) -> dict:
    """
    Validate tool call output against expected schema.
    
    Returns a dict with four binary flags:
    - parses: Does the string parse as valid JSON?
    - has_required: Are all required fields present?
    - types_valid: Do all fields have correct types?
    - schema_valid: Are all three above True?
    
    Always returns dict, never raises exception.
    
    Args:
        output: String output from model (should be JSON)
        expected_schema: OpenAI-style function schema
        required_fields: List of required output field names (if not in schema's "required")
    """
    result = {
        "parses": False,
        "has_required": False,
        "types_valid": False,
        "schema_valid": False
    }
    
    # Level 1: Check if parses as JSON
    parsed = None
    try:
        parsed = json.loads(output)
        result["parses"] = True
    except (json.JSONDecodeError, TypeError):
        return result  # Cannot proceed if not JSON
    
    # Level 2: Check required fields
    # Use provided required_fields, or extract from schema parameters
    required = required_fields if required_fields is not None else []
    if not required:
        # Try to get required from schema parameters (input schema)
        required = expected_schema.get("parameters", {}).get("required", [])
    
    if required:
        result["has_required"] = all(key in parsed for key in required)
    else:
        result["has_required"] = True
    
    # Level 3: Check types (strict)
    # Get properties from schema parameters (for input) or top-level (for output)
    properties = expected_schema.get("parameters", {}).get("properties", {})
    if not properties and "properties" in expected_schema:
        properties = expected_schema.get("properties", {})
    
    type_errors = []
    
    for field, field_schema in properties.items():
        if field not in parsed:
            continue  # Skip missing fields (already caught by has_required)
        
        expected_type = field_schema.get("type")
        actual_value = parsed[field]
        actual_type = type(actual_value)
        
        if expected_type == "string":
            if actual_type is not str:
                type_errors.append(f"{field}: expected str, got {actual_type.__name__}")
        
        elif expected_type == "number":
            if actual_type not in (int, float):
                type_errors.append(f"{field}: expected int/float, got {actual_type.__name__}")
        
        elif expected_type == "boolean":
            if actual_type is not bool:
                type_errors.append(f"{field}: expected bool, got {actual_type.__name__}")
        
        elif expected_type == "object":
            if actual_type is not dict:
                type_errors.append(f"{field}: expected dict, got {actual_type.__name__}")
    
    result["types_valid"] = len(type_errors) == 0
    
    # Level 4: Overall schema valid (all three above)
    result["schema_valid"] = (
        result["parses"] and 
        result["has_required"] and 
        result["types_valid"]
    )
    
    return result


# ============ Test Cases ============

if __name__ == "__main__":
    # Test with weather tool - has 'location' as required input
    # But output should have location, temperature, condition, humidity
    WEATHER_OUTPUT_SCHEMA = {
        "name": "get_weather",
        "description": "Fetch current weather for a location.",
        "properties": {
            "location": {"type": "string"},
            "temperature": {"type": "number"},
            "condition": {"type": "string"},
            "humidity": {"type": "number"}
        }
    }
    WEATHER_REQUIRED = ["location", "temperature", "condition", "humidity"]
    
    # Test 1: Valid input
    valid_input = '{"location": "London", "temperature": 22, "condition": "sunny", "humidity": 65}'
    print(f"Valid input: {validate_tool_call(valid_input, WEATHER_OUTPUT_SCHEMA, WEATHER_REQUIRED)}")
    
    # Test 2: Missing field
    missing_field = '{"location": "London", "temperature": 22, "condition": "sunny"}'
    print(f"Missing field: {validate_tool_call(missing_field, WEATHER_OUTPUT_SCHEMA, WEATHER_REQUIRED)}")
    
    # Test 3: Invalid JSON
    invalid_json = 'not json at all'
    print(f"Invalid JSON: {validate_tool_call(invalid_json, WEATHER_OUTPUT_SCHEMA, WEATHER_REQUIRED)}")
    
    # Test 4: Type error (string instead of number)
    type_error = '{"location": "London", "temperature": "hot", "condition": "sunny", "humidity": 65}'
    print(f"Type error: {validate_tool_call(type_error, WEATHER_OUTPUT_SCHEMA, WEATHER_REQUIRED)}")
