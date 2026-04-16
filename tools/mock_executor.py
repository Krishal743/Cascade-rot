"""Mock executor for deterministic tool execution (Chains A, B, C)."""

import random
import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


def _get_seed_from_string(s: str) -> int:
    """Generate deterministic seed from string using hash."""
    hash_obj = hashlib.sha256(s.encode())
    return int(hash_obj.hexdigest()[:8], 16)


# ============ Chain A: Weather Tools ============

def execute_get_weather(location: str, unit: str = "celsius") -> dict:
    """Get weather data for a location with deterministic output."""
    seed = _get_seed_from_string(location)
    random.seed(seed)
    
    temperature = round(random.uniform(0, 35), 1)
    condition = random.choice(["sunny", "rainy", "cloudy"])
    humidity = round(random.uniform(30, 90), 1)
    
    return {
        "location": location,
        "temperature": temperature,
        "condition": condition,
        "humidity": humidity
    }


def execute_should_bring_umbrella(weather_data: dict) -> dict:
    """Determine if umbrella should be brought based on weather."""
    condition = weather_data.get("condition", "sunny")
    
    bring_umbrella = condition in ["rainy", "snowy"]
    reason = f"Because the weather is {condition}"
    
    return {
        "bring_umbrella": bring_umbrella,
        "reason": reason
    }


# ============ Chain B: Search -> Extract -> Summarize ============

def execute_web_search(query: str, num_results: int = 5) -> dict:
    """Mock web search returning deterministic results."""
    seed = _get_seed_from_string(query)
    random.seed(seed)
    
    results = []
    for i in range(min(num_results, 5)):
        results.append({
            "title": f"Result {i+1} for {query}",
            "url": f"https://example.com/result{i+1}",
            "snippet": f"This is a mock search result about {query}. It contains relevant information about the topic."
        })
    
    return {"results": results}


def execute_extract_facts(search_results: List[dict], topic: str) -> dict:
    """Extract key facts from search results."""
    seed = _get_seed_from_string(topic)
    random.seed(seed)
    
    facts = [
        f"Fact 1: {topic} is an important subject with many aspects.",
        f"Fact 2: Research shows {topic} has significant impact.",
        f"Fact 3: Studies indicate {topic} continues to evolve.",
        f"Fact 4: Experts agree {topic} requires further study.",
        f"Fact 5: The field of {topic} has grown substantially.",
    ]
    
    return {"facts": facts}


def execute_summarize(facts: List[str], max_length: Optional[int] = 50) -> dict:
    """Summarize facts into a concise summary."""
    if not facts:
        summary = "No facts available to summarize."
    else:
        summary = " ".join(facts[:3])[:max_length] if max_length else " ".join(facts[:3])
    
    return {"summary": summary}


# ============ Chain C: Calendar Tools ============

def execute_parse_date(date_string: str) -> dict:
    """Parse a date string into standardized format."""
    now = datetime.now()
    date_string_lower = date_string.lower()
    
    if "today" in date_string_lower:
        parsed_date = now
    elif "tomorrow" in date_string_lower:
        parsed_date = now + timedelta(days=1)
    elif "monday" in date_string_lower:
        days_ahead = (7 - now.weekday() + 0) % 7
        if days_ahead == 0:
            days_ahead = 7
        parsed_date = now + timedelta(days=days_ahead)
    elif "tuesday" in date_string_lower:
        days_ahead = (7 - now.weekday() + 1) % 7
        if days_ahead == 0:
            days_ahead = 7
        parsed_date = now + timedelta(days=days_ahead)
    elif "wednesday" in date_string_lower:
        days_ahead = (7 - now.weekday() + 2) % 7
        if days_ahead == 0:
            days_ahead = 7
        parsed_date = now + timedelta(days=days_ahead)
    elif "thursday" in date_string_lower:
        days_ahead = (7 - now.weekday() + 3) % 7
        if days_ahead == 0:
            days_ahead = 7
        parsed_date = now + timedelta(days=days_ahead)
    elif "friday" in date_string_lower:
        days_ahead = (7 - now.weekday() + 4) % 7
        if days_ahead == 0:
            days_ahead = 7
        parsed_date = now + timedelta(days=days_ahead)
    else:
        parsed_date = now
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    return {
        "date": parsed_date.strftime("%Y-%m-%d"),
        "day_of_week": days[parsed_date.weekday()]
    }


def execute_check_calendar(date: str) -> dict:
    """Check calendar for events on a specific date."""
    seed = _get_seed_from_string(date)
    random.seed(seed)
    
    events = []
    if random.random() > 0.5:
        events = [
            "Team meeting at 10:00 AM",
            "Lunch with client at 12:30 PM"
        ]
    
    return {
        "date": date,
        "events": events
    }


def execute_find_conflicts(events: List[str], duration_minutes: int = 60) -> dict:
    """Find time conflicts in calendar events."""
    has_conflicts = len(events) >= 2
    
    conflicts = []
    if has_conflicts:
        conflicts = [
            f"Conflict detected: {events[0]} overlaps with {events[1]}" if len(events) > 1 else "No conflicts"
        ]
    
    return {
        "has_conflicts": has_conflicts,
        "conflicts": conflicts
    }


def execute_suggest_time(duration_minutes: int, existing_events: List[str]) -> dict:
    """Suggest available time slots for a new event."""
    seed = _get_seed_from_string(str(duration_minutes) + str(len(existing_events)))
    random.seed(seed)
    
    suggested_times = ["09:00 AM", "02:00 PM", "04:30 PM"]
    reason = f"Suggested {suggested_times[0]} based on available slots"
    
    return {
        "suggested_times": suggested_times,
        "reason": reason
    }


def execute_create_event(time: str, title: str, duration_minutes: int) -> dict:
    """Create a new calendar event."""
    seed = _get_seed_from_string(time + title)
    random.seed(seed)
    
    event_id = f"evt_{random.randint(1000, 9999)}"
    
    return {
        "event_id": event_id,
        "status": "created"
    }


# ============ Main Dispatcher ============

def execute_tool(tool_name: str, params: dict) -> dict:
    """Execute a tool with given parameters. Returns dict output."""
    
    # Chain A
    if tool_name == "get_weather":
        location = params.get("location", "")
        unit = params.get("unit", "celsius")
        return execute_get_weather(location, unit)
    
    elif tool_name == "should_bring_umbrella":
        weather_data = params.get("weather_data", {})
        return execute_should_bring_umbrella(weather_data)
    
    # Chain B
    elif tool_name == "web_search":
        query = params.get("query", "")
        num_results = params.get("num_results", 5)
        return execute_web_search(query, num_results)
    
    elif tool_name == "extract_facts":
        search_results = params.get("search_results", [])
        topic = params.get("topic", "")
        return execute_extract_facts(search_results, topic)
    
    elif tool_name == "summarize":
        facts = params.get("facts", [])
        max_length = params.get("max_length", 50)
        return execute_summarize(facts, max_length)
    
    # Chain C
    elif tool_name == "parse_date":
        date_string = params.get("date_string", "")
        return execute_parse_date(date_string)
    
    elif tool_name == "check_calendar":
        date = params.get("date", "")
        return execute_check_calendar(date)
    
    elif tool_name == "find_conflicts":
        events = params.get("events", [])
        duration_minutes = params.get("duration_minutes", 60)
        return execute_find_conflicts(events, duration_minutes)
    
    elif tool_name == "suggest_time":
        duration_minutes = params.get("duration_minutes", 60)
        existing_events = params.get("existing_events", [])
        return execute_suggest_time(duration_minutes, existing_events)
    
    elif tool_name == "create_event":
        time = params.get("time", "")
        title = params.get("title", "")
        duration_minutes = params.get("duration_minutes", 60)
        return execute_create_event(time, title, duration_minutes)
    
    else:
        raise ValueError(f"Unknown tool: {tool_name}")
