"""Tool schemas for Chain A (weather -> umbrella), Chain B (search -> extract -> summarize), and Chain C (calendar chain)."""

from typing import TypedDict, Optional, List

# ============ Chain A: Weather Tools ============

class WeatherInput(TypedDict):
    location: str
    unit: Optional[str]

class WeatherOutput(TypedDict):
    location: str
    temperature: float
    condition: str
    humidity: float

WEATHER_SCHEMA = {
    "name": "get_weather",
    "description": "Fetch current weather for a location. Returns temperature, condition, and humidity.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name (e.g., London, Paris, New York)"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit, default is celsius"
            }
        },
        "required": ["location"]
    }
}

class UmbrellaInput(TypedDict):
    weather_data: dict

class UmbrellaOutput(TypedDict):
    bring_umbrella: bool
    reason: str

UMBRELLA_SCHEMA = {
    "name": "should_bring_umbrella",
    "description": "Determine if user should bring an umbrella based on weather data.",
    "parameters": {
        "type": "object",
        "properties": {
            "weather_data": {
                "type": "object",
                "description": "Weather data from get_weather tool (contains location, temperature, condition, humidity)"
            }
        },
        "required": ["weather_data"]
    }
}

CHAIN_A = [WEATHER_SCHEMA, UMBRELLA_SCHEMA]


# ============ Chain B: Search -> Extract -> Summarize ============

class WebSearchInput(TypedDict):
    query: str
    num_results: Optional[int]

class WebSearchOutput(TypedDict):
    results: List[dict]

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": "Search the web for information on a query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "num_results": {
                "type": "number",
                "description": "Number of results to return, default 5"
            }
        },
        "required": ["query"]
    }
}

class ExtractFactsInput(TypedDict):
    search_results: List[dict]
    topic: str

class ExtractFactsOutput(TypedDict):
    facts: List[str]

EXTRACT_FACTS_SCHEMA = {
    "name": "extract_facts",
    "description": "Extract key facts from web search results.",
    "parameters": {
        "type": "object",
        "properties": {
            "search_results": {
                "type": "array",
                "description": "Results from web_search tool"
            },
            "topic": {
                "type": "string",
                "description": "Topic to extract facts about"
            }
        },
        "required": ["search_results", "topic"]
    }
}

class SummarizeInput(TypedDict):
    facts: List[str]
    max_length: Optional[int]

class SummarizeOutput(TypedDict):
    summary: str

SUMMARIZE_SCHEMA = {
    "name": "summarize",
    "description": "Summarize extracted facts into a concise summary.",
    "parameters": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "description": "List of facts to summarize"
            },
            "max_length": {
                "type": "number",
                "description": "Maximum length of summary in words"
            }
        },
        "required": ["facts"]
    }
}

CHAIN_B = [WEB_SEARCH_SCHEMA, EXTRACT_FACTS_SCHEMA, SUMMARIZE_SCHEMA]


# ============ Chain C: Calendar Tools (5-step) ============

class ParseDateInput(TypedDict):
    date_string: str

class ParseDateOutput(TypedDict):
    date: str
    day_of_week: str

PARSE_DATE_SCHEMA = {
    "name": "parse_date",
    "description": "Parse a date string into a standardized format.",
    "parameters": {
        "type": "object",
        "properties": {
            "date_string": {
                "type": "string",
                "description": "Date string to parse (e.g., 'next Monday', 'tomorrow', 'March 15')"
            }
        },
        "required": ["date_string"]
    }
}

class CheckCalendarInput(TypedDict):
    date: str

class CheckCalendarOutput(TypedDict):
    date: str
    events: List[str]

CHECK_CALENDAR_SCHEMA = {
    "name": "check_calendar",
    "description": "Check calendar for events on a specific date.",
    "parameters": {
        "type": "object",
        "properties": {
            "date": {
                "type": "string",
                "description": "Date to check (YYYY-MM-DD format)"
            }
        },
        "required": ["date"]
    }
}

class FindConflictsInput(TypedDict):
    events: List[str]
    duration_minutes: Optional[int]

class FindConflictsOutput(TypedDict):
    has_conflicts: bool
    conflicts: List[str]

FIND_CONFLICTS_SCHEMA = {
    "name": "find_conflicts",
    "description": "Find time conflicts in calendar events.",
    "parameters": {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "description": "List of events to check"
            },
            "duration_minutes": {
                "type": "number",
                "description": "Duration of new event in minutes"
            }
        },
        "required": ["events"]
    }
}

class SuggestTimeInput(TypedDict):
    duration_minutes: int
    existing_events: List[str]

class SuggestTimeOutput(TypedDict):
    suggested_times: List[str]
    reason: str

SUGGEST_TIME_SCHEMA = {
    "name": "suggest_time",
    "description": "Suggest available time slots for a new event.",
    "parameters": {
        "type": "object",
        "properties": {
            "duration_minutes": {
                "type": "number",
                "description": "Duration needed in minutes"
            },
            "existing_events": {
                "type": "array",
                "description": "List of existing events"
            }
        },
        "required": ["duration_minutes", "existing_events"]
    }
}

class CreateEventInput(TypedDict):
    time: str
    title: str
    duration_minutes: int

class CreateEventOutput(TypedDict):
    event_id: str
    status: str

CREATE_EVENT_SCHEMA = {
    "name": "create_event",
    "description": "Create a new calendar event.",
    "parameters": {
        "type": "object",
        "properties": {
            "time": {
                "type": "string",
                "description": "Time for the event (YYYY-MM-DD HH:MM)"
            },
            "title": {
                "type": "string",
                "description": "Title of the event"
            },
            "duration_minutes": {
                "type": "number",
                "description": "Duration in minutes"
            }
        },
        "required": ["time", "title", "duration_minutes"]
    }
}

CHAIN_C = [PARSE_DATE_SCHEMA, CHECK_CALENDAR_SCHEMA, FIND_CONFLICTS_SCHEMA, SUGGEST_TIME_SCHEMA, CREATE_EVENT_SCHEMA]
