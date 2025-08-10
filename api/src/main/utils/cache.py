import json
import threading
import datetime
from typing import Union

def get_cache_data(tool_call_cache_id: str, tool_call_caches: dict[str, str]) -> str:
    """ Get a specific cache data by name """
    if tool_call_caches and tool_call_cache_id in tool_call_caches:
        return json.dumps(tool_call_caches[tool_call_cache_id], default=str, ensure_ascii=False)
    else:
        return f"Cache '{tool_call_cache_id}' not found"