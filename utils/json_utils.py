
import json
import re
from typing import Any, Dict, List, Union

def extract_json(text: str) -> Union[Dict, List, None]:
    """
    Extracts and parses JSON from a string that might contain other text.
    Handles markdown blocks, comments, and pre/post-amble.
    """
    if not text:
        return None
        
    # 1. Try direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Extract from markdown blocks
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. Find first { or [ and last } or ]
    try:
        # Find start
        start_brace = text.find('{')
        start_bracket = text.find('[')
        
        if start_brace == -1 and start_bracket == -1:
            return None
            
        start = 0
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            start = start_brace
            end = text.rfind('}') + 1
        else:
            start = start_bracket
            end = text.rfind(']') + 1
            
        if end > start:
            json_str = text[start:end]
            return json.loads(json_str)
            
    except json.JSONDecodeError:
        pass
        
    return None
