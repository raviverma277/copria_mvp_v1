import os
import time

_last_llm_call = None
_last_llm_duration = None
_last_llm_model = None
_last_llm_success = None

def record_llm_call_start(model_name: str):
    global _last_llm_call, _last_llm_model
    _last_llm_call = time.time()
    _last_llm_model = model_name
    _last_llm_success = None

def record_llm_call_end(success: bool):
    global _last_llm_duration, _last_llm_success
    if _last_llm_call:
        _last_llm_duration = time.time() - _last_llm_call
    _last_llm_success = success

def get_llm_status():
    return {
        "api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "model": _last_llm_model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "last_call": (
            time.strftime("%H:%M:%S", time.localtime(_last_llm_call))
            if _last_llm_call else None
        ),
        "last_duration": round(_last_llm_duration, 2) if _last_llm_duration else None,
        "last_success": _last_llm_success
    }
