
import json, pathlib
from core.schemas.active import get_active_name, SCHEMA_DIR


def load_schema_by_name(name: str) -> dict:
    # If the name matches a versioned type, read the ACTIVE target;
    # otherwise fall back to {name}.schema.json
    active = get_active_name(name)
    filename = active if active else f"{name}.schema.json"
    path = SCHEMA_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))
