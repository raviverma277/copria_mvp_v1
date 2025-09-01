# core/config.py
import os, yaml

DEFAULTS = {
    "use_llm_miner": False,
    "llm_miner_model": "gpt-4o-mini",
}


def get_config() -> dict:
    """
    Load config from config.yaml if present, then apply env overrides.
    Returns a dict with keys:
      - use_llm_miner (bool)
      - llm_miner_model (str)
    """
    cfg = dict(DEFAULTS)

    # Load from YAML file if present
    try:
        with open("config.yaml", "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    except FileNotFoundError:
        pass

    # Env overrides
    if "USE_LLM_MINER" in os.environ:
        cfg["use_llm_miner"] = os.environ["USE_LLM_MINER"].lower() in {
            "1",
            "true",
            "yes",
        }
    if "LLM_MINER_MODEL" in os.environ:
        cfg["llm_miner_model"] = os.environ["LLM_MINER_MODEL"]

    return cfg
