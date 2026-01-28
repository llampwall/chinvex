"""Gateway configuration loading."""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    requests_per_hour: int = 500


@dataclass
class LimitsConfig:
    max_k: int = 20
    max_chunk_ids: int = 20
    max_query_length: int = 1000
    max_chunk_text_length: int = 5000


@dataclass
class GatewayConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 7778
    token_env: str = "CHINVEX_API_TOKEN"
    context_allowlist: Optional[list[str]] = None
    cors_origins: list[str] = field(default_factory=lambda: [
        "https://chat.openai.com",
        "https://chatgpt.com"
    ])
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    enable_server_llm: bool = False
    audit_log_path: str = "P:\\ai_memory\\gateway_audit.jsonl"

    @property
    def token(self) -> Optional[str]:
        """Get token from environment variable."""
        return os.environ.get(self.token_env)


def load_gateway_config(path: Optional[str] = None) -> GatewayConfig:
    """
    Load gateway configuration from file or defaults.

    Priority:
    1. Explicit path argument
    2. CHINVEX_GATEWAY_CONFIG environment variable
    3. P:\\ai_memory\\gateway.json
    4. Defaults
    """
    if path is None:
        path = os.environ.get("CHINVEX_GATEWAY_CONFIG", "P:\\ai_memory\\gateway.json")

    config_path = Path(path)

    if not config_path.exists():
        return GatewayConfig()

    with open(config_path) as f:
        data = json.load(f)

    gateway_data = data.get("gateway", {})

    # Build rate_limit if present
    rate_limit_data = gateway_data.get("rate_limit", {})
    rate_limit = RateLimitConfig(**rate_limit_data) if rate_limit_data else RateLimitConfig()

    # Build limits if present
    limits_data = gateway_data.get("limits", {})
    limits = LimitsConfig(**limits_data) if limits_data else LimitsConfig()

    return GatewayConfig(
        enabled=gateway_data.get("enabled", True),
        host=gateway_data.get("host", "127.0.0.1"),
        port=gateway_data.get("port", 7778),
        token_env=gateway_data.get("token_env", "CHINVEX_API_TOKEN"),
        context_allowlist=gateway_data.get("context_allowlist"),
        cors_origins=gateway_data.get("cors_origins", [
            "https://chat.openai.com",
            "https://chatgpt.com"
        ]),
        rate_limit=rate_limit,
        limits=limits,
        enable_server_llm=gateway_data.get("enable_server_llm", False)
            or os.environ.get("GATEWAY_ENABLE_SERVER_LLM", "").lower() == "true",
        audit_log_path=gateway_data.get("audit_log_path", "P:\\ai_memory\\gateway_audit.jsonl")
    )
