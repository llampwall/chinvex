import os
import tempfile
from pathlib import Path
from chinvex.gateway.config import load_gateway_config, GatewayConfig


def test_load_default_config_when_file_missing():
    """Should return defaults when config file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_path = Path(tmpdir) / "nonexistent.json"
        os.environ.pop("CHINVEX_GATEWAY_CONFIG", None)

        config = load_gateway_config(str(fake_path))

        assert config.host == "127.0.0.1"
        assert config.port == 7778
        assert config.enabled is True


def test_load_config_from_file():
    """Should load config from JSON file."""
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "gateway.json"
        config_path.write_text(json.dumps({
            "gateway": {
                "host": "0.0.0.0",
                "port": 8080,
                "context_allowlist": ["TestContext"]
            }
        }))

        config = load_gateway_config(str(config_path))

        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.context_allowlist == ["TestContext"]


def test_token_property_reads_from_env():
    """Should read token from environment variable."""
    os.environ["CHINVEX_API_TOKEN"] = "test_token_123"

    config = GatewayConfig()

    assert config.token == "test_token_123"

    del os.environ["CHINVEX_API_TOKEN"]
