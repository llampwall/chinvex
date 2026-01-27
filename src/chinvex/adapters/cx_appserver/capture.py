# src/chinvex/adapters/cx_appserver/capture.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

SAMPLE_DIR = Path("debug/appserver_samples")


def capture_raw_response(data: dict, endpoint_name: str, output_dir: Path) -> Path:
    """
    Capture raw API response to file for schema discovery.

    Args:
        data: Raw JSON response
        endpoint_name: Name of endpoint (e.g., 'thread_list', 'thread_resume')
        output_dir: Directory to write samples (default: debug/appserver_samples/)

    Returns:
        Path to written file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{endpoint_name}_{timestamp}.json"
    filepath = output_dir / filename

    filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return filepath


def capture_sample(endpoint: str, response: dict, context_name: str):
    """
    Save raw API response for schema validation.

    Args:
        endpoint: API endpoint name (e.g., "thread_resume")
        response: Raw API response dict
        context_name: Context this sample belongs to
    """
    sample_dir = SAMPLE_DIR / context_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{endpoint.replace('/', '_')}_{timestamp}.json"

    with open(sample_dir / filename, "w") as f:
        json.dump(response, f, indent=2, default=str)
