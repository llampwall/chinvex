# tests/bootstrap/test_env_vars.py
import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from chinvex.bootstrap.env_vars import set_env_vars, unset_env_vars, validate_paths


def test_validate_paths_creates_missing_dirs(tmp_path: Path):
    """Should create directories if they don't exist"""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"

    # Should not exist yet
    assert not contexts_root.exists()
    assert not indexes_root.exists()

    validate_paths(contexts_root, indexes_root)

    # Should be created
    assert contexts_root.exists()
    assert contexts_root.is_dir()
    assert indexes_root.exists()
    assert indexes_root.is_dir()


def test_set_env_vars_calls_setx(tmp_path: Path):
    """Should call setx to set user environment variables"""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    contexts_root.mkdir()
    indexes_root.mkdir()

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)

        set_env_vars(
            contexts_root=contexts_root,
            indexes_root=indexes_root,
            ntfy_topic="chinvex-alerts"
        )

        # Should have called setx 3 times
        assert mock_run.call_count == 3

        # Extract calls
        calls = [call[0][0] for call in mock_run.call_args_list]

        # Check CHINVEX_CONTEXTS_ROOT set
        contexts_call = [c for c in calls if "CHINVEX_CONTEXTS_ROOT" in c]
        assert len(contexts_call) == 1
        assert str(contexts_root) in contexts_call[0]

        # Check CHINVEX_INDEXES_ROOT set
        indexes_call = [c for c in calls if "CHINVEX_INDEXES_ROOT" in c]
        assert len(indexes_call) == 1

        # Check CHINVEX_NTFY_TOPIC set
        ntfy_call = [c for c in calls if "CHINVEX_NTFY_TOPIC" in c]
        assert len(ntfy_call) == 1
        assert "chinvex-alerts" in ntfy_call[0]


def test_unset_env_vars_removes_variables(tmp_path: Path):
    """Should remove environment variables via reg delete"""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = Mock(returncode=0)

        unset_env_vars()

        # Should delete 3 registry keys
        assert mock_run.call_count == 3

        calls = [call[0][0] for call in mock_run.call_args_list]

        # All should be reg delete commands
        for call in calls:
            assert "reg" in call[0].lower()
            assert "delete" in " ".join(call).lower()
