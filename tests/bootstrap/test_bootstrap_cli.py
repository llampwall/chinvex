# tests/bootstrap/test_bootstrap_cli.py
import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from chinvex.bootstrap.cli import bootstrap_install, bootstrap_status, bootstrap_uninstall


def test_bootstrap_install_full_workflow(tmp_path: Path):
    """Install should configure all components"""
    contexts_root = tmp_path / "contexts"
    indexes_root = tmp_path / "indexes"
    ntfy_topic = "chinvex-test"
    profile_path = tmp_path / "profile.ps1"

    with patch('chinvex.bootstrap.cli.set_env_vars') as mock_env, \
         patch('chinvex.bootstrap.cli.inject_dual_function') as mock_profile, \
         patch('chinvex.bootstrap.cli.register_sweep_task') as mock_sweep, \
         patch('chinvex.bootstrap.cli.register_morning_brief_task') as mock_brief, \
         patch('chinvex.bootstrap.cli._create_global_context') as mock_global, \
         patch('chinvex.bootstrap.cli.subprocess.run') as mock_run:

        mock_run.return_value = Mock(returncode=0, stdout="Started")

        bootstrap_install(
            contexts_root=contexts_root,
            indexes_root=indexes_root,
            ntfy_topic=ntfy_topic,
            profile_path=profile_path,
            morning_brief_time="07:00"
        )

        # Verify all components configured
        mock_env.assert_called_once_with(contexts_root, indexes_root, ntfy_topic)
        mock_profile.assert_called_once_with(profile_path)
        mock_sweep.assert_called_once()
        mock_brief.assert_called_once()
        mock_global.assert_called_once()

        # Verify watcher started
        assert any("sync" in str(call) for call in mock_run.call_args_list)


def test_bootstrap_status_shows_components(tmp_path: Path):
    """Status should show state of all components"""
    # Create a mock module for sync.heartbeat
    import sys
    from unittest.mock import MagicMock

    mock_heartbeat_module = MagicMock()
    mock_heartbeat_module.is_alive = MagicMock(return_value=True)
    sys.modules['chinvex.sync.heartbeat'] = mock_heartbeat_module

    try:
        with patch('chinvex.bootstrap.cli.check_task_exists') as mock_check_task, \
             patch('chinvex.bootstrap.cli.os.getenv') as mock_getenv:

            mock_check_task.side_effect = lambda name: name == "ChinvexSweep"
            mock_getenv.side_effect = lambda key, default=None: {
                "CHINVEX_CONTEXTS_ROOT": "P:/ai_memory/contexts",
                "CHINVEX_NTFY_TOPIC": "chinvex-alerts"
            }.get(key, default)

            status = bootstrap_status()

            # Check status dict
            assert status["watcher_running"] is True
            assert status["sweep_task_installed"] is True
            assert status["brief_task_installed"] is False
            assert status["env_vars_set"] is True
    finally:
        # Cleanup
        if 'chinvex.sync.heartbeat' in sys.modules:
            del sys.modules['chinvex.sync.heartbeat']


def test_bootstrap_uninstall_removes_all(tmp_path: Path):
    """Uninstall should remove all components"""
    profile_path = tmp_path / "profile.ps1"

    with patch('chinvex.bootstrap.cli.unset_env_vars') as mock_env, \
         patch('chinvex.bootstrap.cli.remove_dual_function') as mock_profile, \
         patch('chinvex.bootstrap.cli.unregister_task') as mock_unreg, \
         patch('chinvex.bootstrap.cli.subprocess.run') as mock_run:

        mock_run.return_value = Mock(returncode=0, stdout="Stopped")

        bootstrap_uninstall(profile_path=profile_path)

        # Verify all removals
        mock_env.assert_called_once()
        mock_profile.assert_called_once_with(profile_path)
        assert mock_unreg.call_count == 2  # Sweep + brief tasks

        # Verify watcher stopped
        assert any("sync" in str(call) for call in mock_run.call_args_list)
