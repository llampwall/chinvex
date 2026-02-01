"""Windows Task Scheduler integration."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def register_sweep_task(
    script_path: Path,
    contexts_root: Path,
    ntfy_topic: str = ""
) -> None:
    """
    Register scheduled sweep task in Windows Task Scheduler.

    Args:
        script_path: Path to scheduled_sweep.ps1
        contexts_root: Contexts root directory
        ntfy_topic: ntfy.sh topic for alerts
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Generate task XML
    xml_content = _generate_task_xml(script_path, contexts_root, ntfy_topic)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        xml_path = f.name

    try:
        # Register task
        cmd = [
            "schtasks",
            "/Create",
            "/TN", "ChinvexSweep",
            "/XML", xml_path,
            "/F"  # Force overwrite if exists
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to register task: {result.stderr}")

        log.info("Registered ChinvexSweep task")

    finally:
        # Clean up temp file
        Path(xml_path).unlink(missing_ok=True)


def unregister_sweep_task() -> None:
    """Remove ChinvexSweep task from Task Scheduler."""
    cmd = [
        "schtasks",
        "/Delete",
        "/TN", "ChinvexSweep",
        "/F"  # Force without confirmation
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Task might not exist
        if "cannot find" in result.stderr.lower():
            log.info("ChinvexSweep task not found")
        else:
            raise RuntimeError(f"Failed to unregister task: {result.stderr}")
    else:
        log.info("Unregistered ChinvexSweep task")


def _generate_task_xml(
    script_path: Path,
    contexts_root: Path,
    ntfy_topic: str
) -> str:
    """Generate Task Scheduler XML definition."""
    from datetime import datetime

    # Find VBScript launcher (create if doesn't exist)
    vbs_launcher = script_path.parent / "scheduled_sweep_launcher.vbs"
    if not vbs_launcher.exists():
        # Create VBScript launcher
        vbs_content = f"""' VBScript launcher for scheduled sweep - completely hides all windows
' This prevents any console window flashing when run from Task Scheduler

Dim shell, scriptPath, contextsRoot, ntfyTopic, command

' Configuration
scriptPath = "{script_path}"
contextsRoot = "{contexts_root}"
ntfyTopic = "{ntfy_topic}"

' Build PowerShell command
command = "pwsh.exe -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File """ & scriptPath & """ -ContextsRoot """ & contextsRoot & """"
If ntfyTopic <> "" Then
    command = command & " -NtfyTopic """ & ntfyTopic & """"
End If

' Create shell object
Set shell = CreateObject("WScript.Shell")

' Run command with window style 0 (completely hidden)
shell.Run command, 0, False

Set shell = Nothing
"""
        vbs_launcher.write_text(vbs_content, encoding='utf-8')
        log.info(f"Created VBScript launcher: {vbs_launcher}")

    # Use current time for StartBoundary
    start_boundary = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Chinvex scheduled sweep - ensures watcher running and syncs contexts</Description>
    <URI>\\ChinvexSweep</URI>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <Repetition>
        <Interval>PT30M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>{start_boundary}</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>true</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT10M</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>wscript.exe</Command>
      <Arguments>"{vbs_launcher}" //B //Nologo</Arguments>
    </Exec>
  </Actions>
  <Principals>
    <Principal>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
</Task>"""

    return xml


def register_login_trigger_task() -> None:
    """
    Register task that starts sync watcher at user login.

    This is the PRIMARY mechanism for ensuring watcher is running.
    Sweep task is the backup recovery mechanism.
    """
    xml_content = _generate_login_trigger_xml()

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        xml_path = f.name

    try:
        cmd = [
            "schtasks",
            "/Create",
            "/TN", "ChinvexWatcherStart",
            "/XML", xml_path,
            "/F"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to register login trigger: {result.stderr}")

        log.info("Registered ChinvexWatcherStart task (login trigger)")

    finally:
        Path(xml_path).unlink(missing_ok=True)


def unregister_login_trigger_task() -> None:
    """Remove ChinvexWatcherStart task."""
    cmd = [
        "schtasks",
        "/Delete",
        "/TN", "ChinvexWatcherStart",
        "/F"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if "cannot find" in result.stderr.lower():
            log.info("ChinvexWatcherStart task not found")
        else:
            raise RuntimeError(f"Failed to unregister task: {result.stderr}")
    else:
        log.info("Unregistered ChinvexWatcherStart task")


def _generate_login_trigger_xml() -> str:
    """Generate Task Scheduler XML for login trigger."""
    xml = """<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Chinvex watcher auto-start at user login</Description>
    <URI>\\ChinvexWatcherStart</URI>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT1M</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>chinvex</Command>
      <Arguments>sync ensure-running</Arguments>
    </Exec>
  </Actions>
  <Principals>
    <Principal>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
</Task>"""

    return xml


def register_morning_brief_task(
    contexts_root: Path,
    ntfy_topic: str,
    time: str = "07:00"
) -> None:
    """
    Register morning brief task in Windows Task Scheduler.

    Args:
        contexts_root: Contexts root directory
        ntfy_topic: ntfy.sh topic for morning brief
        time: Time to run (HH:MM format, default 07:00)
    """
    script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "morning_brief.ps1"
    if not script_path.exists():
        raise FileNotFoundError(f"Morning brief script not found: {script_path}")

    # Generate task XML
    xml_content = _generate_morning_brief_xml(script_path, contexts_root, ntfy_topic, time)

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
        f.write(xml_content)
        xml_path = f.name

    try:
        # Register task
        cmd = [
            "schtasks",
            "/Create",
            "/TN", "ChinvexMorningBrief",
            "/XML", xml_path,
            "/F"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to register morning brief task: {result.stderr}")

        log.info(f"Registered ChinvexMorningBrief task (daily at {time})")

    finally:
        Path(xml_path).unlink(missing_ok=True)


def unregister_morning_brief_task() -> None:
    """Remove ChinvexMorningBrief task from Task Scheduler."""
    cmd = [
        "schtasks",
        "/Delete",
        "/TN", "ChinvexMorningBrief",
        "/F"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if "cannot find" in result.stderr.lower():
            log.info("ChinvexMorningBrief task not found")
        else:
            raise RuntimeError(f"Failed to unregister task: {result.stderr}")
    else:
        log.info("Unregistered ChinvexMorningBrief task")


def _generate_morning_brief_xml(
    script_path: Path,
    contexts_root: Path,
    ntfy_topic: str,
    time: str
) -> str:
    """Generate Task Scheduler XML for morning brief."""
    from datetime import datetime, timedelta

    # Build arguments
    args_list = [f"-ContextsRoot \"{contexts_root}\""]
    if ntfy_topic:
        args_list.append(f"-NtfyTopic \"{ntfy_topic}\"")

    args_str = " ".join(args_list)

    # Parse time (HH:MM)
    hour, minute = time.split(":")

    # Use today's date with specified time, or tomorrow if time has passed
    now = datetime.now()
    target_time = now.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)

    if target_time <= now:
        # Time has passed today, use tomorrow
        target_time += timedelta(days=1)

    start_boundary = target_time.strftime("%Y-%m-%dT%H:%M:%S")

    xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Chinvex morning brief - daily status summary</Description>
    <URI>\\ChinvexMorningBrief</URI>
  </RegistrationInfo>
  <Triggers>
    <CalendarTrigger>
      <Repetition>
        <Interval>P1D</Interval>
      </Repetition>
      <StartBoundary>{start_boundary}</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>true</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT5M</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions>
    <Exec>
      <Command>pwsh</Command>
      <Arguments>-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{script_path}" {args_str}</Arguments>
    </Exec>
  </Actions>
  <Principals>
    <Principal>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
</Task>"""

    return xml
