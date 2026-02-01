' VBScript launcher for scheduled sweep - completely hides all windows
' This prevents any console window flashing when run from Task Scheduler

Dim shell, scriptPath, contextsRoot, ntfyTopic, command

' Configuration
scriptPath = "C:\Code\chinvex\scripts\scheduled_sweep.ps1"
contextsRoot = "P:\ai_memory\contexts"
ntfyTopic = "dual-nature"

' Build PowerShell command
command = "pwsh.exe -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File """ & scriptPath & """ -ContextsRoot """ & contextsRoot & """ -NtfyTopic """ & ntfyTopic & """"

' Create shell object
Set shell = CreateObject("WScript.Shell")

' Run command with window style 0 (completely hidden)
' Arguments: command, window_style (0=hidden), wait_for_completion (False=don't wait)
shell.Run command, 0, False

Set shell = Nothing
