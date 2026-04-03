from __future__ import annotations

from datetime import datetime
import ctypes
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - snapshot should degrade gracefully
    np = None


PROCESS_PRIORITY_LABELS = {
    0x40: "idle",
    0x4000: "below_normal",
    0x20: "normal",
    0x8000: "above_normal",
    0x80: "high",
    0x100: "realtime",
}

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _run_command(command: list[str], timeout_s: float = 5.0):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def _run_powershell_json(script: str, timeout_s: float = 5.0):
    for executable in ("pwsh", "powershell"):
        output = _run_command(
            [executable, "-NoProfile", "-Command", script],
            timeout_s=timeout_s,
        )
        if output is None:
            continue
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            continue
    return None


def _run_powershell_text(script: str, timeout_s: float = 5.0):
    for executable in ("pwsh", "powershell"):
        output = _run_command(
            [executable, "-NoProfile", "-Command", script],
            timeout_s=timeout_s,
        )
        if output is not None:
            return output
    return None


def _ensure_list(value: Any):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_power_scheme(raw_text: str | None):
    if not raw_text:
        return {
            "active_scheme_guid": None,
            "active_scheme_name": None,
            "recommended_for_benchmarking": None,
            "raw": None,
        }

    guid = None
    name = None
    if ":" in raw_text:
        after_colon = raw_text.split(":", 1)[1].strip()
        if "(" in after_colon and ")" in after_colon:
            guid, remainder = after_colon.split("(", 1)
            guid = guid.strip()
            name = remainder.rsplit(")", 1)[0].strip()
        else:
            guid = after_colon

    normalized_name = (name or "").strip()
    lowered = normalized_name.lower()
    recommended = None
    if normalized_name:
        recommended = any(
            token in lowered
            for token in (
                "high performance",
                "ultimate performance",
                "best performance",
                "고성능",
                "최고 성능",
            )
        )

    return {
        "active_scheme_guid": guid,
        "active_scheme_name": name,
        "recommended_for_benchmarking": recommended,
        "raw": raw_text,
    }


def _process_priority_snapshot():
    if os.name != "nt":
        return {"priority_class": None, "priority_class_code": None}

    try:
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        priority_code = int(ctypes.windll.kernel32.GetPriorityClass(handle))
    except Exception:  # pragma: no cover - best effort only
        priority_code = 0

    return {
        "priority_class": PROCESS_PRIORITY_LABELS.get(priority_code, "unknown" if priority_code else None),
        "priority_class_code": priority_code or None,
    }


def _ipconfig_addresses():
    output = _run_command(["ipconfig"], timeout_s=5.0)
    if not output:
        return []
    addresses = []
    for line in output.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        if "ipv4" not in label.lower():
            continue
        candidate = value.strip()
        if candidate:
            addresses.append(candidate)
    return sorted(set(addresses))


def _build_windows_runtime_snapshot(expected_host_ip: str | None, target_pid: int):
    inventory = _run_powershell_json(
        f"""
        $ipConfig = Get-NetIPConfiguration | ForEach-Object {{
          [PSCustomObject]@{{
            InterfaceAlias = $_.InterfaceAlias
            InterfaceDescription = $_.InterfaceDescription
            IPv4Addresses = @($_.IPv4Address | ForEach-Object {{ $_.IPAddress }})
            Status = if ($_.NetAdapter) {{ $_.NetAdapter.Status }} else {{ $null }}
            LinkSpeed = if ($_.NetAdapter) {{ $_.NetAdapter.LinkSpeed }} else {{ $null }}
          }}
        }}
        $adapters = Get-NetAdapter | Select-Object Name, InterfaceDescription, Status, LinkSpeed, MacAddress
        $firewall = Get-NetFirewallProfile | Select-Object Name, Enabled, DefaultInboundAction, DefaultOutboundAction
        $priority = $null
        try {{
          $priority = (Get-Process -Id {target_pid}).PriorityClass.ToString()
        }} catch {{
          $priority = $null
        }}
        [PSCustomObject]@{{
          ProcessPriorityClass = $priority
          IpConfiguration = $ipConfig
          Adapters = $adapters
          FirewallProfiles = $firewall
        }} | ConvertTo-Json -Depth 6
        """,
        timeout_s=8.0,
    )

    inventory = inventory or {}
    ip_config_rows = _ensure_list(inventory.get("IpConfiguration"))
    adapter_rows = _ensure_list(inventory.get("Adapters"))
    firewall_rows = _ensure_list(inventory.get("FirewallProfiles"))

    ipv4_addresses: list[str] = []
    for row in ip_config_rows:
        for address in _ensure_list((row or {}).get("IPv4Addresses")):
            if address:
                ipv4_addresses.append(str(address))

    if not ipv4_addresses:
        ipv4_addresses = _ipconfig_addresses()

    ipv4_addresses = sorted(set(ipv4_addresses))
    host_ip_present = None
    if expected_host_ip:
        host_ip_present = expected_host_ip in ipv4_addresses

    return {
        "process_priority_class": inventory.get("ProcessPriorityClass"),
        "expected_host_ip": expected_host_ip,
        "host_ip_present": host_ip_present,
        "ipv4_addresses": ipv4_addresses,
        "ip_configuration": ip_config_rows,
        "adapters": adapter_rows,
        "firewall_profiles": firewall_rows,
    }


def capture_system_snapshot(expected_host_ip: str | None = None):
    power = _parse_power_scheme(_run_command(["powercfg", "/getactivescheme"], timeout_s=4.0))
    network = _build_windows_runtime_snapshot(expected_host_ip, os.getpid())
    priority_snapshot = _process_priority_snapshot()
    process_priority_class = network.pop("process_priority_class", None) or priority_snapshot.get("priority_class")

    return {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "cwd": str(Path.cwd()),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "numpy_version": None if np is None else np.__version__,
        },
        "process": {
            "pid": os.getpid(),
            "cpu_count_logical": os.cpu_count(),
            "priority_class": process_priority_class,
            "priority_class_code": priority_snapshot.get("priority_class_code"),
        },
        "power": power,
        "network": network,
        "env": {
            key: os.environ.get(key)
            for key in THREAD_ENV_KEYS
        },
    }
