# collect data from TI DCA1000 EVM

import time

import serial


CLI_READ_TIMEOUT_S = 1.5
CLI_QUIET_PERIOD_S = 0.05
CLI_POLL_INTERVAL_S = 0.01


class SerialConfig:
    def __init__(self, name, CLIPort, BaudRate):
        self.name = name
        self.CLIPort = serial.Serial(
            CLIPort,
            baudrate=BaudRate,
            timeout=0.2,
            write_timeout=1.0,
        )
        self.CLIPort.reset_input_buffer()
        self.CLIPort.reset_output_buffer()

    def _read_cli_response(self, timeout_s=CLI_READ_TIMEOUT_S, quiet_period_s=CLI_QUIET_PERIOD_S):
        deadline = time.perf_counter() + timeout_s
        last_rx_ts = None
        chunks = []

        while True:
            waiting = getattr(self.CLIPort, "in_waiting", 0)
            if waiting:
                chunk = self.CLIPort.read(waiting)
                if chunk:
                    chunks.append(chunk)
                    last_rx_ts = time.perf_counter()
                    deadline = max(deadline, last_rx_ts + quiet_period_s)
                    continue

            now = time.perf_counter()
            if now >= deadline:
                break
            if last_rx_ts is not None and (now - last_rx_ts) >= quiet_period_s:
                break
            time.sleep(CLI_POLL_INTERVAL_S)

        response_text = b"".join(chunks).decode(errors="ignore")
        response_lines = [
            line.strip()
            for line in response_text.replace("\r", "\n").split("\n")
            if line.strip()
        ]
        return response_text, response_lines

    @staticmethod
    def _has_cli_error(response_lines):
        error_tokens = ("error", "exception", "fail")
        benign_tokens = ("already stopped", "ignored")

        for line in response_lines:
            lowered = line.lower()
            if any(token in lowered for token in benign_tokens):
                continue
            if any(token in lowered for token in error_tokens):
                return True
        return False

    def _send_cli_command(self, command, expect_response=True, raise_on_error=True):
        self.CLIPort.write((command + "\n").encode())
        self.CLIPort.flush()
        print(command)

        response_text, response_lines = self._read_cli_response()
        if expect_response and not response_lines:
            raise RuntimeError(f"No CLI response received for command: {command}")
        if raise_on_error and self._has_cli_error(response_lines):
            raise RuntimeError(
                f"Radar CLI reported an error for command '{command}': "
                + " | ".join(response_lines)
            )
        return response_text, response_lines

    def SendConfig(self, ConfigFileName):
        responses = []
        with open(ConfigFileName) as config_file:
            for line in config_file:
                stripped = line.strip()
                if not stripped or stripped.startswith("%"):
                    continue
                responses.append(self._send_cli_command(stripped, expect_response=True, raise_on_error=True))
                time.sleep(0.01)
        return responses

    def StartRadar(self):
        return self._send_cli_command("sensorStart", expect_response=True, raise_on_error=True)

    def StopRadar(self):
        try:
            return self._send_cli_command("sensorStop", expect_response=False, raise_on_error=False)
        except (serial.SerialException, RuntimeError):
            return "", []

    def DisconnectRadar(self):
        self.StopRadar()
        self.CLIPort.close()
