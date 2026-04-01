from __future__ import annotations

from dataclasses import dataclass
import socket
import time


def build_command(
    code: str,
    *,
    packet_size_bytes: int,
    packet_delay_us: int,
    packet_delay_ticks_per_us: int,
) -> bytes:
    code_map = {
        '3': (0x03).to_bytes(2, byteorder='little', signed=False),
        '5': (0x05).to_bytes(2, byteorder='little', signed=False),
        '6': (0x06).to_bytes(2, byteorder='little', signed=False),
        'B': (0x0B).to_bytes(2, byteorder='little', signed=False),
        '9': (0x09).to_bytes(2, byteorder='little', signed=False),
    }
    if code not in code_map:
        raise ValueError(f'Unsupported DCA1000 command code: {code}')

    header = (0xA55A).to_bytes(2, byteorder='little', signed=False)
    footer = (0xEEAA).to_bytes(2, byteorder='little', signed=False)
    data_size_0 = (0x00).to_bytes(2, byteorder='little', signed=False)
    data_size_6 = (0x06).to_bytes(2, byteorder='little', signed=False)
    data_fpga_config = (0x01020102031E).to_bytes(6, byteorder='big', signed=False)
    packet_delay_ticks = int(packet_delay_us * packet_delay_ticks_per_us)
    data_packet_config = (
        int(packet_size_bytes).to_bytes(2, byteorder='little', signed=False)
        + int(packet_delay_ticks).to_bytes(2, byteorder='little', signed=False)
        + (0).to_bytes(2, byteorder='little', signed=False)
    )

    if code in ('9', '5', '6'):
        return header + code_map[code] + data_size_0 + footer
    if code == '3':
        return header + code_map[code] + data_size_6 + data_fpga_config + footer
    return header + code_map[code] + data_size_6 + data_packet_config + footer


@dataclass(frozen=True)
class DcaResponse:
    command_code: int
    status: int
    payload: bytes


def parse_dca_response(response_bytes: bytes) -> DcaResponse:
    if len(response_bytes) < 8:
        raise RuntimeError(f'DCA1000 response too short: {len(response_bytes)} bytes')

    header = int.from_bytes(response_bytes[:2], byteorder='little', signed=False)
    command_code = int.from_bytes(response_bytes[2:4], byteorder='little', signed=False)
    status = int.from_bytes(response_bytes[4:6], byteorder='little', signed=False)
    footer = int.from_bytes(response_bytes[-2:], byteorder='little', signed=False)

    if header != 0xA55A or footer != 0xEEAA:
        raise RuntimeError(
            'Invalid DCA1000 response packet '
            f'(header=0x{header:04X}, footer=0x{footer:04X})'
        )

    payload = response_bytes[6:-2]
    return DcaResponse(command_code=command_code, status=status, payload=payload)


class DcaConfigClient:
    def __init__(
        self,
        *,
        host_ip: str,
        config_port: int,
        fpga_ip: str,
        fpga_port: int,
        timeout_s: float,
        packet_size_bytes: int,
        packet_delay_us: int,
        packet_delay_ticks_per_us: int,
        event_callback=None,
    ):
        self.config_address = (host_ip, int(config_port))
        self.fpga_address = (fpga_ip, int(fpga_port))
        self.timeout_s = float(timeout_s)
        self.packet_size_bytes = int(packet_size_bytes)
        self.packet_delay_us = int(packet_delay_us)
        self.packet_delay_ticks_per_us = int(packet_delay_ticks_per_us)
        self.event_callback = event_callback
        self.socket_handle = None

    def _emit(self, event_type: str, **payload):
        if self.event_callback is not None:
            self.event_callback(event_type, **payload)

    def open(self):
        if self.socket_handle is not None:
            return
        self.socket_handle = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket_handle.bind(self.config_address)
        self.socket_handle.settimeout(self.timeout_s)

    def configure(self):
        self.open()
        self._emit(
            'dca_config_start',
            config_address=f'{self.config_address[0]}:{self.config_address[1]}',
            fpga_address=f'{self.fpga_address[0]}:{self.fpga_address[1]}',
        )
        for command in ('9', '3', 'B', '5'):
            self.send_command(command)
        self._emit('dca_config_complete')

    def send_command(self, command: str) -> DcaResponse:
        if self.socket_handle is None:
            raise RuntimeError('DCA1000 config socket is not open.')

        expected_command_code = int(command, 16)
        self.socket_handle.sendto(
            build_command(
                command,
                packet_size_bytes=self.packet_size_bytes,
                packet_delay_us=self.packet_delay_us,
                packet_delay_ticks_per_us=self.packet_delay_ticks_per_us,
            ),
            self.fpga_address,
        )
        response_bytes, _ = self.socket_handle.recvfrom(2048)
        response = parse_dca_response(response_bytes)

        if response.command_code != expected_command_code:
            raise RuntimeError(
                f'DCA1000 response command mismatch for 0x{expected_command_code:02X}: '
                f'got 0x{response.command_code:02X}'
            )
        if response.status != 0:
            raise RuntimeError(
                f'DCA1000 command 0x{expected_command_code:02X} failed '
                f'with status 0x{response.status:04X}'
            )

        self._emit(
            'dca_command_ok',
            command=f'0x{expected_command_code:02X}',
            status=f'0x{response.status:04X}',
        )
        time.sleep(0.05)
        return response

    def stop_stream(self):
        if self.socket_handle is None:
            return
        try:
            self.socket_handle.sendto(
                build_command(
                    '6',
                    packet_size_bytes=self.packet_size_bytes,
                    packet_delay_us=self.packet_delay_us,
                    packet_delay_ticks_per_us=self.packet_delay_ticks_per_us,
                ),
                self.fpga_address,
            )
        except OSError:
            pass

    def close(self):
        if self.socket_handle is None:
            return
        self.socket_handle.close()
        self.socket_handle = None
