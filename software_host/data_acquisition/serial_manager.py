"""
Serial communication manager for STM32-based EIS device.
"""

import csv
import serial
import threading
import time
import logging
from typing import Callable, List, Optional, Tuple
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class SerialManager:
    """Manages serial communication with the STM32 EIS device."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: int = 1):
        """
        Initialize serial connection manager.

        Args:
            port: COM port (e.g., 'COM3')
            baudrate: Serial connection speed
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn: Optional[serial.Serial] = None
        self.is_connected = False
        self.read_thread: Optional[threading.Thread] = None
        self.write_thread: Optional[threading.Thread] = None
        self.running = False

        # Queues for thread-safe communication
        self.read_queue: Queue = Queue()
        self.write_queue: Queue = Queue()
        self.on_data_received: Optional[Callable] = None

    def connect(self) -> bool:
        """
        Establish serial connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            self.is_connected = True
            self.running = True

            # Start communication threads
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.write_thread = threading.Thread(target=self._write_loop, daemon=True)
            self.read_thread.start()
            self.write_thread.start()

            logger.info(f"Connected to {self.port} at {self.baudrate} baud")
            return True

        except serial.SerialException as e:
            logger.error(f"Failed to connect to {self.port}: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from serial device."""
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False
        logger.info(f"Disconnected from {self.port}")

    def send_command(self, command: str):
        """
        Queue a command to send to the device.

        Args:
            command: Command string to send
        """
        self.write_queue.put(command)

    def _read_loop(self):
        """Background thread for reading data from serial port."""
        buffer = ""
        while self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting:
                    data = self.serial_conn.read(self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
                    buffer += data

                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            self.read_queue.put(line)
                            if self.on_data_received:
                                self.on_data_received(line)

            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                time.sleep(0.1)

    def _write_loop(self):
        """Background thread for writing commands to serial port."""
        while self.running:
            try:
                try:
                    command = self.write_queue.get(timeout=0.1)
                    if self.serial_conn and self.serial_conn.is_open:
                        self.serial_conn.write((command + '\n').encode('utf-8'))
                        self.serial_conn.flush()
                        logger.debug(f"Sent command: {command}")
                except Empty:
                    pass
            except Exception as e:
                logger.error(f"Error in write loop: {e}")
                time.sleep(0.1)

    def get_data(self, timeout: float = 0.1) -> Optional[str]:
        """
        Get data from the read queue (non-blocking).

        Args:
            timeout: Timeout in seconds

        Returns:
            Data string or None if no data available
        """
        try:
            return self.read_queue.get(timeout=timeout)
        except Empty:
            return None

    def clear_buffers(self):
        """Clear all data queues."""
        while not self.read_queue.empty():
            try:
                self.read_queue.get_nowait()
            except Empty:
                break


# ============================================================================
# Enhanced SerialReader Thread (inspired by old_SeriPlot.py)
# ============================================================================

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


def parse_line_values(line: str) -> Optional[List[float]]:
    """
    Parse a raw serial line (e.g. "0, 1306.478,-12.374,1306.478,-12.374")
    into a list of floats, or None if the line isn't numeric (startup
    banners, log text, etc.).
    """
    parts = line.split(",")
    try:
        return [float(p.strip()) for p in parts]
    except ValueError:
        return None


class SerialReaderThread(QThread):
    """
    Real-time serial data reader thread.
    
    Reads serial port line-by-line, parses impedance/signal data,
    and emits signals for GUI updates. 
    Compatible with both EIS impedance mode and signal mode.
    Inspired by old_SeriPlot.py for robust real-time data handling.
    """

    # Signal mode: full frame of (N, 2) array [adc1, adc2]
    signal_frame = pyqtSignal(np.ndarray)

    # Impedance mode: full sweep of (M, 5) array [idx, mag, phase, accMag, accPhase]
    impedance_sweep = pyqtSignal(np.ndarray)

    # Detected mode string + point count
    mode_detected = pyqtSignal(str, int)

    # Raw line for console
    raw_line = pyqtSignal(str)

    # Status / error messages
    status_msg = pyqtSignal(str)

    def __init__(self, port: str, baudrate: int = 115200, parent=None):
        """
        Initialize serial reader thread.

        Args:
            port: Serial port name
            baudrate: Communication speed
            parent: Parent QObject
        """
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate
        self._running = False

    def run(self):
        """Main thread execution loop."""
        self._running = True
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
        except serial.SerialException as e:
            self.status_msg.emit(f"Serial error: {e}")
            return

        self.status_msg.emit(f"Connected to {self.port} @ {self.baudrate}")

        signal_buf = []
        impedance_buf = []
        last_idx = -1
        current_mode = None

        while self._running:
            try:
                raw = ser.readline()
            except serial.SerialException:
                self.status_msg.emit("Serial connection lost")
                break

            if not raw:
                continue

            try:
                line = raw.decode("ascii", errors="ignore").strip()
            except Exception:
                continue

            if not line:
                continue

            # Emit raw line for console
            self.raw_line.emit(line)

            values = parse_line_values(line)
            if values is None:
                # Text line (startup messages etc.) – skip
                continue

            n_fields = len(values)

            # --- Signal mode: 2 fields (ADC1, ADC2) ---
            if n_fields == 2:
                if current_mode != "signal":
                    current_mode = "signal"
                    signal_buf.clear()
                signal_buf.append(values)
                if len(signal_buf) >= 2048:
                    arr = np.array(signal_buf[:2048])
                    self.signal_frame.emit(arr)
                    self.mode_detected.emit("signal", 2048)
                    signal_buf.clear()

            # --- Impedance mode: 5 fields (idx, mag, phase, accMag, accPhase) ---
            elif n_fields == 5:
                if current_mode != "impedance":
                    current_mode = "impedance"
                    impedance_buf.clear()
                    last_idx = -1

                idx = int(values[0])

                # Detect sweep boundary: index resets to 0
                if idx == 0 and impedance_buf:
                    arr = np.array(impedance_buf)
                    self.impedance_sweep.emit(arr)
                    self.mode_detected.emit("impedance", len(impedance_buf))
                    impedance_buf.clear()

                impedance_buf.append(values)
                last_idx = idx

        ser.close()
        self.status_msg.emit("Disconnected")

    def stop(self):
        """Stop the reader thread."""
        self._running = False
        self.wait(2000)


# ============================================================================
# CSV Replay: rebuild sweeps/frames from a file exported by this GUI
# ============================================================================
#
# Two export formats exist in the GUI (CleanMainWindow):
#   - "Record" (RECORD button): header ["timestamp", "data"], where "data" is
#     the raw serial line exactly as received from the STM32.
#   - "Buffer" (SAVE/EXPORT button): header
#     ["sweep", "idx", "mag", "phase", "acc_mag", "acc_phase"].
#
# load_recorded_csv() detects which one it's looking at and reconstructs the
# same (M, 5) impedance-sweep / (N, 2) signal-frame arrays that
# SerialReaderThread emits live, so old recordings can be replotted.

def load_recorded_csv(filepath: str) -> Tuple[List["np.ndarray"], List["np.ndarray"]]:
    """
    Rebuild impedance sweeps (and signal frames, if any) from a previously
    exported CSV file.

    Returns:
        (impedance_sweeps, signal_frames): lists of numpy arrays, in
        chronological order. impedance_sweeps entries are (M, 5) arrays of
        [idx, mag, phase, accMag, accPhase]; signal_frames entries are
        (N, 2) arrays of [adc1, adc2].
    """
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        rows = [row for row in reader if row]

    header_norm = [h.strip().lower() for h in header]

    if header_norm[:2] == ["sweep", "idx"]:
        return _load_buffer_format_rows(rows), []

    return _load_record_format_rows(rows)


def _load_buffer_format_rows(rows: List[List[str]]) -> List["np.ndarray"]:
    """Parse rows from the SAVE/EXPORT "buffer" CSV format."""
    sweeps = {}
    order = []
    for row in rows:
        if len(row) < 6:
            continue
        try:
            sweep_id = int(float(row[0]))
            values = [float(v) for v in row[1:6]]
        except ValueError:
            continue
        if sweep_id not in sweeps:
            sweeps[sweep_id] = []
            order.append(sweep_id)
        sweeps[sweep_id].append(values)

    return [np.array(sweeps[sid]) for sid in order if sweeps[sid]]


def _load_record_format_rows(
    rows: List[List[str]],
) -> Tuple[List["np.ndarray"], List["np.ndarray"]]:
    """Parse rows from the RECORD "timestamp,data" raw-line CSV format."""
    impedance_sweeps = []
    signal_frames = []

    impedance_buf = []
    signal_buf = []
    current_mode = None

    for row in rows:
        if len(row) < 2:
            continue
        values = parse_line_values(row[1])
        if values is None:
            continue

        n_fields = len(values)

        if n_fields == 2:
            if current_mode != "signal":
                current_mode = "signal"
                if impedance_buf:
                    impedance_sweeps.append(np.array(impedance_buf))
                    impedance_buf = []
            signal_buf.append(values)
            if len(signal_buf) >= 2048:
                signal_frames.append(np.array(signal_buf[:2048]))
                signal_buf = []

        elif n_fields == 5:
            if current_mode != "impedance":
                current_mode = "impedance"
                if signal_buf:
                    signal_frames.append(np.array(signal_buf))
                    signal_buf = []

            idx = int(values[0])
            if idx == 0 and impedance_buf:
                impedance_sweeps.append(np.array(impedance_buf))
                impedance_buf = []
            impedance_buf.append(values)

    # Unlike the live reader, there's no "next sweep" coming to trigger a
    # final flush - save whatever is left in the buffers so the last
    # (possibly partial) sweep/frame in the file isn't lost.
    if impedance_buf:
        impedance_sweeps.append(np.array(impedance_buf))
    if signal_buf:
        signal_frames.append(np.array(signal_buf))

    return impedance_sweeps, signal_frames
