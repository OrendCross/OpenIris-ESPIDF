#!/usr/bin/env python3
"""
OpenIris JPEG-LS UVC Viewer

Captures raw frames from the OpenIris UVC webcam and decodes them,
supporting both standard JPEG and JPEG-LS (ISO 14495-1) formats.

Usage:
    pip install pillow pillow-jpls opencv-python numpy
    python jpegls_viewer.py

    # Use OpenCV capture (default, works for JPEG; JPEG-LS may show as black):
    python jpegls_viewer.py

    # Use ffmpeg raw capture (needed for JPEG-LS decoding):
    python jpegls_viewer.py --ffmpeg

    # Specify device by index or name:
    python jpegls_viewer.py --device 0
    python jpegls_viewer.py --ffmpeg --device "openiristracker"

    # With serial port for mode switching:
    python jpegls_viewer.py --serial COM12

Controls:
    q     - Quit
    s     - Save current frame to disk
    j     - Switch device to JPEG mode (via serial)
    l     - Switch device to JPEG-LS mode (via serial)
    space - Toggle format info overlay
"""

import subprocess
import sys
import io
import time2
import threading
import argparse
import numpy as np

try:
    from PIL import Image
except ImportError:
    print("ERROR: pip install pillow")
    sys.exit(1)

try:
    import pillow_jpls  # noqa: F401 - registers JPEG-LS codec with Pillow
except ImportError:
    print("ERROR: pip install pillow-jpls")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: pip install opencv-python")
    sys.exit(1)


def list_dshow_devices():
    """List available DirectShow video devices using ffmpeg."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
            capture_output=True, text=True, timeout=10,
        )
        devices = []
        print("\n=== Available DirectShow devices ===")
        for line in result.stderr.splitlines():
            if '"' in line and ("video" in line.lower() or "camera" in line.lower()
                                or "dshow" in line.lower()):
                start = line.find('"')
                end = line.find('"', start + 1)
                if start >= 0 and end > start:
                    name = line[start + 1:end]
                    devices.append(name)
                    print(f"  [{len(devices)-1}] {name}")
        if not devices:
            # Print all stderr for debugging
            print("  (no video devices found)")
            print("\nffmpeg output:")
            for line in result.stderr.splitlines():
                print(f"  {line}")
        print()
        return devices
    except FileNotFoundError:
        print("ERROR: ffmpeg not found in PATH. Install ffmpeg or use --cv2 mode.")
        return []
    except Exception as e:
        print(f"ERROR listing devices: {e}")
        return []


def find_device_name():
    """Auto-detect the OpenIris UVC device via ffmpeg/DirectShow."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stderr.splitlines():
            low = line.lower()
            if "openiristracker" in low or "openiris" in low:
                start = line.find('"')
                end = line.find('"', start + 1)
                if start >= 0 and end > start:
                    return line[start + 1 : end]
    except Exception:
        pass
    return None


def identify_format(data: bytes) -> str:
    """Identify whether raw frame bytes are JPEG or JPEG-LS."""
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        return "unknown"
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xF7:
            return "JPEG-LS"
        if marker in (0xC0, 0xC1, 0xC2):
            return "JPEG"
        if marker == 0xDA:
            break
        if marker in (0x00, 0xFF):
            i += 1
            continue
        if i + 3 < len(data):
            seg_len = (data[i + 2] << 8) | data[i + 3]
            i += 2 + seg_len
        else:
            break
    return "JPEG"


def decode_frame(data: bytes):
    """Decode JPEG or JPEG-LS bytes into a BGR numpy array for cv2."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force decode
        arr = np.array(img)
        if len(arr.shape) == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception as e:
        global _decode_error_count
        _decode_error_count = getattr(sys.modules[__name__], '_decode_error_count', 0) + 1
        if _decode_error_count <= 5:
            print(f"[decode] Pillow error: {e} (size={len(data)}, header={data[:16].hex()})")
        # For JPEG, try OpenCV as fallback
        try:
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None


def extract_frames(stream):
    """
    Yield complete JPEG/JPEG-LS frames from a raw byte stream.
    Scans for SOI (FF D8) and EOI (FF D9) markers.
    """
    buf = bytearray()
    while True:
        chunk = stream.read(8192)
        if not chunk:
            break
        buf.extend(chunk)

        while True:
            soi = buf.find(b"\xff\xd8")
            if soi < 0:
                buf = buf[-1:]
                break

            eoi = buf.find(b"\xff\xd9", soi + 2)
            if eoi < 0:
                if soi > 0:
                    buf = buf[soi:]
                break

            frame = bytes(buf[soi : eoi + 2])
            buf = buf[eoi + 2 :]
            yield frame


class SerialLogReader:
    """Background thread that reads log lines from CDC serial port."""

    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.last_line = ""
        self.lock = threading.Lock()
        self._ser = None
        self._running = False
        self._thread = None

    def start(self):
        try:
            import serial
            self._ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self._ser.reset_input_buffer()
            self._running = True
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            print(f"Serial log reader started on {self.port}")
        except Exception as e:
            print(f"Could not open serial port {self.port}: {e}")

    def _read_loop(self):
        buf = ""
        while self._running:
            try:
                data = self._ser.read(256)
                if data:
                    buf += data.decode(errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line:
                            print(f"[CDC] {line}")
                            if "[JLS]" in line:
                                with self.lock:
                                    self.last_line = line
            except Exception:
                time.sleep(0.1)

    def get_last_timing(self):
        with self.lock:
            return self.last_line

    def write_command(self, command_json):
        """Send a JSON command over the serial port."""
        import json
        if self._ser and self._ser.is_open:
            try:
                self._ser.write((json.dumps(command_json) + "\n").encode())
                time.sleep(0.5)
                return True
            except Exception as e:
                print(f"Serial write error: {e}")
        return False

    def stop(self):
        self._running = False
        if self._ser:
            self._ser.close()


# Global serial log reader (set when --serial is used)
_serial_reader = None


def send_serial_command(port, command_json):
    """Send a JSON command to the device over serial."""
    global _serial_reader
    if _serial_reader:
        _serial_reader.write_command(command_json)
        return None
    try:
        import serial
        import json

        ser = serial.Serial(port, 115200, timeout=2)
        time.sleep(0.3)
        ser.reset_input_buffer()
        ser.write((json.dumps(command_json) + "\n").encode())
        time.sleep(1)
        resp = ser.read(4096).decode(errors="replace")
        ser.close()
        for line in resp.splitlines():
            if line.strip().startswith("{"):
                return json.loads(line.strip())
        return None
    except Exception as e:
        print(f"Serial error: {e}")
        return None


def handle_key(key, args, last_format, last_size, last_frame_bytes, show_overlay):
    """Handle keyboard input. Returns (should_quit, show_overlay)."""
    if key == ord("q"):
        return True, show_overlay
    elif key == ord("s"):
        ext = "jls" if "LS" in last_format else "jpg"
        fname = f"openiris_frame_{int(time.time())}.{ext}"
        with open(fname, "wb") as f:
            f.write(last_frame_bytes)
        print(f"Saved: {fname} ({last_size:,} bytes, {last_format})")
    elif key == ord(" "):
        show_overlay = not show_overlay
    elif key == ord("j") and args.serial:
        print("Switching to JPEG mode...")
        resp = send_serial_command(
            args.serial,
            {"commands": [{"command": "set_encoding_mode", "data": {"mode": 0}}]},
        )
        if resp:
            print(f"  -> {resp}")
    elif key == ord("l") and args.serial:
        print("Switching to JPEG-LS mode...")
        resp = send_serial_command(
            args.serial,
            {"commands": [{"command": "set_encoding_mode", "data": {"mode": 1}}]},
        )
        if resp:
            print(f"  -> {resp}")
    return False, show_overlay


def run_cv2_mode(args):
    """
    Capture using cv2.VideoCapture (DirectShow on Windows).
    Works well for JPEG mode. For JPEG-LS, the DirectShow MJPEG decoder
    can't decode the frames, so they'll appear black/garbled.
    Use --ffmpeg mode for proper JPEG-LS decoding.
    """
    device = args.device
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            pass  # keep as string

    if device is None:
        device = 0  # default to first camera

    print(f"Opening device {device} with OpenCV (DirectShow)...")
    if isinstance(device, int):
        cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"ERROR: Could not open device {device}")
        print("Try specifying --device <index> (0, 1, 2, ...)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # Request MJPEG format from DirectShow
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    print(f"Capture opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("NOTE: OpenCV mode decodes via DirectShow - JPEG-LS frames may appear black.")
    print("      Use --ffmpeg mode for proper JPEG-LS decoding.")
    print("Controls: q=quit  s=save  j=JPEG mode  l=JPEG-LS mode  space=toggle overlay")

    show_overlay = True
    frame_count = 0
    fps_time = time.time()
    fps = 0.0
    last_frame_bytes = b""
    last_format = "JPEG (cv2)"
    last_size = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1
            now = time.time()
            dt = now - fps_time
            if dt >= 1.0:
                fps = frame_count / dt
                frame_count = 0
                fps_time = now

            # Encode current frame for saving
            _, buf = cv2.imencode('.jpg', frame)
            last_frame_bytes = buf.tobytes()
            last_size = len(last_frame_bytes)

            if args.scale != 1.0:
                h, w = frame.shape[:2]
                frame = cv2.resize(
                    frame,
                    (int(w * args.scale), int(h * args.scale)),
                    interpolation=cv2.INTER_NEAREST,
                )

            if show_overlay:
                info = f"{last_format} | {fps:.1f} fps"
                cv2.putText(frame, info, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(frame, info, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                if _serial_reader:
                    enc_info = _serial_reader.get_last_timing()
                    if enc_info:
                        cv2.putText(frame, enc_info, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
                        cv2.putText(frame, enc_info, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            cv2.imshow("OpenIris Viewer (cv2)", frame)

            key = cv2.waitKey(1) & 0xFF
            should_quit, show_overlay = handle_key(
                key, args, last_format, last_size, last_frame_bytes, show_overlay
            )
            if should_quit:
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_ffmpeg_mode(args):
    """
    Capture using ffmpeg with -c:v copy to get raw compressed frames.
    This preserves JPEG-LS data without transcoding, allowing proper decoding
    via Pillow + pillow-jpls.
    """
    device_name = args.device
    if device_name is None:
        print("Searching for OpenIris UVC device...")
        device_name = find_device_name()
    if device_name is None:
        print("Auto-detect failed. Listing available devices...")
        devices = list_dshow_devices()
        if devices:
            device_name = devices[0]
            print(f"Using first device: {device_name}")
        else:
            print("ERROR: No devices found. Specify --device <name>")
            return

    print(f"Using device: {device_name}")

    # Use -f avi with -c:v copy to wrap raw frames in a simple container.
    # AVI doesn't validate JPEG internals, so JPEG-LS passes through intact.
    # We then extract frames by scanning for SOI/EOI markers in the output.
    cmd = [
        "ffmpeg",
        "-f", "dshow",
        "-rtbufsize", "100M",
        "-video_size", f"{args.width}x{args.height}",
        "-i", f"video={device_name}",
        "-c:v", "copy",
        "-f", "avi",
        "pipe:1",
    ]

    print(f"Starting capture: {' '.join(cmd)}")
    print("Controls: q=quit  s=save  j=JPEG mode  l=JPEG-LS mode  space=toggle overlay")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("ERROR: ffmpeg not found in PATH.")
        print("Install ffmpeg: https://ffmpeg.org/download.html")
        return

    # Collect stderr and print it for debugging
    stderr_lines = []
    stderr_lock = threading.Lock()

    def drain_stderr():
        while True:
            line = proc.stderr.readline()
            if not line:
                break
            decoded = line.decode(errors="replace").rstrip()
            with stderr_lock:
                stderr_lines.append(decoded)
            # Print errors/warnings to help debug
            if any(kw in decoded.lower() for kw in ["error", "fail", "cannot", "no such", "invalid"]):
                print(f"[ffmpeg] {decoded}")

    t = threading.Thread(target=drain_stderr, daemon=True)
    t.start()

    # Give ffmpeg a moment to start and potentially fail
    time.sleep(2)

    # Check if ffmpeg already exited
    if proc.poll() is not None:
        print(f"\nERROR: ffmpeg exited with code {proc.returncode}")
        print("\nffmpeg stderr:")
        with stderr_lock:
            for line in stderr_lines:
                print(f"  {line}")
        print("\nTips:")
        print("  - Run with --list to see available devices")
        print("  - Specify device name exactly: --device \"Device Name Here\"")
        print("  - Try cv2 mode (default, without --ffmpeg flag)")
        return

    show_overlay = True
    frame_count = 0
    total_frames = 0
    fps_time = time.time()
    fps = 0.0
    last_format = "?"
    last_size = 0
    last_frame_bytes = b""

    print("Waiting for frames...")

    try:
        for frame_bytes in extract_frames(proc.stdout):
            fmt = identify_format(frame_bytes)
            last_format = fmt
            last_size = len(frame_bytes)
            last_frame_bytes = frame_bytes

            if total_frames == 0:
                print(f"First frame received: {fmt}, {last_size:,} bytes, header: {frame_bytes[:16].hex()}")
                # Auto-save first frame for debugging
                debug_fname = "openiris_debug_frame.jls" if "LS" in fmt else "openiris_debug_frame.jpg"
                with open(debug_fname, "wb") as f:
                    f.write(frame_bytes)
                print(f"  Saved debug frame to {debug_fname}")

            frame = decode_frame(frame_bytes)

            frame_count += 1
            total_frames += 1
            now = time.time()
            dt = now - fps_time
            if dt >= 1.0:
                fps = frame_count / dt
                frame_count = 0
                fps_time = now

            if frame is not None:
                if args.scale != 1.0:
                    h, w = frame.shape[:2]
                    frame = cv2.resize(
                        frame,
                        (int(w * args.scale), int(h * args.scale)),
                        interpolation=cv2.INTER_NEAREST,
                    )

                if show_overlay:
                    info = f"{fmt} | {last_size:,} bytes | {fps:.1f} fps"
                    cv2.putText(frame, info, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                    cv2.putText(frame, info, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                    if _serial_reader:
                        enc_info = _serial_reader.get_last_timing()
                        if enc_info:
                            cv2.putText(frame, enc_info, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
                            cv2.putText(frame, enc_info, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            else:
                dw = int(args.width * args.scale)
                dh = int(args.height * args.scale)
                frame = np.zeros((dh, dw, 3), dtype=np.uint8)
                cv2.putText(frame, f"Decode failed: {fmt}", (10, dh // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(frame, f"Size: {last_size} bytes", (10, dh // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(frame, f"Header: {frame_bytes[:16].hex()}", (10, dh // 2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)

            cv2.imshow("OpenIris JPEG-LS Viewer (ffmpeg)", frame)

            key = cv2.waitKey(1) & 0xFF
            should_quit, show_overlay = handle_key(
                key, args, last_format, last_size, last_frame_bytes, show_overlay
            )
            if should_quit:
                break

    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        cv2.destroyAllWindows()

    if total_frames == 0:
        print("\nNo frames were received.")
        print("\nffmpeg stderr:")
        with stderr_lock:
            for line in stderr_lines[-20:]:
                print(f"  {line}")


def main():
    parser = argparse.ArgumentParser(
        description="OpenIris JPEG-LS Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # OpenCV capture (device 0)
  %(prog)s --device 1               # OpenCV capture (device 1)
  %(prog)s --ffmpeg                  # ffmpeg raw capture (auto-detect device)
  %(prog)s --ffmpeg --device "name"  # ffmpeg with specific device name
  %(prog)s --list                    # List available devices
  %(prog)s --serial COM12            # Enable serial mode switching (j/l keys)
""",
    )
    parser.add_argument(
        "--device", "-d", type=str, default=None,
        help="Device index (0,1,2..) for cv2 mode, or device name for ffmpeg mode",
    )
    parser.add_argument(
        "--ffmpeg", action="store_true",
        help="Use ffmpeg for raw frame capture (needed for JPEG-LS decoding)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available DirectShow devices and exit",
    )
    parser.add_argument(
        "--serial", "-s", type=str, default=None,
        help="Serial port for mode switching commands (e.g. COM12)",
    )
    parser.add_argument(
        "--width", type=int, default=240, help="Frame width (default: 240)",
    )
    parser.add_argument(
        "--height", type=int, default=240, help="Frame height (default: 240)",
    )
    parser.add_argument(
        "--scale", type=float, default=2.0, help="Display scale factor (default: 2.0)",
    )
    args = parser.parse_args()

    if args.list:
        list_dshow_devices()
        return

    print("OpenIris JPEG-LS Viewer")
    print("=" * 40)

    # Start serial log reader if --serial is specified
    global _serial_reader
    if args.serial:
        _serial_reader = SerialLogReader(args.serial)
        _serial_reader.start()

    try:
        if args.ffmpeg:
            run_ffmpeg_mode(args)
        else:
            run_cv2_mode(args)
    finally:
        if _serial_reader:
            _serial_reader.stop()


if __name__ == "__main__":
    main()
