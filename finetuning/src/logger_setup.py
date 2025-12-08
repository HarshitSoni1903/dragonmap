# logger_setup.py
"""
Redirect stdout/stderr into a unified log file, with timestamps.

Every printed line becomes:
    [YYYY-MM-DD HH:MM:SS] message

This module should be activated FIRST inside prep.py.
"""

from __future__ import annotations
import sys
import datetime
from pathlib import Path
from typing import Optional
from config_loader import PrepConfig


class TimestampedWriter:
    """
    Wraps a file-like object so that each write() call
    is prefixed with a timestamp.
    """

    def __init__(self, fp):
        self.fp = fp

    def write(self, msg: str):
        if msg.strip():
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.fp.write(f"[{timestamp}] {msg}")
        else:
            # preserve blank lines cleanly
            self.fp.write(msg)

    def flush(self):
        self.fp.flush()


class StreamRedirector:
    """
    Redirect stdout and/or stderr to a timestamped log file.
    Keeps references to originals so we can restore them later.
    """

    def __init__(self, cfg: PrepConfig):
        self.cfg = cfg
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_fp: Optional[object] = None

    def activate(self):
        """
        Begin log redirection with timestamps.
        """
        log_path: Path = self.cfg.log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self.log_fp = open(log_path, "a", encoding="utf-8")

        wrapped = TimestampedWriter(self.log_fp)

        if self.cfg.redirect_stdout:
            sys.stdout = wrapped
        if self.cfg.redirect_stderr:
            sys.stderr = wrapped

        print("=== Logging redirected with timestamps ===")
        print(f"Log file: {log_path}")
        print(f"stdout → {self.cfg.redirect_stdout}, stderr → {self.cfg.redirect_stderr}")
        print("==========================================\n")

    def deactivate(self):
        """
        Restore original stdout/stderr streams.
        """
        if self.log_fp is not None:
            self.log_fp.flush()

        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

        if self.log_fp is not None:
            self.log_fp.close()
            self.log_fp = None


def setup_logging(cfg: PrepConfig) -> StreamRedirector:
    """Used by prep.py to begin timestamped logging."""
    redirector = StreamRedirector(cfg)
    redirector.activate()
    return redirector


# ============================================================
# UNIT TEST
# ============================================================

if __name__ == "__main__":
    from config_loader import load_config

    cfg = load_config()
    r = setup_logging(cfg)

    print("[TEST] This line should appear in log with a timestamp.")
    print("Another line.")

    r.deactivate()
    print("[TEST] Logging restored.")
