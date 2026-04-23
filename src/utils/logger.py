"""Lightweight CSV / console logger."""

import csv
import os
import time
from typing import Any, Dict, Optional


class Logger:
    """Logs scalar metrics to both the console and a CSV file.

    Always call :meth:`close` (or use as a context manager) when done to
    guarantee the CSV file handle is flushed and released.
    """

    def __init__(self, log_dir: str, filename: str = "progress.csv") -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self._csv_path = os.path.join(log_dir, filename)
        self._csv_file = None
        self._writer = None
        self._fieldnames = None
        self._start_time = time.time()
        self._closed = False

    # ------------------------------------------------------------------
    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dictionary of scalar values."""
        if self._closed:
            raise RuntimeError("Logger has already been closed.")
        if step is not None:
            data = {"step": step, **data}
        data["wall_time"] = round(time.time() - self._start_time, 2)

        # Lazy CSV initialisation so we can infer headers from the first call
        if self._writer is None:
            self._fieldnames = list(data.keys())
            self._csv_file = open(self._csv_path, "w", newline="")  # noqa: WPS515
            self._writer = csv.DictWriter(
                self._csv_file, fieldnames=self._fieldnames
            )
            self._writer.writeheader()

        self._writer.writerow(data)
        self._csv_file.flush()
        self._print(data)

    # ------------------------------------------------------------------
    def _print(self, data: Dict[str, Any]) -> None:
        parts = [f"{k}: {v}" for k, v in data.items()]
        print(" | ".join(parts))

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush and close the underlying CSV file."""
        if not self._closed and self._csv_file is not None:
            self._csv_file.close()
        self._closed = True

    # ------------------------------------------------------------------
    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *args) -> None:
        self.close()
