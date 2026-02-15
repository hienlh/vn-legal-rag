"""
Simple Progress Tracker for Testing.

Minimal implementation that provides disable() method to suppress logging during tests.
This is a simplified version of semantica's ProgressTracker.

Usage:
    >>> from vn_legal_rag.utils import ProgressTracker
    >>> ProgressTracker.get_instance().disable()
"""

import threading
from typing import Optional


class ProgressTracker:
    """
    Simple progress tracker singleton.

    Provides minimal interface for disabling progress output during tests.
    """

    _instance: Optional["ProgressTracker"] = None
    _lock = threading.Lock()

    def __init__(self, enabled: bool = True):
        """Initialize progress tracker."""
        self._enabled = enabled
        self._explicitly_disabled = False

    @property
    def enabled(self) -> bool:
        """Check if progress tracking is enabled."""
        return self._enabled and not self._explicitly_disabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set enabled state."""
        if not self._explicitly_disabled:
            self._enabled = value

    @classmethod
    def get_instance(cls) -> "ProgressTracker":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def disable(self) -> None:
        """Explicitly disable progress tracking."""
        self._explicitly_disabled = True
        self._enabled = False

    def enable(self) -> None:
        """Explicitly enable progress tracking."""
        self._explicitly_disabled = False
        self._enabled = True


def get_progress_tracker() -> ProgressTracker:
    """Get global progress tracker instance."""
    return ProgressTracker.get_instance()
