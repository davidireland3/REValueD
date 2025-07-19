"""Metrics tracking and aggregation utilities."""
from collections import defaultdict, deque
from typing import Dict, List

import numpy as np


class MetricTracker:
    """Track and aggregate training metrics."""

    def __init__(self, window_size: int = 100):
        """Initialise metric tracker.

        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.all_metrics: Dict[str, List[float]] = defaultdict(list)

    def update(self, **kwargs) -> None:
        """Update metrics with new values.

        Args:
            **kwargs: Metric names and values
        """
        for key, value in kwargs.items():
            self.metrics[key].append(value)
            self.all_metrics[key].append(value)

    def get_average(self, key: str) -> float:
        """Get moving average of metric.

        Args:
            key: Metric name

        Returns:
            Moving average value
        """
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return np.mean(self.metrics[key])

    def get_current(self, key: str) -> float:
        """Get most recent value of metric.

        Args:
            key: Metric name

        Returns:
            Most recent value
        """
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return self.metrics[key][-1]

    def get_all_averages(self) -> Dict[str, float]:
        """Get all moving averages.

        Returns:
            Dictionary of metric names and moving averages
        """
        return {key: self.get_average(key) for key in self.metrics}

    def get_history(self, key: str) -> List[float]:
        """Get full history of metric.

        Args:
            key: Metric name

        Returns:
            List of all values
        """
        return self.all_metrics.get(key, [])

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.all_metrics.clear()
