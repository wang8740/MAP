"""Utility functions for the alignmap package."""

from alignmap.utils.device import (
    get_device,
    get_device_info,
    is_gpu_available,
    set_device_environment,
    get_optimal_device_settings
)

__all__ = [
    "get_device",
    "get_device_info",
    "is_gpu_available",
    "set_device_environment",
    "get_optimal_device_settings"
] 