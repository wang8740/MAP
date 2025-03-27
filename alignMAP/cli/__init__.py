"""Command-line interfaces for AlignMAP.

This package provides command-line interfaces for using the AlignMAP framework.
"""

from alignmap.cli.align import AlignValuesCommand, run_cli as run_align_cli

__all__ = [
    'AlignValuesCommand',
    'run_align_cli',
] 