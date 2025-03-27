"""Job runners for different computing environments."""

from alignmap.cli.runners.pbs import submit_pbs_job

__all__ = [
    "submit_pbs_job"
] 