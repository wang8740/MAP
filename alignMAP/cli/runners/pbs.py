"""PBS job runner for alignmap."""

import os
import random
import string
import logging
import subprocess
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

def generate_random_string(length: int = 8) -> str:
    """Generate a random string of specified length.
    
    Args:
        length (int): Length of random string to generate
        
    Returns:
        str: Random string
    """
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def submit_pbs_job(
    commands: List[str],
    job_name: Optional[str] = None,
    template_path: Optional[str] = None,
    output_dir: str = "pbs-jobs",
    resources: Optional[Dict[str, Any]] = None,
    submit: bool = True,
    notify_email: Optional[str] = None
) -> str:
    """Submit a job to a PBS cluster.
    
    Args:
        commands (List[str]): Commands to run in the job
        job_name (Optional[str]): Name for the job
        template_path (Optional[str]): Path to PBS template file
        output_dir (str): Directory to save job files
        resources (Optional[Dict[str, Any]]): Resource specifications
        submit (bool): Whether to submit the job
        notify_email (Optional[str]): Email to notify on job completion
        
    Returns:
        str: Path to the created job file
    """
    # Create default job name if not provided
    if job_name is None:
        job_name = f"alignmap_{generate_random_string()}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read template or use default
    if template_path and os.path.exists(template_path):
        with open(template_path, 'r') as file:
            template = file.read()
    else:
        template = _get_default_pbs_template()
    
    # Apply resources to template
    template = _apply_resources(template, resources)
    
    # Add email notification if specified
    if notify_email:
        template = _add_email_notification(template, notify_email)
    
    # Replace job name
    template = template.replace("JOB_NAME_PLACEHOLDER", job_name)
    
    # Add commands
    commands_text = "\n".join(commands)
    template = template.replace("COMMAND_PLACEHOLDER", commands_text)
    
    # Create job file
    job_file_path = os.path.join(output_dir, f"{job_name}.pbs")
    with open(job_file_path, 'w') as file:
        file.write(template)
    
    logger.info(f"Created PBS job file: {job_file_path}")
    
    # Submit job if requested
    if submit:
        try:
            result = subprocess.run(
                ['qsub', job_file_path],
                check=True,
                capture_output=True,
                text=True
            )
            job_id = result.stdout.strip()
            logger.info(f"Submitted job {job_name}, ID: {job_id}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to submit job: {e}")
            logger.error(f"Error output: {e.stderr}")
    
    return job_file_path

def _get_default_pbs_template() -> str:
    """Get a default PBS job template.
    
    Returns:
        str: Default PBS template
    """
    return """#!/bin/bash
#PBS -N JOB_NAME_PLACEHOLDER
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=8
#PBS -l mem=16gb
#PBS -j oe
#PBS -o logs/
#PBS -e logs/

# Load environment modules if needed
# module load python/3.8

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "Starting time: $(date)"

# Commands to execute
COMMAND_PLACEHOLDER

echo "Finished at: $(date)"
"""

def _apply_resources(template: str, resources: Optional[Dict[str, Any]]) -> str:
    """Apply resource specifications to the template.
    
    Args:
        template (str): PBS template
        resources (Optional[Dict[str, Any]]): Resource specifications
        
    Returns:
        str: Updated template
    """
    if not resources:
        return template
    
    # Common resource mappings
    resource_lines = []
    
    if 'nodes' in resources:
        ppn = resources.get('ppn', 1)
        resource_lines.append(f"#PBS -l nodes={resources['nodes']}:ppn={ppn}")
    
    if 'mem' in resources:
        resource_lines.append(f"#PBS -l mem={resources['mem']}")
    
    if 'walltime' in resources:
        resource_lines.append(f"#PBS -l walltime={resources['walltime']}")
    
    if 'queue' in resources:
        resource_lines.append(f"#PBS -q {resources['queue']}")
    
    if 'gpu' in resources and resources['gpu']:
        resource_lines.append("#PBS -l ngpus=1")
    
    # Remove default resource lines and add new ones
    lines = template.split('\n')
    filtered_lines = [line for line in lines if not line.startswith("#PBS -l")]
    
    # Find where to insert resource lines
    insert_idx = -1
    for i, line in enumerate(filtered_lines):
        if line.startswith("#PBS"):
            insert_idx = i + 1
    
    if insert_idx >= 0:
        for i, res_line in enumerate(resource_lines):
            filtered_lines.insert(insert_idx + i, res_line)
    else:
        # If no PBS lines found, add at the top
        filtered_lines = resource_lines + filtered_lines
    
    return '\n'.join(filtered_lines)

def _add_email_notification(template: str, email: str) -> str:
    """Add email notification to the template.
    
    Args:
        template (str): PBS template
        email (str): Email address for notifications
        
    Returns:
        str: Updated template
    """
    email_lines = [
        f"#PBS -M {email}",
        "#PBS -m abe"  # a=abort, b=begin, e=end
    ]
    
    lines = template.split('\n')
    
    # Find where to insert email lines (after other PBS directives)
    insert_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("#PBS"):
            insert_idx = i + 1
    
    if insert_idx >= 0:
        for i, email_line in enumerate(email_lines):
            lines.insert(insert_idx + i, email_line)
    else:
        # If no PBS lines found, add at the top
        lines = email_lines + lines
    
    return '\n'.join(lines) 