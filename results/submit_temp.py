# temp adjust the perplexity for opt models

import os
import random
import string
from utils import ALL_SUPPORTED_VALUES, convert_ppo_modelname_to_huggingface_valid

def random_string(length=8):
    return ''.join(random.choice(string.ascii_letters + string.digits) for i in range(length))

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":

    '''
        Setup, given ppo-aligned models
        This script will generate data and assess the original c-levels

        basemodel_name: "llama2_chat", "opt1.3b", "gpt2"
        data_name: "Anthropic-harmless" or "Imdb"
    '''

    basemodel_name, data_name = "opt1.3b", "Imdb"
    value_list, lam_list = "all", "0.241,0.077,0.117,0.033,0.070,0.065"

    json_filepath = f"{ppoModel_relative_path}-{data_name}.json"

    # Convert model_name to a abs path for HF loading
    ppoModel_abs_path = os.path.abspath(ppoModel_relative_path)
    print(f"abs path of ppoModel_name: {ppoModel_abs_path}")


    seqCommands = []

    # # calculate rewards of the generated data file 
    seqCommands += [f'python rewardProcessor.py --value="{value}" --file_path={json_filepath} --basemodel_for_perplexity={basemodel_name} add_reward' for value in ["perplexity"]]    
    seqCommands.append(f'python mergeProcessor.py --original_file_path={json_filepath} merge_added_rewards')

    # read into a pbs file
    template_path = 'main-jd-4a100.pbs' # candidate partitions: 'main-jd-4a100.pbs' 'main-a100-8.pbs' 'main-a100-4.pbs'
    jobs_dir = 'pbs-files'
    ensure_dir(jobs_dir)  # Ensure the directory exists

    with open(template_path, 'r') as template_file:
        content = template_file.read()
    
    # Combine all commands into a single string, each command on a new line
    commands_combined = "\n".join(seqCommands)
    content = content.replace("COMMAND_PLACEHOLDER", commands_combined)

    # Generate a unique job file name
    job_file_name = os.path.join(jobs_dir, f'job_{random_string()}.pbs')
    
    # Write the modified content to the new PBS job file
    with open(job_file_name, 'w') as job_file:
        job_file.write(content)
    print(f'Created job file {job_file_name}')

    # Submit the job
    os.system(f'sbatch {job_file_name}')
