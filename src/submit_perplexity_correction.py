import os
import random
import string
from utils import ALL_SUPPORTED_VALUES


def random_string(length=8):
    return ''.join(random.choice(string.ascii_letters + string.digits) for i in range(length))


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

        
if __name__ == "__main__":

    '''
        This script will gen data and determine pallete of values to align

        basemodel_name: "llama2_chat", "opt1.3b", "gpt2"
        data_name: "Anthropic-harmless" or "Imdb"
    '''
    basemodel_name = "opt1.3b"
    data_name = "Imdb"

    '''
        Gen original data
    '''
    basemodel_name, data_name = "opt1.3b", "Anthropic-harmless" #"llama2_chat", "Anthropic-harmless"

    commands = [
        f'python rewardProcessor.py --value="perplexity" --file_path="results/{basemodel_name}-{data_name}.json" add_reward' + '\n' +\
            f'python rewardProcessor.py --file_path="results/{basemodel_name}-{data_name}.json" --values_to_evaluate="all" assess_original_value',
    ]

    template_path = 'main-jd-4a100.pbs' # candidate partitions: 'main-jd-4a100.pbs' 'main-a100-8.pbs' 'main-a100-4.pbs'
    jobs_dir = 'pbs-files'
    ensure_dir(jobs_dir)  # Ensure the directory exists

    for idx, command in enumerate(commands):
        # Create a new PBS file for each command
        with open(template_path, 'r') as template_file:
            content = template_file.read()
            content = content.replace("COMMAND_PLACEHOLDER", command)

        job_file_name = os.path.join(jobs_dir, f'job_{idx}_{random_string()}.pbs')
        with open(job_file_name, 'w') as job_file:
            job_file.write(content)
        # print(f'created sbatch {job_file_name} into {bash_file_name}')

        # Submit the job
        os.system(f'sbatch {job_file_name}')
