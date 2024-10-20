import os
import random
import string
from utils import ALL_SUPPORTED_VALUES
from utils import convert_ppo_modelname_to_huggingface_valid

def random_string(length=8):
    return ''.join(random.choice(string.ascii_letters + string.digits) for i in range(length))


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

        
if __name__ == "__main__":

    '''
        DPO soup

        Prerequisite: run submit_DPO_job.py to first get two models with harmless_ratio = 0 or 1
        Based on that we then apply linear interpolation
    '''
    basemodel_name = "opt1.3b"
    sample_size = 2000
    beta = 0.1
    harmless_lambda = 0.1
    gen_data_name = "Anthropic-harmless"

    soup_model_name = f"{basemodel_name}-{sample_size}sample-{beta}beta-{harmless_lambda}soup"
    soupModel_relative_path = f"modelsDPO/soup/{soup_model_name}"
    soupModel_relative_path = convert_ppo_modelname_to_huggingface_valid(soupModel_relative_path)
    json_filepath = f"{soupModel_relative_path}-{gen_data_name}.json"

    # Convert model_name to an abs path for HF loading
    soupModel_abs_path = os.path.abspath(soupModel_relative_path)
    print(f"abs path of soupModel_name: {soupModel_abs_path}")

    seqCommands = []
    
    seqCommands.append(f'python getDPOsoup.py --basemodel_name={basemodel_name} --sample_size={sample_size} --beta={beta} --harmless_lambda={harmless_lambda} --save_path={soupModel_relative_path}')

    # gen data, save to {soupModel_abs_path}-{data_name}.json
    seqCommands.append(f'python gendata.py --basemodel_name="{soupModel_abs_path}" --data_name="{gen_data_name}" --save_directory="modelsDPO/soup" generate_from_original_model')

    # calculate rewards of the generated data file 
    seqCommands += [f'python rewardProcessor.py --value="{value}" --file_path={json_filepath} --basemodel_for_perplexity={soupModel_abs_path} add_reward' for value in ALL_SUPPORTED_VALUES]    
    seqCommands.append(f'python mergeProcessor.py --original_file_path={json_filepath} merge_added_rewards')

    # evaluate the true c-level
    seqCommands.append(f'python rewardProcessor.py --file_path={json_filepath} --values_to_evaluate="all" --evaluation_mode=True assess_original_value')

    # remove the model to save disk memory
    seqCommands.append(f'rm -rf {soupModel_relative_path}')


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