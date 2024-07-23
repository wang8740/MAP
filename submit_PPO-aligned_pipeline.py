# This script is for automated evaluation of PPO-aligned models and save results under ppoModels/
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

    basemodel_name, data_name = "llama2_chat", "Anthropic-harmless"
    batch_size, mini_batch_size = 20, 1
    value_list, lam_list = "all", "0.269,0.202,0.210,0.096,0.035,0.087" 
    # value_list, lam_list = "all", "0.852,0.782,0.790,0.011,0.020,0.071" 
    # value_list, lam_list = "all", "2.018,1.393,1.498,0.008,0.015,0.088" 
    # value_list, lam_list = "all", "5.942,2.432,2.923,0.006,0.011,0.147" 
    # value_list, lam_list = "humor", "2.887"
    # value_list, lam_list = "gpt2-helpful", "0.693"
    # value_list, lam_list = "gpt2-harmless", "0.988"

    # basemodel_name, data_name = "opt1.3b", "Anthropic-harmless"
    # batch_size, mini_batch_size = 20, 2
    # value_list, lam_list = "all", "2.533,0.233,0.278,0.020,0.046,0.051" 
    # value_list, lam_list = "all", "6.297,0.834,0.929,0.013,0.026,0.032" 
    # value_list, lam_list = "all", "12.766,1.526,1.689,0.012,0.019,0.023" 
    # value_list, lam_list = "humor", "16.442"
    # value_list, lam_list = "gpt2-helpful", "0.721"
    # value_list, lam_list = "gpt2-harmless", "0.952"

    # basemodel_name, data_name = "opt1.3b", "Imdb"
    # batch_size, mini_batch_size = 20, 20
    # value_list, lam_list = "all", "0.241,0.077,0.117,0.033,0.070,0.065" 
    # value_list, lam_list = "all", "2.234,0.412,0.790,0.009,0.031,0.038" 
    # value_list, lam_list = "all", "3.834,0.895,1.481,0.007,0.018,0.028" 
    # value_list, lam_list = "all", "9.765,1.418,2.265,0.004,0.012,0.027" 
    # value_list, lam_list = "positive", "10.975"
    # value_list, lam_list = "gpt2-helpful", "0.949"
    # value_list, lam_list = "gpt2-harmless", "1.425"

    nepoch = 2
    ppoModel_relative_path = f"ppoModels/{basemodel_name}-{data_name}-lam={lam_list}-val={value_list}"
    ppoModel_relative_path = convert_ppo_modelname_to_huggingface_valid(ppoModel_relative_path)
    json_filepath = f"{ppoModel_relative_path}-{data_name}.json"

    # Convert model_name to an abs path for HF loading
    ppoModel_abs_path = os.path.abspath(ppoModel_relative_path)
    print(f"abs path of ppoModel_name: {ppoModel_abs_path}")

    seqCommands = []

    seqCommands.append(f'python trainPPO.py --model_name={basemodel_name} --data_name={data_name} --value_list={value_list} --lam_list={lam_list} --learning_rate=1e-6 --nepoch={nepoch} --batch_size={batch_size} --mini_batch_size={mini_batch_size}')

    # gen data, save to {ppoModel_relative_path}-{data_name}.json
    seqCommands.append(f'python gendata.py --basemodel_name="{ppoModel_abs_path}" --data_name="{data_name}" --save_directory="ppoModels" generate_from_original_model')

    # calculate rewards of the generated data file 
    seqCommands += [f'python rewardProcessor.py --value="{value}" --file_path={json_filepath} --basemodel_for_perplexity={ppoModel_abs_path} add_reward' for value in ALL_SUPPORTED_VALUES]    
    seqCommands.append(f'python mergeProcessor.py --original_file_path={json_filepath} merge_added_rewards')

    # evaluate the true c-level
    seqCommands.append(f'python rewardProcessor.py --file_path={json_filepath} --values_to_evaluate="all" --evaluation_mode=True assess_original_value')

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
