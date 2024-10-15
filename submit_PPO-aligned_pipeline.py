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

    ### MAP Tabular results in the paper ###
    # basemodel_name, data_name = "llama2_chat", "Anthropic-harmless"
    # batch_size, mini_batch_size = 20, 1
    # nepoch = 1
    # value_list, lam_list = "all", "0.269,0.202,0.210,0.096,0.035,0.087" 
    # value_list, lam_list = "all", "0.852,0.782,0.790,0.011,0.020,0.071" 
    # value_list, lam_list = "all", "2.018,1.393,1.498,0.008,0.015,0.088" 
    # value_list, lam_list = "all", "5.942,2.432,2.923,0.006,0.011,0.147" 
    # value_list, lam_list = "humor", "2.887"
    # value_list, lam_list = "gpt2-helpful", "0.693"
    # value_list, lam_list = "gpt2-harmless", "0.988"

    ### MAP Tabular results in the paper ###
    # basemodel_name, data_name = "opt1.3b", "Anthropic-harmless"
    # batch_size, mini_batch_size = 20, 2
    # nepoch = 1
    # value_list, lam_list = "all", "2.533,0.233,0.278,0.020,0.046,0.051" 
    # value_list, lam_list = "all", "6.297,0.834,0.929,0.013,0.026,0.032" 
    # value_list, lam_list = "all", "12.766,1.526,1.689,0.012,0.019,0.023" 
    # value_list, lam_list = "humor", "16.442"
    # value_list, lam_list = "gpt2-helpful", "0.721"
    # value_list, lam_list = "gpt2-harmless", "0.952"

    ### MAP Tabular results in the paper ###
    # basemodel_name, data_name = "opt1.3b", "Imdb"
    # batch_size, mini_batch_size = 20, 20
    # nepoch = 1
    # value_list, lam_list = "all", "0.241,0.077,0.117,0.033,0.070,0.065" 
    # value_list, lam_list = "all", "2.234,0.412,0.790,0.009,0.031,0.038" 
    # value_list, lam_list = "all", "3.834,0.895,1.481,0.007,0.018,0.028" 
    # value_list, lam_list = "all", "9.765,1.418,2.265,0.004,0.012,0.027" 
    # value_list, lam_list = "positive", "10.975"
    # value_list, lam_list = "gpt2-helpful", "0.949"
    # value_list, lam_list = "gpt2-harmless", "1.425"

    ### Random lambda results for success rate comparison with MAP that aligns on HH ###
    # basemodel_name, data_name = "opt1.3b", "Anthropic-harmless"
    # batch_size, mini_batch_size = 20, 1
    # nepoch = 1
    # value_list, lam_list = "all", "0.269,0.202,0.210,0.096,0.035,0.087" 

    ### Random lambda results for success rate comparison with MAP that aligns on HH, 
    # first ten "optimized_lambda" from plot_rand_lambda_6scale_2D.csv ###
    basemodel_name, data_name = "opt1.3b", "Anthropic-harmless"
    batch_size, mini_batch_size = 20, 1
    nepoch = 1
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.548,1.148"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.556,1.172"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.854,1.912"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.452,0.867"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "2.983,1.392"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.333,1.690"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.141,1.088"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.259,3.034"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.301,2.402"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "0.831,0.942"
    # second batch of 10
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "2.556,1.317"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "2.489,1.153"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.324,3.406"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.437,1.040"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.673,3.305"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "2.390,3.044"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "0.613,0.657"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.427,3.043"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "3.074,2.536"
    value_list, lam_list = "gpt2-helpful,gpt2-harmless", "2.130,1.484"
    save_dir = f"modelsPPO/MAP-lambda2"
    ppoModel_relative_path = f"{save_dir}/{basemodel_name}-{data_name}-lam={lam_list}-val={value_list}"

    ### Random lambda results for success rate comparison with MAP that aligns on HH, 
    # first ten "Dirichlet_lambda_ref" from plot_rand_lambda_6scale_2D.csv ###
    # basemodel_name, data_name = "opt1.3b", "Anthropic-harmless"
    # batch_size, mini_batch_size = 20, 1
    nepoch = 1
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.865,0.358"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "2.834,0.970"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "3.498,0.328"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.363,0.260"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "2.276,1.145"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "4.132,0.208"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "0.767,0.422"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.473,0.845"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "0.815,0.649"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "0.029,0.220"
    # second batch of 10
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "2.990,0.489"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "3.750,0.853"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "3.179,0.439"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "0.719,2.330"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "3.535,0.359"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "0.536,1.111"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "0.521,0.815"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "4.513,0.297"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "1.890,3.980"
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "3.173,1.608"
    # save_dir = f"modelsPPO/random-lambda2"
    # ppoModel_relative_path = f"{save_dir}/{basemodel_name}-{data_name}-lam={lam_list}-val={value_list}"

    ppoModel_relative_path = convert_ppo_modelname_to_huggingface_valid(ppoModel_relative_path)
    json_filepath = f"{ppoModel_relative_path}-{data_name}.json"
    # Convert model_name to an abs path for HF loading
    ppoModel_abs_path = os.path.abspath(ppoModel_relative_path)
    print(f"abs path of ppoModel_name: {ppoModel_abs_path}")

    seqCommands = []

    seqCommands.append(f'python trainPPO.py --model_name={basemodel_name} --data_name={data_name} --value_list={value_list} --lam_list={lam_list} --learning_rate=1e-6 --nepoch={nepoch} --batch_size={batch_size} --mini_batch_size={mini_batch_size}  --save_path={ppoModel_relative_path}')

    # gen data, save to {ppoModel_relative_path}-{data_name}.json
    seqCommands.append(f'python gendata.py --basemodel_name="{ppoModel_abs_path}" --data_name="{data_name}" --save_directory="{save_dir}" generate_from_original_model')

    # calculate rewards of the generated data file 
    seqCommands += [f'python rewardProcessor.py --value="{value}" --file_path={json_filepath} --basemodel_for_perplexity={ppoModel_abs_path} add_reward' for value in ALL_SUPPORTED_VALUES]    
    seqCommands.append(f'python mergeProcessor.py --original_file_path={json_filepath} merge_added_rewards')

    # evaluate the true c-level
    seqCommands.append(f'python rewardProcessor.py --file_path={json_filepath} --values_to_evaluate="all" --evaluation_mode=True assess_original_value')

    # remove the model to save disk memory
    seqCommands.append(f'rm -rf {ppoModel_relative_path}')


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
