# This script is for automated evaluation of MC-aligned models and save results under results/
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
        Setup, given lambda values and values to align
        This script will assess the post-alignment c-levels

        basemodel_name: "llama2_chat", "opt1.3b", "gpt2"
        data_name: "Anthropic-harmless" or "Imdb"

        NOTE: before the random lambda experiments, we used the notation values_str, lams_str rather than values_list, lam_list
        and save_dir = "results"
    '''
    # basemodel_name = "llama2_chat"
    # data_name = "Anthropic-harmless"
    # values_str, lams_str = "humor", "2.887"
    # values_str, lams_str = "gpt2-helpful", 0.693
    # values_str, lams_str = "gpt2-harmless", 0.988

    # basemodel_name = "opt1.3b"
    # data_name = "Anthropic-harmless"
    # values_str, lams_str = "all", "2.533,0.233,0.278,0.020,0.046,0.051"
    # values_str, lams_str = "all", "6.297,0.834,0.929,0.013,0.026,0.032"
    # values_str, lams_str = "all", "12.766,1.526,1.689,0.012,0.019,0.023"
    # values_str, lams_str = "humor", "16.442"
    # values_str, lams_str = "gpt2-helpful", "0.721"
    # values_str, lams_str = "gpt2-harmless", "0.952"

    # basemodel_name = "opt1.3b"
    # data_name = "Imdb"
    # 0.8 positive
    # values_str, lams_str = "all", "2.234,0.412,0.790,0.009,0.031,0.038"
    # values_str, lams_str = "positive", "2.377"
    # values_str, lams_str = "positive,gpt2-harmless,diversity,coherence,perplexity", "2.443,0.531,0.031,0.030,0.043"
    # values_str, lams_str = "positive,gpt2-helpful,diversity,coherence,perplexity", "2.313,0.097,0.029,0.038,0.050"
    # values_str, lams_str = "positive,gpt2-helpful,gpt2-harmless", "2.237,0.418,0.795"

    # values_str, lams_str = "all", "0.241,0.077,0.117,0.033,0.070,0.065"
    # values_str, lams_str = "all", "2.234,0.412,0.790,0.009,0.031,0.038"
    # values_str, lams_str = "all", "3.834,0.895,1.481,0.007,0.018,0.028"
    # values_str, lams_str = "all", "9.765,1.418,2.265,0.004,0.012,0.027"
    # values_str, lams_str = "positive", "10.975"
    # values_str, lams_str = "gpt2-helpful", "0.949"
    # values_str, lams_str = "gpt2-harmless", "1.425"

    ### Random lambda results for success rate comparison with MAP that aligns on HH, 
    # first ten "optimized_lambda" from plot_rand_lambda_6scale_2D.csv ###
    # basemodel_name = "opt1.3b"
    # data_name = "Anthropic-harmless"
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
    # value_list, lam_list = "gpt2-helpful,gpt2-harmless", "2.130,1.484"
    # save_dir = f"modelsDecoding/MAP-lambda"
    # setting = f"{basemodel_name}-{data_name}"
    # palette = f"lam={lam_list}_val={value_list}"


    ### Random lambda results for success rate comparison with MAP that aligns on HH, 
    # first ten "Dirichlet_lambda_ref" from plot_rand_lambda_6scale_2D.csv ###
    basemodel_name = "opt1.3b"
    data_name = "Anthropic-harmless"
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
    value_list, lam_list = "gpt2-helpful,gpt2-harmless", "3.173,1.608"
    save_dir = f"modelsDecoding/random-lambda"
    setting = f"{basemodel_name}-{data_name}"
    palette = f"lam={lam_list}_val={value_list}"

    seqCommands = []

    # assess_postalignment_multivalue, use numerical
    # seqCommands.append(f'python rewardProcessor.py --file_path="results/{setting}.json"  --values_to_evaluate="all" --values_to_align="{value_list}" --lam={lam_list} assess_postalignment_multivalue')

    # assess_postalignment_multivalue, use MC-generation
    # seqCommands.append(f'python gendata.py generate_from_MC_aligned_model --basemodel_name="{basemodel_name}" --data_name="{data_name}" --lam_list={lam_list} --value_list="{value_list}" --MC_nsamples=16 --start_index=0 --end_index=10')
    seqCommands.append(f'python gendata.py --basemodel_name="{basemodel_name}" --data_name="{data_name}" generate_from_MC_aligned_model --save_directory={save_dir} --lam_list={lam_list} --value_list="{value_list}" --MC_nsamples=16')
    seqCommands.append(f'python mergeProcessor.py --json_file_pattern="{save_dir}/temp/{setting}_{palette}_*to*.json" merge_gendata_bypattern')

    # re-calculate rewards of the MC-generated file 
    seqCommands += [f'python rewardProcessor.py --value="{value}" --file_path="{save_dir}/{setting}_{palette}.json" add_reward' for value in ALL_SUPPORTED_VALUES]    
    seqCommands.append(f'python mergeProcessor.py --original_file_path="{save_dir}/{setting}_{palette}.json" merge_added_rewards')

    # re-evaluate the c-level
    seqCommands.append(f'python rewardProcessor.py --file_path="{save_dir}/{setting}_{palette}.json" --values_to_evaluate="all" --evaluation_mode=True assess_original_value')


    # read into a pbs file
    template_path = 'main-a100-4.pbs' # candidate partitions: 'main-jd-4a100.pbs' 'main-a100-8.pbs' 'main-a100-4.pbs'
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
