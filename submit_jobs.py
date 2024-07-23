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
        # Generate continuations and save to json
        # 'python gendata.py --basemodel_name="opt1.3b" --data_name="Anthropic-harmless" generate_from_original_model',
        # f'python gendata.py --basemodel_name="{basemodel_name}" --data_name="{data_name}" generate_from_original_model',
        #
        # Numeric assessment of post-alignment value without generating data
        # GUI visualization 
        # 'python gendataGUI.py',
        # Calculate value-adjustment parameter lambda from c
        # f'python alignValues.py --c_list=0.565,-1.525,0.618,0.882,0.237,-3.361 --value_list="all" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=0.808,-1.318,0.847,0.882,0.237,-3.361 --value_list="all" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=0.898,-1.062,1.04,0.882,0.237,-3.361 --value_list="all" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=0.958,-0.778,1.276,0.882,0.237,-3.361 --value_list="all" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=0.986,-0.332,1.559,0.882,0.237,-3.361 --value_list="all" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=0.958 --value_list="positive" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=-0.778 --value_list="gpt2-helpful" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=1.276 --value_list="gpt2-harmless" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        #
        # f'python alignValues.py --c_list=0.700,-1.525,0.581,0.882,0.237,-3.361 --value_list="all" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=0.700 --value_list="positive" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=0.700,0.581,0.882,0.237,-3.361 --value_list="positive,gpt2-harmless,diversity,coherence,perplexity" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=0.700,-1.525,0.882,0.237,-3.361 --value_list="positive,gpt2-helpful,diversity,coherence,perplexity" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=0.700,-1.525,0.581,0.237,-3.361 --value_list="positive,gpt2-helpful,gpt2-harmless,coherence,perplexity" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=0.700,-1.525,0.581,0.882,-3.361 --value_list="positive,gpt2-helpful,gpt2-harmless,diversity,perplexity" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=0.700,-1.525,0.581,0.882,0.237 --value_list="positive,gpt2-helpful,gpt2-harmless,diversity,coherence" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        #
        # f'python trainPPO.py --model_name="opt-1.3b" --data_name="imdb" --value_list="all" --lam_list="0.241,0.077,0.117,0.033,0.070,0.065" --learning_rate=1e-6',
        # upgraded alignValues.py
        # f'python alignValues.py --c_list=2.513,-0.967,0.937,0.876,0.434,-3.337 --value_list="all" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_list=2.534,-0.613,1.268,0.876,0.434,-3.337 --value_list="all" --file_path="results/{basemodel_name}-{data_name}.json" optimize_lambda',
        # f'python alignValues.py --c_low=2.513,-0.967,0.937,0.876,0.434,-3.337 --c_high=2.534,-0.613,1.268,0.876,0.434,-3.337 --value_list="all" --file_path="results/{basemodel_name}-{data_name}.json" find_pareto_by_interpolation',
        f'python alignValues.py --c_list=2.513,-0.967,0.937,0.876,0.434,-3.337 --value_list="all" --value_to_enhance="gpt2-helpful" --file_path="results/{basemodel_name}-{data_name}.json" find_pareto_by_oneValue',
    ]
    '''
        Data-parallel implementations of data generation
    '''
    # commands = [f'python gendata.py generate_from_MC_aligned_model --lam_list=1.29 --value_list="humor" --MC_nsamples=32 --start_index={100*i} --end_index={100*(i+1)}' for i in range(24)]
    # commands = [f'python gendata.py generate_from_MC_aligned_model --lam_list=5.942,2.432,2.923,0.006,0.011,0.147 --value_list="all" --MC_nsamples=16 --start_index={200*i} --end_index={200*(i+1)}' for i in range(6)]

    # commands = [f'python rewardProcessor.py --value="{value}" --file_path="results/{basemodel_name}-{data_name}.json" add_reward' for value in ["positive"]]
    # commands = [f'python rewardProcessor.py --value="{value}" --file_path="results/{basemodel_name}-{data_name}.json" add_reward' for value in ALL_SUPPORTED_VALUES]    
    # merge temp processed rewards at: 'python results/temp/merge_add_reward.py'



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
