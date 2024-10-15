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
        Comparison with DPO and PPO
    '''
 
    basemodel_name, data_name = "opt1.3b", "Anthropic-harmless"
    commands = [
        # run gen_rand_MAP_lambda to randomly draw MAP-allowed lambda subject to l1 norm no larger than default 4
        # f'python alignValues.py --c_list=2.07,-1.471,0.245,0.876,0.434,-3.337, --value_list="all" --file_path="results/{basemodel_name}-{data_name}.json" --num_lambda=10 gen_rand_MAP_lambda',
        # avg level
        f'python alignValues.py --c_list=2.07,-1.471,0.245,0.876,0.434,-3.337 --value_list="humor,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity" --file_path="results/{basemodel_name}-{data_name}.json" --num_lambda=100 --scaling_MAX=6 --save_prefix="plot_rand_lambda_6scale_6D" gen_rand_MAP_lambda',
        f'python alignValues.py --c_list=2.07,-1.471,0.245,0.876,0.434 --value_list="humor,gpt2-helpful,gpt2-harmless,diversity,coherence" --file_path="results/{basemodel_name}-{data_name}.json" --num_lambda=100 --scaling_MAX=6 --save_prefix="plot_rand_lambda_6scale_5D" gen_rand_MAP_lambda',
        f'python alignValues.py --c_list=2.07,-1.471,0.245,0.876 --value_list="humor,gpt2-helpful,gpt2-harmless,diversity" --file_path="results/{basemodel_name}-{data_name}.json" --num_lambda=100 --scaling_MAX=6 --save_prefix="plot_rand_lambda_6scale_4D" gen_rand_MAP_lambda',
        # f'python alignValues.py --c_list=2.07,-1.471,0.245 --value_list="humor,gpt2-helpful,gpt2-harmless" --file_path="results/{basemodel_name}-{data_name}.json" --num_lambda=100 --scaling_MAX=6 --save_prefix="plot_rand_lambda_6scale_3D" gen_rand_MAP_lambda',
        # f'python alignValues.py --c_list=-1.471,0.245 --value_list="gpt2-helpful,gpt2-harmless" --file_path="results/{basemodel_name}-{data_name}.json" --num_lambda=100 --scaling_MAX=6 --save_prefix="plot_rand_lambda_6scale_2D" gen_rand_MAP_lambda',
        # 60% level -- took too much time
        # f'python alignValues.py --c_list=2.492,-1.278,0.675, --value_list="humor,gpt2-helpful,gpt2-harmless" --file_path="results/{basemodel_name}-{data_name}.json" --num_lambda=10 --save_prefix="rand_MAP_lambda_forPPO" gen_rand_MAP_lambda',
        #
        # run submit_PPO-aligned_pipeline.py to PPO train the random lambda
        # 
        # run assess_postalignment_multivalue to numericall assess the expected reward under particular lambda
        # f'python rewardProcessor.py --file_path="results/{basemodel_name}-{data_name}.json"  --values_to_evaluate="all" --values_to_align="humor,gpt2-helpful,gpt2-harmless" --lam=0.300,0.347,0.337 assess_postalignment_multivalue',
        # f'python rewardProcessor.py --file_path="results/{basemodel_name}-{data_name}.json"  --values_to_evaluate="all" --values_to_align="humor,gpt2-helpful,gpt2-harmless" --lam=0.935,1.045,1.741 assess_postalignment_multivalue',
        #
    ]


    template_path = 'main-jd-4a100.pbs'  # 'main-a100-8.pbs' # 'main-jd-4a100.pbs' 
    jobs_dir = 'pbs-files'
    ensure_dir(jobs_dir)

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
