# Sequence of commands, exemplified by conversational task (Llama2-7b-chat, harmless prompts)

## NOTE: a list input of one or more numbers in either form of --var=a,b,... or --var="a,b,..." can be well interpreted 

## generate original data using the prompts and base model (supporting tasks: 'conversation', 'movie_review')
!python gendata.py --model_generate="{basemodel_name}" --data_name="{data_name}" generate_from_original_model

## add rewards
!python submit_jobs.py with
commands = [f'python rewardProcessor.py --value="{value}" --file_path="results/{basemodel_name}-{data_name}.json" add_reward' for value in ALL_SUPPORTED_VALUES]

## merge processed value-specific rewards of the original data stored under temp/
!python mergeProcessor.py --original_file_path="results/{basemodel_name}-{data_name}.json" merge_added_rewards
<!-- python mergeProcessor.py --original_file_path="results/opt1.3b-Imdb.json" merge_added_rewards -->

## assess the hist of score-based rewards
!python plot_results.py with
plot_hist("results/{basemodel_name}-{data_name}.json")

## assess c-levels of the original data
!python rewardProcessor.py --file_path="results/{basemodel_name}-{data_name}.json" --values_to_evaluate="all" assess_original_value
<!-- !python rewardProcessor.py --file_path="results/opt1.3b-Imdb.json" --values_to_evaluate="all" assess_original_value -->

## optimize_lambda
!python alignValues.py --c_list=2.46,0.473,2.513,0.848,0.521,-1.375 --value_list="all" --file_path="results/llama2_chat-Anthropic-harmless.json" optimize_lambda
!python alignValues.py --c_list=-0.108 --value_list="gpt2-harmless" --file_path="results/llama2_chat-Anthropic-harmless.json" optimize_lambda

## assess_postalignment_multivalue, use numerical
!python rewardProcessor.py --file_path="results/{basemodel_name}-{data_name}.json" --values_to_align="all" --lam="5.942,2.432,2.923,0.006,0.011,0.147" --values_to_evaluate="all" assess_postalignment_multivalue

## assess_postalignment_multivalue, use MC-generation
!python submit_jobs.py with
commands = [f'python gendata.py generate_from_MC_aligned_model --lam_list=5.942,2.432,2.923,0.006,0.011,0.147 --value_list="all" --MC_nsamples=16 --start_index={200*i} --end_index={200*(i+1)}' for i in range(12)]

## merge parallel-generated data
!python mergeProcessor.py --json_file_pattern="results/temp/{basemodel_name}-{data_name}_lam=2.018,1.393,1.498,0.008,0.015,0.088_val=all_*to*.json" merge_gendata_bypattern

## re-calculate rewards of the MC-generated file and evaluate the c-level
!python submit_jobs.py with
commands = [f'python rewardProcessor.py --value="{value}" --file_path="results/temp/{basemodel_name}-{data_name}_lam=5.942,2.432,2.923,0.006,0.011,0.147_val=all_*to*.json" add_reward' for value in ALL_SUPPORTED_VALUES]

## merge processed value-specific rewards of MC-generated data stored under temp/
!python mergeProcessor.py --original_file_path="results/{basemodel_name}-{data_name}_lam=5.942,2.432,2.923,0.006,0.011,0.147_val=all_*to*.json" merge_added_rewards

## assess c-levels of the MC-generated data
!python rewardProcessor.py --file_path="results/{basemodel_name}-{data_name}_lam=5.942,2.432,2.923,0.006,0.011,0.147_val=all_*to*.json" --values_to_evaluate="all" --evaluation_mode=True assess_original_value



# For visualization purpose 

## gen a csv file of randomly created lambda and correspondingly, numerically calculated c levels
python rewardProcessor.py --file_path="results/opt1.3b-Imdb.json" --values_to_align="all" --values_to_evaluate="all" --scaling=-1 assess_postalignment_multivalue


## get quantile of the aligned c-levels wrt to a datafile that contains calculated rewards:
!python rewardProcessor.py --file_path="results/{basemodel_name}-{data_name}.json" --c_list=0.269,0.202,0.21,0.096,0.035,0.087 quantile_transform_single_c

## TODO: PPO-finetune model and generate data