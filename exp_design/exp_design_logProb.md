# Llama2-7b-chat, harmless prompts, conversational task

## add rewards
!python submit_jobs.py with
commands = [f'python rewardProcessor.py --value="{value}" --file_path="results/Llama27b-chat-Anthropic-harmless.json" add_reward' for value in ALL_SUPPORTED_VALUES]

merge temp processed rewards at
!python results/temp/merge_add_reward.py with
merge_results(original_file_path="results/Llama27b-chat-Anthropic-harmless.json")

## assess_original_value
!python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_evaluate="all" assess_original_value

Statistic,humor,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity
avg,-1.239,-1.437,-0.362,0.848,0.521,-1.375
50%,-0.077,-1.367,-0.227,0.848,0.521,-1.375
60%,-0.021,-1.155,-0.178,0.848,0.521,-1.375
70%,-0.01,-0.951,-0.141,0.848,0.521,-1.375
80%,-0.007,-0.749,-0.109,0.848,0.521,-1.375
90%,-0.005,-0.485,-0.078,0.848,0.521,-1.375
99%,-0.004,-0.17,-0.034,0.848,0.521,-1.375

## optimize_lambda
!python alignValues.py --c_list=-0.077,-1.367,-0.227,0.848,0.521,-1.375 --value_list="all" --file_path="results/Llama27b-chat-Anthropic-harmless.json" optimize_lambda

50%,2.137,0.804,2.620,0.036,0.031,0.072
60%,13.638,1.683,6.177,0.010,0.021,0.037
70%,85.460,2.919,12.481,0.005,0.022,0.075
80%,321.402,6.117,29.249,0.003,1.108,0.718
divergence

## assess_postalignment_multivalue, use numerical
!python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="all" --lam=2.137,0.804,2.620,0.036,0.031,0.072 --values_to_evaluate="all" assess_postalignment_multivalue

50%,-0.077,-1.367,-0.227,0.85,0.529,-1.371
60%,-0.021,-1.155,-0.178,0.857,0.536,-1.358
70%,-0.01,-0.951,-0.141,0.874,0.538,-1.374
80%,-0.007,-0.749,-0.109,0.902,0.521,-1.375

## assess_postalignment_multivalue, use MC-generation
!python submit_jobs.py with
commands = [f'python gendata.py generate_from_MC_aligned_model --lam_list=2.137,0.804,2.620,0.036,0.031,0.072 --value_list="all" --MC_nsamples=16 --start_index={200*i} --end_index={200*(i+1)}' for i in range(12)]

!python utils.py with
merge_json_files("results/temp/Llama27b-chat-Anthropic-harmless_lam=(-0.021, -1.155, -0.178, 0.857, 0.536, -1.358)_val=all_*to*.json")

## re-calculate rewards of the MC-generated file and evaluate the c-level
!python submit_jobs.py with
commands = [f'python rewardProcessor.py --value="{value}" --file_path="results/Llama27b-chat-Anthropic-harmless_lam=2.137,0.804,2.62,0.036,0.031,0.072_val=all_*to*.json" add_reward' for value in ALL_SUPPORTED_VALUES]

merge temp processed rewards at
!python results/temp/merge_add_reward.py with
merge_results(original_file_path="results/Llama27b-chat-Anthropic-harmless_lam=2.137,0.804,2.62,0.036,0.031,0.072_val=all_*to*.json")

!python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless_lam=2.137,0.804,2.62,0.036,0.031,0.072_val=all_*to*.json" --values_to_evaluate="all" assess_original_value

<!-- 
Statistic,humor,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity
avg,-0.142,-1.457,-0.255,0.836,0.528,-1.386
50%,-0.016,-1.379,-0.192,0.836,0.528,-1.386
60%,-0.01,-1.188,-0.161,0.836,0.528,-1.386
70%,-0.008,-1.018,-0.132,0.836,0.528,-1.386
80%,-0.006,-0.828,-0.102,0.836,0.528,-1.386
90%,-0.005,-0.594,-0.074,0.836,0.528,-1.386
99%,-0.004,-0.227,-0.033,0.836,0.528,-1.386 -->

<!-- pre-align avg: -1.239,-1.437,-0.362,0.848,0.521,-1.375
target 50%: -0.077,-1.367,-0.227,0.848,0.521,-1.375
post-align avg: -0.142,-1.457,-0.255,0.836,0.528,-1.386 -->


