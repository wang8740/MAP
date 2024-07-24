
# Pareto curve between helpful and harmless/diversity

## assess_original_value
!python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_evaluate="all" --evaluation_mode=True assess_original_value

<!-- Statistic,humor,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity
avg,0.604,-1.011,1.248,0.848,0.521,-1.375
50%,1.196,-1.072,1.367,0.891,0.536,-1.293
60%,1.83,-0.776,1.638,0.908,0.582,-1.198
70%,2.168,-0.462,1.885,0.919,0.625,-1.106
80%,2.363,-0.108,2.164,0.93,0.667,-1.015
90%,2.46,0.473,2.513,0.941,0.736,-0.908
99%,2.544,1.688,3.35,0.955,0.838,-0.703 -->


## plot pareto frontier

!python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="gpt2-helpful,gpt2-harmless" --values_to_evaluate="gpt2-helpful,gpt2-harmless" --scaling=-1 --scaling_MAX=7 --k=500 assess_postalignment_multivalue

!python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="gpt2-helpful,diversity" --values_to_evaluate="gpt2-helpful,diversity" --scaling=-1 --scaling_MAX=7 --k=500 assess_postalignment_multivalue


## align helpful & harmless/diversity at 70%, optimize_lambda
!python alignValues.py --c_list=-0.462,1.885 --value_list="gpt2-helpful,gpt2-harmless" --file_path="results/Llama27b-chat-Anthropic-harmless.json" optimize_lambda

Optimized lambda: 0.927,1.139

!python alignValues.py --c_list=-0.462,0.919 --value_list="gpt2-helpful,diversity" --file_path="results/Llama27b-chat-Anthropic-harmless.json" optimize_lambda

Optimized lambda: 0.488,19.402

## assess_postalignment_multivalue, use numerical
!python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="gpt2-helpful,gpt2-harmless" --lam=0.927,1.139 --values_to_evaluate="all" assess_postalignment_multivalue

<!-- humor,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity
0.111,-0.462,1.885,0.862,0.527,-1.243 -->

!python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="gpt2-helpful,diversity" --lam=0.488,19.402 --values_to_evaluate="all" assess_postalignment_multivalue

<!-- humor,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity
0.276,-0.462,1.128,0.919,0.498,-1.38 -->

## align helpful only at 70%, optimize_lambda
!python alignValues.py --c_list=-0.462 --value_list="gpt2-helpful" --file_path="results/Llama27b-chat-Anthropic-harmless.json" optimize_lambda

Optimized lambda: 0.428

## assess_postalignment_multivalue, use numerical
!python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="gpt2-helpful" --lam=0.428 --values_to_evaluate="all" assess_postalignment_multivalue

<!-- humor,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity
0.353,-0.462,0.981,0.846,0.543,-1.34 -->

## align harmless/diversity at 70%, optimize_lambda
!python alignValues.py --c_list=1.885 --value_list="gpt2-harmless" --file_path="results/Llama27b-chat-Anthropic-harmless.json" optimize_lambda

Optimized lambda: 0.633

!python alignValues.py --c_list=0.919 --value_list="diversity" --file_path="results/Llama27b-chat-Anthropic-harmless.json" optimize_lambda

Optimized lambda: 18.451

## assess_postalignment_multivalue, use numerical
!python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="gpt2-harmless" --lam=0.633 --values_to_evaluate="all" assess_postalignment_multivalue

<!-- humor,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity
0.589,-1.32,1.885,0.857,0.501,-1.317 -->

!python rewardProcessor.py --file_path="results/Llama27b-chat-Anthropic-harmless.json" --values_to_align="diversity" --lam=18.451 --values_to_evaluate="all" assess_postalignment_multivalue

<!-- humor,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity
0.562,-1.068,1.386,0.919,0.475,-1.411 -->
