# conversational task 
basemodel_name = "llama2_chat"
data_name = "Anthropic-harmless"

## assess c-levels of the original data
Statistic,humor,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity
avg,0.604,-1.011,1.248,0.848,0.521,-1.375
avg_std,0.037,0.023,0.023,0.002,0.004,0.009
50%,1.196,-1.011,1.367,0.848,0.521,-1.375
60%,1.83,-0.776,1.638,0.848,0.521,-1.375
70%,2.168,-0.462,1.885,0.848,0.521,-1.375
80%,2.363,-0.108,2.164,0.848,0.521,-1.375
90%,2.46,0.473,2.513,0.848,0.521,-1.375
99%,2.544,1.688,3.35,0.848,0.521,-1.375

## optimized lambda
50% all,0.269,0.202,0.210,0.096,0.035,0.087
60% all,0.852,0.782,0.790,0.011,0.020,0.071
70% all,2.018,1.393,1.498,0.008,0.015,0.088
80%,all,5.942,2.432,2.923,0.006,0.011,0.147
diverge
80% humor,2.887
80% gpt2-helpful,0.693
80% gpt2-harmless,0.988


## assess_postalignment_multivalue, use numerical
see files

## assess_postalignment_multivalue, use MC-generation

### all-50% alignment
<!-- pre-align avg: 0.604,-1.011,1.248,0.848,0.521,-1.375
target 50%: 1.196,-1.011,1.367,0.848,0.521,-1.375
lam: 0.269,0.202,0.21,0.096,0.035,0.087
post-align avg: 0.984,-1.057,1.362,0.843,0.527,-1.367 -->

### all-60% alignment
<!-- pre-align avg: 0.604,-1.011,1.248,0.848,0.521,-1.375
target 60%: 1.83,-0.776,1.638,0.848,0.521,-1.375
lam: 0.852,0.782,0.79,0.011,0.02,0.071
post-align avg: 1.564,-0.926,1.473,0.835,0.534,-1.368 -->

### all-70% alignment
<!-- pre-align avg: 0.604,-1.011,1.248,0.848,0.521,-1.375
target 70%: 2.168,-0.462,1.885,0.848,0.521,-1.375
lam: 2.018,1.393,1.498,0.008,0.015,0.088
post-align avg: 2.005,-0.869,1.569,0.829,0.546,-1.374 -->

### all-80% alignment
<!-- pre-align avg: 0.604,-1.011,1.248,0.848,0.521,-1.375
target 80%: 2.363,-0.108,2.164,0.848,0.521,-1.375
lam: 5.942,2.432,2.923,0.006,0.011,0.147
post-align avg: 2.172,-0.934,1.675,0.824,0.55,-1.371 -->

### 80% humor alignment
<!-- pre-align avg: 0.604,-1.011,1.248,0.848,0.521,-1.375
target 80% humor: 2.363
lam: 2.887
post-align avg: 2.208,-1.257,1.055,0.815,0.552,-1.469 -->

### 80% gpt2-helpful alignment
<!-- pre-align avg: 0.604,-1.011,1.248,0.848,0.521,-1.375
target 80% gpt2-helpful: -0.108
lam: 0.693
post-align avg: 0.501,-0.391,0.941,0.856,0.527,-1.362 -->

### 80% gpt2-harmless alignment
<!-- pre-align avg: 0.604,-1.011,1.248,0.848,0.521,-1.375
target 80% gpt2-harmless: 2.164
lam: 0.988
post-align avg: 0.47,-1.328,1.911,0.859,0.501,-1.335 -->





# conversational task 
basemodel_name = "opt1.3b"
data_name = "Anthropic-harmless"

## assess c-levels of the original data
Statistic,humor,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity
avg,2.07,-1.471,0.245,0.876,0.434,-3.337
avg_std,0.021,0.022,0.024,0.002,0.004,0.011
50%,2.460,-1.471,0.354,0.876,0.434,-3.337
60%,2.492,-1.278,0.675,0.876,0.434,-3.337
70%,2.513,-0.967,0.937,0.876,0.434,-3.337
80%,2.534,-0.613,1.268,0.876,0.434,-3.337
90%,2.554,-0.127,1.677,0.876,0.434,-3.337
99%,2.586,1.162,2.499,0.876,0.434,-3.337


## optimized lambda
50% all,2.533,0.233,0.278,0.020,0.046,0.051
60% all,6.297,0.834,0.929,0.013,0.026,0.032
70% all,12.766,1.526,1.689,0.012,0.019,0.023
80%,all,diverge
diverge
80% humor,16.442
80% gpt2-helpful,0.721
80% gpt2-harmless,0.952


## assess_postalignment_multivalue, use numerical
50% all,2.46,-1.471,0.353,0.883,0.449,-3.29
60% all,2.492,-1.277,0.674,0.887,0.466,-3.244
70% all,2.513,-0.967,0.937,0.89,0.488,-3.177
80%,all,diverge
diverge
80% humor,2.534,-1.521,0.009,0.889,0.443,-3.34
80% gpt2-helpful,1.941,-0.613,-0.3,0.88,0.454,-3.28
80% gpt2-harmless,1.954,-2.014,1.268,0.874,0.429,-3.308

## assess_postalignment_multivalue, use MC-generation

pre-align avg: 2.07,-1.471,0.245,0.876,0.434,-3.337

### all-50% alignment
MC: avg,2.437,-1.382,0.207,0.884,0.426,-3.17
ppo: avg,2.204,-1.8,0.655,0.86,0.406,-2.801


### all-60% alignment
MC: avg,2.476,-1.325,0.481,0.883,0.433,-3.149
PPO: avg,2.469,-2.255,0.499,0.891,0.257,-3.486

### all-70% alignment
MC: avg,2.494,-1.287,0.661,0.879,0.446,-3.136
PPO: avg,2.174,-2.279,0.97,0.823,0.097,-5.039

### 80% humor alignment
MC: avg,2.516,-1.419,0.012,0.889,0.429,-3.205
PPO: avg,2.083,-2.494,0.394,0.793,0.085,-5.853

### 80% gpt2-helpful alignment
MC: avg,1.992,-0.754,-0.35,0.88,0.427,-3.196
PPO: avg,2.024,-0.663,-0.577,0.879,0.412,-2.729

### 80% gpt2-harmless alignment
MC: avg,1.97,-1.864,0.968,0.877,0.417,-3.166
PPO: avg,2.049,-2.017,0.935,0.869,0.397,-2.628






# sentiment_control task 
basemodel_name = "opt1.3b"
data_name = "Imdb"

## assess c-levels of the original data
Statistic,positive,gpt2-helpful,gpt2-harmless,diversity,coherence,perplexity
avg,0.520,-1.525,0.581,0.882,0.237,-3.361
avg_std,0.012,0.028,0.024,0.003,0.006,0.016
50%,0.565,-1.525,0.618,0.882,0.237,-3.361
60%,0.808,-1.318,0.847,0.882,0.237,-3.361
70%,0.898,-1.062,1.04,0.882,0.237,-3.361
80%,0.958,-0.778,1.276,0.882,0.237,-3.361
90%,0.986,-0.332,1.559,0.882,0.237,-3.361
99%,0.995,0.445,2.097,0.882,0.237,-3.361

## aligned lam
50% all,0.241,0.077,0.117,0.033,0.070,0.065
60% all,2.234,0.412,0.790,0.009,0.031,0.038
70% all,3.834,0.895,1.481,0.007,0.018,0.028
80%,all,9.765,1.418,2.265,0.004,0.012,0.027
diverge
80% positive,10.975
80% gpt2-helpful,0.949
80% gpt2-harmless,1.425


### all-50% alignment
MC:  avg,0.616,-1.398,0.572,0.883,0.239,-3.187
PPO: avg,0.549,-1.495,0.49,0.882,0.228,-2.799

### all-60% alignment
MC:  avg,0.905,-1.164,0.674,0.892,0.251,-3.134
PPO: avg,0.884,-0.685,0.378,0.871,0.243,-2.464

### all-70% alignment
MC:  avg,0.933,-1.101,0.743,0.888,0.242,-3.129
PPO: avg,0.940,-0.329,0.239,0.799,0.177,-2.636

### all-80% alignment
MC:  avg,0.933,-1.062,0.637,0.89,0.246,-3.138
PPO: avg,0.954,-1.219,0.242,0.731,0.182,-2.990

### 80% positive alignment
MC:  avg,0.942,-1.099,0.469,0.887,0.25,-3.168
PPO: avg,0.924,-1.266,0.366,0.732,0.171,-2.998

### 80% gpt2-helpful alignment
MC:  avg,0.561,-0.859,0.285,0.889,0.228,-3.17
PPO: avg,0.549,-1.284,0.34,0.88,0.226,-2.832

### 80% gpt2-harmless alignment
MC:  avg,0.528,-1.798,1.213,0.891,0.238,-3.165
PPO: avg,0.474,-1.771,0.908,0.891,0.228,-2.624












## the following three studies are pointless as the three last values are not much affected 
so we changed to the same setting as anthro data

## palette settings (use submit_jobs.py to batch optimize lam)
1) increase prob of positive to 80%, maintain others
--c_list=0.800,-1.525,0.581,0.882,0.237,-3.361 --value_list="all"
optimized lambda: 2.253,0.039,0.077,0.016,0.040,0.051

2) single value
--c_list=0.800 --value_list="positive"
optimized lambda: 2.273

3) ablation of not maintaining one particular value: 
--c_list=0.800,0.581,0.882,0.237,-3.361 --value_list="positive,gpt2-harmless,diversity,coherence,perplexity"
optimized lambda: 2.275,0.077,0.043,0.030,0.049
--c_list=0.800,-1.525,0.882,0.237,-3.361 --value_list="positive,gpt2-helpful,diversity,coherence,perplexity"
optimized lambda:  2.241,0.044,0.030,0.040,0.052
--c_list=0.800,-1.525,0.581,0.237,-3.361 --value_list="positive,gpt2-helpful,gpt2-harmless,coherence,perplexity"
optimized lambda: 2.253,0.039,0.077,0.040,0.053
--c_list=0.800,-1.525,0.581,0.882,-3.361 --value_list="positive,gpt2-helpful,gpt2-harmless,diversity,perplexity"
optimized lambda: 2.256,0.038,0.077,0.017,0.051
--c_list=0.800,-1.525,0.581,0.882,0.237 --value_list="positive,gpt2-helpful,gpt2-harmless,diversity,coherence"
optimized lambda: 2.259,0.038,0.077,0.040,0.040

## assess_postalignment_multivalue, use numerical

## assess_postalignment_multivalue, use MC-generation MCsize=16
maintain all: avg,0.913,-1.201,0.492,0.891,0.242,-3.152
maintain none: avg,0.916,-1.204,0.516,0.892,0.243,-3.165
maintain all except for one: 
            avg,0.917,-1.231,0.535,0.894,0.252,-3.147
            avg,0.917,-1.143,0.473,0.888,0.244,-3.183
            avg,0.913,-1.201,0.492,0.891,0.242,-3.152
            avg,0.911,-1.24,0.527,0.887,0.234,-3.149
            avg,0.915,-1.195,0.513,0.891,0.241,-3.15




## palette settings (use submit_jobs.py to batch optimize lam)
1) increase prob of positive to 70%, maintain others
--c_list=0.700,-1.525,0.581,0.882,0.237,-3.361 --value_list="all"
optimized lambda: 1.246,0.048,0.071,0.017,0.050,0.056

2) single value
--c_list=0.700 --value_list="positive"
optimized lambda: 1.273

3) ablation of not maintaining one particular value: 
--c_list=0.700,0.581,0.882,0.237,-3.361 --value_list="positive,gpt2-harmless,diversity,coherence,perplexity"
optimized lambda: 1.271,0.071,0.053,0.040,0.055
--c_list=0.700,-1.525,0.882,0.237,-3.361 --value_list="positive,gpt2-helpful,diversity,coherence,perplexity"
optimized lambda: 1.238,0.052,0.037,0.050,0.057
--c_list=0.700,-1.525,0.581,0.237,-3.361 --value_list="positive,gpt2-helpful,gpt2-harmless,coherence,perplexity"
optimized lambda: 1.246,0.048,0.072,0.049,0.058
--c_list=0.700,-1.525,0.581,0.882,-3.361 --value_list="positive,gpt2-helpful,gpt2-harmless,diversity,perplexity"
optimized lambda: 1.248,0.047,0.071,0.019,0.057
--c_list=0.700,-1.525,0.581,0.882,0.237 --value_list="positive,gpt2-helpful,gpt2-harmless,diversity,coherence"
optimized lambda: 1.251,0.046,0.071,0.047,0.050

## assess_postalignment_multivalue, use numerical

## assess_postalignment_multivalue, use MC-generation
maintain all: avg,0.854,-1.264,0.534,0.885,0.243,-3.147
maintain none: avg,0.874,-1.291,0.523,0.893,0.246,-3.2
maintain all except for one: 
            avg,0.859,-1.303,0.553,0.891,0.25,-3.143
            avg,0.856,-1.228,0.508,0.887,0.246,-3.169
            avg,0.854,-1.264,0.534,0.885,0.243,-3.147
            avg,0.863,-1.27,0.574,0.887,0.242,-3.173
            avg,0.863,-1.244,0.547,0.888,0.244,-3.17





## palette settings (use submit_jobs.py to batch optimize lam)
1) increase prob of positive to 80%, meanwhile increase all (also including positive) to 60% quantile
--c_list=0.808,-1.318,0.847,0.882,0.237,-3.361 --value_list="all"
optimized lambda: 2.234,0.412,0.790,0.009,0.031,0.038

2) single value
--c_list=0.808 --value_list="positive"
optimized lambda: 2.377

3) ablation of not maintaining one particular value: 
--c_list=0.808,0.847,0.882,0.237,-3.361 --value_list="positive,gpt2-harmless,diversity,coherence,perplexity"
optimized lambda: 2.443,0.531,0.031,0.030,0.043
--c_list=0.808,-1.318,0.882,0.237,-3.361 --value_list="positive,gpt2-helpful,diversity,coherence,perplexity"
optimized lambda: 2.313,0.097,0.029,0.038,0.050

4) ablation of not maintaining the three general metrics
--c_list=0.808,-1.318,0.847 --value_list="positive,gpt2-helpful,gpt2-harmless"
optimized lambda: 2.237,0.418,0.795


## assess_postalignment_multivalue, use numerical

## assess_postalignment_multivalue, use MC-generation
maintain all: avg,0.904,-1.196,0.697,0.888,0.249,-3.144
maintain none: avg,0.913,-1.162,0.472,0.891,0.241,-3.171
maintain all except for one: 
            avg,0.914,-1.266,0.645,0.888,0.239,-3.15
            avg,0.91,-1.136,0.479,0.893,0.245,-3.196
maintain all except for three:
            avg,0.916,-1.14,0.673,0.895,0.248,-3.131



