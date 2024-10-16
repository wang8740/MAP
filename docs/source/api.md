<a id="alignValues"></a>

# alignValues

<a id="alignValues.AlignValues"></a>

## AlignValues Objects

```python
class AlignValues()
```

<a id="alignValues.AlignValues.save_results_to_text"></a>

#### save\_results\_to\_text

```python
def save_results_to_text(optimized_lambda,
                         success,
                         save_prefix='results/alignValues')
```

Save the results to text file. This is used to generate the results file and save it to disk

**Arguments**:

- `optimized_lambda`: list of optimized lambda values
- `success`: True if success False if failure ( NaN in case of failure

<a id="alignValues.AlignValues.save_results_to_csv"></a>

#### save\_results\_to\_csv

```python
def save_results_to_csv(optimized_lambda,
                        dirichlet_lambda,
                        save_prefix='results/alignValues')
```

Save the results to a CSV file. This function appends new data each time it's called.

**Arguments**:

- `optimized_lambda`: list of optimized lambda values
- `dirichlet_lambda`: list of Dirichlet reference lambda values
- `save_prefix`: prefix for the save file path

<a id="alignValues.AlignValues.gen_rand_MAP_lambda"></a>

#### gen\_rand\_MAP\_lambda

```python
def gen_rand_MAP_lambda(num_lambda,
                        scaling_MAX,
                        save_prefix='rand_MAP_lambda')
```

Generate random MAP lambda values by drawing each c_i randomly between the current c_i

and the maximum reward corresponding to value i. This function modifies the c values,
recalculates lambda, and returns a list of lambda values constrained by scaling_MAX.

**Arguments**:

- `num_lambda`: Number of valid lambda values to generate
- `scaling_MAX`: Maximum allowed L1 norm for the generated lambda values

**Returns**:

Tuple containing list of generated lambda values and success rate

<a id="genDPOdata"></a>

# genDPOdata

<a id="genDPOdata.preprocess_data"></a>

#### preprocess\_data

```python
def preprocess_data(examples, max_words=50)
```

Note that many examples have the chosen=rejected in the first round of dialog, so the sample size cannot be too small.

<a id="genDPOdata.gen_mixed_preference_data"></a>

#### gen\_mixed\_preference\_data

```python
def gen_mixed_preference_data(data_source, sample_size, split)
```

data_source is a dict that currently supports {"harmless": p, "helpful": 1-p} format
split = 'train' or 'test'

<a id="gendata"></a>

# gendata

<a id="gendataGUI"></a>

# gendataGUI

<a id="gendataGUI.generateGUI_from_original_model"></a>

#### generateGUI\_from\_original\_model

```python
def generateGUI_from_original_model(prompt,
                                    temperature=1.0,
                                    top_k=50,
                                    num_beams=1,
                                    max_new_tokens=50)
```

Generate text based on the input prompt and other parameters.

<a id="generate-alpacaFinetune"></a>

# generate-alpacaFinetune

<a id="getDPOsoup"></a>

# getDPOsoup

<a id="getDPOsoup.soup"></a>

#### soup

```python
def soup(basemodel_name, sample_size, beta, harmless_lambda, save_path)
```

beta: same as in trainDPO.py
harmless_lambda: Interpolation factor in [0, 1]

<a id="mergeProcessor"></a>

# mergeProcessor

<a id="mergeProcessor.MergeProcessor"></a>

## MergeProcessor Objects

```python
class MergeProcessor()
```

<a id="mergeProcessor.MergeProcessor.merge_added_rewards"></a>

#### merge\_added\_rewards

```python
def merge_added_rewards(original_file_path, save_to_temp_folder=False)
```

save_to_temp_folder=True if we save to a temp folder, otherwise overwriting the original file with more entries (rewards) in each row

<a id="mergeProcessor.MergeProcessor.merge_gendata_bypattern"></a>

#### merge\_gendata\_bypattern

```python
def merge_gendata_bypattern(json_file_pattern)
```

Merges JSON files matched by a specific pattern into a single file and moves it to the directory level
one above 'temp/'. Assumes that the JSON file pattern includes a 'temp/' directory at its end
and that all files are located within this directory. The final filename has the pattern '_*to*'
removed before saving.

**Arguments**:

- `json_file_pattern` _str_ - The glob pattern to match JSON files for merging.
- `Example` - 'results/temp/*_val=all_*to*.json'

<a id="plot_cal_winrate"></a>

# plot\_cal\_winrate

<a id="plot_cal_winrate.collect_multiple_results"></a>

#### collect\_multiple\_results

```python
def collect_multiple_results(model_files,
                             base_model_file,
                             file_prefix,
                             metrics=None)
```

summarize multiple results into a list and save to json

<a id="plot_cal_winrate.render_latex_table"></a>

#### render\_latex\_table

```python
def render_latex_table(win_rate_results_list, file_prefix)
```

Render and save LaTeX table

<a id="plot_cal_winrate.plot_helpful_vs_harmless"></a>

#### plot\_helpful\_vs\_harmless

```python
def plot_helpful_vs_harmless(win_rate_results_list, harmless_ratios,
                             file_prefix)
```

Plot helpful and harmless win rates over the harmless_ratios

<a id="plot_pairwise_pareto"></a>

# plot\_pairwise\_pareto

<a id="plot_pairwise_pareto.plot_matrix_scatterplot"></a>

#### plot\_matrix\_scatterplot

```python
def plot_matrix_scatterplot(csv_path)
```

Intend to input a path to randomly generated c values to show Tradeoffs and Pareto Fronts

<a id="plot_radar_tabresults"></a>

# plot\_radar\_tabresults

<a id="plot_rand_lambda_MAP_region"></a>

# plot\_rand\_lambda\_MAP\_region

<a id="plot_rand_lambda_MAP_region.plot_lambdas_3D"></a>

#### plot\_lambdas\_3D

```python
def plot_lambdas_3D(file_path, x_value, y_value, z_value, output_prefix)
```

Function to plot optimized_lambda and Dirichlet_lambda_ref in a 3D plot

with semi-transparent dots, overlaying one over the other.

**Arguments**:

- `file_path`: Path to the CSV file containing lambda values
- `x_value`: The name of the value for the x-axis (e.g., 'gpt2-helpful')
- `y_value`: The name of the value for the y-axis (e.g., 'gpt2-harmless')
- `z_value`: The name of the value for the z-axis
- `output_prefix`: Path to save the output PDF

<a id="plot_rand_lambda_MAP_region.plot_lambdas_2D_subplots"></a>

#### plot\_lambdas\_2D\_subplots

```python
def plot_lambdas_2D_subplots(file_paths, x_value, y_value, output_prefix)
```

Function to create several subplots, one for each file, in a 2D plot

with semi-transparent dots. The Dirichlet_lambda_ref is plotted file-specific 
for each subplot.

**Arguments**:

- `file_paths`: List of paths to the CSV files containing lambda values
- `x_value`: The name of the value for the x-axis (e.g., 'gpt2-helpful')
- `y_value`: The name of the value for the y-axis (e.g., 'gpt2-harmless')
- `output_prefix`: Path to save the output PDF

<a id="plot_reward_dist"></a>

# plot\_reward\_dist

<a id="plot_reward_dist.plot_hist"></a>

#### plot\_hist

```python
def plot_hist(json_file)
```

Intend to input a path to sentences along with calculated rewards to show the distribution of each reward

<a id="plot_reward_dist.plot_hist_positive"></a>

#### plot\_hist\_positive

```python
def plot_hist_positive(json_file)
```

Intends to input a path to a JSON file with a "positive" column and plot the distribution of these values.

<a id="plot_sequential_baseline"></a>

# plot\_sequential\_baseline

<a id="plot_tab_in_quantiles"></a>

# plot\_tab\_in\_quantiles

This script turns the tab results namely the real c levels into quantile representations. 
But we leave the details out of the paper as they are not perfectly close to the target and can cause reader confusions.

<a id="rewardProcessor"></a>

# rewardProcessor

<a id="rewardProcessor.RewardProcessor"></a>

## RewardProcessor Objects

```python
class RewardProcessor()
```

<a id="rewardProcessor.RewardProcessor.add_reward"></a>

#### add\_reward

```python
def add_reward(value, basemodel_for_perplexity=None)
```

The process is non-invasive, meaning that if the original json files has existing rewards for values not equal to value, it will not be overwritten or emptied

<a id="rewardTrainer"></a>

# rewardTrainer

python rewardTrainer.py --model_name_or_path=gpt2 --output_dir="results/reward_modeling_anthropic_hh" --per_device_train_batch_size=64 --num_train_epochs=1 --gradient_accumulation_steps=16 --gradient_checkpointing=True --learning_rate=1.41e-5 --report_to="wandb" --remove_unused_columns=False --optim="adamw_torch" --logging_steps=10 --evaluation_strategy="steps" --max_length=512

<a id="submit_DPO_job"></a>

# submit\_DPO\_job

<a id="submit_MAP-DPO-PPO_comparison"></a>

# submit\_MAP-DPO-PPO\_comparison

<a id="submit_MC-aligned_pipeline"></a>

# submit\_MC-aligned\_pipeline

<a id="submit_PPO-aligned_pipeline"></a>

# submit\_PPO-aligned\_pipeline

<a id="submit_jobs"></a>

# submit\_jobs

<a id="submit_perplexity_correction"></a>

# submit\_perplexity\_correction

<a id="submit_soup"></a>

# submit\_soup

<a id="tokenValueDecoder"></a>

# tokenValueDecoder

<a id="tokenValueDecoder.generate_value_aligned"></a>

#### generate\_value\_aligned

```python
def generate_value_aligned(prompt,
                           max_decode_length,
                           top_k,
                           model,
                           tokenizer,
                           value,
                           value_param=1,
                           interval=3,
                           debug=False)
```

max_decode_length: maximum length of the generated text

<a id="trainDPO"></a>

# trainDPO

<a id="trainPPO"></a>

# trainPPO

<a id="trainPPO.build_dataset"></a>

#### build\_dataset

```python
def build_dataset(config, tokenizer, data_name)
```

Build dataset for training using prompts from get_prompts_from_imdb and apply tokenization.

**Arguments**:

  data_name (`str`):
  The name of the dataset to be loaded.
  

**Returns**:

  dataloader (`torch.utils.data.DataLoader`):
  The dataloader for the dataset.

<a id="utils"></a>

# utils

<a id="utils.TASK_NAME"></a>

#### TASK\_NAME

'conversation', 'sentiment_control'

<a id="utils.convert_ppo_modelname_to_huggingface_valid"></a>

#### convert\_ppo\_modelname\_to\_huggingface\_valid

```python
def convert_ppo_modelname_to_huggingface_valid(ppo_model_name)
```

convert a ppo trained model name to ensure that it is a valid huggingface model path name

