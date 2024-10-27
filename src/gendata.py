import torch
import json
import fire
from transformers import GenerationConfig
from utils import (
    clean_and_trim_to_last_sentence,
    get_prompts_from_Anthropic_harmless,
    get_prompts_from_imdb,
    get_model_and_tokenizer,
    get_reward,
    ALL_SUPPORTED_VALUES,
    save_results_to_json
)
import os

class TextGeneration:
    """
    Generates text continuations based on existing prompts, either using a pretrained model or an aligned model with Monte Carlo sampling.

    Attributes:
        device (str): The device for model inference, typically 'cuda' if available.
        basemodel_name (str): The name of the base model for text generation.
        data_name (str): The source of the prompts, supporting "Anthropic-harmless" and "Imdb".
        file_path (str): Path to save the generated output in JSON format.
        top_k (int): Number of highest probability vocabulary tokens to keep for generation.
        max_new_tokens (int): Maximum number of new tokens to generate.
        generation_config (GenerationConfig): Configuration settings for text generation.

    Example usage:
        Run the following commands from the command line to use the `TextGeneration` class:
        
        ``bash
        # Generate text directly from the original model
        python gendata.py generate_from_original_model

        # Generate text with Monte Carlo sampling from an aligned model
        python gendata.py generate_from_MC_aligned_model --lam_list=-0.5 --value_list="humor" --MC_nsamples=50
        ``
    """
    def __init__(self, basemodel_name, data_name, save_directory="results"):

        """
        Initializes the TextGeneration class with model and tokenizer setup.

        Args:
            basemodel_name (str): The base model name for generation.
            data_name (str): The data source name, supports "Anthropic-harmless" and "Imdb".
            save_directory (str, optional): Directory to save generated results. Defaults to "results".
        """
        print("\nRunning TextGeneration.__init__")

        self.device = "cuda"
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

        self.basemodel_name = basemodel_name
        self.data_name = data_name

        # Load model and tokenizer for generation model
        self.basemodel, self.tokenizer_generate = get_model_and_tokenizer(basemodel_name)
        basemodel_filename = basemodel_name.rsplit('/', 1)[-1]
        self.file_path = os.path.join(save_directory, f"{basemodel_filename}-{data_name}.json")

        self.top_k = 50
        self.max_new_tokens = 50

        # Configuration for generation
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer_generate.pad_token_id,
            eos_token_id=self.tokenizer_generate.eos_token_id,
            temperature=1,
            top_k=self.top_k,
            num_beams=1,
            do_sample=True
        )

    def generate_from_original_model(self, batch_size=32):
        """
        Generates text continuations directly from the original model using predefined generation configuration.

        Args:
            batch_size (int, optional): Number of prompts processed per batch. Defaults to 32.

        Raises:
            ValueError: If an unsupported `data_name` is provided.
        """

        print("\nRunning TextGeneration.generate_from_original_model")

        results = []
        if self.data_name == "Anthropic-harmless":
            prompts = get_prompts_from_Anthropic_harmless()
        elif self.data_name == "Imdb":
            prompts = get_prompts_from_imdb()
        else:
            raise ValueError(f"Unknown data_name: {self.data_name}")

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = self.tokenizer_generate(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                generate_ids = self.basemodel.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=False,
                    output_scores=False
                )

            decoded_outputs = self.tokenizer_generate.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            perplexities = get_reward(decoded_outputs, "perplexity", model=self.basemodel, tokenizer=self.tokenizer_generate)
            for prompt, gen_text, perplexity in zip(batch_prompts, decoded_outputs, perplexities):
                results.append({"prompt": prompt, "generated": gen_text, "perplexity": perplexity})

        save_results_to_json(results, self.file_path)

        return

    def generate_from_MC_aligned_model(self, lam_list, value_list, MC_nsamples=32, start_index=0, end_index=None):
        """
        Samples multiple continuations from each prompt using Monte Carlo sampling and lambda-weighted rewards.

        The generation probability is proportional to an exponential of the reward:
        
        $$
        p(y \mid x) \propto p(y \mid x) * e^{r * \lambda}
        $$

        Args:
            lam_list (Union[List[float], float]): Lambda weights for aligning generation with specific rewards.
            value_list (Union[List[str], str]): Values to align the generated text with.
            MC_nsamples (int, optional): Number of Monte Carlo samples per prompt. Defaults to 32.
            start_index (int, optional): Start index of the prompts to process. Defaults to 0.
            end_index (Optional[int], optional): End index of the prompts to process. Defaults to None.

        Raises:
            ValueError: If an unsupported `data_name` is provided.

        """

        print("\nRunning TextGeneration.generate_from_MC_aligned_model")

        if self.data_name == "Anthropic-harmless":
            prompts = get_prompts_from_Anthropic_harmless()
        elif self.data_name == "Imdb":
            prompts = get_prompts_from_imdb()
        else:
            raise ValueError(f"Unknown data_name: {self.data_name}")
        print(f"\nTotal number of prompts: {len(prompts)}")

        if end_index:
            end_index = min(end_index, len(prompts))
        else:
            end_index = len(prompts)

        if not isinstance(lam_list, (list, tuple)):
            lam_list = [lam_list]
        elif not isinstance(lam_list, list):
            lam_list = list(lam_list)
        else:
            lam_list = lam_list
        lam = torch.tensor(lam_list, dtype=torch.float32)
        # Use a formatted string to enforce three decimal places
        lam_str = ','.join(f"{l:.3f}" for l in lam_list)
        print(f"lam_list: {lam_list}, lam_str: {lam_str}")
        # lam_str = ','.join(str(l) for l in lam_list)

        if isinstance(value_list, str):
            if value_list == "all":
                values = ALL_SUPPORTED_VALUES
            else:
                # values = [value_list]
                values = value_list.split(',')  # Split the string into a list

        # prepare file path to save results
        if value_list == "all":
            values_str = "all"
        else:
            values_str = ','.join(values)
        temp_folder = os.path.join(os.path.dirname(self.file_path), "temp")
        base_name = os.path.basename(self.file_path).rsplit('.', 1)[0]
        file_path = os.path.join(temp_folder, f"{base_name}_lam={lam_str}_val={values_str}_{start_index}to{end_index}.json")
        
        model_rewards, tokenizer_rewards = {}, {}
        feasible_values = [v for v in values if v not in ["diversity", "perplexity"]]
        for value in feasible_values:
            try:
                model_rewards[value], tokenizer_rewards[value] = get_model_and_tokenizer(value)
            except RuntimeError as e:
                print(f"Failed to load model or tokenizer for value {value}: {e}")
                continue

        print(f"\nInput lambda is {lam_str} for aligning {values_str} during generation")

        results = []

        for i in range(start_index, end_index):
            prompt = prompts[i]
            inputs = self.tokenizer_generate(prompt, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                generate_ids = self.basemodel.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=self.max_new_tokens,
                    num_return_sequences=MC_nsamples,
                    return_dict_in_generate=False,
                    output_scores=False
                )

            decoded_outputs = self.tokenizer_generate.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # clean_outputs = clean_and_trim_to_last_sentence(batch_prompts, decoded_outputs)
                
            # Memory cleanup after a large operation
            del generate_ids
            torch.cuda.empty_cache()

            # Calculate rewards and select a sentence based on lambda-weighted probabilities
            reward_vectors = []
            for value in values:
                if value == "diversity":
                    rewards = get_reward(decoded_outputs, value)
                elif value == "perplexity":
                    rewards = get_reward(decoded_outputs, value, model=self.basemodel, tokenizer=self.tokenizer_generate)
                else:
                    prompts_repeated = [prompt] * MC_nsamples
                    rewards = get_reward(decoded_outputs, value, model=model_rewards[value], tokenizer=tokenizer_rewards[value], prompts=prompts_repeated)
                reward_vectors.append(rewards)

            # print("\nGenerated candidate sentences and their rewards are:")
            # for sentence, reward in zip(decoded_outputs, rewards):
            #     print(f"\n%%% Sentence: {sentence} | Reward: {reward} %%%")

            reward_matrix = torch.tensor(reward_vectors, dtype=torch.float32)
            # print(f"\nlam shape is {lam.size()}, reward_matrix shape is {reward_matrix.size()}")
            # print(f"\nlam is {lam}, reward_matrix is {reward_matrix}")

            exp_scores = torch.exp(torch.matmul(lam, reward_matrix))
            # print(f"\nexp_scores is {exp_scores}, exp_scores shape is {exp_scores.size()}")
            probabilities = exp_scores / torch.sum(exp_scores)
            selected_index = torch.multinomial(probabilities, 1).item()
            gen_text = decoded_outputs[selected_index]

            # TODO: improve the following code to directly add calculated rewards to the selected sentences to reduce future processing
            # print(f"\nProbabilities: {probabilities}, finally selected sentence: {gen_text}")
            results.append({"prompt": prompt, "generated": gen_text})

            # Save results to the same file path, overwriting it, every 50 iterations
            if (i - start_index + 1) % 50 == 0:
                print(f"\nAt index {i+1}, saving results to: {file_path}")
                save_results_to_json(results, file_path)

        print(f"\nSaving results to: {file_path}")
        save_results_to_json(results, file_path)

        return

if __name__ == '__main__':
    fire.Fire(TextGeneration)