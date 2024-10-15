from datasets import load_dataset, concatenate_datasets

seed = 6

def preprocess_data(examples, max_words=50):
    '''
        Note that many examples have the chosen=rejected in the first round of dialog, so the sample size cannot be too small.
    '''
    def truncate_to_max_words(text, max_words):
        # Split text into words, limit to max_words, and join back into a string
        return ' '.join(text.split()[:max_words])
    
    prompts = []
    chosen_responses = []
    rejected_responses = []
    processed_count = 0

    for chosen, rejected in zip(examples['chosen'], examples['rejected']):
        # Print the raw chosen and rejected data for inspection
        # print("\nRaw chosen response:")
        # print(chosen)
        # print("\nRaw rejected response:")
        # print(rejected)

        # Extract prompt (first Human question)
        prompt_start = chosen.find("Human: ")
        prompt_end = chosen.find("Assistant:", prompt_start + len("Human: "))

        # Skip if the expected pattern is not found
        if prompt_start == -1 or prompt_end == -1:
            continue

        prompt = chosen[prompt_start + len("Human: "):prompt_end].strip()

        # Extract chosen response (first Assistant response)
        chosen_start = prompt_end + len("Assistant:")
        chosen_end = chosen.find("Human:", chosen_start)
        if chosen_end == -1:  # If there's no second "Human:", take the rest of the string
            chosen_end = len(chosen)
        chosen_response = chosen[chosen_start:chosen_end].strip()

        # Extract rejected response (first Assistant response)
        rejected_start = rejected.find("Assistant:") + len("Assistant:")
        rejected_end = rejected.find("Human:", rejected_start)
        if rejected_end == -1:  # If there's no second "Human:", take the rest of the string
            rejected_end = len(rejected)
        rejected_response = rejected[rejected_start:rejected_end].strip()

        # Truncate each text to max_words
        truncated_prompt = truncate_to_max_words(prompt, max_words)
        truncated_chosen = truncate_to_max_words(chosen_response, max_words)
        truncated_rejected = truncate_to_max_words(rejected_response, max_words)

        prompts.append(truncated_prompt)
        chosen_responses.append(truncated_chosen)
        rejected_responses.append(truncated_rejected)

        # Increment processed count
        processed_count += 1

    return {
        'prompt': prompts,
        'chosen': chosen_responses,
        'rejected': rejected_responses
    }


def gen_mixed_preference_data(data_source, sample_size, split):
    '''
    data_source is a dict that currently supports {"harmless": p, "helpful": 1-p} format
    split = 'train' or 'test'
    '''
    ds_mix = None

    # Load and process each dataset
    for k, v in data_source.items():
        if v > 0:
            if k == "harmless":
                ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split=split)
            elif k == "helpful":
                ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split=split)
            else:
                raise ValueError(f"Unsupported data source: {k}")

            # Randomly select a subset
            subset_size = int(v * sample_size)
            ds = ds.shuffle(seed=seed).select(range(min(len(ds), subset_size)))

            # Process the dataset
            processed_ds = ds.map(
                preprocess_data,
                batched=True,
                batch_size=100,  # Adjust this based on your memory constraints
                remove_columns=ds.column_names
            )

            # Merge datasets
            if ds_mix is None:
                ds_mix = processed_ds
            else:
                ds_mix = concatenate_datasets([ds_mix, processed_ds])

    # Shuffle the final mixed dataset
    ds_mix = ds_mix.shuffle(seed=seed)

    # Format and print the first entry with all columns
    example = ds_mix[0]
    for col_name, col_value in example.items():
        print(f"\n{col_name}:")
        print(f"{col_value}")

    return ds_mix

if __name__ == "__main__":
    # data_source = {"harmless": 1, "helpful": 0}
    # data_source = {"harmless": 0, "helpful": 1}
    data_source = {"harmless": 0.5, "helpful": 0.5}
    sample_size = 2000
    ds_mix = gen_mixed_preference_data(data_source, sample_size, split="train")
    