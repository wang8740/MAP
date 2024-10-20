# proposed decoding - top k 
import time
import copy
import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import numpy as np
import re

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "balanced"
print(device)

llama2_path = "/home/aanwar/wang8740/llama/model_output/llama-2-7b-chat"
model = LlamaForCausalLM.from_pretrained(llama2_path, device_map=device)
tokenizer = LlamaTokenizer.from_pretrained(llama2_path, padding_side='left')


def generate_value_aligned(prompt, max_decode_length, top_k, model, tokenizer, value, value_param=1, interval=3, debug=False):
    '''
        max_decode_length: maximum length of the generated text
    '''
    
    eos_token_id = tokenizer.eos_token_id  # Get the EOS token ID
    period_token_id = tokenizer.convert_tokens_to_ids('.')  # Get the token ID for period
#     print(f"eos_token_id: {eos_token_id}, period_token_id: {period_token_id}")

    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt") #.to(device)
        outputs = model(**inputs)
    
        if debug: print(f"outputs.logits size: {outputs.logits.size()}")

        generated_token_ids = inputs['input_ids'].tolist()[0]  # Start with the original prompt's token IDs

        for i in range(max_decode_length):
            if debug: print(f"\n=== generating the {i}th token ===")
            next_token_logits = outputs.logits[0, -1, :] # only batch size 1 is supported
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)

            # Filter tokens within the top k
            sorted_indices_to_keep = sorted_indices[:top_k]
            sorted_logits_to_keep = sorted_logits[:top_k]
            # print(f"sorted_indices_to_keep: {sorted_indices_to_keep}")

            probs = torch.nn.functional.softmax(sorted_logits_to_keep, dim=-1)

            if debug: print("words of selected indices: ", indices_to_words(sorted_indices_to_keep, tokenizer))
            if debug: print("probs of selected indices before preference adjustment: ", probs)

            if value_param != 0 and (i+1) % interval == 0:
                # apply the value adjustment

                log_probs = torch.log(probs)
                tentative_generated_token_ids = copy.deepcopy(generated_token_ids)

                # Apply reward to tentative sentences adding each possible candidate token
                tentative_text_batch = []
                for i, idx in enumerate(sorted_indices_to_keep):
                    # Tentatively append the token ID to the generated text
                    tentative_generated_token_ids.append(idx.item())
                    tentative_text = tokenizer.decode(tentative_generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    tentative_text_batch.append(tentative_text)

                    # Remove the tentatively added token ID to reset for the next token
                    tentative_generated_token_ids.pop()

                # Adjust the logits based on the reward
#                 rewards = get_reward(tentative_text_batch, self.value, reward_model, reward_tokenizer)
                    rewards = [random.gauss(0, 0.001) for _ in tentative_text_batch]

                log_probs += torch.tensor(rewards) * value_param

                # Sample from the filtered distribution
                probs = torch.nn.functional.softmax(log_probs, dim=-1)
                if debug: print("probs of selected indices after preference adjustment: ", probs)

            next_token_index = torch.multinomial(probs, 1).item()
            next_token_id = sorted_indices_to_keep[next_token_index].item()
            if debug: print(f"next_token_id: {next_token_id}, next_word: {tokenizer.decode([next_token_id])}")

            # Append the generated token ID
            generated_token_ids.append(next_token_id)
            generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if debug: print(f"updated text: {generated_text}")

            if next_token_id == eos_token_id or next_token_id == period_token_id:
                if debug: print(f"Stopping token generated (EOS or period), stopping at iteration {i}.")
                break

            # Prepare for the next iteration
            inputs = tokenizer(generated_text, return_tensors="pt") #.to(device)        
            outputs = model(**inputs)

    return generated_text


start_time = time.time()
# profiled usage
# %lprun -f \
generated_text = generate_value_aligned(
    prompt="If this is true, this just proves again that the idiots running this team are a bunch of", 
    max_decode_length=50, 
    top_k=50, 
    model=model, 
    tokenizer=tokenizer, 
    value = "humor",
    value_param=1,
    interval=3,
    debug=False)

print(generated_text)
print(f"took {time.time()-start_time} seconds")
