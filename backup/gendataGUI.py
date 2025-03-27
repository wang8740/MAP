import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import clean_and_trim_to_last_sentence, get_model_and_tokenizer, get_reward

# Load the model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model_generate, tokenizer_generate = get_model_and_tokenizer("llama2_chat")


def generateGUI_from_original_model(prompt, temperature=1.0, top_k=50, num_beams=1, max_new_tokens=50):
    """
    Generates text based on a given prompt using an existing pre-trained model.

    Args:
        prompt (str): Input text prompt to generate continuation.
        temperature (float, optional): Sampling temperature for text generation. Defaults to 1.0.
        top_k (int, optional): Limits the number of high-probability tokens to sample from. Defaults to 50.
        num_beams (int, optional): Number of beams for beam search. Defaults to 1.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 50.

    Returns:
        str: Generated text based on the input prompt.
    """
    inputs = tokenizer_generate([prompt], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        generate_ids = model_generate.generate(
            **inputs,
            pad_token_id=tokenizer_generate.pad_token_id,
            eos_token_id=tokenizer_generate.eos_token_id,
            temperature=temperature,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=True
        )

    decoded_output = tokenizer_generate.decode(generate_ids[0], skip_special_tokens=True)
    return decoded_output


def generateGUI_from_MC_aligned_model(prompt, model_choice, value_list, lam_list, MC_nsamples, temperature, top_k, num_beams, max_new_tokens):
    """
    Generates text using a Monte Carlo aligned model, adjusting output based on lambda-weighted rewards.

    Args:
        prompt (str): Input text prompt for generation.
        model_choice (str): Indicates whether to use the original or aligned model.
        value_list (str): Comma-separated values indicating alignment criteria.
        lam_list (str): Comma-separated lambda values for reward weighting.
        MC_nsamples (int, optional): Number of Monte Carlo samples for each prompt.
        temperature (float, optional): Sampling temperature for text generation.
        top_k (int, optional): Limits the number of high-probability tokens to sample from.
        num_beams (int, optional): Number of beams for beam search.
        max_new_tokens (int, optional): Maximum number of new tokens to generate.

    Returns:
        str: Selected generated text based on alignment.
    """
    # Debug: Check the input types and values
    print(f"\nValues List: {value_list} {type(value_list)}")
    print(f"Lambda List: {lam_list} {type(lam_list)}")

    if model_choice == "Original":
        return generateGUI_from_original_model(prompt, temperature, top_k, num_beams, max_new_tokens)
    
    if not isinstance(lam_list, (list, tuple)):
        lam_list = [lam_list]
    lam = torch.tensor(lam_list, dtype=torch.float32)
    
    if isinstance(value_list, str):
        value_list = [value_list]

    model_rewards, tokenizer_rewards = {}, {}
    for value in value_list:
        try:
            model_rewards[value], tokenizer_rewards[value] = get_model_and_tokenizer(value)
        except RuntimeError as e:
            print(f"Failed to load model or tokenizer for value {value}: {e}")
            continue

    print(f"\nInput lambda is {lam} for aligning {value_list} during generation")

    inputs = tokenizer_generate(prompt, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        generate_ids = model_generate.generate(
            **inputs,
            pad_token_id=tokenizer_generate.pad_token_id,
            eos_token_id=tokenizer_generate.eos_token_id,
            num_return_sequences=MC_nsamples,
            temperature=temperature,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=True
        )
 
    decoded_outputs = tokenizer_generate.batch_decode(generate_ids, skip_special_tokens=True)
    # clean_outputs = clean_and_trim_to_last_sentence(batch_prompts, decoded_outputs)

    # Calculate rewards and select a sentence based on lambda-weighted probabilities
    reward_vectors = []
    for value in value_list:
        rewards = get_reward(decoded_outputs, value, model_rewards[value], tokenizer_rewards[value])
        reward_vectors.append(rewards)

    reward_matrix = torch.tensor(reward_vectors, dtype=torch.float32)
    exp_scores = torch.exp(torch.matmul(lam, reward_matrix))
    probabilities = exp_scores / torch.sum(exp_scores)
    selected_index = torch.multinomial(probabilities, 1).item()
    decoded_output = decoded_outputs[selected_index]

    return decoded_output

# Initialize Gradio Interface for text generation
iface = gr.Interface(
    fn=generateGUI_from_MC_aligned_model,
    inputs=[
        gr.Textbox(lines=2, label="Enter your prompt"),
        gr.Radio(["Original", "Aligned"], value="Original", label="Model", info="Use aligned model at inference-stage?"),
        gr.Textbox(lines=1, value="humor", label="Enter the values to align, separated by ','"),
        gr.Textbox(lines=1, value="1.29", label="Enter the hyperparameter 'lambda' associated with each value"),
        gr.Slider(minimum=1, maximum=50, step=1, value=32, label="Num Monte Carlo Samples"),
        gr.Slider(minimum=0, maximum=1, step=0.01, value=1, label="Temperature"),
        gr.Slider(minimum=0, maximum=100, step=1, value=50, label="Top k"),
        gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Num Beams"),
        gr.Slider(minimum=1, maximum=200, step=1, value=50, label="Max New Tokens")
    ],
    outputs=gr.Textbox(lines=10, label="Generated Text"),
    title="Text Generation - original model",
    description="Generate text using the original llama-2-7B-chat model."
)

if __name__ == '__main__':
    _, _, url = iface.queue().launch(server_name="0.0.0.0", share=True)
    print("Public URL:", url)
