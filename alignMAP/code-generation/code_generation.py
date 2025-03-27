import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict
import re
import tempfile
import subprocess

# Load pre-trained code generation model from HuggingFace
def load_llm_model(model_name: str = 'Salesforce/codegen-2B-multi'): # also try "codellama/CodeLlama-7b-hf"
    """Load a pre-trained code generation model and tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    model.eval()
    return tokenizer, model


# Generate code for a given prompt
def generate_code(prompt: str, tokenizer, model, max_length: int = 1024, num_return_sequences: int = 1) -> List[str]:
    """Generate code snippets based on a given prompt using the loaded LLM model."""
    
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(model.device)
    prompt_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    return [tokenizer.decode(output[prompt_length:], skip_special_tokens=True) for output in outputs]

# Load prompts from JSON file
def load_prompts(filename: str = 'code_prompts.json', top_n: int = 2) -> List[Dict[str, str]]:
    """Load top N prompts from a JSON file for debugging."""
    with open(filename, 'r') as f:
        prompts = json.load(f)
    return prompts[:top_n]

# Save generated code to JSON file
def save_generated_codes(data: List[Dict[str, str]], filename: str = 'generated_codes.json'):
    """Save generated code snippets along with prompts to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Generated codes saved to {filename}")

def create_filtered_code(filename: str, output_filename: str) -> None:
    """
    Load generated codes from filename, clean them, and save the filtered result.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        raw_code = entry.get('continuation', '')
        entry['code'] = extract_python_code(raw_code)
    
    with open(output_filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Filtered codes saved to {output_filename}")


def extract_python_code(code: str) -> str:
    """
    Extract valid Python code from mixed content.
    Handles references, markdown headers, and malformed code.
    Then validates and formats using `ruff`.
    """
    # Step 1: Remove non-code text
    code = re.sub(r'Reference:.*\n', '', code, flags=re.IGNORECASE)  # Remove references
    code = re.sub(r'\*\*.*\*\*', '', code)  # Remove markdown-style bold text
    
    # Step 2: Extract valid Python-like lines
    python_lines = [
        line for line in code.split('\n')
        if line.strip().startswith((
            'def ', 'class ', 'import ', 'from ', '#', 'if ', 'for ', 'while ',
            'try ', 'except ', 'return ', '@'
        )) or line.strip() == ''  # Keep blank lines for readability
    ]
    
    filtered_code = '\n'.join(python_lines).strip()
    
    # Step 3: Validate and auto-correct with `ruff`
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode='w+', delete=False) as tmp_file:
            tmp_file.write(filtered_code)
            tmp_file.seek(0)
            
            # Run `ruff check --fix` with detailed output
            result = subprocess.run(
                ['ruff', 'check', '--fix', '--verbose', tmp_file.name],
                check=False,
                capture_output=True,
                text=True
            )
            
            print("Ruff STDOUT:\n", result.stdout)
            print("Ruff STDERR:\n", result.stderr)
            
            if result.returncode != 0:
                print("Failed to auto-correct code with ruff.")
                return filtered_code  # Return the raw filtered code
            
            # Read the corrected code
            tmp_file.seek(0)
            corrected_code = tmp_file.read()
        
        print("Code successfully formatted with ruff.")
        return corrected_code.strip()
    
    except subprocess.CalledProcessError as e:
        print("Failed to auto-correct code with ruff:", e.stderr)
        return filtered_code  # Return the extracted code if `ruff` fails



# Main function to generate code for each prompt
if __name__ == '__main__':
    # model_name = 'Salesforce/codegen-2B-multi'
    # tokenizer, model = load_llm_model(model_name)
    
    # prompts = load_prompts('code_prompts.json', top_n=2)
    # generated_data = []
    
    # for prompt_entry in prompts:
    #     prompt_text = prompt_entry.get('prompt', '')
    #     print(f"Generating code for prompt: {prompt_text[:50]}...")
    #     codes = generate_code(prompt_text, tokenizer, model, num_return_sequences=1)
    #     for raw_code in codes:
    #         generated_data.append({
    #             'prompt': prompt_text,
    #             'continuation': raw_code,
    #         })

    # save_generated_codes(generated_data, filename='generated_codes.json')
    create_filtered_code('generated_codes.json', 'generated_codes.json')
