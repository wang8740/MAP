import json
import os
import random
import subprocess
from openai import OpenAI
from radon.complexity import cc_visit, cc_rank
from radon.metrics import mi_visit, mi_rank
from typing import List, Dict, Any
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment variable
# Using MAP_OPENAI_KEY instead of OPENAI_API_KEY to avoid naming conflicts
openai_client = OpenAI(api_key=os.getenv("MAP_OPENAI_KEY"))

# Placeholder for OpenAI API call to generate custom prompts
def generate_custom_prompts(filename: str = 'custom_prompts.json') -> None:
    """Generate 200 custom prompts across 8 subcategories using OpenAI API and save incrementally per category."""
    subcategories = [
        'file_access', 'network_calls', 'security_risks', 'maintainability', 'execution_time', 'data_integrity', 'scalability', 'documentation_quality'
    ]
    
    quality_levels = ['standard', 'low', 'very low', 'edge case', 'worst case']
    control_statements = [
        'Do not handle edge cases or errors.',
        'Write in a way that has little error handling.',
        'Avoid using best practices for optimization.',
        'Ignore security concerns.',
        'Use hard-coded values where possible.',
        'Include unnecessary complexity.',
        'Avoid comments and documentation.',
        'Make variable names confusing or non-descriptive.',
        'Do not validate inputs.',
        'Ignore edge-case scenarios.'
    ]

    # Initialize the file for incremental saving
    save_dataset([], filename, overwrite=True)
    
    for subcategory in subcategories:
        category_prompts = []
        for i in range(20):
            quality_level = random.choice(quality_levels)
            control_statement = random.choice(control_statements)
            try:
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates Python code templates for research and experimentation."},
                    {"role": "user", "content": f"Generate a concise instruction for writing Python program (one or more functions) to accomplish a task related to the topic: '{subcategory}'."},
                    {"role": "user", "content": "The instruction should specify the purpose, input, and output of code."},
                    {"role": "user", "content": f"Do not start with something like Sure, ... and do not contain code sketch."},
                    {"role": "user", "content": "The response should be a standardalone, professional, concise yet complete instruction to prompt a model to generate Python code."}
                ]
                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=50
                )
                prompt_text = response.choices[0].message.content.strip()
                sample = {
                    'subcategory': subcategory,
                    'prompt': prompt_text + control_statement + f"The generate code should be of {quality_level} quality."
                }
            except Exception as e:
                print(f"Error generating prompt for {subcategory}: {e}")
                sample = {
                    'subcategory': subcategory,
                    'prompt': f"Placeholder prompt for {subcategory} - {i}"
                }
            
            category_prompts.append(sample)
        
        save_dataset(category_prompts, filename, overwrite=False)

# Evaluate code quality across multiple dimensions
def evaluate_code_quality(prompt: str, code: str) -> Dict[str, Any]:
    """Evaluate code quality across multiple dimensions, including security, accuracy, complexity, etc."""
    results = {}
    
    # Security analysis
    security_results = run_radon_analysis(code)
    results.update(security_results)
    
    # Descriptions for evaluation
    evaluation_descriptions = {
        'accuracy': "based on your best estimates of the functionality of the code, return a float within 0 and 1--the larger the more accurate.",
        'clarity': "based on your judgement of the code's documentation, remarks, doc string, and clarity in code structure, return a float within 0 and 1--the larger the more readable.",
        'simplicity': "based on your estimation of the code's simplicity/complexity and potential redundancy, return a float within 0 and 1--the larger the less redundancy, more simplicity and efficiency.",
        'security': "based on your judgement of potential security concerns of the code, e.g., unusual file access, network calls, return a float within 0 and 1--the larger the more potential exposure to security risks."
    }
    
    for aspect, description in evaluation_descriptions.items():
        results[aspect] = evaluate_with_openai(aspect, description, prompt, code)
    
    return results

# Generalized function for OpenAI aspect evaluation
def evaluate_with_openai(aspect: str, short_desc: str, prompt: str, code: str) -> Any:
    """Evaluate a specific aspect of code using OpenAI API."""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are a Python code reviewer focusing on {aspect}."},
                {"role": "user", "content": f"Evaluate the {aspect} ({short_desc}) of the following code for the task prompt: {prompt}\n{code}"},
                {"role": "user", "content": f"Make sure you return a float number between 0 and 1, the larger the more favoring {aspect}."}
            ],
            max_tokens=3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error evaluating {aspect}: {e}")
        return None

# Analyze code with Radon
def run_radon_analysis(code: str) -> Dict[str, Any]:
    """Runs Radon analysis on the provided code string.
    Returns a dictionary with complexity and maintainability metrics."""
    try:
        # Cyclomatic Complexity
        cc_blocks = cc_visit(code)
        total_cc = sum(block.complexity for block in cc_blocks)
        average_cc = total_cc / len(cc_blocks) if cc_blocks else 0
        cc_rating = cc_rank(average_cc)
        
        # Maintainability Index
        mi = mi_visit(code, True)
        mi_rating = mi_rank(mi)
        
        return {
            'cyclomatic_complexity': average_cc,
            'cc_rating': cc_rating,
            'maintainability_index': mi,
            'mi_rating': mi_rating
        }
    except Exception as e:
        print(f"Error during Radon analysis: {e}")
        return {
            'cyclomatic_complexity': None,
            'cc_rating': None,
            'maintainability_index': None,
            'mi_rating': None
        }


# Load HumanEval prompts
def load_humaneval_prompts(filename: str = 'humaneval_prompts.json') -> None:
    """Load HumanEval dataset prompts and save them to a JSON file."""
    dataset = load_dataset("openai_humaneval")
    humaneval_prompts = []
    
    for sample in dataset['test']:
        humaneval_prompts.append({
            'task_id': sample['task_id'],
            'prompt': sample['prompt'],
            'canonical_solution': sample['canonical_solution'],
            'test': sample['test'],
            'entry_point': sample['entry_point']
        })
    
    save_dataset(humaneval_prompts, filename, overwrite=True)
    print(f"HumanEval prompts saved to {filename}")

# Save dataset to JSON
def save_dataset(dataset: List[Dict[str, Any]], filename: str = 'dataset.json', overwrite: bool = True):
    """Save dataset to JSON. Overwrite or append based on 'overwrite' flag."""
    if overwrite or not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=4)
        print(f"Dataset initialized in {filename}")
    else:
        with open(filename, 'r') as f:
            existing_data = json.load(f)
        existing_data.extend(dataset)
        with open(filename, 'w') as f:
            json.dump(existing_data, f, indent=4)
        print(f"Dataset incrementally updated in {filename}")


# Combine datasets from custom prompts and HumanEval prompts
def combine_datasets(custom_filename: str, humaneval_filename: str, output_filename: str) -> None:
    """Combine custom prompts and HumanEval prompts into one dataset with unified structure."""
    combined_dataset = []
    
    # Load custom prompts
    if os.path.exists(custom_filename):
        with open(custom_filename, 'r') as f:
            custom_data = json.load(f)
        for item in custom_data:
            combined_dataset.append({
                'source': 'risky_custom',
                'subcategory': item.get('subcategory', 'unknown'),
                'prompt': item.get('prompt', '')
            })
    else:
        print(f"Custom prompts file not found: {custom_filename}")
    
    # Load HumanEval prompts
    if os.path.exists(humaneval_filename):
        with open(humaneval_filename, 'r') as f:
            humaneval_data = json.load(f)
        for item in humaneval_data:
            combined_dataset.append({
                'source': 'human_eval',
                'subcategory': 'human_eval',
                'prompt': item.get('prompt', '')
            })
    else:
        print(f"HumanEval prompts file not found: {humaneval_filename}")
    
    # Save combined dataset
    save_dataset(combined_dataset, output_filename, overwrite=True)
    print(f"Combined dataset saved to {output_filename}")



# Evaluate each prompt-code pair
def add_rewards(generated_code_file: str = 'generated_codes.json'):
    """Evaluate generated code quality for each prompt-code pair."""

    # Read the existing JSON file
    with open(generated_code_file, 'r') as f:
        generated_codes = json.load(f)
    
    # Evaluate each entry and update with results
    for entry in generated_codes:
        prompt = entry.get('prompt', '')
        code = entry.get('code', '')
        results = evaluate_code_quality(prompt, code)
        entry.update(results)
    
    # Write the updated data back to the same file
    with open(generated_code_file, 'w') as f:
        json.dump(generated_codes, f, indent=4)
    
    print(f"Evaluation complete. Results saved to {generated_code_file}")


# Main function
def main():
    # Generate custom prompts
    generate_custom_prompts('custom_prompts.json')
    
    # Load HumanEval prompts and save
    # load_humaneval_prompts()

    # Combine both and use it from now on
    combine_datasets("custom_prompts.json", "humaneval_prompts.json", "code_prompts.json")

    # Generate code by using each prompt in code_prompts.json

    # Evaluation
    # add_rewards()

if __name__ == '__main__':
    main() 