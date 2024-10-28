import json
import os
import glob
from shutil import copyfile
import fire

class MergeProcessor:

    def merge_added_rewards(self, original_file_path: str, save_to_temp_folder: bool = False) -> None:
        """
        Merges rewards from various human values stored in temporary files into a single JSON file, either
        saving to a temporary folder or overwriting the original file.

        This method checks for files in the temporary folder with names matching the pattern
        of the original file, updates the original JSON file by adding reward entries
        from these files, and saves the result either in the temp folder or overwrites
        the original.

        Args:
            original_file_path (str): Path to the original JSON file.
            save_to_temp_folder (bool, optional): If True, saves the merged file in a temp folder
                                                  instead of overwriting the original. Defaults to False.

        Example:
            >>> processor = MergeProcessor()
            >>> processor.merge_added_rewards("results/Llama27b-chat-Anthropic-harmless.json", save_to_temp_folder=True)

        Command-line usage:
            python mergeProcessor.py merge_added_rewards --original_file_path="results/Llama27b-chat-Anthropic-harmless.json" --save_to_temp_folder=True
        """

        print("\nRunning MergeProcessor.merge_added_rewards\n")

        # Step 1: Copy the original file to a new merged file in the temp folder
        temp_folder = os.path.join(os.path.dirname(original_file_path), "temp")
        base_name = os.path.basename(original_file_path).rsplit('.', 1)[0]
        # print(f"base_name: {base_name}")

        if save_to_temp_folder:
            merged_file_path = os.path.join(temp_folder, base_name+'.json')
            copyfile(original_file_path, merged_file_path)
            print(f"merging results at {merged_file_path}")

            # Load the content of the newly copied merged file
            with open(merged_file_path, 'r') as file:
                merged_data = json.load(file)
        else:
            with open(original_file_path, 'r') as file:
                merged_data = json.load(file)

        # Step 2: Process each temp file and update the merged data
        for file_name in os.listdir(temp_folder):
            if file_name.startswith(base_name) and "temp" in file_name:
                temp_file_path = os.path.join(temp_folder, file_name)

                # Identify the value from the file name
                parts = file_name.split('_temp_')
                if len(parts) == 2:
                    value = parts[1].rsplit('.', 1)[0]  # Get 'value' from the filename
                    print(f"Processing {value} from {temp_file_path}")
                    # Load the temp file data
                    with open(temp_file_path, 'r') as file:
                        temp_data = json.load(file)

                    # Update merged_data based on temp_data
                    for entry, temp_entry in zip(merged_data, temp_data):
                        entry[value] = temp_entry[value]

                # Remove the temporary file
                os.remove(temp_file_path)

        # Step 3: Save the updated merged data back to the merged file
        # print(f"merged_data len: {len(merged_data)}, merged_data[0]: {merged_data[0]}")
        if save_to_temp_folder:
            with open(merged_file_path, 'w') as file:
                json.dump(merged_data, file, indent=4)
            print(f"Updated results saved back to the temp file: {merged_file_path}")
        else:
            with open(original_file_path, 'w') as file:
                json.dump(merged_data, file, indent=4)
            print(f"Updated results saved back to the original file: {original_file_path}")

        return


    def merge_gendata_bypattern(self, json_file_pattern: str) -> None:
        """
        Merges multiple JSON files matched by a pattern into a single output file.

        This function collects JSON files based on the specified glob pattern, merges
        the data into one JSON array, and saves the result at a directory level above
        'temp/'. The function also removes '_*to*' from the filename before saving.

        Args:
            json_file_pattern (str): The glob pattern to match JSON files for merging.
                                     Example: 'results/temp/*_val=all_*to*.json'

        Example:
            >>> processor = MergeProcessor()
            >>> processor.merge_gendata_bypattern("results/temp/Llama27b-chat-Anthropic-harmless_lam=2.018,1.393,1.498,0.008,0.015,0.088_val=all_*to*.json")

        Command-line usage:
            python mergeProcessor.py merge_gendata_bypattern --json_file_pattern="results/temp/Llama27b-chat-Anthropic-harmless_lam=2.018,1.393,1.498,0.008,0.015,0.088_val=all_*to*.json"
        """
  
        print("\nRunning MergeProcessor.merge_gendata_bypattern\n")
        all_results = []
        for file_name in glob.glob(json_file_pattern):
            with open(file_name, 'r') as file:
                data = json.load(file)
                all_results.extend(data)
            
            # Remove the temporary file
            os.remove(file_name)
        
        # Construct the output file path to save the merged file at the same level as temp/
        output_dir = os.path.dirname(os.path.dirname(json_file_pattern))  # Go up one level from temp/
        output_base_name = os.path.basename(json_file_pattern).replace('_*to*', '')  # Remove '_*to*' pattern
        output_file_path = os.path.join(output_dir, output_base_name)

        # Save the merged results to the new location
        with open(output_file_path, 'w') as file:
            json.dump(all_results, file, indent=4)

        print(f"Merged data saved to {output_file_path}")
        
if __name__ == "__main__":
    fire.Fire(MergeProcessor)
