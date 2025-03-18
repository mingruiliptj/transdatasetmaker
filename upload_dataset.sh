#!/bin/bash

# Help function
show_help() {
    echo "Usage: $0 [options] <dataset_directory>"
    echo
    echo "Upload a dataset to Hugging Face Hub"
    echo
    echo "Options:"
    echo "  -n, --name NAME       Name for the dataset (default: zh_en_translation)"
    echo "  -h, --help           Show this help message"
    echo
    echo "Example:"
    echo "  $0 -n my_translation_dataset ./translation_dataset"
    echo "  $0 --name custom_dataset ./output_dir"
}

# Default values
DATASET_NAME="zh_en_translation"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            DATASET_NAME="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            DATASET_DIR="$1"
            shift
            ;;
    esac
done

# Check if dataset directory is provided
if [ -z "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not provided"
    show_help
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Directory '$DATASET_DIR' does not exist"
    exit 1
fi

# Check if required files exist (either Arrow or JSON format)
if [ ! -f "$DATASET_DIR/data-00000-of-00001.arrow" ] && [ ! -f "$DATASET_DIR/finetune_data.json" ]; then
    echo "Error: No valid dataset file found in $DATASET_DIR"
    echo "Expected either:"
    echo "  - data-00000-of-00001.arrow (Arrow format)"
    echo "  - finetune_data.json (JSON format)"
    exit 1
fi

# Check if Hugging Face CLI is installed and user is logged in
if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli not found. Please install it with:"
    echo "pip install huggingface_hub"
    exit 1
fi

# Check if user is logged in
if ! huggingface-cli whoami &> /dev/null; then
    echo "Error: Not logged in to Hugging Face. Please login with:"
    echo "huggingface-cli login"
    exit 1
fi

echo "Uploading dataset to Hugging Face Hub..."
echo "Dataset name: $DATASET_NAME"
echo "Source directory: $DATASET_DIR"
echo "Privacy setting: public"

# Create Python script for upload
TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" << EOL
from huggingface_hub import HfApi, create_repo, whoami
from datasets import load_from_disk, Dataset
import os
import sys
import time
import json
from datetime import datetime

dataset_dir = "$DATASET_DIR"
repo_name = "$DATASET_NAME"

try:
    # Get user information
    user_info = whoami()
    namespace = user_info.get("name")
    if not namespace:
        raise ValueError("Could not determine your Hugging Face username. Please make sure you are logged in.")
    
    # Create full repository name with namespace
    full_repo_name = f"{namespace}/{repo_name}"
    print(f"Using repository name: {full_repo_name}")
    
    # Create the repository
    try:
        repo_url = create_repo(
            repo_id=full_repo_name,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        print(f"Repository ready at: {repo_url}")
    except Exception as e:
        print(f"Error creating repository: {str(e)}")
        sys.exit(1)
    
    # Wait a moment for the repository to be fully created
    time.sleep(2)
    
    # Initialize API
    api = HfApi()
    
    # Check if we have Arrow or JSON format
    is_arrow_format = os.path.exists(os.path.join(dataset_dir, "data-00000-of-00001.arrow"))
    
    if is_arrow_format:
        # Load dataset in Arrow format
        dataset = load_from_disk(dataset_dir)
        total_examples = len(dataset)
        
        # Format for Sloth - each example needs a "conversations" field with a list of messages
        sloth_data = []
        for example in dataset:
            # Create a conversation with human and assistant messages
            conversation = [
                {"role": "user", "content": f"translate following chinese into english -- {example['translation']['zh']}"},
                {"role": "assistant", "content": f"{example['translation']['en']}"}
            ]
            
            # Add to our dataset
            sloth_data.append({"conversations": conversation})
        
        # Split the data
        train_size = int(0.8 * total_examples)
        val_size = int(0.1 * total_examples)
        
        train_data = sloth_data[:train_size]
        val_data = sloth_data[train_size:train_size + val_size]
        test_data = sloth_data[train_size + val_size:]
    else:
        # Load JSON format
        with open(os.path.join(dataset_dir, "finetune_data.json"), 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Format for Sloth
        sloth_data = []
        for message_pair in all_data:
            # Convert from human/gpt format to user/assistant format
            conversation = [
                {"role": "user", "content": message_pair[0]["value"]},
                {"role": "assistant", "content": message_pair[1]["value"]}
            ]
            
            # Add to our dataset
            sloth_data.append({"conversations": conversation})
        
        total_examples = len(sloth_data)
        train_size = int(0.8 * total_examples)
        val_size = int(0.1 * total_examples)
        
        train_data = sloth_data[:train_size]
        val_data = sloth_data[train_size:train_size + val_size]
        test_data = sloth_data[train_size + val_size:]
    
    # Save and upload splits as Hugging Face datasets
    splits = {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        # Create dataset
        hf_dataset = Dataset.from_list(split_data)
        
        # Save split temporarily
        temp_dir = f"{split_name}_temp"
        os.makedirs(temp_dir, exist_ok=True)
        hf_dataset.save_to_disk(temp_dir)
        
        # Push to hub
        hf_dataset.push_to_hub(full_repo_name, split=split_name)
        print(f"Pushed {split_name} split with {len(split_data)} examples")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
    
    # Create a README file with YAML metadata
    current_date = datetime.now().strftime("%Y-%m-%d")
    readme_content = f"""---
language:
- zh
- en
license: apache-2.0
pretty_name: {repo_name}
size_categories:
- 10K<n<100K
task_categories:
- text2text-generation
task_ids:
- machine-translation
tags:
- translation
- chinese
- english
- fine-tuning
- sloth
---

# {repo_name}

This dataset contains Chinese to English translation pairs formatted for fine-tuning LLMs, compatible with Sloth fine-tuning framework.

## Dataset Description

- **Languages:** Chinese (zh) and English (en)
- **Task:** Machine Translation
- **Size:** {total_examples} examples
- **Format:** Conversation pairs for Sloth fine-tuning
- **License:** Apache 2.0
- **Last Updated:** {current_date}

## Format

Each example is formatted with a "conversations" key containing a list of messages:

```json
{{
  "conversations": [
    {{
      "role": "user",
      "content": "translate following chinese into english -- [Chinese Text]"
    }},
    {{
      "role": "assistant", 
      "content": "[English Translation]"
    }}
  ]
}}
```

## Sloth Fine-tuning Example

When using Sloth for fine-tuning, use code like this:

```python
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {{"text": texts}}
```

## Statistics

The dataset contains the following splits:
- train: {len(train_data)} examples
- validation: {len(val_data)} examples
- test: {len(test_data)} examples

## Usage

This dataset is specifically formatted for fine-tuning language models on Chinese to English translation tasks using the Sloth framework.
"""
    
    # Save and upload README
    readme_path = "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=full_repo_name,
        repo_type="dataset"
    )
    print("Created and pushed README.md with YAML metadata")
    
    # Clean up
    os.remove(readme_path)
    
    print(f"\\nDataset successfully uploaded to: https://huggingface.co/datasets/{full_repo_name}")
    print(f"The dataset is formatted for Sloth fine-tuning with the 'conversations' field.")
    
except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1)
EOL

# Run the Python script
python "$TMP_SCRIPT"
rm "$TMP_SCRIPT" 