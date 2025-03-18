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

# Check if required files exist
if [ ! -f "$DATASET_DIR/finetune_data.json" ]; then
    echo "Error: 'finetune_data.json' not found in $DATASET_DIR"
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
    
    # Load and split the dataset
    with open(os.path.join(dataset_dir, "finetune_data.json"), 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    total_examples = len(all_data)
    train_size = int(0.8 * total_examples)
    val_size = int(0.1 * total_examples)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    # Save and upload splits
    splits = {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        # Save split temporarily
        temp_path = f"{split_name}_temp.json"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        # Upload the split
        api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo=f"{split_name}.json",
            repo_id=full_repo_name,
            repo_type="dataset"
        )
        print(f"Pushed {split_name}.json with {len(split_data)} examples")
        
        # Clean up
        os.remove(temp_path)
    
    # Upload sample.json if it exists
    sample_path = os.path.join(dataset_dir, "sample.json")
    if os.path.exists(sample_path):
        api.upload_file(
            path_or_fileobj=sample_path,
            path_in_repo="sample.json",
            repo_id=full_repo_name,
            repo_type="dataset"
        )
        print("Pushed sample.json file")
    
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
---

# {repo_name}

This dataset contains Chinese to English translation pairs formatted for fine-tuning LLMs.

## Dataset Description

- **Languages:** Chinese (zh) and English (en)
- **Task:** Machine Translation
- **Size:** {total_examples} examples
- **Format:** Fine-tuning conversation pairs
- **License:** Apache 2.0
- **Last Updated:** {current_date}

## Format

Each example is formatted as a conversation pair:

```json
[
  {{
    "from": "human",
    "value": "translate following chinese into english -- [Chinese Text]"
  }},
  {{
    "from": "gpt",
    "value": "[English Translation]"
  }}
]
```

## Statistics

The dataset contains the following splits:
- train: {len(train_data)} examples
- validation: {len(val_data)} examples
- test: {len(test_data)} examples

## Usage

This dataset is specifically formatted for fine-tuning language models on Chinese to English translation tasks. Each example is structured as a conversation pair with a human prompt and the expected GPT response.

## Data Processing

The dataset has been processed to:
1. Ensure proper formatting for fine-tuning
2. Split into train/validation/test sets (80/10/10 split)
3. Verify content quality and format consistency
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
    
except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1)
EOL

# Run the Python script
python "$TMP_SCRIPT"
rm "$TMP_SCRIPT" 