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

# Check if required splits exist
for split in train validation test; do
    if [ ! -d "$DATASET_DIR/$split" ]; then
        echo "Warning: '$split' split not found in $DATASET_DIR"
    fi
done

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
from datasets import load_from_disk
from huggingface_hub import HfApi, create_repo, whoami
import os
import sys
import time
import json

dataset_dir = "$DATASET_DIR"
repo_name = "$DATASET_NAME"

def convert_to_finetune_format(dataset):
    """Convert translation dataset to the finetune format with human/gpt message pairs."""
    finetune_data = []
    
    for example in dataset:
        # Get the Chinese and English text
        chinese_text = example['translation']['zh']
        english_text = example['translation']['en']
        
        # Create the message pairs
        message_pair = [
            {"from": "human", "value": f"translate following chinese into english -- {chinese_text}"},
            {"from": "gpt", "value": f"{english_text}"}
        ]
        
        finetune_data.append(message_pair)
    
    return finetune_data

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
    
    # Convert and upload each split
    splits = ['train', 'validation', 'test']
    for split in splits:
        split_path = os.path.join(dataset_dir, split)
        if os.path.exists(split_path):
            # Load the dataset
            split_dataset = load_from_disk(split_path)
            print(f"Loaded {split} split with {len(split_dataset)} examples")
            
            # Convert to finetune format
            finetune_data = convert_to_finetune_format(split_dataset)
            print(f"Converted {len(finetune_data)} examples to finetune format")
            
            # Save to a temporary JSON file
            temp_json_path = f"{split}_finetune.json"
            with open(temp_json_path, 'w', encoding='utf-8') as f:
                json.dump(finetune_data, f, ensure_ascii=False, indent=2)
            
            # Upload the JSON file
            api.upload_file(
                path_or_fileobj=temp_json_path,
                path_in_repo=f"{split}.json",
                repo_id=full_repo_name,
                repo_type="dataset"
            )
            print(f"Pushed {split}.json with {len(finetune_data)} examples")
            
            # Clean up the temporary file
            os.remove(temp_json_path)
    
    # Upload sample.json if it exists
    sample_path = os.path.join(dataset_dir, "sample.json")
    if os.path.exists(sample_path):
        # Read the original sample file
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        # Convert to finetune format
        finetune_sample = []
        for item in sample_data:
            message_pair = [
                {"from": "human", "value": f"translate following chinese into english -- {item['chinese']}"},
                {"from": "gpt", "value": f"{item['english']}"}
            ]
            finetune_sample.append(message_pair)
        
        # Save to temporary file
        temp_sample_path = "finetune_sample.json"
        with open(temp_sample_path, 'w', encoding='utf-8') as f:
            json.dump(finetune_sample, f, ensure_ascii=False, indent=2)
        
        # Upload the converted sample
        api.upload_file(
            path_or_fileobj=temp_sample_path,
            path_in_repo="sample.json",
            repo_id=full_repo_name,
            repo_type="dataset"
        )
        print("Pushed sample.json file in finetune format")
        
        # Clean up
        os.remove(temp_sample_path)
    
    # Create a README file explaining the dataset format
    readme_content = f"""# {repo_name}

This dataset contains Chinese to English translation pairs formatted for fine-tuning LLMs.

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
"""
    
    # Add statistics for each split
    for split in splits:
        split_path = os.path.join(dataset_dir, split)
        if os.path.exists(split_path):
            split_dataset = load_from_disk(split_path)
            readme_content += f"- {split}: {len(split_dataset)} examples\\n"
    
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
    print("Created and pushed README.md")
    
    # Clean up
    os.remove(readme_path)
    
    print(f"\nDataset successfully uploaded to: https://huggingface.co/datasets/{full_repo_name}")
    print("The dataset is formatted for fine-tuning with human/gpt message pairs")
    
except Exception as e:
    print(f"Error: {str(e)}", file=sys.stderr)
    sys.exit(1)
EOL

# Execute the upload script
python "$TMP_SCRIPT"
RESULT=$?

# Clean up temporary script
rm "$TMP_SCRIPT"

# Check if upload was successful
if [ $RESULT -eq 0 ]; then
    echo "Upload completed successfully!"
    # Get username for the URL
    USERNAME=$(huggingface-cli whoami)
    echo "Dataset URL: https://huggingface.co/datasets/${USERNAME}/${DATASET_NAME}"
    echo "The dataset is now formatted for fine-tuning with human/gpt message pairs"
else
    echo "Upload failed. Please check the error messages above."
    exit 1
fi 