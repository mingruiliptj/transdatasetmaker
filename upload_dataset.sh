#!/bin/bash

# Help function
show_help() {
    echo "Usage: $0 [options] <dataset_directory>"
    echo
    echo "Upload a dataset to Hugging Face Hub with automatic versioning"
    echo
    echo "Options:"
    echo "  -n, --name NAME       Base name for the dataset (default: zh_en_translation)"
    echo "  -p, --public          Make the dataset public (default: private)"
    echo "  -h, --help           Show this help message"
    echo
    echo "Example:"
    echo "  $0 -n my_translation_dataset ./translation_dataset"
    echo "  $0 --name custom_dataset --public ./output_dir"
}

# Default values
DATASET_NAME="zh_en_translation"
PRIVATE="--private"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            DATASET_NAME="$2"
            shift 2
            ;;
        -p|--public)
            PRIVATE=""
            shift
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

# Generate timestamp for unique dataset name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_DATASET_NAME="${DATASET_NAME}_${TIMESTAMP}"

echo "Uploading dataset to Hugging Face Hub..."
echo "Dataset name: $FULL_DATASET_NAME"
echo "Source directory: $DATASET_DIR"
echo "Privacy setting: ${PRIVATE:+private}"

# Create Python script for upload
TMP_SCRIPT=$(mktemp)
cat > "$TMP_SCRIPT" << EOL
from datasets import load_from_disk
from huggingface_hub import HfApi, create_repo
import os
import sys

dataset_dir = "$DATASET_DIR"
repo_name = "$FULL_DATASET_NAME"
private = ${PRIVATE:+True}

try:
    # Create the repository
    repo_url = create_repo(
        repo_id=repo_name,
        repo_type="dataset",
        private=${PRIVATE:+True},
        exist_ok=False
    )
    print(f"Created repository: {repo_url}")
    
    # Upload each split
    splits = ['train', 'validation', 'test']
    for split in splits:
        split_path = os.path.join(dataset_dir, split)
        if os.path.exists(split_path):
            split_dataset = load_from_disk(split_path)
            split_dataset.push_to_hub(
                repo_name,
                split=split,
                private=${PRIVATE:+True}
            )
            print(f"Pushed {split} split with {len(split_dataset)} examples")
    
    # Upload sample.json if it exists
    sample_path = os.path.join(dataset_dir, "sample.json")
    if os.path.exists(sample_path):
        api = HfApi()
        api.upload_file(
            path_or_fileobj=sample_path,
            path_in_repo="sample.json",
            repo_id=repo_name,
            repo_type="dataset"
        )
        print("Pushed sample.json file")
    
    print(f"\nDataset successfully uploaded to: https://huggingface.co/datasets/{repo_name}")
    
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
    echo "Dataset URL: https://huggingface.co/datasets/${FULL_DATASET_NAME}"
else
    echo "Upload failed. Please check the error messages above."
    exit 1
fi 