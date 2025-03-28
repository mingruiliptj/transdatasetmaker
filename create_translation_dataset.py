import os
import argparse
import re
import json
import jieba
import nltk
from datasets import Dataset, load_from_disk
from huggingface_hub import HfApi, create_repo
from datetime import datetime
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer
import opencc
import sys
import random

# Set seed for language detection
DetectorFactory.seed = 0

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

def convert_to_simplified(text):
    """Convert traditional Chinese text to simplified Chinese."""
    converter = opencc.OpenCC('t2s')  # Traditional to Simplified
    return converter.convert(text)

def read_text_file(file_path, max_lines=None):
    """Read text from a plain text file, handling paragraphs separated by double newlines.

    Args:
        file_path: Path to the text file.
        max_lines: Maximum number of lines to process (not paragraphs). If None, process all content.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split content into paragraphs based on double newlines (Windows or Linux)
        paragraphs = re.split(r'(\r?\n){2,}', content)
        text_content = [p.strip() for p in paragraphs if p.strip()]

        if max_lines:
            text_content = text_content[:max_lines] # Note: max_lines is applied to paragraphs now, not lines

        if not text_content:
            raise ValueError(f"No valid text content could be extracted from: {file_path}")

        print(f"Successfully extracted {len(text_content)} paragraphs from {file_path}")
        return text_content

    except UnicodeDecodeError:
        # Try with different encodings if UTF-8 fails
        encodings = ['gb18030', 'big5', 'latin1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                paragraphs = re.split(r'(\r?\n){2,}', content)
                text_content = [p.strip() for p in paragraphs if p.strip()]

                if max_lines:
                    text_content = text_content[:max_lines]

                if not text_content:
                    continue

                print(f"Successfully extracted {len(text_content)} paragraphs from {file_path} using {encoding} encoding")
                return text_content

            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not decode file {file_path} with any supported encoding")

def preprocess_chinese_text(pages, to_simplified=False):
    """Preprocess Chinese text and segment into paragraphs.
    
    Args:
        pages: List of text pages
        to_simplified: Whether to convert traditional Chinese to simplified Chinese
    """
    paragraphs = []
    
    # Initialize converter if needed
    converter = opencc.OpenCC('t2s') if to_simplified else None
    
    for page in pages:
        # Convert to simplified Chinese if requested
        if to_simplified:
            page = converter.convert(page)
            
        # Split by double newlines to get paragraphs
        page_paragraphs = re.split(r'\n\s*\n', page)
        
        for para in page_paragraphs:
            # Clean the paragraph
            para = para.strip()
            if para:
                # Use jieba for word segmentation
                if len(para) > 20:
                    seg_list = jieba.cut(para)
                    processed_para = " ".join(seg_list)
                else:
                    processed_para = para

                paragraphs.append({"text": para, "processed": processed_para.replace("\n", "")})
    
    return paragraphs

def preprocess_english_text(pages):
    """Preprocess English text and segment into paragraphs."""
    paragraphs = []
    
    for page in pages:
        # Split by double newlines to get paragraphs
        page_paragraphs = re.split(r'\n\s*\n', page)
        
        for para in page_paragraphs:
            # Clean the paragraph
            para = para.strip().replace("\n", "")
            if para:  # Filter out very short paragraphs
                # Tokenize into words
                words = re.findall(r'\b\w+\b', para.lower())
                processed_para = " ".join(words).replace("\n", "")
                paragraphs.append({"text": para, "processed": processed_para.replace("\n", "")})
    
    return paragraphs

def verify_language(paragraphs, expected_lang):
    """Verify and filter paragraphs based on language detection."""
    verified_paragraphs = []
    
    for para in tqdm(paragraphs, desc=f"Verifying {expected_lang} text"):
        try:
            # Skip very short texts that might cause detection errors
            if len(para['text']) < 30:
                verified_paragraphs.append(para)
                continue
                
            detected_lang = detect(para['text'])
            
            # For Chinese, 'zh-cn', 'zh-tw', etc. are all acceptable
            if expected_lang == 'zh' and detected_lang.startswith('zh'):
                verified_paragraphs.append(para)
            # For English, only 'en' is acceptable
            elif expected_lang == 'en' and detected_lang == 'en':
                verified_paragraphs.append(para)
        except:
            # If language detection fails, keep the paragraph
            verified_paragraphs.append(para)
    
    return verified_paragraphs

def create_huggingface_dataset(chinese_paragraphs, english_paragraphs, output_dir="translation_dataset", finetune_format=False):
    """Create and save the dataset in Hugging Face format.
    Assumes paragraphs are already aligned in the input files.
    
    Args:
        chinese_paragraphs: List of Chinese paragraph dictionaries
        english_paragraphs: List of English paragraph dictionaries
        output_dir: Directory to save the dataset
        finetune_format: Whether to create the dataset in finetune format
    """
    # Take the minimum length to ensure we have pairs
    min_len = min(len(chinese_paragraphs), len(english_paragraphs))
    
    if finetune_format:
        # Create data in finetune format (for LLM fine-tuning)
        finetune_data = []
        for i in range(min_len):
            chinese_text = chinese_paragraphs[i]["text"]
            english_text = english_paragraphs[i]["text"]
            
            # Create the message pairs
            message_pair = [
                {"from": "human", "value": f"translate following chinese into english -- {chinese_text}"},
                {"from": "gpt", "value": f"{english_text}"}
            ]
            finetune_data.append(message_pair)
        
        # Save in JSON format
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "finetune_data.json"), "w", encoding="utf-8") as f:
            json.dump(finetune_data, f, ensure_ascii=False, indent=2)
        
        # Also save splits
        train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
        train_size = int(len(finetune_data) * train_ratio)
        val_size = int(len(finetune_data) * val_ratio)
        
        # Randomly shuffle the data
        random.seed(42)
        random.shuffle(finetune_data)
        
        train_data = finetune_data[:train_size]
        val_data = finetune_data[train_size:train_size+val_size]
        test_data = finetune_data[train_size+val_size:]
        
        # Save the splits
        os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "validation"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
        
        with open(os.path.join(output_dir, "train", "data.json"), "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, "validation", "data.json"), "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, "test", "data.json"), "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        # Save a sample
        sample_size = min(100, len(finetune_data))
        with open(os.path.join(output_dir, "sample.json"), "w", encoding="utf-8") as f:
            json.dump(finetune_data[:sample_size], f, ensure_ascii=False, indent=2)
        
        print(f"Created finetune dataset with {len(finetune_data)} translation pairs")
        print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
        
        # Return the data as a dictionary
        return {
            "train": train_data,
            "validation": val_data,
            "test": test_data,
            "full": finetune_data
        }
    else:
        # Create pairs directly from the paragraphs in order (original format)
        aligned_pairs = [
            {
                "chinese": chinese_paragraphs[i]["text"],
                "english": english_paragraphs[i]["text"]
            }
            for i in range(min_len)
        ]
        
        # Prepare data in the format expected by the Dataset.from_dict method
        dataset_dict = {
            "translation": [
                {"zh": pair["chinese"], "en": pair["english"]} 
                for pair in aligned_pairs
            ]
        }
        
        # Create the dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Save dataset statistics
        print(f"Created dataset with {len(dataset)} translation pairs")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the dataset in Hugging Face Arrow format
        dataset.save_to_disk(output_dir)
        
        # Also save a sample in JSON format for inspection
        sample_size = min(100, len(aligned_pairs))
        with open(os.path.join(output_dir, "sample.json"), "w", encoding="utf-8") as f:
            json.dump(aligned_pairs[:sample_size], f, ensure_ascii=False, indent=2)
        
        return dataset

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split the dataset into train, validation, and test sets."""
    assert train_ratio + val_ratio + test_ratio == 1.0
    
    # Split the dataset
    splits = dataset.train_test_split(
        test_size=(val_ratio + test_ratio), 
        shuffle=True, 
        seed=seed
    )
    
    train_dataset = splits["train"]
    
    # Further split the test portion into validation and test
    if test_ratio > 0:
        remaining_ratio = test_ratio / (val_ratio + test_ratio)
        test_val_splits = splits["test"].train_test_split(
            test_size=remaining_ratio, 
            shuffle=True, 
            seed=seed
        )
        val_dataset = test_val_splits["train"] 
        test_dataset = test_val_splits["test"]
    else:
        val_dataset = splits["test"]
        test_dataset = Dataset.from_dict({"translation": []})
    
    return {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    }

def generate_paragraph_report(chinese_paragraphs, english_paragraphs, output_dir):
    """Generate an HTML report showing how texts are split into paragraphs.
    
    Args:
        chinese_paragraphs: List of Chinese paragraph dictionaries
        english_paragraphs: List of English paragraph dictionaries
        output_dir: Directory to save the report
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px;
                background-color: #f5f5f5;
            }
            .header {
                background-color: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .grid-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                align-items: start;
            }
            .column-header {
                background-color: white;
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .paragraph-pair {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 10px;
            }
            .paragraph { 
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                height: 100%;
            }
            .paragraph-number {
                font-weight: bold;
                color: #444;
                margin-bottom: 10px;
                padding: 5px 10px;
                background-color: #f8f9fa;
                border-radius: 4px;
                display: inline-block;
            }
            .text {
                white-space: pre-wrap;
                word-wrap: break-word;
                line-height: 1.5;
                margin-bottom: 10px;
            }
            .processed {
                font-size: 0.9em;
                color: #666;
                border-top: 1px dashed #ddd;
                padding-top: 10px;
                margin-top: 10px;
            }
            h1 { 
                color: #333;
                margin: 0;
            }
            h2 { 
                color: #444;
                margin: 0;
            }
            .stats {
                margin-top: 10px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Paragraph Split Report</h1>
            <div class="stats">
                Total paragraphs - Chinese: ${len(chinese_paragraphs)}, English: ${len(english_paragraphs)}
            </div>
        </div>
        
        <div class="grid-container">
            ${paragraph_pairs_html}
        </div>
    </body>
    </html>
    """
    
    # Generate HTML for paragraph pairs
    paragraph_pairs = []
    max_pairs = max(len(chinese_paragraphs), len(english_paragraphs))
    
    for i in range(max_pairs):
        zh_para = chinese_paragraphs[i] if i < len(chinese_paragraphs) else None
        en_para = english_paragraphs[i] if i < len(english_paragraphs) else None
        
        zh_html = f"""
        <div class="paragraph">
            <div class="paragraph-number">Chinese #{i+1}</div>
            <div class="text">{zh_para['text'] if zh_para else '(No paragraph)'}</div>
            <div class="processed">Processed: {zh_para['processed'] if zh_para else ''}</div>
        </div>
        """ if zh_para or i < len(english_paragraphs) else ""
        
        en_html = f"""
        <div class="paragraph">
            <div class="paragraph-number">English #{i+1}</div>
            <div class="text">{en_para['text'] if en_para else '(No paragraph)'}</div>
            <div class="processed">Processed: {en_para['processed'] if en_para else ''}</div>
        </div>
        """ if en_para or i < len(chinese_paragraphs) else ""
        
        if zh_html or en_html:
            paragraph_pairs.append(f"""
            <div class="paragraph-pair">
                {zh_html}
                {en_html}
            </div>
            """)
    
    # Replace placeholder in template
    html_content = html_content.replace("${len(chinese_paragraphs)}", str(len(chinese_paragraphs)))
    html_content = html_content.replace("${len(english_paragraphs)}", str(len(english_paragraphs)))
    html_content = html_content.replace("${paragraph_pairs_html}", "\n".join(paragraph_pairs))
    
    # Save the report
    report_path = os.path.join(output_dir, "paragraph_split_report.html")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\nGenerated paragraph split report at: {report_path}")
    return report_path

def push_to_hub(dataset_dir, repo_name, private=True):
    """Push the dataset to Hugging Face Hub.
    
    Args:
        dataset_dir: Directory containing the dataset
        repo_name: Base name for the repository
        private: Whether to create a private repository
    """
    full_repo_name = f"{repo_name}"
    
    print(f"\nPushing dataset to Hugging Face Hub as '{full_repo_name}'...")
    
    try:
        # Initialize Hugging Face API
        api = HfApi()
        
        # Create the repository
        repo_url = create_repo(
            repo_id=full_repo_name,
            repo_type="dataset",
            private=private,
            exist_ok=False
        )
        print(f"Created repository: {repo_url}")
        
        # Load and push each split
        splits = ['train', 'validation', 'test']
        for split in splits:
            split_path = os.path.join(dataset_dir, split)
            if os.path.exists(split_path):
                # Load the split
                split_dataset = load_from_disk(split_path)
                
                # Push to hub
                split_dataset.push_to_hub(
                    full_repo_name,
                    split=split,
                    private=private
                )
                print(f"Pushed {split} split with {len(split_dataset)} examples")
        
        # Also push the sample.json file
        sample_path = os.path.join(dataset_dir, "sample.json")
        if os.path.exists(sample_path):
            api.upload_file(
                path_or_fileobj=sample_path,
                path_in_repo="sample.json",
                repo_id=full_repo_name,
                repo_type="dataset"
            )
            print("Pushed sample.json file")
            
        print(f"\nDataset successfully pushed to: https://huggingface.co/datasets/{full_repo_name}")
        return full_repo_name
        
    except Exception as e:
        print(f"Error pushing to Hugging Face Hub: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Create translation dataset from Chinese and English text files")
    parser.add_argument("--chinese_text", required=True, help="Path to Chinese text file")
    parser.add_argument("--english_text", required=True, help="Path to English text file")
    parser.add_argument("--output_dir", default="translation_dataset", help="Directory to save the dataset")
    parser.add_argument("--verify_language", action="store_true", help="Verify language of paragraphs")
    parser.add_argument("--max_lines", type=int, default=None, 
                        help="Maximum number of lines to process from each text file. If not specified, process all lines.")
    parser.add_argument("--to_simplified", action="store_true",
                        help="Convert traditional Chinese to simplified Chinese")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the dataset to Hugging Face Hub")
    parser.add_argument("--repo_name", default="zh_en_translation",
                        help="Base name for the Hugging Face dataset repository")
    parser.add_argument("--private", action="store_true", default=False,
                        help="Whether to create a private repository")
    parser.add_argument("--finetune_format", action="store_true",
                        help="Create dataset in fine-tuning format with human/gpt message pairs")
    args = parser.parse_args()
    
    try:
        # Extract text from text files
        print("Reading text files...")
        chinese_text = read_text_file(args.chinese_text, args.max_lines)
        english_text = read_text_file(args.english_text, args.max_lines)
        
        if not chinese_text or not english_text:
            raise ValueError("No text could be extracted from one or both text files")
        
        # Preprocess text and segment into paragraphs
        print("Preprocessing Chinese text...")
        if args.to_simplified:
            print("Converting traditional Chinese to simplified Chinese...")
        chinese_paragraphs = preprocess_chinese_text(chinese_text, to_simplified=args.to_simplified)
        print(f"Extracted {len(chinese_paragraphs)} Chinese paragraphs")
        
        print("Preprocessing English text...")
        english_paragraphs = preprocess_english_text(english_text)
        print(f"Extracted {len(english_paragraphs)} English paragraphs")
        
        # Generate paragraph split report before language verification
        print("\nGenerating paragraph split report...")
        report_path = generate_paragraph_report(chinese_paragraphs, english_paragraphs, args.output_dir)
        print(f"Please check the report at {report_path} to verify paragraph splits")
        
        if not chinese_paragraphs or not english_paragraphs:
            raise ValueError("No valid paragraphs could be extracted from one or both text files")
        
        # Print some statistics about the paragraphs
        print("\nParagraph Statistics:")
        print(f"Chinese paragraphs: {len(chinese_paragraphs)}")
        print(f"English paragraphs: {len(english_paragraphs)}")
        print(f"Ratio (Chinese/English): {len(chinese_paragraphs)/len(english_paragraphs):.2f}")
        
        # Print sample of first few paragraphs
        print("\nFirst Chinese paragraph:")
        print(chinese_paragraphs[0]['text'][:200] + "..." if len(chinese_paragraphs) > 0 else "None")
        print("\nFirst English paragraph:")
        print(english_paragraphs[0]['text'][:200] + "..." if len(english_paragraphs) > 0 else "None")
        
        # Optionally verify language
        if args.verify_language:
            print("\nVerifying language of paragraphs...")
            chinese_paragraphs = verify_language(chinese_paragraphs, 'zh')
            print(f"{len(chinese_paragraphs)} Chinese paragraphs after language verification")
            
            english_paragraphs = verify_language(english_paragraphs, 'en')
            print(f"{len(english_paragraphs)} English paragraphs after language verification")
            
            if not chinese_paragraphs or not english_paragraphs:
                raise ValueError("No valid paragraphs remained after language verification")
        
        # Create dataset directly from paragraphs
        print("\nCreating dataset...")
        if args.finetune_format:
            print("Using fine-tuning format with human/gpt message pairs")
            result = create_huggingface_dataset(chinese_paragraphs, english_paragraphs, 
                                                output_dir=args.output_dir,
                                                finetune_format=True)
            # No need to split the dataset as it's already split in the function
        else:
            dataset = create_huggingface_dataset(chinese_paragraphs, english_paragraphs, 
                                              output_dir=args.output_dir)
            
            # Split dataset
            print("Splitting dataset into train/validation/test sets...")
            dataset_splits = split_dataset(dataset)
            
            # Save split datasets
            for split_name, split_data in dataset_splits.items():
                split_dir = os.path.join(args.output_dir, split_name)
                split_data.save_to_disk(split_dir)
                print(f"Saved {split_name} split with {len(split_data)} examples to {split_dir}")
            
        # Push to Hugging Face Hub if requested
        if args.push_to_hub:
            try:
                if args.finetune_format:
                    print("\nFor fine-tuning format, please use the upload_dataset.sh script to push to Hub")
                else:
                    repo_name = push_to_hub(args.output_dir, args.repo_name, args.private)
                    print(f"\nDataset is now available at: https://huggingface.co/datasets/{repo_name}")
            except Exception as e:
                print(f"\nFailed to push to Hugging Face Hub: {str(e)}")
                print("The dataset was created locally but could not be pushed to the Hub")
                print("Please check your Hugging Face credentials and internet connection")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check that:")
        print("1. Both text files exist and are readable")
        print("2. Text files contain extractable text")
        print("3. Text files contain sufficient non-empty lines")
        print("4. Text is properly encoded and contains actual content")
        print("5. Review the paragraph split report to see how the text was divided")
        sys.exit(1)

if __name__ == "__main__":
    main()
