import os
import argparse
import re
import json
import jieba
import nltk
from datasets import Dataset
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer
import opencc
import sys

# Set seed for language detection
DetectorFactory.seed = 0

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

def convert_to_simplified(text):
    """Convert traditional Chinese text to simplified Chinese."""
    converter = opencc.OpenCC('t2s')  # Traditional to Simplified
    return converter.convert(text)

def read_text_file(file_path, max_lines=None):
    """Read text from a plain text file.
    
    Args:
        file_path: Path to the text file
        max_lines: Maximum number of lines to process. If None, process all lines.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")
        
    text_content = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if max_lines:
            lines = lines[:max_lines]
            
        # Group lines into paragraphs (separated by empty lines)
        current_paragraph = []
        for line in lines:
            line = line.strip()
            if line:
                current_paragraph.append(line)
            elif current_paragraph:  # Empty line and we have content
                text_content.append(' '.join(current_paragraph))
                current_paragraph = []
                
        # Add the last paragraph if it exists
        if current_paragraph:
            text_content.append(' '.join(current_paragraph))
            
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
                    lines = f.readlines()
                if max_lines:
                    lines = lines[:max_lines]
                    
                # Group lines into paragraphs
                current_paragraph = []
                for line in lines:
                    line = line.strip()
                    if line:
                        current_paragraph.append(line)
                    elif current_paragraph:
                        text_content.append(' '.join(current_paragraph))
                        current_paragraph = []
                        
                if current_paragraph:
                    text_content.append(' '.join(current_paragraph))
                    
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
            if para and len(para) > 10:  # Filter out very short paragraphs
                # Use jieba for word segmentation
                seg_list = jieba.cut(para)
                processed_para = " ".join(seg_list)
                paragraphs.append({"text": para, "processed": processed_para})
    
    return paragraphs

def preprocess_english_text(pages):
    """Preprocess English text and segment into paragraphs."""
    paragraphs = []
    
    for page in pages:
        # Split by double newlines to get paragraphs
        page_paragraphs = re.split(r'\n\s*\n', page)
        
        for para in page_paragraphs:
            # Clean the paragraph
            para = para.strip()
            if para and len(para) > 10:  # Filter out very short paragraphs
                # Tokenize into words
                words = re.findall(r'\b\w+\b', para.lower())
                processed_para = " ".join(words)
                paragraphs.append({"text": para, "processed": processed_para})
    
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

def align_paragraphs(chinese_paragraphs, english_paragraphs, method="tfidf_similarity"):
    """
    Align Chinese and English paragraphs.
    
    Methods:
    - length_based: Simple alignment based on relative paragraph lengths
    - tfidf_similarity: Alignment based on TF-IDF vector similarity
    - sentence_transformer: Alignment based on multilingual sentence embeddings
    """
    # Validate input
    if not chinese_paragraphs or not english_paragraphs:
        raise ValueError("Both Chinese and English paragraphs must not be empty")
    
    aligned_pairs = []
    
    if method == "length_based":
        # Simple length-based alignment assuming paragraphs appear in same order
        # This is a naive approach and works only if both PDFs have the same structure
        total_zh = len(chinese_paragraphs)
        total_en = len(english_paragraphs)
        
        # Use the shorter document as reference
        shorter_len = min(total_zh, total_en)
        
        # Calculate ratio for alignment
        ratio = total_zh / total_en if total_en < total_zh else total_en / total_zh
        
        for i in range(shorter_len):
            if total_zh <= total_en:
                zh_idx = i
                en_idx = int(i * ratio)
            else:
                zh_idx = int(i * ratio)
                en_idx = i
                
            if zh_idx < total_zh and en_idx < total_en:
                aligned_pairs.append({
                    "chinese": chinese_paragraphs[zh_idx]["text"],
                    "english": english_paragraphs[en_idx]["text"]
                })
    
    elif method == "tfidf_similarity":
        # More advanced method using TF-IDF and cosine similarity
        print("Using TF-IDF similarity for paragraph alignment")
        
        # Create TF-IDF vectors for processed paragraphs
        all_processed_texts = [p["processed"] for p in chinese_paragraphs + english_paragraphs]
        
        try:
            # Increase min_df to filter out rare terms that might just be noise
            # and max_df to filter out terms that appear in too many documents
            vectorizer = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(all_processed_texts)
            
            # Extract Chinese and English parts of the matrix
            zh_vectors = tfidf_matrix[:len(chinese_paragraphs)]
            en_vectors = tfidf_matrix[len(chinese_paragraphs):]
            
            # Compute similarity between all Chinese and English paragraphs
            print("Computing similarity matrix...")
            similarity_matrix = cosine_similarity(zh_vectors, en_vectors)
            
            # For each Chinese paragraph, find the most similar English paragraph
            used_en_indices = set()
            
            # First pass: High confidence matches
            for zh_idx, similarities in enumerate(similarity_matrix):
                en_idx = np.argmax(similarities)
                score = similarities[en_idx]
                
                # Only keep alignments with high similarity
                if score > 0.15 and en_idx not in used_en_indices:  # Higher threshold for first pass
                    aligned_pairs.append({
                        "chinese": chinese_paragraphs[zh_idx]["text"],
                        "english": english_paragraphs[en_idx]["text"],
                        "score": float(score)
                    })
                    used_en_indices.add(en_idx)
            
            # Second pass: For Chinese paragraphs without a match, use a lower threshold
            for zh_idx, similarities in enumerate(similarity_matrix):
                # Skip if this Chinese paragraph already has a match
                if any(pair["chinese"] == chinese_paragraphs[zh_idx]["text"] for pair in aligned_pairs):
                    continue
                    
                # Find best unused English paragraph
                sorted_indices = np.argsort(-similarities)  # Sort in descending order
                
                for en_idx in sorted_indices:
                    score = similarities[en_idx]
                    
                    # Use a lower threshold but still ensure some similarity
                    if score > 0.08 and en_idx not in used_en_indices:
                        aligned_pairs.append({
                            "chinese": chinese_paragraphs[zh_idx]["text"],
                            "english": english_paragraphs[en_idx]["text"],
                            "score": float(score)
                        })
                        used_en_indices.add(en_idx)
                        break
                        
        except ValueError as e:
            print(f"Warning: TF-IDF alignment failed ({str(e)}), falling back to length-based alignment")
            return align_paragraphs(chinese_paragraphs, english_paragraphs, method="length_based")
    
    elif method == "sentence_transformer":
        if not chinese_paragraphs or not english_paragraphs:
            print("Warning: Empty paragraphs detected, falling back to length-based alignment")
            return align_paragraphs(chinese_paragraphs, english_paragraphs, method="length_based")
            
        print("Using SentenceTransformer for paragraph alignment")
        
        # Load a multilingual sentence transformer model
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        
        # Extract text for embedding
        chinese_texts = [p["text"] for p in chinese_paragraphs]
        english_texts = [p["text"] for p in english_paragraphs]
        
        # Create embeddings in batches
        print("Generating Chinese embeddings...")
        chinese_embeddings = model.encode(chinese_texts, show_progress_bar=True, batch_size=16)
        
        print("Generating English embeddings...")
        english_embeddings = model.encode(english_texts, show_progress_bar=True, batch_size=16)
        
        # Compute cosine similarity
        print("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(chinese_embeddings, english_embeddings)
        
        # For each Chinese paragraph, find the most similar English paragraph
        used_en_indices = set()
        
        # First pass with higher threshold
        for zh_idx, similarities in enumerate(similarity_matrix):
            en_idx = np.argmax(similarities)
            score = similarities[en_idx]
            
            if score > 0.6 and en_idx not in used_en_indices:  # Higher threshold for first pass
                aligned_pairs.append({
                    "chinese": chinese_paragraphs[zh_idx]["text"],
                    "english": english_paragraphs[en_idx]["text"],
                    "score": float(score)
                })
                used_en_indices.add(en_idx)
        
        # Second pass with lower threshold
        for zh_idx, similarities in enumerate(similarity_matrix):
            if any(pair["chinese"] == chinese_paragraphs[zh_idx]["text"] for pair in aligned_pairs):
                continue
                
            sorted_indices = np.argsort(-similarities)
            
            for en_idx in sorted_indices:
                score = similarities[en_idx]
                
                if score > 0.4 and en_idx not in used_en_indices:
                    aligned_pairs.append({
                        "chinese": chinese_paragraphs[zh_idx]["text"],
                        "english": english_paragraphs[en_idx]["text"],
                        "score": float(score)
                    })
                    used_en_indices.add(en_idx)
                    break
    
    # Sort the aligned pairs by score if available
    if aligned_pairs and "score" in aligned_pairs[0]:
        aligned_pairs.sort(key=lambda x: x["score"], reverse=True)
    
    if not aligned_pairs:
        raise ValueError("No paragraph pairs could be aligned. Please check your input PDFs.")
    
    return aligned_pairs

def create_huggingface_dataset(aligned_pairs, output_dir="translation_dataset"):
    """Create and save the dataset in Hugging Face format."""
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

def main():
    parser = argparse.ArgumentParser(description="Create translation dataset from Chinese and English text files")
    parser.add_argument("--chinese_text", required=True, help="Path to Chinese text file")
    parser.add_argument("--english_text", required=True, help="Path to English text file")
    parser.add_argument("--output_dir", default="translation_dataset", help="Directory to save the dataset")
    parser.add_argument("--alignment_method", choices=["length_based", "tfidf_similarity", "sentence_transformer"], 
                        default="tfidf_similarity", help="Method for paragraph alignment")
    parser.add_argument("--verify_language", action="store_true", help="Verify language of paragraphs")
    parser.add_argument("--min_score", type=float, default=0.08, help="Minimum similarity score for alignment")
    parser.add_argument("--max_lines", type=int, default=None, 
                        help="Maximum number of lines to process from each text file. If not specified, process all lines.")
    parser.add_argument("--to_simplified", action="store_true",
                        help="Convert traditional Chinese to simplified Chinese")
    args = parser.parse_args()
    
    try:
        # Extract text from text files
        print("Reading text files...")
        chinese_paragraphs = read_text_file(args.chinese_text, args.max_lines)
        english_paragraphs = read_text_file(args.english_text, args.max_lines)
        
        if not chinese_paragraphs or not english_paragraphs:
            raise ValueError("No text could be extracted from one or both text files")
        
        # Preprocess text and segment into paragraphs
        print("Preprocessing Chinese text...")
        if args.to_simplified:
            print("Converting traditional Chinese to simplified Chinese...")
        chinese_paragraphs = preprocess_chinese_text(chinese_paragraphs, to_simplified=args.to_simplified)
        print(f"Extracted {len(chinese_paragraphs)} Chinese paragraphs")
        
        print("Preprocessing English text...")
        english_paragraphs = preprocess_english_text(english_paragraphs)
        print(f"Extracted {len(english_paragraphs)} English paragraphs")
        
        if not chinese_paragraphs or not english_paragraphs:
            raise ValueError("No valid paragraphs could be extracted from one or both text files")
        
        # Optionally verify language
        if args.verify_language:
            print("Verifying language of paragraphs...")
            chinese_paragraphs = verify_language(chinese_paragraphs, 'zh')
            print(f"{len(chinese_paragraphs)} Chinese paragraphs after language verification")
            
            english_paragraphs = verify_language(english_paragraphs, 'en')
            print(f"{len(english_paragraphs)} English paragraphs after language verification")
            
            if not chinese_paragraphs or not english_paragraphs:
                raise ValueError("No valid paragraphs remained after language verification")
        
        # Align paragraphs
        print(f"Aligning paragraphs using {args.alignment_method} method...")
        aligned_pairs = align_paragraphs(chinese_paragraphs, english_paragraphs, method=args.alignment_method)
        print(f"Created {len(aligned_pairs)} aligned paragraph pairs")
        
        # Filter by score if applicable
        if args.alignment_method in ["tfidf_similarity", "sentence_transformer"] and args.min_score > 0:
            original_count = len(aligned_pairs)
            aligned_pairs = [pair for pair in aligned_pairs if pair.get("score", 1.0) >= args.min_score]
            print(f"Filtered down to {len(aligned_pairs)} pairs with score >= {args.min_score} (removed {original_count - len(aligned_pairs)} pairs)")
            
            if not aligned_pairs:
                raise ValueError(f"No pairs remained after filtering with min_score={args.min_score}")
        
        # Create dataset
        print("Creating Hugging Face dataset...")
        dataset = create_huggingface_dataset(aligned_pairs, output_dir=args.output_dir)
        
        # Split dataset
        print("Splitting dataset into train/validation/test sets...")
        splits = split_dataset(dataset)
        
        # Save split datasets
        for split_name, split_dataset in splits.items():
            split_dir = os.path.join(args.output_dir, split_name)
            split_dataset.save_to_disk(split_dir)
            print(f"Saved {split_name} split with {len(split_dataset)} examples to {split_dir}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check that:")
        print("1. Both text files exist and are readable")
        print("2. Text files contain extractable text (not scanned images)")
        print("3. Text files contain sufficient non-empty lines")
        print("4. Text is properly encoded and contains actual content")
        sys.exit(1)

if __name__ == "__main__":
    main() 