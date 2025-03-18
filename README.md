# Chinese to English Translation Dataset Creator

This tool helps you create a high-quality parallel corpus from Chinese and English PDFs for fine-tuning translation models with Unsloth.

## Features

- Extracts text from Chinese and English PDF files
- Aligns corresponding paragraphs using three methods:
  - Length-based matching (for structurally similar documents)
  - **Enhanced TF-IDF similarity matching** (robust for documents with different structures)
  - Sentence Transformer embeddings (best for semantically equivalent content in different languages)
- Processes Chinese text with Jieba for word segmentation
- Language verification to filter out incorrect content
- Two-pass alignment for higher quality matches
- Creates dataset in Hugging Face format ready for fine-tuning
- Automatically splits data into train/validation/test sets
- Supports processing partial PDFs for testing or handling large documents

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install pdfplumber jieba nltk datasets tqdm scikit-learn numpy langdetect sentence-transformers
```

## Usage

Basic usage:

```bash
python create_translation_dataset.py --chinese_pdf path/to/chinese.pdf --english_pdf path/to/english.pdf
```

Process only first 20 pages (useful for testing or large PDFs):

```bash
python create_translation_dataset.py --chinese_pdf path/to/chinese.pdf --english_pdf path/to/english.pdf --max_pages 20
```

Advanced usage with optimal settings for different document structures:

```bash
python create_translation_dataset.py --chinese_pdf path/to/chinese.pdf --english_pdf path/to/english.pdf --alignment_method tfidf_similarity --verify_language --min_score 0.1
```

### Arguments

- `--chinese_pdf`: Path to the Chinese PDF file (required)
- `--english_pdf`: Path to the English PDF file (required)
- `--output_dir`: Directory to save the dataset (default: "translation_dataset")
- `--alignment_method`: Method for paragraph alignment (choices: "length_based", "tfidf_similarity", "sentence_transformer", default: "tfidf_similarity")
- `--verify_language`: If provided, verify language of extracted paragraphs
- `--min_score`: Minimum similarity score for alignment (default: 0.08)
- `--max_pages`: Maximum number of pages to process from each PDF. If not specified, process all pages.

## Alignment Methods

1. **Length-based** (--alignment_method length_based)
   - Best for: Documents with identical structure and page/paragraph flow
   - Fastest method but least accurate for differently structured documents

2. **TF-IDF Similarity** (--alignment_method tfidf_similarity)
   - Best for: Documents with different structures but similar vocabulary
   - Default method with good balance of speed and accuracy
   - Uses a two-pass approach to maximize matching quality

3. **Sentence Transformer** (--alignment_method sentence_transformer)
   - Best for: Documents with completely different structures but same content
   - Most accurate but much slower and requires more memory
   - Uses multilingual embeddings for semantic matching

## Output

The script creates a dataset in Hugging Face format with the following structure:

```
translation_dataset/
├── train/            # Training split (80% by default)
├── validation/       # Validation split (10% by default)
├── test/             # Test split (10% by default)
└── sample.json       # Sample of aligned paragraphs for inspection
```

Each dataset contains translation pairs in the format:

```json
{
  "translation": {
    "zh": "Chinese text",
    "en": "English translation"
  }
}
```

## Using with Unsloth

After creating your dataset, you can fine-tune a translation model with Unsloth:

```python
from datasets import load_from_disk
from unsloth import FastLanguageModel

# Load your dataset
dataset = load_from_disk("translation_dataset")

# Load and prepare the model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    "your/base-model-path",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)

# Format your dataset to the right format for the model
# Prepare your model for training with Unsloth's utilities
# Train the model

# See Unsloth documentation for complete fine-tuning code
```

## Tips for Better Results

1. For best results, use PDFs that are direct translations of each other
2. If your document structure differs significantly between languages, use the "tfidf_similarity" method
3. For documents with vastly different structures, try "sentence_transformer" method
4. Always use the `--verify_language` flag to ensure content is in the correct language
5. Adjust the `--min_score` parameter (between 0.05 and 0.3) to control quality vs. quantity of aligned pairs
6. The sample.json file shows the alignment quality - review it to ensure good matches 