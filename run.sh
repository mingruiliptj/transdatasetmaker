#/bin/bash

huggingface-cli login

pip install -r requirements

python create_translation_dataset.py --chinese_text towest_cn.txt --english_text towest_en.txt  --to_simplified --push_to_hub --repo_name xyshyniaphy/cn2en_s