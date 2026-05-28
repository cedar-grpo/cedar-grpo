import os
import json
import glob

base_dir = "./dataset_general_reasoning"

train_data = []
val_data =[]

dataset_dirs =["Big-Bench", "CommonsenseQA", "FOLIO", "GSM8K", "MMLU", "VitaminC"]

for d in dataset_dirs:
    dir_path = os.path.join(base_dir, d)
    if not os.path.isdir(dir_path):
        continue
    
    for f in glob.glob(os.path.join(dir_path, "*_train.json")):
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                item['datasetName'] = d

            train_data.extend(data)

            
    for f in glob.glob(os.path.join(dir_path, "*_val.json")):
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                item['datasetName'] = d
            val_data.extend(data)

with open(os.path.join(base_dir, "train_split.json"), "w", encoding='utf-8') as f:
    json.dump(train_data, f, indent=2)

with open(os.path.join(base_dir, "val_split.json"), "w", encoding='utf-8') as f:
    json.dump(val_data, f, indent=2)

print(f"✅ Created train_split.json with {len(train_data)} records.")
print(f"✅ Created val_split.json with {len(val_data)} records.")
