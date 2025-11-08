import json
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  
# ======================
# 提前加载 Sentence-BERT 模型（只加载一次）
# Preload Sentence-BERT model (only once)
# ======================
try:
    print("Loading text encoding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0')  # 可自动下载或从缓存读取   Auto-download or load from cache
except Exception as e:
    print(f"Failed to load the model. Please check your network connection or download the model manually.\nError: {str(e)}")
    model = None

# 配置基础路径和文件映射  Configure paths and dataset mapping
base_path = "./data/"
datasets = {
    "iemocap": "iemocap.train.json",
    "meld": "meld.train.json",
   # "emorynlp": "emorynlp.train.json"
}

label_mapping = {
    "iemocap": ['happy','sad','neutral','angry','excited','frustrated'],
    "emorynlp": ['Joyful', 'Mad', 'Peaceful', 'Neutral', 'Sad', 'Powerful', 'Scared'],
    "meld": ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger'],
    "dailydialog": ['no_emotion', 'happiness', 'sadness', 'surprise', 'anger', 'fear', 'disgust']
}

# 文本向量编码（不再重复加载模型） Text vector encoding (no repeated model loading)
def generate_text_vector(sample):
    if model is None:
        raise RuntimeError("Text encoding model not loaded.")
    return model.encode(sample, convert_to_tensor=False).tolist()

def unify_label(original_label, dataset_name):
    mapping = {
        "no_emotion": "neutral",
        "happiness": "happy",
        "joy": "happy",
        "Joyful": "happy",
        "Mad": "angry",
        "sadness": "sad",
        "Sad": "sad",
        "Scared": "fear"
    }
    return mapping.get(original_label, original_label)

def process_dataset(dataset_name, file_path):
    samples = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset_pbar = tqdm(data.items(), 
                           desc=f"Processing {dataset_name}", 
                           unit="conversation",
                           position=0,
                           leave=True)
        for conv_id, conv_data in dataset_pbar:
            sentences = conv_data.get("sentences", [])
            labels = conv_data.get("labels", [])
            if len(sentences) != len(labels):
                print(f"Warning: {dataset_name} conversation {conv_id} has mismatched sentences and labels.")
                continue

            for i, (text, label_idx) in enumerate(zip(sentences, labels)):
                try:
                    raw_label = label_mapping[dataset_name][label_idx]
                    unified_label = unify_label(raw_label, dataset_name)
                    text_vector = generate_text_vector(text.strip())
                except IndexError:
                    print(f"Error: {dataset_name} conversation {conv_id} index out of range (index={label_idx}).")
                    continue
                except Exception as e:
                    print(f"Encoding error in {dataset_name} conversation {conv_id}, sentence {i}: {str(e)}")
                    continue

                sample = {
                    "text": text.strip(),
                    "label": unified_label,
                    "dataset": dataset_name,
                    "conversation_id": conv_id,
                    "utterance_id": i,
                    "vector": text_vector
                }
                samples.append(sample)

        print(f"Successfully processed {dataset_name}: {len(samples)} samples.")
        return samples

    except Exception as e:
        print(f"Error while processing {dataset_name}: {str(e)}")
        return []

def load_old_data_and_convert(old_json_path, dataset_name="old_data"):
    samples = []
    try:
        with open(old_json_path, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
        for i, entry in enumerate(tqdm(old_data, desc="Processing my data", unit="条")):
            text = entry.get("sentence", "").strip()
            label = entry.get("label", "unknown")
            try:
                vector = generate_text_vector(text)
            except Exception as e:
                print(f"Encoding failed, skipping: {text[:30]}..., reason: {str(e)}")
                continue
            sample = {
                "text": text,
                "label": label,
                "dataset": dataset_name,
                "conversation_id": None,
                "utterance_id": i,
                "vector": vector
            }
            samples.append(sample)
        print(f"Successfully processed my data: {len(samples)} samples")
    except Exception as e:
        print(f"Failed to load my data: {str(e)}")
    return samples


if __name__ == "__main__":
    if model is None:
        print("Model not loaded, exiting program.")
        exit(1)


    my_data_path = "sentence_label_data_with_emotion.json"
    my_samples = load_old_data_and_convert(my_data_path, dataset_name="my_data")
    all_samples = []
    all_samples.extend(my_samples)

    for dataset, filename in datasets.items():
        file_path = os.path.join(base_path, filename)
        if os.path.exists(file_path):
            samples = process_dataset(dataset, file_path)
            all_samples.extend(samples)
        else:
            print(f"文件不存在: {file_path}")

    # Save as Emotion Retrieval Library
    retrieval_library_path = os.path.join(base_path, "Emotion_Retrieval_Library.json")
    with open(retrieval_library_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    stats = {ds: 0 for ds in datasets}
    stats["my_data"] = 0
    for sample in all_samples:
        ds = sample.get("dataset", "unknown")
        if ds not in stats:
            stats[ds] = 0
        stats[ds] += 1

    print("\n===== Demo retrieval library construction completed =====")
    print(f"Total emotion samples: {len(all_samples)}")
    for ds, count in stats.items():
        print(f"- {ds}: {count} samples")
    print(f"Retrieval library path: {retrieval_library_path}")
    print("Tip: This library can be used directly for RAG-based emotion retrieval")
