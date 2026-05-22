import pickle
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import torch
import pandas as pd
from torchvision import datasets, models, transforms
import os
import numpy as np
from PIL import Image


def read_image():
    image_list = {}
    file_list = ['/home/fxy/project/OVLAFND/src/weibo21/nonrumor_images/',
                 '/home/fxy/project/OVLAFND/src/weibo21/rumor_images/']
    for path in file_list:
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist")
            continue
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if not os.listdir(path):
            print(f"Warning: Directory {path} is empty")
            continue

        for filename in os.listdir(path):
            try:
                im = Image.open(os.path.join(path, filename)).convert('RGB')
                im = data_transforms(im)
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    print("image length " + str(len(image_list)))
    return image_list


def _init_fn(worker_id):
    np.random.seed(2024)


def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t


def df_filter(df_data):
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data


def word2input(texts, vocab_file, max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids = []
    masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            max_length=max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        token_ids.append(encoded['input_ids'])
        masks.append(encoded['attention_mask'])

    if token_ids:  # 确保列表不为空
        token_ids = torch.cat(token_ids, dim=0)
        masks = torch.cat(masks, dim=0)
    else:
        token_ids = torch.empty(0, max_len, dtype=torch.long)
        masks = torch.empty(0, max_len, dtype=torch.long)

    return token_ids, masks


class bert_data():
    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data(self, path, imagepath, clipimagepath, shuffle, text_only=False):
        if not os.path.exists(path):
            print(f"Error: Data file {path} does not exist")
            return None
        self.data = pd.read_excel(path)
        print(f"Loaded data with {len(self.data)} rows")
        required_columns = ['content', 'label', 'category']
        for col in required_columns:
            if col not in self.data.columns:
                print(f"Error: Required column '{col}' not found in data")
                return None
        if 'content_neg' not in self.data.columns:
            print("Warning: 'content_neg' column not found. Using original content as fallback.")
            self.data['content_neg'] = self.data['content']
        self.data['content_neg'] = self.data['content_neg'].apply(
            lambda x: '' if pd.isna(x) else str(x)
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clipmodel, _ = load_from_name("ViT-B-16", device=device, download_root='./')

        self.data['content'] = self.data['content'].apply(
            lambda x: '' if pd.isna(x) else str(x)
        )

        content = self.data['content'].astype('object').to_numpy()
        label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
        category = torch.tensor(
            self.data['category'].astype('object').apply(lambda c: self.category_dict[c]).to_numpy())

        categories = []
        for cat1, cat2 in zip(self.data['category'], self.data['领域']):
            if cat2 == "无领域":
                categories.append([self.category_dict[cat1]])
            else:
                categories.append([self.category_dict[cat1], self.category_dict[cat2]])

        num_domains = 9
        labels_multi_domain = torch.zeros(len(categories), num_domains)

        for i, cat_list in enumerate(categories):
            labels_multi_domain[i, cat_list] = 1

        mul_category = labels_multi_domain

        token_ids, masks = word2input(content, self.vocab_file, self.max_len)
        # print(f"Text tokens: {token_ids.shape}")

        ordered_image = None
        if os.path.exists(imagepath):
            try:
                ordered_image = pickle.load(open(imagepath, 'rb'))
                print(f"Loaded ordered_image: {ordered_image.shape}")
            except Exception as e:
                print(f"Error loading {imagepath}: {e}")
                return None
        else:
            print(f"Error: Image file {imagepath} does not exist")
            return None
        clip_image = None
        if os.path.exists(clipimagepath):
            try:
                clip_image = pickle.load(open(clipimagepath, 'rb'))
                print(f"Loaded clip_image: {clip_image.shape}")
            except Exception as e:
                print(f"Error loading {clipimagepath}: {e}")
                return None
        else:
            print(f"Error: CLIP image file {clipimagepath} does not exist")
            return None

        clip_text = clip.tokenize(content)
        clip_content_neg = clip.tokenize(self.data['content_neg'].astype('object').to_numpy())  # 新增：处理否定文本
        # print(f"CLIP text: {clip_text.shape}")

        if 'enhanced' in self.data.columns:
            # 将 True/1 转为 1.0, 其他转为 0.0
            enhanced_val = self.data['enhanced'].apply(lambda x: 1.0 if str(x).lower() in ['true', '1', '1.0'] else 0.0)
            enhanced = torch.tensor(enhanced_val.values, dtype=torch.float)
        else:
            enhanced = torch.zeros(len(self.data), dtype=torch.float)

        print(f"正在加载 weibo21 的 LLM 语义特征锚点...")
        pt_dir = "/home/fxy/project/OVLAFND/llm_topic_features/weibo21"
        llm_topic_embs = []
        for i in range(len(self.data)):
            pt_path = os.path.join(pt_dir, f"row_{i}_topic.pt")
            if os.path.exists(pt_path):
                emb = torch.load(pt_path, map_location='cpu').squeeze(0)
            else:
                emb = torch.zeros(512)
            llm_topic_embs.append(emb)

        llm_topic_embs = torch.stack(llm_topic_embs)
        print(f"Loaded LLM topic embeddings: {llm_topic_embs.shape}")
        tensor_shapes = {
            "token_ids": token_ids.shape[0],
            "masks": masks.shape[0],
            "label": label.shape[0],
            "category": category.shape[0],
            "ordered_image": ordered_image.shape[0],
            "clip_image": clip_image.shape[0],
            "clip_text": clip_text.shape[0],
            "mul_category": mul_category.shape[0],
            "clip_content_neg": clip_content_neg.shape[0],
            "enhanced": enhanced.shape[0],  
            "llm_topic_embs": llm_topic_embs.shape[0]  
        }

        print("Tensor shapes:", tensor_shapes)

        min_len = min(tensor_shapes.values())

        if len(set(tensor_shapes.values())) > 1:
            print(f"Warning: Tensor size mismatch. Truncating to {min_len} samples")
            token_ids = token_ids[:min_len]
            masks = masks[:min_len]
            label = label[:min_len]
            category = category[:min_len]
            ordered_image = ordered_image[:min_len]
            clip_image = clip_image[:min_len]
            clip_text = clip_text[:min_len]
            mul_category = mul_category[:min_len]
            clip_content_neg = clip_content_neg[:min_len]
            enhanced = enhanced[:min_len]
            llm_topic_embs = llm_topic_embs[:min_len]  
        datasets = TensorDataset(
            token_ids,  # 0: content (BERT ids)
            masks,  # 1: content_masks
            label,  # 2: label
            category,  # 3: category
            ordered_image,  # 4: image (MAE)
            clip_image,  # 5: clip_image
            clip_text,  # 6: clip_text
            mul_category,  # 7: mul_category
            clip_content_neg,  # 8: clip_content_neg
            enhanced,  # 9: enhanced
            llm_topic_embs  # 10: llm_topic_emb 
        )

        dataloader = DataLoader(
            dataset=datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )

        return dataloader
