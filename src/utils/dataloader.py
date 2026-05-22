import pickle
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer
import torch
from torchvision import transforms
import os
from PIL import Image
import cn_clip.clip as clip

def _init_fn(worker_id):
    np.random.seed(2024)


def word2input(texts, vocab_file, max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                                          truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i, token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids, masks

class bert_data():
    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data(self, path, ttv, shuffle, text_only=False):

        if path.endswith('.csv'):
            self.data = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip', engine='python')
        elif path.endswith('.xlsx'):
            self.data = pd.read_excel(path)
        else:
            raise ValueError(f"不支持的文件格式: {path}")
        task_type = "weibo21" if "weibo21" in path.lower() else "weibo"
        pt_dir = f"/home/fxy/project/OVLAFND/llm_topic_features/{task_type}"
        content = self.data['content'].astype('object').to_numpy()
        label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
        category = torch.tensor(
            self.data['category'].astype('object').apply(lambda c: self.category_dict[c]).to_numpy())
        token_ids, masks = word2input(content, self.vocab_file, self.max_len)
        ordered_image = pickle.load(open(ttv, 'rb'))
        if 'enhanced' in self.data.columns:
            enhanced_col = self.data['enhanced'].apply(
                lambda x: 1.0 if (str(x).lower() == 'true' or str(x) == '1' or str(x) == '1.0') else 0.0)
            enhanced = torch.tensor(enhanced_col.to_numpy(), dtype=torch.float)
        else:
            enhanced = torch.zeros(len(content), dtype=torch.float)
        if 'content_neg' in self.data.columns:
            content_neg_list = self.data['content_neg'].fillna("").astype(str).tolist()
        else:
            content_neg_list = [""] * len(content)

        # print("正在进行 CLIP Tokenize (这可能需要几秒钟)...")
        clip_text = clip.tokenize(list(content), context_length=77)
        clip_content_neg = clip.tokenize(content_neg_list, context_length=77)
        clip_image = ordered_image
        multi_category = category
        # print(f"正在加载 {task_type} 的 LLM 语义特征锚点...")
        llm_topic_embs = []
        for i in range(len(content)):
            pt_path = os.path.join(pt_dir, f"row_{i}_topic.pt")
            if os.path.exists(pt_path):
                emb = torch.load(pt_path, map_location='cpu').squeeze(0)
            else:
                emb = torch.zeros(512)
            llm_topic_embs.append(emb)
        llm_topic_embs = torch.stack(llm_topic_embs)
        print(
            f"Dataset summary: ids={token_ids.shape}, img={ordered_image.shape}, enhanced={enhanced.shape}, llm_emb={llm_topic_embs.shape}")

        datasets = TensorDataset(
            token_ids,  # 0: content (BERT ids)
            masks,  # 1: content_masks
            label,  # 2: label
            category,  # 3: category
            ordered_image,  # 4: image (MAE)
            clip_image,  # 5: clip_image
            clip_text,  # 6: clip_text (Tensor)
            multi_category,  # 7: multi_category
            clip_content_neg,  # 8: content_neg (CLIP Tokenized Tensor)
            enhanced,  # 9: enhanced (Float Tensor)
            llm_topic_embs  
        )
        print(f"TensorDataset items: {len(datasets.tensors)}")

        dataloader = DataLoader(
            dataset=datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn
        )
        return dataloader
