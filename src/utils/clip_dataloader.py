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


import cn_clip.clip as clip  # 确保导入了 clip


def read_image():
    image_list = {}
    file_list = ['data/nonrumor_images/', 'data/rumor_images/']
    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        for i, filename in enumerate(os.listdir(path)):  # assuming gif

            # print(filename)
            try:
                im = Image.open(path + filename).convert('RGB')
                im = data_transforms(im)
                #im = 1
                image_list[filename.split('/')[-1].split(".")[0].lower()] = im
            except:
                print("wrong"+filename)
    #print("image length " + str(len(image_list)))
    #print("image names are " + str(image_list.keys()))
    return image_list

def _init_fn(worker_id):
    np.random.seed(2024)

def read_pkl(path):
    with open(path,"rb")as f:
        t = pickle.load(f)
    return t
def df_filter(df_data):
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data

def word2input(texts,vocab_file,max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids =[]
    for i,text in enumerate(texts):
        token_ids.append(tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.size())
    for i,token in enumerate(token_ids):
        masks[i] = (token != 0)
    return token_ids,masks

class bert_data():
    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers=2):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data(self, path, imagepath, clipimagepath, shuffle, text_only=False):
        self.data = pd.read_csv(path, encoding='utf-8')

        # 1. 基础字段
        content = self.data['content'].astype('object').to_numpy()
        label = torch.tensor(self.data['label'].astype('object').astype(int).to_numpy())
        category = torch.tensor(
            self.data['category'].astype('object').apply(lambda c: self.category_dict[c]).to_numpy())

        # 2. 多领域标签 (mul_category)
        num_domains = 9
        labels_multi_domain = torch.zeros(len(self.data), num_domains)
        for i, (cat1, cat2) in enumerate(zip(self.data['category'], self.data['领域'])):
            indices = [self.category_dict[cat1]]
            if cat2 != "无领域":
                indices.append(self.category_dict[cat2])
            labels_multi_domain[i, indices] = 1
        mul_category = labels_multi_domain

        # 3. BERT Tokenize
        token_ids, masks = word2input(content, self.vocab_file, self.max_len)

        # 4. 图像特征加载
        ordered_image = pickle.load(open(imagepath, 'rb'))
        clip_image = pickle.load(open(clipimagepath, 'rb'))

        # 5. CLIP 文本 Tokenize
        clip_text = clip.tokenize(list(content))

        # 6. 处理否定文本 content_neg
        if 'content_neg' in self.data.columns:
            neg_content = self.data['content_neg'].fillna("无").astype(str).tolist()
        else:
            neg_content = ["无"] * len(self.data)
        clip_content_neg = clip.tokenize(neg_content)

        # 7. 处理增强标记 enhanced (用于 Logic Loss Mask)
        if 'enhanced' in self.data.columns:
            # 将 True/1 转为 1.0, 其他转为 0.0
            enhanced_val = self.data['enhanced'].apply(lambda x: 1.0 if str(x).lower() in ['true', '1', '1.0'] else 0.0)
            enhanced = torch.tensor(enhanced_val.values, dtype=torch.float)
        else:
            enhanced = torch.zeros(len(self.data), dtype=torch.float)

        # === [核心新增：8. 批量加载 LLM 语义锚点 (.pt 特征)] ===
        print(f"正在加载 weibo 的 LLM 语义特征锚点...")
        pt_dir = "/home/fxy/project/DAMMFND/llm_topic_features/weibo"
        llm_topic_embs = []
        for i in range(len(self.data)):
            # 根据行号寻找对应的特征文件
            pt_path = os.path.join(pt_dir, f"row_{i}_topic.pt")

            if os.path.exists(pt_path):
                # 读取并去掉 batch 维度，变成 [512] 的一维张量
                emb = torch.load(pt_path, map_location='cpu').squeeze(0)
            else:
                # 兜底：如果没找到，用全0填充
                emb = torch.zeros(512)
            llm_topic_embs.append(emb)

        # 将列表堆叠成一个大 Tensor，形状为 [数据总量, 512]
        llm_topic_embs = torch.stack(llm_topic_embs)
        print(f"Loaded LLM topic embeddings: {llm_topic_embs.shape}")
        # ========================================================

        # === [新增修复：添加维度诊断与强制对齐逻辑] ===
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

        print("📊 [数据诊断] 各特征样本数量:", tensor_shapes)

        # 找出最短的数据长度
        min_len = min(tensor_shapes.values())

        # 如果长度不一致，强制全部截断到最小长度，防止 TensorDataset 崩溃
        if len(set(tensor_shapes.values())) > 1:
            print(f"⚠️ [警告] 发现数据长度不匹配！已强制对齐截断至最小长度: {min_len} 条")
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
        # ==============================================

        # === [核心修复：更新 TensorDataset 塞入 11 个特征] ===
        datasets = TensorDataset(
            token_ids,  # 0
            masks,  # 1
            label,  # 2
            category,  # 3
            ordered_image,  # 4
            clip_image,  # 5
            clip_text,  # 6
            mul_category,  # 7
            clip_content_neg,  # 8
            enhanced,  # 9
            llm_topic_embs  # 10
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