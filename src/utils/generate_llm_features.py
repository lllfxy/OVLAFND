import os
import pandas as pd
import requests
import base64
import torch
import cn_clip.clip as clip
from tqdm import tqdm

# ================= 配置区 =================
OLLAMA_API_URL = "http://localhost:11435/api/generate"
MODEL_NAME = "qwen3-vl:8b"

# 数据集目录
DIR_WEIBO21 = "/home/fxy/project/DAMMFND/weibo21_enhance"
DIR_WEIBO = "/home/fxy/project/DAMMFND/weibo_enhance"

# 特征保存目录
OUTPUT_DIR_BASE = "/home/fxy/project/DAMMFND/llm_topic_features/"
OUTPUT_DIR_WEIBO21 = os.path.join(OUTPUT_DIR_BASE, "weibo21")
OUTPUT_DIR_WEIBO = os.path.join(OUTPUT_DIR_BASE, "weibo")

os.makedirs(OUTPUT_DIR_WEIBO21, exist_ok=True)
os.makedirs(OUTPUT_DIR_WEIBO, exist_ok=True)

# 加载本地 CN-CLIP 权重
device = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_WEIGHT_DIR = "/home/fxy/project/DAMMFND/src"
print(f"Loading local CN-CLIP from {LOCAL_WEIGHT_DIR} on {device}...")

from cn_clip.clip import load_from_name
clip_model, preprocess = load_from_name("ViT-B-16", device=device, download_root=LOCAL_WEIGHT_DIR)
clip_model.eval()
# ==========================================


def get_image_root(task_type):
    if task_type == 'weibo':
        return "/home/fxy/project/DAMMFND/src/data"
    elif task_type == 'weibo21':
        return "/home/fxy/project/DAMMFND/src/weibo21"
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def find_image_path(row, task_type):
    img_name = None
    image_root = get_image_root(task_type)

    if task_type == 'weibo':
        if 'image_id' in row and pd.notna(row['image_id']):
            url_str = str(row['image_id'])
            if url_str.lower() != 'null':
                urls = [url.strip() for url in url_str.split('|') if url.strip()]
                if urls:
                    first_url = urls[0]
                    img_name = os.path.basename(first_url).split('?')[0]
        elif 'image' in row and pd.notna(row['image']):
            img_path = str(row['image'])
            if 'rumor_images/' in img_path:
                img_name = img_path.split('rumor_images/')[1]
            elif 'nonrumor_images/' in img_path:
                img_name = img_path.split('nonrumor_images/')[1]
            else:
                img_name = os.path.basename(img_path)

    elif task_type == 'weibo21':
        if 'image' in row and pd.notna(row['image']):
            img_path = str(row['image'])
            img_name = os.path.basename(img_path)

    if img_name and '/' in img_name:
        img_name = os.path.basename(img_name)

    if img_name and not img_name.lower().endswith('.jpg'):
        img_name += '.jpg'

    if img_name is None:
        return None

    possible_paths = [
        os.path.join(image_root, "nonrumor_images", img_name),
        os.path.join(image_root, "rumor_images", img_name),
        os.path.join(image_root, img_name)
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


def image_to_base64(image_path):
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"  [Warning] 读取图片失败 {image_path}: {e}")
        return None


def generate_topic_phrases(text, image_path):
    if pd.isna(text):
        text = "无文本内容"

    img_b64 = image_to_base64(image_path)

    # === [核心修改] 完美契合论文的 Open-Vocabulary Prompt ===
    prompt = (
        f"你是一个强大的多模态新闻主题分析引擎。请观察图片并阅读文本。\n"
        f"请在你的内部理解中自动提取：\n"
        f"1. 图像描述 [image caption]\n"
        f"2. 图像中的文字 [ocr text]\n"
        f"3. 文本核心摘要 [text summary]\n"
        f"4. 核心实体词 [entity keywords]\n\n"
        f"This news is about: {text}\n\n"
        f"Provide 3-5 concise topical phrases (请综合以上信息，输出3-5个精准的新闻主题短语，用逗号分隔，不要输出任何推理过程):"
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    if img_b64:
        payload["images"] = [img_b64]

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        print(f"  [Error] Ollama API 请求失败: {e}")
        return ""


def process_file(file_path):
    print(f"\n开始处理文件: {file_path}")

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        print(f"不支持的文件格式: {file_path}")
        return

    if "weibo21" in file_path:
        task_type = "weibo21"
        ID_COLUMN = 'Unnamed: 0'
        TEXT_COLUMN = 'content'
        current_output_dir = OUTPUT_DIR_WEIBO21
    else:
        task_type = "weibo"
        ID_COLUMN = 'post_id'
        TEXT_COLUMN = 'content'
        current_output_dir = OUTPUT_DIR_WEIBO

    for index, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(file_path)):
        try:
            if ID_COLUMN in df.columns:
                news_id = str(row[ID_COLUMN]).replace('.0', '').strip()
            else:
                news_id = f"row_{index}"

            save_path = os.path.join(current_output_dir, f"{news_id}_topic.pt")
            if os.path.exists(save_path):
                continue

            text_content = row[TEXT_COLUMN] if TEXT_COLUMN in df.columns else ""
            real_img_path = find_image_path(row, task_type)

            phrases = generate_topic_phrases(text_content, real_img_path)
            if not phrases:
                phrases = "未知主题"

            with torch.no_grad():
                text_tokens = clip.tokenize([phrases]).to(device)
                topic_emb = clip_model.encode_text(text_tokens)
                topic_emb /= topic_emb.norm(dim=-1, keepdim=True)

            torch.save(topic_emb.cpu(), save_path)

        except Exception as e:
            print(f"  [Error] 处理第 {index} 行数据时出错: {e}")


if __name__ == "__main__":
    files_to_process = []

    # === [核心修改] 仅筛选带有 "train" 关键字的文件 ===
    if os.path.exists(DIR_WEIBO21):
        for f in os.listdir(DIR_WEIBO21):
            if f.endswith('.xlsx') and not f.startswith('~$') and 'train' in f.lower():
                files_to_process.append(os.path.join(DIR_WEIBO21, f))

    if os.path.exists(DIR_WEIBO):
        for f in os.listdir(DIR_WEIBO):
            if f.endswith('.csv') and 'train' in f.lower():
                files_to_process.append(os.path.join(DIR_WEIBO, f))

    print(f"共找到 {len(files_to_process)} 个训练集文件需要处理。")

    for file_path in files_to_process:
        process_file(file_path)

    print("\n🎉 训练集的主题特征锚点提取完毕！")