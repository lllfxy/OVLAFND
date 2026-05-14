import os
import pandas as pd
import requests
import base64
import torch
from tqdm import tqdm
import cn_clip.clip as clip
import time

# ================= 配置区 =================
OLLAMA_API_URL = "http://localhost:11435/api/generate"
MODEL_NAME = "qwen3-vl:8b"

# 仅处理 weibo 数据集
DIR_WEIBO = "/home/fxy/project/DAMMFND/weibo_enhance"
IMAGE_ROOT_WEIBO = "/home/fxy/project/DAMMFND/src/data"

# 专属特征保存目录
OUTPUT_DIR_WEIBO = "/home/fxy/project/DAMMFND/llm_topic_features/weibo/"
os.makedirs(OUTPUT_DIR_WEIBO, exist_ok=True)

# 加载本地 CN-CLIP 权重
device = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_WEIGHT_DIR = "/home/fxy/project/DAMMFND/src"
print(f"Loading local CN-CLIP from {LOCAL_WEIGHT_DIR} on {device}...")

from cn_clip.clip import load_from_name

clip_model, preprocess = load_from_name("ViT-B-16", device=device, download_root=LOCAL_WEIGHT_DIR)
clip_model.eval()

# 提前计算出“失败占位符”的特征向量，用于侦测并删除坏文件
with torch.no_grad():
    fallback_tokens = clip.tokenize(["未知主题"]).to(device)
    FALLBACK_EMB = clip_model.encode_text(fallback_tokens)
    FALLBACK_EMB /= FALLBACK_EMB.norm(dim=-1, keepdim=True)
    FALLBACK_EMB = FALLBACK_EMB.cpu().squeeze()


# ==========================================

def clean_id(x):
    """清理 ID，防止 Pandas 读取时变成浮点数（如 123.0）"""
    return str(x).replace('.0', '').strip()


def find_image_path(row):
    """仅针对 weibo 数据集寻找真实的本地图片路径"""
    img_name = None

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

    if img_name and '/' in img_name:
        img_name = os.path.basename(img_name)
    if img_name and not img_name.lower().endswith('.jpg'):
        img_name += '.jpg'

    if img_name is None:
        return None

    possible_paths = [
        os.path.join(IMAGE_ROOT_WEIBO, "nonrumor_images", img_name),
        os.path.join(IMAGE_ROOT_WEIBO, "rumor_images", img_name),
        os.path.join(IMAGE_ROOT_WEIBO, img_name)
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


def generate_topic_phrases(text, image_path, max_retries=3):
    if pd.isna(text):
        text = "无文本内容"

    img_b64 = image_to_base64(image_path)

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

    for attempt in range(max_retries):
        try:
            # 宽容的 120 秒超时时间
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"\n  [Warning] Ollama API 请求异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("  等待 5 秒后重试...")
                time.sleep(5)
            else:
                print("  [Error] 达到最大重试次数，放弃此条目。")
                return None
    return None


def process_file(file_path):
    print(f"\n开始处理 Weibo 训练集文件: {file_path}")

    df = pd.read_csv(file_path, on_bad_lines='skip', engine='python', encoding='utf-8-sig')
    TEXT_COLUMN = 'content'

    for index, row in tqdm(df.iterrows(), total=len(df), desc=os.path.basename(file_path)):
        try:
            # === [彻底修改] 强制使用 row_索引 命名，无视可能损坏的 post_id ===
            news_id = f"row_{index}"

            save_path = os.path.join(OUTPUT_DIR_WEIBO, f"{news_id}_topic.pt")

            # 检查断点续传与坏文件剔除
            if os.path.exists(save_path):
                try:
                    existing_emb = torch.load(save_path, map_location='cpu').squeeze()
                    if torch.allclose(existing_emb, FALLBACK_EMB, atol=1e-5):
                        os.remove(save_path)  # 是坏文件，删除重跑
                    else:
                        continue  # 是你之前成功跑完的健康文件，直接【秒跳过】！
                except Exception:
                    os.remove(save_path)  # 文件损坏，删除重跑

            text_content = row[TEXT_COLUMN] if TEXT_COLUMN in df.columns else ""
            real_img_path = find_image_path(row)

            phrases = generate_topic_phrases(text_content, real_img_path)

            # 如果彻底失败（重试都挂了），直接跳过本条，不保存坏文件
            if phrases is None:
                continue

            if not phrases:
                phrases = "未知主题"

            with torch.no_grad():
                text_tokens = clip.tokenize([phrases]).to(device)
                topic_emb = clip_model.encode_text(text_tokens)
                topic_emb /= topic_emb.norm(dim=-1, keepdim=True)

            torch.save(topic_emb.cpu(), save_path)

        except Exception as e:
            print(f"  [Error] 处理第 {index} 行 (ID: {news_id}) 时出错: {e}")


if __name__ == "__main__":
    files_to_process = []

    if os.path.exists(DIR_WEIBO):
        for f in os.listdir(DIR_WEIBO):
            if f.endswith('.csv') and 'train' in f.lower():
                files_to_process.append(os.path.join(DIR_WEIBO, f))

    print(f"共找到 {len(files_to_process)} 个 Weibo 训练集文件需要处理。")

    for file_path in files_to_process:
        process_file(file_path)

    print("\n🎉 Weibo 训练集的主题特征锚点提取完毕！")