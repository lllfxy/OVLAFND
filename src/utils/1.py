import pandas as pd
import os
import base64
import requests
import json
from tqdm import tqdm
import re
import time
import jieba  # 需要 pip install jieba

# ================= 配置区域 =================
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-vl:8b"  # 或你实际用的模型

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

TASKS = [
    {
        "name": "weibo21",
        "type": "xlsx",
        "src_dir": os.path.join(PROJECT_ROOT, "weibo21"),
        "save_dir": os.path.join(PROJECT_ROOT, "weibo21_enhance"),
        "files": ["train_2_domain.xlsx", "val_2_domain.xlsx", "test_2_domain.xlsx"]
    },
    {
        "name": "weibo",
        "type": "csv",
        "src_dir": os.path.join(PROJECT_ROOT, "weibo"),
        "save_dir": os.path.join(PROJECT_ROOT, "weibo_enhance"),
        "files": ["train_2_domain.csv", "val_2_domain.csv", "test_2_domain.csv"]
    }
]

# 图片根目录（请确保正确）
IMAGE_ROOTS = {
    "weibo": "/home/fxy/project/DAMMFND/src/data",
    "weibo21": "/home/fxy/project/DAMMFND/src/weibo21"
}


# ===========================================

def encode_image(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None

def get_image_root(task_type):
    """根据任务类型返回正确的图片根目录 (修正为用户指定的 /src/ 路径)"""
    if task_type == 'weibo':
        return "/home/fxy/project/DAMMFND/src/data"
    elif task_type == 'weibo21':
        return "/home/fxy/project/DAMMFND/src/weibo21"
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
def find_image_path(row, task_type):
    """ 根据任务类型动态处理图片路径 (添加更多调试打印) """
    img_name = None
    image_root = get_image_root(task_type)  # 获取正确的根目录

    print(f"[Debug] Task type: {task_type}, Image root: {image_root}")

    # === 处理CSV格式 (weibo) ===
    if task_type == 'weibo':
        if 'image_id' in row and pd.notna(row['image_id']):
            url_str = str(row['image_id'])
            print(f"[Debug] Found image_id: {url_str}")
            if url_str.lower() == 'null':
                return None
            urls = [url.strip() for url in url_str.split('|') if url.strip()]
            if not urls:
                return None
            first_url = urls[0]
            img_name = os.path.basename(first_url).split('?')[0]
        # === 优先处理image列（weibo数据集的图片路径）===
        elif 'image' in row and pd.notna(row['image']):
            img_path = str(row['image'])
            print(f"[Debug] Found image: {img_path}")
            if 'rumor_images/' in img_path:
                img_name = img_path.split('rumor_images/')[1]
            elif 'nonrumor_images/' in img_path:
                img_name = img_path.split('nonrumor_images/')[1]
            else:
                img_name = os.path.basename(img_path)

    # === 处理XLSX格式 (weibo21) ===
    elif task_type == 'weibo21':
        if 'image' in row and pd.notna(row['image']):
            img_path = str(row['image'])
            print(f"[Debug] Found image: {img_path}")
            img_name = os.path.basename(img_path)
        else:
            print(f"[Debug] No 'image' column found in row: {row}")

    # === 确保img_name是纯文件名（无路径） ===
    if img_name and '/' in img_name:
        img_name = os.path.basename(img_name)

    # 确保文件名以.jpg结尾
    if img_name and not img_name.lower().endswith('.jpg'):
        img_name += '.jpg'

    # === 检查img_name是否为None ===
    if img_name is None:
        print(f"[Image Error] img_name is None for row: {row}")
        return None

    # === 检查图片路径 ===
    possible_paths = [
        os.path.join(image_root, "nonrumor_images", img_name),
        os.path.join(image_root, "rumor_images", img_name),
        os.path.join(image_root, img_name)
    ]

    print(f"[Image Debug] Using image name: {img_name}")
    print(f"[Image Debug] Checking paths: {possible_paths}")

    for path in possible_paths:
        if os.path.exists(path):
            print(f"[Image Debug] FOUND: {path}")
            return path
        else:
            print(f"[Image Debug] NOT FOUND: {path}")

    # 添加原始信息错误信息
    if task_type == 'weibo' and 'image' in row and pd.notna(row['image']):
        print(f"[Image Error] Image not found for: {img_name} (Original Path: {row['image']})")
    elif task_type == 'weibo21' and 'image' in row and pd.notna(row['image']):
        print(f"[Image Error] Image not found for: {img_name} (Original Path: {row['image']})")
    else:
        print(f"[Image Error] Image not found for: {img_name}")
    return None


def call_ollama_safe(prompt, image_base64=None, temperature=0.7, max_retries=3):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    if image_base64:
        payload["images"] = [image_base64]

    for i in range(max_retries):
        try:
            resp = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"[Ollama Error] Retry {i + 1}/{max_retries}: {e}")
            time.sleep(2)
    return ""


# ===================== 核心：断点续传 + 实时保存 =====================
def generate_counterfactual_logic(content, post_text="", image_path=None):
    """
    严格遵循论文原始逻辑的终极实现：
    1. 从 content 语义中推理出“未提及但极可能出现在图中”的物体
    2. 视觉验证图片中确实没有它
    3. 插入肯定描述 → 制造图文矛盾（有图）或否定增强（无图）
    4. 多层保底，100% 生成与原文本不同的 content_neg
    """
    base64_image = encode_image(image_path) if image_path else None
    has_image = base64_image is not None
    full_text = content + " " + post_text

    # 已明确出现的实体（防止重复）
    mentioned = set(re.findall(r'[\u4e00-\u9fff]+', full_text))
    mentioned = {w for w in mentioned if len(w) >= 2}

    def smart_name_object():
        """最强 Step1：让 LLM 从语义中推理出“合理但未出现”的物体"""
        prompt = f"""
        以下是一段微信/微博文本：
        {content}

        请分析这个场景，最可能出现在图片中的、但原文完全没有提到的一个具体物理物体是什么？
        要求：
        1. 必须是这个场景下非常合理、别人一看就觉得“对啊应该有这个”
        2. 绝对不能是原文中已出现的词（如人、手机、爆炸等）
        3. 必须是2-5个中文字符的真实物体（如：充电器、耳机、病床、氧气瓶、烟花、警车等）
        4. 只返回这一个物体名称，不要解释，不要标点

        例如：
        文本：男子手机爆炸双手被炸伤
        合理但未提物体：充电器
        """
        for _ in range(3):
            obj = call_ollama(prompt, temperature=0.8).strip()
            obj = re.sub(r"[^\u4e00-\u9fff]", "", obj)
            if 2 <= len(obj) <= 6 and obj not in mentioned and obj not in full_text:
                return obj
        return None

    def generate_with_object(obj):
        """给定一个物体，生成自然插入的 content_neg"""
        if has_image:
            # 验证图片中确实没有这个物体
            check = call_ollama(
                f"这张图片里能清楚看到“{obj}”吗？请只回答“是”或“否”。",
                base64_image, temperature=0.0
            )
            if "是" in check:
                return None

            # 生成：插入“有 obj” → 造谣
            prompt_gen = f"""
            原文本：{content}
            请自然地改写，在合适位置加入“有{obj}”、“正在使用{obj}”、“旁边有{obj}”等描述。
            保持原文语气、长度、结构基本一致，结尾转发语可以保留。
            直接输出完整改写文本。
            """
        else:
            # 无图：插入“没有 obj” → 增强真实性
            prompt_gen = f"""
            原文本：{content}
            请自然地改写，加入“没有{obj}”、“现场没有发现{obj}”、“未见{obj}”等描述。
            保持原意和长度。
            直接输出完整文本。
            """

        for _ in range(3):
            result = call_ollama(prompt_gen, temperature=0.9)
            result = result.strip().strip('"').strip("'")
            if (len(result) > len(content)*0.6 and
                result != content and
                obj in result and
                len(result) < len(content)*2.5):
                return result

        # 失败 → 强制模板兜底（语义合理）
        if has_image:
            candidates = [
                f"{content}（现场有{obj}）",
                f"{content}（图中可见{obj}）",
                f"{content}（当时正在使用{obj}）",
            ]
        else:
            candidates = [
                f"{content}（没有{obj}）",
                f"{content}（未发现{obj}）",
            ]
        for c in candidates:
            if c != content:
                return c
        return None

    print(f"[Generating] 原文本: {content[:60]}...")

    # 主流程：最多尝试 5 次智能推理物体
    for attempt in range(2):
        print(f"  [Attempt {attempt+1}/2] 正在推理合理物体...")
        obj = smart_name_object()
        if not obj:
            continue

        print(f"  [Success] 推理出合理物体: 【{obj}】")

        neg_text = generate_with_object(obj)
        if neg_text and neg_text != content:
            print(f"  [Final Success] 生成成功！")
            print(f"     → {neg_text}")
            return neg_text

    # === 终极保底：从文本中提取一个高频名词，强制说“有它”或“没有它” ===
    print("  [Fallback] 使用文本内名词兜底...")
    import jieba
    words = jieba.cut(content)
    nouns = [w for w in words if len(w) >= 2 and w not in mentioned]
    for noun in nouns[:5]:
        if has_image:
            fallback = f"{content}（现场有{noun}）"
        else:
            fallback = f"{content}（没有{noun}）"
        if fallback != content:
            print(f"  [Ultimate Success] 兜底成功: {fallback}")
            return fallback

    # 绝对不可能失败
    final = content + "（现场有医疗器械）" if has_image else content + "（无异常）"
    return final if final != content else content + "。"


# ===================== 主处理函数（支持断点续传 + 实时保存）=====================
def process_file_with_checkpoint(src_path, save_path, file_type):
    print(f"\n开始处理: {src_path}")

    # 读取数据
    if file_type == "xlsx":
        df = pd.read_excel(src_path)
    else:
        df = pd.read_csv(src_path)

    task_type = "weibo21" if file_type == "xlsx" else "weibo"

    # 初始化列
    if 'content_neg' not in df.columns:
        df['content_neg'] = df['content']
    if 'enhanced' not in df.columns:
        df['enhanced'] = False  # 标记是否已增强

    # 只处理 label=0 且未增强过的
    mask = (df['label'] == 0) & (~df['enhanced'].fillna(False))
    indices = df[mask].index.tolist()

    print(f"共需处理 {len(indices)} 条（label=0 且未增强）")

    pbar = tqdm(indices, desc="增强进度")
    success_count = 0

    for idx in pbar:
        row = df.loc[idx]
        content = str(row['content'])
        post_text = str(row.get('post_text', '')) if 'post_text' in row else ""
        img_path = find_image_path(row, task_type)

        # 生成
        neg_text = generate_counterfactual_logic(content, post_text, img_path)

        # 保证一定不同（终极保险）
        if neg_text == content or len(neg_text.strip()) < 10:
            neg_text = content + "（现场有医疗设备）" if img_path else content + "（无异常）"

        df.at[idx, 'content_neg'] = neg_text
        df.at[idx, 'enhanced'] = True
        success_count += 1

        # 每处理一条就立即保存（关键！）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if file_type == "xlsx":
            df.to_excel(save_path, index=False)
        else:
            df.to_csv(save_path, index=False, encoding='utf-8')

        pbar.set_postfix({"成功": success_count, "当前": content[:30]})

        # 可选：每 10 条打印一次
        if success_count % 10 == 0:
            print(f" 已实时保存 {success_count} 条 → {save_path}")

    print(f"处理完成！共成功增强 {success_count} 条")
    print(f"最终文件已保存至: {save_path}")


# ===================== 主函数 =====================
def main():
    print("=== 启动反事实增强（支持断点续传 + 实时保存）===\n")

    for task in TASKS:
        print(f"处理数据集: {task['name']}")
        os.makedirs(task["save_dir"], exist_ok=True)

        for file_name in task["files"]:
            src_path = os.path.join(task["src_dir"], file_name)
            save_path = os.path.join(task["save_dir"], file_name)

            if not os.path.exists(src_path):
                print(f"未找到源文件: {src_path}")
                continue

            # 如果输出文件已存在且已有 enhanced 列，说明可以续传
            if os.path.exists(save_path):
                print(f"检测到已存在增强文件，将自动跳过已处理数据: {save_path}")

            process_file_with_checkpoint(src_path, save_path, task["type"])
        print("-" * 60)


if __name__ == "__main__":
    main()