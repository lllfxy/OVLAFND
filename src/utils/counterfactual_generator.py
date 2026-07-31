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
OLLAMA_API_URL = "http://localhost:11435/api/generate"
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
            resp = requests.post(OLLAMA_API_URL, json=payload, timeout=200)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"[Ollama Error] Retry {i + 1}/{max_retries}: {e}")
            time.sleep(2)
    return ""


# ===================== 核心：断点续传 + 实时保存 =====================
import random

PROMPT_EXISTENCE = """
你是一个制造“视觉逻辑陷阱”的专家。
请观察图片，并阅读原新闻：“{content}”

任务：构造一个【存在性否定】的假新闻。
1. 找到图片中**最显眼、最核心**的一个物体或人物（例如：警察、火灾、人群、横幅）。
2. 在原句基础上进行修改，**睁眼说瞎话**，断言该物体**根本不在现场**。
3. 必须使用显性的否定词，如“没有”、“并未发现”、“空无一物”、“并无”。

示例：
原句：图中有一辆警车停在路边。
修改：现场并没有发现任何警车，路边空空荡荡。

请输出修改后的句子（仅输出句子）：
"""

PROMPT_COUNTING = """
你是一个假新闻生成器。
请观察图片，并阅读原句：“{content}”

任务：构造一个【数量/程度反转】的假新闻。
1. 关注原句或图片中的“数量”、“密度”或“程度”。
2. 将其修改为**逻辑完全相反**的状态。
   - 如果图中有很多人，说“现场冷清，只有零星几人”。
   - 如果图中有大火，说“只有轻微烟雾，并未起火”。
   - 如果有多个物体，说“只有一个”。
3. 句式要像正规新闻报道，但事实完全相反。

请输出修改后的句子（仅输出句子）：
"""

PROMPT_STATE = """
请观察图片和原句：“{content}”

任务：构造一个【状态互斥】的逻辑陷阱。
请修改原句中的**形容词**或**动词**，使其与图片内容**逻辑互斥**。
   - 如果人在“打架”，改成他们在“拥抱”或“并没有肢体接触”。
   - 如果物体是“破碎的”，改成“完好无损的”。
   - 如果是“白天”，改成“夜间”。
   - 使用“其实是”、“看似...实则...”的句式来增强欺骗性。

请输出修改后的句子（仅输出句子）：
"""

PROMPT_COT_NEGATION = """
你是一个造假专家。请按步骤执行：

第一步（视觉分析）：
仔细看图，找出图中真真切切存在的一个核心物体（例如：车、人、血迹、标语）。

第二步（逻辑反转）：
针对你看到的这个物体，构造一句话，声称**它不存在**或者**完全是另一种东西**。

原句参考：“{content}”

要求：
1. 修改后的句子必须包含“没有”、“并非”、“不存在”等否定词。
2. 句子要通顺，像一条辟谣微博，但内容是假的（因为你否定了图中真实存在的东西）。
3. 不要输出分析过程，只输出最后生成的句子。
"""


# 定义 Prompts 字典
PROMPT_TEMPLATES = {
    "existence": PROMPT_EXISTENCE,
    "counting": PROMPT_COUNTING,
    "state": PROMPT_STATE,
    "cot": PROMPT_COT_NEGATION
}


def generate_counterfactual_logic(content, post_text="", image_path=None):
    """
    VFL-Logic 增强版生成函数
    随机选择攻击策略，全面覆盖 Recognition, Counting, Grounding 层
    """
    base64_image = encode_image(image_path) if image_path else None

    # 策略选择逻辑：
    # 1. 如果有图，随机选择一种视觉攻击策略
    # 2. 这里的权重可以调整，目前侧重 'cot' (思维链) 因为它效果最好
    if base64_image:
        strategy = random.choices(
            ["existence", "counting", "state", "cot"],
            weights=[0.2, 0.2, 0.2, 0.4],
            k=1
        )[0]
    else:
        # 无图模式只能做纯文本逻辑取反
        strategy = "text_logic"
        prompt_template = f"""
        原句：“{content}”
        请将这句话改写为逻辑完全相反的假新闻。
        必须包含“没有”、“未发生”、“并非”等否定词。
        例如把“发生了爆炸”改成“并未发生爆炸，纯属谣言”。
        直接输出改写后的句子。
        """

    # 构造 Prompt
    if strategy != "text_logic":
        # 格式化 Prompt
        prompt = PROMPT_TEMPLATES[strategy].format(content=content)
        print(f"[Strategy: {strategy}] 正在生成...")
    else:
        prompt = prompt_template
        print(f"[Strategy: Text Only] 正在生成...")

    # 调用 Ollama (复用你现有的 call_ollama_safe)
    # 注意：温度稍微调高一点 (0.7-0.9) 以增加多样性，因为我们有强逻辑约束
    result = call_ollama_safe(prompt, base64_image, temperature=0.8, max_retries=2)

    # 清洗结果
    result = result.strip().strip('"').strip("'").replace("修改后的句子：", "").replace("修改：", "")

    # === 质量校验 (VFL-Logic Check) ===
    # 检查是否真的生成了否定词，如果没生成，强制回退到简单否定
    negation_keywords = ["没有", "无", "并未", "未发现", "不是", "并非", "0", "零"]
    has_negation = any(kw in result for kw in negation_keywords)

    if not has_negation or len(result) < 5:
        print(f"[Warn] 策略 {strategy} 生成失败 (无否定词)，尝试兜底生成...")
        # 兜底：简单暴力的否定
        fallback_prompt = f"原句：{content}\n请直接改为否定句，说原句里提到的东西都没有、都没发生。直接输出句子。"
        result = call_ollama_safe(fallback_prompt, base64_image, temperature=0.5)
        result = result.strip().strip('"')

    print(f"原句: {content[:20]}...")
    print(f"反转: {result[:20]}...\n")

    return result
# ===================== 主处理函数（支持断点续传 + 实时保存）=====================
def process_file_with_checkpoint(src_path, save_path, file_type):
    print(f"\n开始处理: {src_path} → {save_path}")

    # === 关键修复：优先读取已增强的文件，如果存在就续传！===
    if os.path.exists(save_path):
        print(f"检测到已存在增强文件，正在加载以实现断点续传: {save_path}")
        if file_type == "xlsx":
            df = pd.read_excel(save_path)
        else:
            df = pd.read_csv(save_path)
        print(f"已加载增强文件，共 {len(df)} 行，其中已增强: {df['enhanced'].sum()} 条")
    else:
        print("未检测到增强文件，从原始文件开始")
        if file_type == "xlsx":
            df = pd.read_excel(src_path)
        else:
            df = pd.read_csv(src_path)
        # 初始化增强列
        if 'content_neg' not in df.columns:
            df['content_neg'] = df['content']
        if 'enhanced' not in df.columns:
            df['enhanced'] = False

    task_type = "weibo21" if file_type == "xlsx" else "weibo"

    # 筛选：label==0 且 未增强
    mask = (df['label'] == 0) & (~df['enhanced'].astype(bool))
    indices = df[mask].index.tolist()

    if len(indices) == 0:
        print("所有 label=0 的样本均已增强，跳过此文件！")
        return

    print(f"本次需处理 {len(indices)} 条未增强样本（label=0）")

    pbar = tqdm(indices, desc="增强进度")
    success_count = 0

    for idx in pbar:
        row = df.loc[idx]
        content = str(row['content'])
        post_text = str(row.get('post_text', '')) if 'post_text' in row else ""
        img_path = find_image_path(row, task_type)

        # 生成反事实文本
        neg_text = generate_counterfactual_logic(content, post_text, img_path)

        # 终极保底
        if neg_text == content or len(neg_text.strip()) < 10:
            neg_text = content + "（现场有医疗设备）" if img_path else content + "（无异常）"

        df.at[idx, 'content_neg'] = neg_text
        df.at[idx, 'enhanced'] = True
        success_count += 1

        # 每处理一条立即保存（关键！）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if file_type == "xlsx":
            df.to_excel(save_path, index=False)
        else:
            df.to_csv(save_path, index=False, encoding='utf-8')

        pbar.set_postfix({
            "已完成": success_count,
            "剩余": len(indices) - success_count,
            "当前": content[:25].replace("\n", " ")
        })

        # 每10条打印一次
        if success_count % 10 == 0:
            print(f" 已实时保存 {success_count} 条 → {save_path}")

    print(f"本文件处理完成！本次新增增强 {success_count} 条")
    print(f"最终文件: {save_path}")


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