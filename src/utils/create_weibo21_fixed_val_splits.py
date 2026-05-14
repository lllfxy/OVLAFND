import pandas as pd
import os


def create_weibo21_fixed_val_splits(train_file, val_file, test_file, output_base_dir):
    """
    针对 Weibo21 数据集 (.xlsx)
    策略：验证集固定为"社会生活"（测"社会生活"时验证集改为"文体娱乐"）。
    """
    print("================ 开始生成 Weibo21 (固定验证集) 数据集 ================")

    try:
        # 1. 读取并合并 3 份原始数据
        print(f"正在读取文件...\nTrain: {train_file}\nVal: {val_file}\nTest: {test_file}")
        train_df = pd.read_excel(train_file)
        val_df = pd.read_excel(val_file)
        test_df = pd.read_excel(test_file)

        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        print(f"✅ 成功合并原始文件！初始总数据量: {len(df)} 条")
    except Exception as e:
        raise ValueError(f"❌ 读取文件失败，请检查路径或格式是否为 xlsx: {e}")

    # 剔除空类别
    df = df.dropna(subset=['category'])

    # 2. 清洗：只保留单一领域的数据
    df['领域'] = df['领域'].fillna('无领域').astype(str).str.strip()
    df = df[df['领域'] == '无领域']
    print(f"✅ 过滤完成！纯单一领域数据量: {len(df)} 条")

    # 获取标准领域
    domains = sorted(df['category'].unique().tolist())
    print(f"✅ 提取到 {len(domains)} 个领域：\n{domains}\n")
    print("-" * 60)

    os.makedirs(output_base_dir, exist_ok=True)

    # 3. 核心分配逻辑
    for test_domain in domains:
        # === 【核心修改：验证集选择逻辑】 ===
        if test_domain == '社会生活':
            val_domain = '文体娱乐'
        else:
            val_domain = '社会生活'
        # ===================================

        # 剩下的 7 个领域作为训练集
        train_domains = [d for d in domains if d not in [test_domain, val_domain]]

        # 执行 DataFrame 切分
        test_split = df[df['category'] == test_domain]
        val_split = df[df['category'] == val_domain]
        train_split = df[df['category'].isin(train_domains)]

        # 创建并保存到专属文件夹
        fold_dir = os.path.join(output_base_dir, f"test_{test_domain}")
        os.makedirs(fold_dir, exist_ok=True)

        # 保存为 .xlsx 格式
        train_split.to_excel(os.path.join(fold_dir, "train.xlsx"), index=False)
        val_split.to_excel(os.path.join(fold_dir, "val.xlsx"), index=False)
        test_split.to_excel(os.path.join(fold_dir, "test.xlsx"), index=False)

        # 打印日志
        print(
            f"[Test] {test_domain:<6} | [Val] {val_domain:<6} | Train({len(train_domains)}域): {len(train_split):>5} 条")

    print("-" * 60)
    print(f"🎉 Weibo21 划分完毕！数据已保存在 '{output_base_dir}'。")


if __name__ == "__main__":
    WEIBO21_TRAIN = "/home/fxy/project/OVLAFND/weibo21_enhance/train_2_domain.xlsx"
    WEIBO21_VAL = "/home/fxy/project/OVLAFND/weibo21_enhance/val_2_domain.xlsx"
    WEIBO21_TEST = "/home/fxy/project/OVLAFND/weibo21_enhance/test_2_domain.xlsx"

    # 建立一个全新的专属文件夹
    WEIBO21_OUTPUT = "/home/fxy/project/OVLAFND/weibo21_enhance/weibo21_fixed_val_experiments"

    create_weibo21_fixed_val_splits(WEIBO21_TRAIN, WEIBO21_VAL, WEIBO21_TEST, WEIBO21_OUTPUT)