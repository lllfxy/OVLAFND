import pandas as pd
import os


def fix_politics_val_split(train_file, val_file, test_file, output_base_dir):
    """
    针对 Weibo 数据集 (.csv)
    策略：仅针对测试集为"政治"的划分进行修改，将其验证集强制指定为"娱乐"，其他领域跳过不覆盖。
    """
    print("================ 开始修正 LTDO (政治测试集 -> 娱乐验证集) ================")

    try:
        # 1. 读取并合并 3 份原始数据 (CSV格式)
        train_df = pd.read_csv(train_file, encoding='utf-8-sig', engine='python', on_bad_lines='skip')
        val_df = pd.read_csv(val_file, encoding='utf-8-sig', engine='python', on_bad_lines='skip')
        test_df = pd.read_csv(test_file, encoding='utf-8-sig', engine='python', on_bad_lines='skip')
        df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        print(f"✅ 成功合并原始文件！初始总数据量: {len(df)} 条")
    except Exception as e:
        raise ValueError(f"❌ 读取文件失败: {e}")

    df = df.dropna(subset=['category'])

    # 2. 清洗：只保留单一领域的数据
    df['领域'] = df['领域'].fillna('无领域').astype(str).str.strip()
    df = df[df['领域'] == '无领域']

    # 3. 获取所有标准领域
    domains = sorted(df['category'].unique().tolist())
    print(f"✅ 提取到 {len(domains)} 个领域：\n{domains}\n")
    os.makedirs(output_base_dir, exist_ok=True)

    # 4. 核心逻辑：只更新政治领域
    for test_domain in domains:
        # 拦截机制：如果不是"政治"，直接跳过，不覆盖原有文件
        if test_domain != '政治':
            continue

        # 强行指定政治的验证集为娱乐
        val_domain = '娱乐'

        # 剩下的领域作为训练集
        train_domains = [d for d in domains if d not in [test_domain, val_domain]]

        # 切分 DataFrame
        test_split = df[df['category'] == test_domain]
        val_split = df[df['category'] == val_domain]
        train_split = df[df['category'].isin(train_domains)]

        # 创建并保存到专属文件夹
        fold_dir = os.path.join(output_base_dir, f"test_{test_domain}")
        os.makedirs(fold_dir, exist_ok=True)

        # 保存为 .csv 格式
        train_split.to_csv(os.path.join(fold_dir, "train.csv"), index=False, encoding='utf-8-sig')
        val_split.to_csv(os.path.join(fold_dir, "val.csv"), index=False, encoding='utf-8-sig')
        test_split.to_csv(os.path.join(fold_dir, "test.csv"), index=False, encoding='utf-8-sig')

        print(
            f"✅ 修正完成: [Test] {test_domain:<6} | [Val] {val_domain:<6} | Train({len(train_domains)}域): {len(train_split)} 条")

    print(f"\n🎉 针对政治领域的修正划分完毕！数据已覆盖/保存在 '{output_base_dir}'。")


if __name__ == "__main__":
    WEIBO_TRAIN = "/home/fxy/project/OVLAFND/weibo_enhance/train_2_domain.csv"
    WEIBO_VAL = "/home/fxy/project/OVLAFND/weibo_enhance/val_2_domain.csv"
    WEIBO_TEST = "/home/fxy/project/OVLAFND/weibo_enhance/test_2_domain.csv"

    # 指向你存放 7+1+1 实验数据的目标文件夹，这样只会覆盖里面的 test_政治 文件夹
    WEIBO_OUTPUT = "/home/fxy/project/OVLAFND/weibo_enhance/ltdo_7_1_1_experiments"

    fix_politics_val_split(WEIBO_TRAIN, WEIBO_VAL, WEIBO_TEST, WEIBO_OUTPUT)