import pandas as pd
import os


def clear_content_neg_with_keywords(file_path):
    """
    清空content_neg列中包含特定关键词的内容
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        print(f"读取文件: {file_path}")
        print(f"原始数据行数: {len(df)}")

        # 定义需要匹配的关键词列表
        keywords = [
            "作为AI助手",
            "作为AI助手，我无法按照您的要求生成虚假或误导性内容。",
            "作为AI助手，我必须遵守法律法规，不参与任何造假、编造虚假信息的行为。",
            "请您理解并尊重事实与法律，"
        ]

        # 记录被修改的行数
        modified_count = 0

        # 遍历每一行，检查content_neg列是否包含关键词
        for index, row in df.iterrows():
            content_neg = str(row['content_neg']) if pd.notna(row['content_neg']) else ''

            # 检查是否包含任一关键词
            should_clear = any(keyword in content_neg for keyword in keywords)

            if should_clear:
                # 清空content_neg列
                df.at[index, 'content_neg'] = ''
                modified_count += 1

                # 可选：打印修改的行（用于调试）
                # print(f"修改第 {index} 行: {content_neg[:50]}...")

        # 保存修改后的文件（覆盖原文件）
        df.to_csv(file_path, index=False, encoding='utf-8')

        print(f"处理完成！共修改了 {modified_count} 行")
        print(f"保存到原文件: {file_path}")

        return df

    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None


def process_directory(directory_path):
    """
    处理目录下的所有CSV文件
    """
    if not os.path.exists(directory_path):
        print(f"目录不存在: {directory_path}")
        return

    # 获取目录下所有CSV文件
    csv_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.csv')]

    if not csv_files:
        print(f"在目录 {directory_path} 中未找到CSV文件")
        return

    print(f"在目录中找到 {len(csv_files)} 个CSV文件")

    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        print(f"\n处理文件: {csv_file}")
        print("-" * 50)
        clear_content_neg_with_keywords(file_path)
        print("-" * 50)


# 主程序
if __name__ == "__main__":
    directory_path = "/home/fxy/project/DAMMFND/weibo_enhance"

    # 处理目录下的所有CSV文件
    process_directory(directory_path)

    print("\n所有文件处理完成！")