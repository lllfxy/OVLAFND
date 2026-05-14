import torch
import numpy as np
import tqdm
import os

from main import config
from run import Run
from utils.utils import clipdata2gpu

# 注意：这里导入的模型，如果是 weibo 和 weibo21 代码分开了，请确保导入正确的类
from model.ovlafnd_weibo import OVLAFNDMODEL  # 根据实际路径调整


def extract_features():
    dataset = config['dataset']
    print(f"=== 开始提取 OVLAFND 特征 ({dataset} 数据集) ===")

    run_instance = Run(config=config)
    _, _, test_loader = run_instance.get_dataloader(dataset)

    # === [核心修改] 从 config 中动态获取当前数据集对应的 dropout ===
    current_dropout = config['model']['mlp']['dropout']
    print(f"当前加载的 Dropout 参数为: {current_dropout}")

    # 将写死的 0.3 替换为 current_dropout
    model = OVLAFNDMODEL(768, [384], '/home/fxy/chinese-roberta-wwm-ext', 320, current_dropout).cuda()

    # === [优化] 动态适配模型权重路径 ===
    if dataset == 'weibo':
        model_path = "/home/fxy/project/OVLAFND/src/param_model/ovlafnd_weibo/parameter_ovlafnd_weibo.pkl"
    else:
        model_path = "/home/fxy/project/OVLAFND/src/param_model/ovlafnd/parameter_OVLAFND_weibo21.pkl"

    print(f"加载权重: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_fake_feats, all_domain_feats, all_labels, all_categories = [], [], [], []

    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            batch_data = clipdata2gpu(batch)
            outputs = model(**batch_data)

            # === 智能区分返回值 ===
            if dataset == 'weibo21':
                # weibo21: 索引 6 是 topic_rep, 7 是 fake_news_feature
                dom_feat = outputs[6]
                fake_feat = outputs[7]
            else:
                # weibo: 刚刚追加在末尾的两个特征 (-1代表倒数第一, -2代表倒数第二)
                fake_feat = outputs[-2]  # fake_news_feature
                dom_feat = outputs[-1]  # multi_label_feature

            all_domain_feats.append(dom_feat.cpu().numpy())
            all_fake_feats.append(fake_feat.cpu().numpy())
            all_labels.append(batch_data['label'].cpu().numpy())

            cat_key = 'category' if 'category' in batch_data else 'multi_category'
            all_categories.append(batch_data[cat_key].cpu().numpy())

    # 自动保存到对应的 dataset 目录
    save_dir = f"/home/fxy/project/OVLAFND/src/plot_results/{dataset}"
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'ovlafnd_fake_feat.npy'), np.concatenate(all_fake_feats, axis=0))
    np.save(os.path.join(save_dir, 'ovlafnd_dom_feat.npy'), np.concatenate(all_domain_feats, axis=0))
    np.save(os.path.join(save_dir, 'test_labels.npy'), np.concatenate(all_labels, axis=0))
    np.save(os.path.join(save_dir, 'test_categories.npy'), np.concatenate(all_categories, axis=0))

    print(f"OVLAFND 特征已成功保存至 {save_dir}！样本数验证: {len(np.concatenate(all_labels))}")


if __name__ == "__main__":
    extract_features()