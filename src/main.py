import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='ovlafnd_weibo')
parser.add_argument('--dataset', default='weibo')#weibo21 %% weibo
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--max_len', type=int, default=197) # raw is 197
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--bert_vocab_file', default='/home/fxy/chinese-roberta-wwm-ext/vocab.txt')
parser.add_argument('--root_path', default='/home/fxy/project/OVLAFND/weibo_enhance/')#weibo21 %% data
parser.add_argument('--bert', default='/home/fxy/chinese-roberta-wwm-ext')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=3074)
parser.add_argument('--gpu', default='0')
parser.add_argument('--bert_emb_dim', type=int, default=768)
parser.add_argument('--w2v_emb_dim', type=int, default=200)
# parser.add_argument('--lr', type=float, default=0.001) # weibo
parser.add_argument('--lr', type=float, default=0.0001) # weibo21
parser.add_argument('--emb_type', default='bert')
parser.add_argument('--w2v_vocab_file', default='./pretrained_model/w2v/Tencent_AILab_Chinese_w2v_model.kv')
parser.add_argument('--save_param_dir', default= './param_model')
# === [新增] 支持输入不同的留一法文件夹路径 ===
parser.add_argument('--fold_dir', type=str, default='/home/fxy/project/OVLAFND/weibo_enhance/ltdo_7_1_1_experiments/test_政治', help='当前实验的数据文件夹路径')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from run import Run
import torch
import numpy as np
import random

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True


if args.emb_type == 'bert':
    emb_dim = args.bert_emb_dim
    vocab_file = args.bert_vocab_file
elif args.emb_type == 'w2v':
    emb_dim = args.w2v_emb_dim
    vocab_file = args.w2v_vocab_file

print('lr: {}; model name: {}; emb_type: {}; batchsize: {}; epoch: {}; gpu: {}; emb_dim: {}'.format(args.lr, args.model_name, args.emb_type,  args.batchsize, args.epoch, args.gpu, emb_dim))
# print(f"当前 LODO 实验数据目录: {args.fold_dir}")

# 根据数据集类型设置不同的root_path
if args.dataset == "weibo21":
    root_path = '/home/fxy/project/OVLAFND/weibo21_enhance'
elif args.dataset == "weibo":
    root_path = '/home/fxy/project/OVLAFND/weibo_enhance/'

# === [新增] 根据数据集动态设置超参数 ===
if args.dataset == "weibo21":
    current_weight_decay = 5e-5
    current_dropout = 0.2
else:  # weibo 数据集的默认设置
    current_weight_decay = 5e-4
    current_dropout = 0.3
# =======================================
config = {
        'use_cuda': True,

        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'num_workers': args.num_workers,
        'vocab_file': vocab_file,
        'emb_type': args.emb_type,
        'bert': args.bert,
        'root_path': args.root_path,  # 使用根据数据集类型设置的路径
        # === [修改] 把写死的 root_path 换成你传进来的 fold_dir ===
        'fold_dir': args.fold_dir,
        'weight_decay': current_weight_decay,
        'model':
            {
            'mlp': {'dims': [384], 'dropout': current_dropout}
            },
        'emb_dim': emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_param_dir': args.save_param_dir,
        'dataset':args.dataset
        }




if __name__ == '__main__':
    Run(config = config).main()