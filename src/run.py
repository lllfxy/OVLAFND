import os
from utils.clip_dataloader import bert_data as weibo_data
from utils.weibo21_clip_dataloader import bert_data as weibo21_data
from model.ovlafnd import Trainer as OVLAFNDTrainer
from model.ovlafnd_weibo import Trainer as OVLAFNDWeiboTrainer

import torch
class Run():
    def __init__(self,
                 config
                 ):
        self.configinfo = config
        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_type = config['emb_type']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert = config['bert']
        self.root_path = config['root_path']
        #
        self.fold_dir = config['fold_dir']
        #
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']
        self.dataset = config['dataset']
        if config['dataset']=="weibo":
            self.root_path = '/home/fxy/project/OVLAFND/weibo_enhance/'

            self.train_path = self.root_path + 'train_2_domain.csv'  
            self.val_path = self.root_path + 'val_2_domain.csv'  
            self.test_path = self.root_path + 'test_2_domain.csv' 
            self.category_dict = {
                "经济": 0,
                "健康": 1,
                "军事": 2,
                "科学": 3,
                "政治": 4,
                "国际": 5,
                "教育": 6,
                "娱乐": 7,
                "社会": 8
            }

            # self.train_path = os.path.join(self.fold_dir, 'train.csv')
            # self.val_path = os.path.join(self.fold_dir, 'val.csv')
            # self.test_path = os.path.join(self.fold_dir, 'test.csv')
            # self.category_dict = {
            #     "经济": 0,
            #     "健康": 1,
            #     "军事": 2,
            #     "科学": 3,
            #     "政治": 4,
            #     "国际": 5,
            #     "教育": 6,
            #     "娱乐": 7,
            #     "社会": 8
            # }
        if config['dataset']=="weibo21":
            self.root_path = '/home/fxy/project/OVLAFND/weibo21_enhance/'

            self.train_path = self.root_path + 'train_2_domain.xlsx'#weibo21
            self.val_path = self.root_path + 'val_2_domain.xlsx'#weibo21
            self.test_path = self.root_path + 'test_2_domain.xlsx'#weibo21
            self.category_dict = {
                "科技": 0,
                "军事": 1,
                "教育考试": 2,
                "灾难事故": 3,
                "政治": 4,
                "医药健康": 5,
                "财经商业": 6,
                "文体娱乐": 7,
                "社会生活": 8
            }

            # self.train_path = os.path.join(self.fold_dir, 'train.xlsx')
            # self.val_path = os.path.join(self.fold_dir, 'val.xlsx')
            # self.test_path = os.path.join(self.fold_dir, 'test.xlsx')
            # self.category_dict = {
            #     "科技": 0, "军事": 1, "教育考试": 2, "灾难事故": 3,
            #     "政治": 4, "医药健康": 5, "财经商业": 6, "文体娱乐": 7, "社会生活": 8
            # }

    def get_dataloader(self, dataset):
        if self.emb_type == 'bert':
            if dataset == "weibo":
                loader = weibo_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                                    category_dict=self.category_dict, num_workers=self.num_workers)
            if dataset == "weibo21":
                loader = weibo21_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                                      category_dict=self.category_dict, num_workers=self.num_workers)

        # clip_weibo
        if dataset == "weibo":
            train_loader = loader.load_data(self.train_path, './data/train_loader.pkl',
                                            './data/train_clip_loader.pkl', True)
            val_loader = loader.load_data(self.val_path, './data/val_loader.pkl', './data/val_clip_loader.pkl', False)
            test_loader = loader.load_data(self.test_path, './data/test_loader.pkl', './data/test_clip_loader.pkl',
                                           False)

        if dataset == "weibo21":
            train_loader = loader.load_data(self.train_path, './weibo21/train_loader.pkl',
                                            './weibo21/train_clip_loader.pkl', True)
            val_loader = loader.load_data(self.val_path, './weibo21/val_loader.pkl', './weibo21/val_clip_loader.pkl',
                                          False)
            test_loader = loader.load_data(self.test_path, './weibo21/test_loader.pkl',
                                           './weibo21/test_clip_loader.pkl', False)


        return train_loader, val_loader, test_loader

    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        train_loader, val_loader, test_loader = self.get_dataloader(self.dataset)
        if self.model_name == 'ovlafnd':
            trainer = OVLAFNDTrainer(
                emb_dim=self.emb_dim, mlp_dims=self.mlp_dims, bert=self.bert,
                use_cuda=self.use_cuda, lr=self.lr, train_loader=train_loader,
                dropout=self.dropout, weight_decay=self.weight_decay,
                val_loader=val_loader, test_loader=test_loader,
                category_dict=self.category_dict, early_stop=self.early_stop, epoches=self.epoch,
                save_param_dir=os.path.join(self.save_param_dir, self.model_name)
            )

        elif self.model_name == 'ovlafnd_weibo':
            trainer = OVLAFNDWeiboTrainer(  
                emb_dim=self.emb_dim, mlp_dims=self.mlp_dims, bert=self.bert,
                use_cuda=self.use_cuda, lr=self.lr, train_loader=train_loader,
                dropout=self.dropout, weight_decay=self.weight_decay,
                val_loader=val_loader, test_loader=test_loader,
                category_dict=self.category_dict, early_stop=self.early_stop, epoches=self.epoch,
                save_param_dir=os.path.join(self.save_param_dir, self.model_name)
            )

        else:
            raise ValueError(f"Model name '{self.model_name}' is not recognized. Check your arguments.")

        trainer.train()
