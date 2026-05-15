import os
import tqdm
import torch
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncodingPermute3D
from transformers import BertModel
import torch.nn as nn
import models_mae
from utils.utils import data2gpu, Averager, metrics, Recorder, clipdata2gpu
from utils.utils import metricsTrueFalse
from .layers import *
from .pivot import *
from timm.models.vision_transformer import Block
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import datetime
import torch.nn.functional as F

import matplotlib.pyplot as plt
import json


class DomainAwareTransformer(nn.Module):
    def __init__(self, dim=512, num_heads=8, ffn_dim=2048, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.global_proj = nn.Linear(dim, dim)
        self.weight_proj = nn.Linear(dim, 3)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, modality_reps, domain_rep):
        batch_size = domain_rep.shape[0]

        global_rep = torch.mean(torch.stack(modality_reps, dim=1), dim=1)
        global_rep = self.global_proj(global_rep)

        q = self.q_proj(domain_rep + global_rep).view(batch_size, self.num_heads, self.head_dim)

        k = torch.stack([self.k_proj(rep) for rep in modality_reps], dim=1)
        v = torch.stack([self.v_proj(rep) for rep in modality_reps], dim=1)

        k = k.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 3, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q.unsqueeze(2), k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).squeeze(2)
        out = out.reshape(batch_size, -1)

        domain_rep = domain_rep + self.dropout(out)
        domain_rep = self.norm1(domain_rep)

        ffn_output = self.ffn(domain_rep)

        domain_rep = domain_rep + self.dropout(ffn_output)
        domain_rep = self.norm2(domain_rep)

        weights = self.weight_proj(domain_rep)
        weights = F.softmax(weights, dim=-1)

        return weights


class OVLAFNDMODEL(torch.nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, out_channels, dropout):
        super(OVLAFNDMODEL, self).__init__()
        self.num_expert = 6
        self.task_num = 2
        self.domain_num = self.task_num
        self.gate_num = 3
        self.num_share = 1
        self.unified_dim, self.text_dim = emb_dim, 768
        self.image_dim = 768
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.text_token_len = 197
        self.image_token_len = 197

        text_expert_list = []
        for i in range(self.domain_num):
            text_expert = []
            for j in range(self.num_expert):
                text_expert.append(cnn_extractor(emb_dim, feature_kernel))
            text_expert = nn.ModuleList(text_expert)
            text_expert_list.append(text_expert)
        self.text_experts = nn.ModuleList(text_expert_list)

        image_expert_list = []
        for i in range(self.domain_num):
            image_expert = []
            for j in range(self.num_expert):
                image_expert.append(cnn_extractor(self.image_dim, feature_kernel))
            image_expert = nn.ModuleList(image_expert)
            image_expert_list.append(image_expert)
        self.image_experts = nn.ModuleList(image_expert_list)

        fusion_expert_list = []
        for i in range(self.domain_num):
            fusion_expert = []
            for j in range(self.num_expert):
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       nn.Linear(320, 320),
                                       )
                fusion_expert.append(expert)
            fusion_expert = nn.ModuleList(fusion_expert)
            fusion_expert_list.append(fusion_expert)
        self.fusion_experts = nn.ModuleList(fusion_expert_list)

        final_expert_list = []
        for i in range(self.domain_num):
            final_expert = []
            for j in range(self.num_expert):
                final_expert.append(Block(dim=320, num_heads=8))
            final_expert = nn.ModuleList(final_expert)
            final_expert_list.append(final_expert)
        self.final_experts = nn.ModuleList(final_expert_list)

        text_share_expert, image_share_expert, fusion_share_expert, final_share_expert = [], [], [], []
        for i in range(self.num_share):
            text_share = []
            image_share = []
            fusion_share = []
            final_share = []
            for j in range(self.num_expert * 2):
                text_share.append(cnn_extractor(emb_dim, feature_kernel))
                image_share.append(cnn_extractor(self.image_dim, feature_kernel))
                expert = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       nn.Linear(320, 320),
                                       )
                fusion_share.append(expert)
                final_share.append(Block(dim=320, num_heads=8))
            text_share = nn.ModuleList(text_share)
            text_share_expert.append(text_share)
            image_share = nn.ModuleList(image_share)
            image_share_expert.append(image_share)
            fusion_share = nn.ModuleList(fusion_share)
            fusion_share_expert.append(fusion_share)
            final_share = nn.ModuleList(final_share)
            final_share_expert.append(final_share)
        self.text_share_expert = nn.ModuleList(text_share_expert)
        self.image_share_expert = nn.ModuleList(image_share_expert)
        self.fusion_share_expert = nn.ModuleList(fusion_share_expert)
        self.final_share_expert = nn.ModuleList(final_share_expert)

        image_gate_list, text_gate_list, fusion_gate_list, fusion_gate_list0, final_gate_list = [], [], [], [], []
        for i in range(self.domain_num):
            image_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.unified_dim, self.num_expert * 3),
                                       nn.Dropout(0.2),
                                       nn.Softmax(dim=1)
                                       )
            text_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.unified_dim, self.num_expert * 3),
                                      nn.Dropout(0.2),
                                      nn.Softmax(dim=1)
                                      )
            fusion_gate = nn.Sequential(nn.Linear(self.unified_dim, self.unified_dim),
                                        nn.SiLU(),
                                        nn.Linear(self.unified_dim, self.num_expert * 4),
                                        nn.Dropout(0.2),
                                        nn.Softmax(dim=1)
                                        )
            fusion_gate0 = nn.Sequential(nn.Linear(320, 160),
                                         nn.SiLU(),
                                         nn.Linear(160, self.num_expert * 3),
                                         nn.Dropout(0.2),
                                         nn.Softmax(dim=1)
                                         )
            final_gate = nn.Sequential(nn.Linear(320, 320),
                                       nn.SiLU(),
                                       nn.Linear(320, 160),
                                       nn.SiLU(),
                                       nn.Linear(160, self.num_expert * 3),
                                       nn.Dropout(0.2),
                                       nn.Softmax(dim=1)
                                       )
            image_gate_list.append(image_gate)
            text_gate_list.append(text_gate)
            fusion_gate_list.append(fusion_gate)
            fusion_gate_list0.append(fusion_gate0)
            final_gate_list.append(final_gate)
        self.image_gate_list = nn.ModuleList(image_gate_list)
        self.text_gate_list = nn.ModuleList(text_gate_list)
        self.fusion_gate_list = nn.ModuleList(fusion_gate_list)
        self.fusion_gate_list0 = nn.ModuleList(fusion_gate_list0)
        self.final_gate_list = nn.ModuleList(final_gate_list)

        self.text_attention = MaskAttention(self.unified_dim)
        self.image_attention = TokenAttention(self.unified_dim)
        self.fusion_attention = TokenAttention(self.unified_dim * 2)
        self.final_attention = TokenAttention(320)

        self.text_classifier = MLP(320, mlp_dims, dropout)
        self.image_classifier = MLP(320, mlp_dims, dropout)
        self.fusion_classifier = MLP(320, mlp_dims, dropout)
        num_latent_topics = 16
        self.text_classifier_Mu = TopicPrototypeRouting(320, num_topics=num_latent_topics, dropout=dropout)
        self.image_classifier_Mu = TopicPrototypeRouting(320, num_topics=num_latent_topics, dropout=dropout)
        self.fusion_classifier_Mu = TopicPrototypeRouting(320, num_topics=num_latent_topics, dropout=dropout)
        self.semantic_projector = nn.Sequential(
            nn.Linear(160, 320),
            nn.GELU(),
            nn.LayerNorm(320),
            nn.Linear(320, 512)
        )
        self.max_classifier = MLP(320 * 1, mlp_dims, dropout)
        self.domain_aware_text_classifier = MLP(320 * 1, mlp_dims, dropout)
        self.domain_aware_image_classifier = MLP(320 * 1, mlp_dims, dropout)
        self.domain_aware_fusion_classifier = MLP(320 * 1, mlp_dims, dropout)
        self.domain_aware_total_classifier = MLP(320 * 1, mlp_dims, dropout)
        share_classifier_list = []
        for i in range(self.domain_num):
            share_classifier = MLP(320, mlp_dims, dropout)
            share_classifier_list.append(share_classifier)
        self.share_classifier_list = nn.ModuleList(share_classifier_list)

        dom_classifier_list = []
        for i in range(self.domain_num):
            dom_classifier = MLP(320, mlp_dims, dropout)
            dom_classifier_list.append(dom_classifier)
        self.dom_classifier_list = nn.ModuleList(dom_classifier_list)

        final_classifier_list = []
        for i in range(self.domain_num):
            final_classifier = MLP(320, mlp_dims, dropout)
            final_classifier_list.append(final_classifier)
        self.final_classifier_list = nn.ModuleList(final_classifier_list)

        self.MLP_fusion = MLP_fusion(960, 320, [348], 0.1)
        self.domain_fusion = MLP_fusion(320, 320, [348], 0.1)
        self.MLP_fusion0 = MLP_fusion(768 * 2, 768, [348], 0.1)
        self.clip_fusion = clip_fuion(1024, 320, [348], 0.1)
        self.att_mlp_text = MLP_fusion(320, 2, [174], 0.1)
        self.att_mlp_img = MLP_fusion(320, 2, [174], 0.1)
        self.att_mlp_mm = MLP_fusion(320, 2, [174], 0.1)

        self.model_size = "base"
        self.image_model = models_mae.__dict__["mae_vit_{}_patch16".format(self.model_size)](norm_pix_loss=False)
        self.image_model.cuda()
        checkpoint = torch.load('./mae_pretrain_vit_{}.pth'.format(self.model_size), map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)
        for param in self.image_model.parameters():
            param.requires_grad = False
        self.ClipModel, _ = load_from_name("ViT-B-16", device="cuda", download_root='./')
        self.fake_news_layernorm = LayerNorm(320 * 3, eps=1e-12)
        self.domain_classification_layernorm = LayerNorm(320 * 1, eps=1e-12)
        self.gate_trans = nn.Sequential(
            nn.Linear(320 * 1, 3 * 320, bias=False),
            nn.GELU(),
            nn.Linear(3 * 320, 320 * 1, bias=False),
            nn.GELU(),
        )
        self.query_text = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 1, bias=False)
        )
        self.query_image = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 1, bias=False)
        )
        self.query_fusion = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=-1)

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 320),
            nn.Sigmoid()
        )
        self.gate_text_prefer = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 320),
            nn.Sigmoid()
        )

        self.gate_fusion_prefer = nn.Sequential(
            nn.Linear(320, 320),
            torch.nn.BatchNorm1d(320),
            nn.GELU(),
            nn.Linear(320, 320),
            nn.Sigmoid()
        )

        self.attention = DomainAwareTransformer(dim=320, num_heads=8)
        self.tau = 0.5

        self.negation_adapter = NegationAdapter(input_dim=512)
        self.logic_module = LogicConsistencyModule(img_dim=512, text_dim=512, projection_dim=320)
        self.logic_integrator = nn.Linear(320 + 320 * 4, 320)

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        text_feature = self.bert(inputs, attention_mask=masks)[0]
        image = kwargs['image']
        image_feature = self.image_model.forward_ying(image)

        clip_image = kwargs['clip_image']
        clip_text = kwargs['clip_text']
        clip_content_neg = kwargs.get('clip_content_neg', None)

        with torch.no_grad():
            clip_image_feature = self.ClipModel.encode_image(clip_image)
            clip_text_feature = self.ClipModel.encode_text(clip_text)
            clip_image_feature /= clip_image_feature.norm(dim=-1, keepdim=True)
            clip_text_feature /= clip_text_feature.norm(dim=-1, keepdim=True)

            clip_neg_feature = None
            if clip_content_neg is not None:
                clip_neg_feature = self.ClipModel.encode_text(clip_content_neg)
                clip_neg_feature /= clip_neg_feature.norm(dim=-1, keepdim=True)

        adapted_text_feature = clip_text_feature.float()
        clip_fusion_feature_origin = torch.cat((clip_image_feature.float(), adapted_text_feature), dim=-1)
        clip_fusion_feature_base = self.clip_fusion(clip_fusion_feature_origin)

        logic_logit, logic_feats = self.logic_module(clip_image_feature.float(), adapted_text_feature)

        logic_logit_neg = None
        if clip_neg_feature is not None:
            adapted_neg_feature = clip_neg_feature.float()

            logic_logit_neg, _ = self.logic_module(clip_image_feature.float(), adapted_neg_feature)

        combined_features = torch.cat([clip_fusion_feature_base, logic_feats], dim=-1)
        clip_fusion_feature = torch.relu(self.logic_integrator(combined_features))

        text_atn_feature = self.text_attention(text_feature, masks)
        image_atn_feature, _ = self.image_attention(image_feature)
        fusion_feature = torch.cat((image_feature, text_feature), dim=-1)
        fusion_atn_feature, _ = self.fusion_attention(fusion_feature)
        fusion_atn_feature = self.MLP_fusion0(fusion_atn_feature)

        text_gate_input = text_atn_feature
        image_gate_input = image_atn_feature
        fusion_gate_input = fusion_atn_feature

        text_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.text_gate_list[i](text_gate_input)
            text_gate_out_list.append(gate_out)
        self.text_gate_out_list = text_gate_out_list

        image_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.image_gate_list[i](image_gate_input)
            image_gate_out_list.append(gate_out)
        self.image_gate_out_list = image_gate_out_list

        fusion_gate_out_list = []
        for i in range(self.domain_num):
            gate_out = self.fusion_gate_list[i](fusion_gate_input)
            fusion_gate_out_list.append(gate_out)
        self.fusion_gate_out_list = fusion_gate_out_list

        text_gate_expert_value = []
        text_experts_feature = 0
        text_gate_share_expert_value = []
        for i in range(1):
            gate_expert = 0
            gate_share_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.text_experts[i][j](text_feature)
                gate_expert += (tmp_expert * text_gate_out_list[i][:, j].unsqueeze(1))
            for j in range(self.num_expert * 2):
                tmp_expert = self.text_share_expert[0][j](text_feature)
                gate_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
                gate_share_expert += (tmp_expert * text_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
            text_experts_feature = gate_expert
            text_gate_share_expert_value.append(gate_share_expert)

        att = F.softmax(self.att_mlp_text(text_experts_feature), dim=-1)
        text_experts_feature0 = att[:, 0].view(-1, 1) * text_experts_feature
        text_experts_feature1 = att[:, 1].view(-1, 1) * text_experts_feature
        text_gate_expert_value.append(text_experts_feature0)
        text_gate_expert_value.append(text_experts_feature1)

        image_gate_expert_value = []
        image_experts_feature = 0
        image_gate_share_expert_value = []
        for i in range(1):
            gate_expert = 0
            gate_share_expert = 0
            for j in range(self.num_expert):
                tmp_expert = self.image_experts[i][j](image_feature)
                gate_expert += (tmp_expert * image_gate_out_list[i][:, j].unsqueeze(1))
            for j in range(self.num_expert * 2):
                tmp_expert = self.image_share_expert[0][j](image_feature)
                gate_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
                gate_share_expert += (tmp_expert * image_gate_out_list[i][:, (self.num_expert + j)].unsqueeze(1))
            image_experts_feature = gate_expert
            image_gate_share_expert_value.append(gate_share_expert)

        att = F.softmax(self.att_mlp_img(image_experts_feature), dim=-1)
        image_experts_feature0 = att[:, 0].view(-1, 1) * image_experts_feature
        image_experts_feature1 = att[:, 1].view(-1, 1) * image_experts_feature
        image_gate_expert_value.append(image_experts_feature0)
        image_gate_expert_value.append(image_experts_feature1)

        text = text_gate_share_expert_value[0]
        image = image_gate_share_expert_value[0]
        fusion_share_feature = torch.cat((clip_fusion_feature, text, image), dim=-1)

        fusion_share_feature = self.MLP_fusion(fusion_share_feature)
        fusion_gate_input0 = self.domain_fusion(fusion_share_feature)
        fusion_gate_out_list0 = []
        for k in range(self.domain_num):
            gate_out = self.fusion_gate_list0[k](fusion_gate_input0)
            fusion_gate_out_list0.append(gate_out)
        self.fusion_gate_out_list0 = fusion_gate_out_list0

        fusion_gate_expert_value0 = []
        fusion_experts_feature = 0
        fusion_gate_share_expert_value0 = []
        for m in range(1):
            share_gate_expert0 = 0
            gate_share_expert = 0
            for n in range(self.num_expert):
                fusion_tmp_expert0 = self.fusion_experts[m][n](fusion_share_feature)
                share_gate_expert0 += (fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, n].unsqueeze(1))
            for n in range(self.num_expert * 2):
                fusion_tmp_expert0 = self.fusion_share_expert[0][n](fusion_share_feature)
                share_gate_expert0 += (
                        fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
                gate_share_expert += (
                        fusion_tmp_expert0 * self.fusion_gate_out_list0[m][:, (self.num_expert + n)].unsqueeze(1))
            fusion_gate_share_expert_value0.append(gate_share_expert)
            fusion_experts_feature = fusion_tmp_expert0

        att = F.softmax(self.att_mlp_mm(fusion_experts_feature), dim=-1)
        fusion_experts_feature0 = att[:, 0].view(-1, 1) * fusion_experts_feature
        fusion_experts_feature1 = att[:, 1].view(-1, 1) * fusion_experts_feature
        fusion_gate_expert_value0.append(fusion_experts_feature0)
        fusion_gate_expert_value0.append(fusion_experts_feature1)

        text_two_task = []
        image_two_task = []
        fusion_two_task = []
        text_two_task.append(self.text_classifier(text_gate_expert_value[0]).squeeze(1))
        text_two_task.append(self.text_classifier_Mu(text_gate_expert_value[1]).squeeze(1))
        image_two_task.append(self.image_classifier(image_gate_expert_value[0]).squeeze(1))
        image_two_task.append(self.image_classifier_Mu(image_gate_expert_value[1]).squeeze(1))
        fusion_two_task.append(self.fusion_classifier(fusion_gate_expert_value0[0]).squeeze(1))
        fusion_two_task.append(self.fusion_classifier_Mu(fusion_gate_expert_value0[1]).squeeze(1))

        text_fake_news = torch.softmax(text_two_task[0], -1)
        image_fake_news = torch.softmax(image_two_task[0], -1)
        fusion_fake_news = torch.softmax(fusion_two_task[0], -1)

        # 这里的 multi_domain 现在代表的是 Open Topic 分布 (16维)
        text_multi_domain = text_two_task[1]
        image_multi_domain = image_two_task[1]
        fusion_multi_domain = fusion_two_task[1]

        multi_label_feature = text_gate_expert_value[0] + image_gate_expert_value[0] + fusion_gate_expert_value0[0]
        fake_news_feature = text_gate_expert_value[1] + image_gate_expert_value[1] + fusion_gate_expert_value0[1]

        text_domain_features = text_gate_expert_value[0]
        image_domain_features = image_gate_expert_value[0]
        fusion_domain_features = fusion_gate_expert_value0[0]

        text_domain_features = self.gate_text_prefer(fake_news_feature) * text_domain_features
        image_domain_features = self.gate_image_prefer(fake_news_feature) * image_domain_features
        fusion_domain_features = self.gate_fusion_prefer(fake_news_feature) * fusion_domain_features

        domain_aware_text_view = torch.sigmoid(
            self.domain_aware_text_classifier(text_gate_expert_value[0] + text_domain_features).squeeze())
        domain_aware_image_view = torch.sigmoid(
            self.domain_aware_image_classifier(image_gate_expert_value[0] + image_domain_features).squeeze())
        domain_aware_fusion_view = torch.sigmoid(
            self.domain_aware_fusion_classifier(fusion_gate_expert_value0[0] + fusion_domain_features).squeeze())

        weight_common = self.attention(
            [text_gate_expert_value[0], image_gate_expert_value[0], fusion_gate_expert_value0[0]], multi_label_feature)

        fake_news_sigmoid = weight_common[:, 0].squeeze() * domain_aware_text_view + \
                            weight_common[:, 1].squeeze() * domain_aware_image_view + \
                            weight_common[:, 2].squeeze() * domain_aware_fusion_view

        fake_news_sigmoid = torch.clamp(fake_news_sigmoid, min=0.0, max=1.0)
        expected_topic_feature = torch.matmul(fusion_multi_domain, self.fusion_classifier_Mu.prototypes)
        projected_semantics = self.semantic_projector(expected_topic_feature)
        return fake_news_sigmoid, text_fake_news, text_multi_domain, image_fake_news, image_multi_domain, fusion_fake_news, fusion_multi_domain, domain_aware_text_view, domain_aware_image_view, domain_aware_fusion_view, logic_logit, logic_logit_neg, projected_semantics, fake_news_feature, multi_label_feature

class Trainer():
    def __init__(self, emb_dim, mlp_dims, bert, use_cuda, lr, dropout, train_loader, val_loader, test_loader,
                 category_dict, weight_decay, save_param_dir, loss_weight=[1, 0.006, 0.009, 5e-5], early_stop=5,
                 epoches=100):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout
        if not os.path.exists(save_param_dir):
            self.save_param_dir = os.makedirs(save_param_dir)
        else:
            self.save_param_dir = save_param_dir

            self.history = {
                'epoch': [],
                'total_loss': [],
                'primary_loss': [],
                'logic_loss_total': [],
                'logic_loss_pos': [],
                'logic_loss_neg': [],
                'val_acc': [],
                'val_f1': [],
                'val_details': []
            }
            base_log_dir = '/home/fxy/project/OVLAFND/src/log/weibo'
            run_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.current_run_dir = os.path.join(base_log_dir, f"run_{run_timestamp}")

            if not os.path.exists(self.current_run_dir):
                os.makedirs(self.current_run_dir)

            self.log_file = os.path.join(self.current_run_dir, "training_log.txt")

    def train(self):
        self.model = OVLAFNDMODEL(self.emb_dim, self.mlp_dims, self.bert, 320, self.dropout)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()

        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epoches, eta_min=1e-6)
        recorder = Recorder(self.early_stop)

        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = clipdata2gpu(batch)
                label = batch_data['label']
                label0, text_fake_news, text_multi_domain, image_fake_news, \
                image_multi_domain, fusion_fake_news, fusion_multi_domain, \
                domain_aware_text_view, domain_aware_image_view, domain_aware_fusion_view, \
                logic_logit, logic_logit_neg, projected_semantics,fake_news_feature, multi_label_feature = self.model(**batch_data)
                loss0 = loss_fn(label0, label.float())
                loss12_aux = loss_fn(domain_aware_text_view.squeeze(), label.float())
                loss22_aux = loss_fn(domain_aware_image_view.squeeze(), label.float())
                loss32_auc = loss_fn(domain_aware_fusion_view.squeeze(), label.float())
                num_topics = 16
                text_topic_log_prob = torch.log(text_multi_domain + 1e-8)
                image_topic_log_prob = torch.log(image_multi_domain + 1e-8)
                fusion_topic_log_prob = torch.log(fusion_multi_domain + 1e-8)
                uniform_target = torch.ones_like(text_multi_domain, dtype=torch.float).cuda() / num_topics
                loss12 = F.kl_div(text_topic_log_prob, uniform_target, reduction='batchmean')
                loss22 = F.kl_div(image_topic_log_prob, uniform_target, reduction='batchmean')
                loss32 = F.kl_div(fusion_topic_log_prob, uniform_target, reduction='batchmean')
                loss_topic_consistency = F.mse_loss(text_multi_domain, image_multi_domain) + \
                                         F.mse_loss(fusion_multi_domain, text_multi_domain)
                smooth_factor = 0.05
                smooth_target = torch.where(
                    label.float() == 1,
                    torch.tensor(1.0 - smooth_factor).cuda(),
                    torch.tensor(smooth_factor).cuda()
                )
                loss_logic_abs = torch.nn.functional.binary_cross_entropy_with_logits(
                    logic_logit.squeeze(), smooth_target
                )
                loss_logic_rel = torch.tensor(0.0).cuda()
                if logic_logit_neg is not None and 'enhanced' in batch_data:
                    enhanced_mask = batch_data['enhanced'].float()
                    if enhanced_mask.sum() > 0:
                        m_base = 0.6
                        m_bonus = 0.2
                        dynamic_margin = m_base + (1.0 - label.float()) * m_bonus
                        contrastive_loss = F.relu(
                            logic_logit.squeeze() - logic_logit_neg.squeeze() + dynamic_margin
                        )
                        loss_logic_rel = (contrastive_loss * enhanced_mask).sum() / (enhanced_mask.sum() + 1e-8)
                lambda_abs_rel = 0.5
                total_logic_loss = loss_logic_abs + lambda_abs_rel * loss_logic_rel
                lambda_logic = 0.2 * min(1.0, epoch / 10.0)
                loss_semantic_align = torch.tensor(0.0).cuda()
                loss_proto_ortho = torch.tensor(0.0).cuda()
                if 'llm_topic_embs' in batch_data:
                    llm_semantics = batch_data['llm_topic_embs'].cuda()
                    proj_norm = F.normalize(projected_semantics, p=2, dim=-1)
                    llm_norm = F.normalize(llm_semantics, p=2, dim=-1)
                    temperature = 0.07
                    sim_matrix = torch.matmul(proj_norm, llm_norm.T) / temperature
                    labels = torch.arange(proj_norm.size(0)).cuda()
                    loss_semantic_align = F.cross_entropy(sim_matrix, labels)
                    protos = self.model.fusion_classifier_Mu.prototypes
                    protos_norm = F.normalize(protos, p=2, dim=-1)
                    proto_sim = torch.matmul(protos_norm, protos_norm.T)
                    eye_matrix = torch.eye(protos_norm.size(0)).cuda()
                    loss_proto_ortho = F.mse_loss(proto_sim, eye_matrix)
                loss = loss0 + \
                       (loss12_aux + loss22_aux + loss32_auc) / 3.0 + \
                       0.1 * (loss12 + loss22 + loss32) + \
                       0.5 * loss_topic_consistency + \
                       lambda_logic * total_logic_loss
                loss += 0.1 * loss_semantic_align
                loss += 0.05 * loss_proto_ortho
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                if (scheduler is not None):
                    scheduler.step()
                avg_loss.add(loss.item())
            print('Epoch {}; Total: {:.4f}; Logic(Pos): {:.4f}; Logic(Neg): {:.4f}'.format(
                epoch + 1, avg_loss.item(), loss_logic_abs.item(), loss_logic_rel.item()))

            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {epoch + 1}:\n")
                f.write(f"  Total Loss: {avg_loss.item()}\n")
                f.write(f"  Primary Loss (loss0): {loss0.item()}\n")
                f.write(
                    f"  Logic Loss: {total_logic_loss.item()} (Pos: {loss_logic_abs.item()}, Neg: {loss_logic_rel.item()})\n")
                f.write("\n")
            print("----- self.save_param_dir", self.save_param_dir)
            results0 = self.test(self.val_loader)
            try:
                val_acc = results0.get('acc', 0)
                val_f1 = results0.get('metric', 0)
            except AttributeError:
                val_acc = 0
                val_f1 = 0

            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            import copy
            self.history['val_details'].append(copy.deepcopy(results0))
            self.history['epoch'].append(epoch + 1)
            self.history['total_loss'].append(avg_loss.item())
            self.history['primary_loss'].append(loss0.item())
            self.history['logic_loss_total'].append(total_logic_loss.item())
            self.history['logic_loss_pos'].append(loss_logic_abs.item())
            self.history['logic_loss_neg'].append(loss_logic_rel.item())
            self.save_training_report()
            mark = recorder.add(results0)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"=== Epoch {epoch + 1} Validation Results ===\n")
                f.write(f"  Macro-F1: {val_f1:.4f} | Accuracy: {val_acc:.4f}\n")
                f.write(f"  [Domain Details]:\n")
                for key, value in results0.items():
                    if isinstance(value, dict):
                        f.write(f"    {key}: {value}\n")
                if mark == 'save':
                    f.write(f"  >>> 🌟 【突破记录】发现更高 F1，保存该 Epoch 为最佳参数！\n")
                elif mark == 'esc':
                    f.write(f"  >>> 🛑 【Early Stopping】触发早停！最佳模型锁定在 Epoch {recorder.maxindex}。\n")
                else:
                    f.write(f"  >>> 📉 【未提升】模型未达到历史最佳，继续训练...\n")

                f.write("-" * 50 + "\n\n")
            if mark == 'save':
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_param_dir, 'parameter_ovlafnd_weibo.pkl'))
            elif mark == 'esc':
                break
            else:
                continue
        self.model.load_state_dict(torch.load(os.path.join(self.save_param_dir, 'parameter_ovlafnd_weibo.pkl')))
        print("开始进行最后的测试")
        results0 = self.test(self.test_loader)
        print("final: ", results0)

        with open(self.log_file, 'a') as f:
            f.write("Final Test Results:\n")
            f.write(str(results0) + "\n")

        return results0, os.path.join(self.save_param_dir, 'parameter_ovlafnd_weibo.pkl')

    def save_training_report(self):
        import matplotlib.pyplot as plt
        import json

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['epoch'], self.history['total_loss'], label='Total Loss', color='blue')
        plt.plot(self.history['epoch'], self.history['primary_loss'], label='Primary Task Loss', color='red',
                 linestyle='--')
        plt.title('Training Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(self.history['epoch'], self.history['logic_loss_total'], label='Total Logic Loss', color='purple')
        plt.plot(self.history['epoch'], self.history['logic_loss_pos'], label='Positive Logic', color='green',
                 alpha=0.6)
        plt.plot(self.history['epoch'], self.history['logic_loss_neg'], label='Negative Logic', color='orange',
                 alpha=0.6)
        plt.title('Logic Consistency Loss Analysis')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.current_run_dir, 'loss_analysis.png'))
        plt.close()
        plt.figure(figsize=(8, 5))
        plt.plot(self.history['epoch'], self.history['val_acc'], label='Val Accuracy', marker='o')
        plt.plot(self.history['epoch'], self.history['val_f1'], label='Val F1', marker='s')
        plt.title('Validation Performance')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.current_run_dir, 'metric_curve.png'))
        plt.close()
        report_path = os.path.join(self.current_run_dir, 'detailed_report.txt')
        curr_loss0 = self.history['primary_loss'][-1]
        curr_logic_neg = self.history['logic_loss_neg'][-1]
        best_f1 = max(self.history['val_f1']) if self.history['val_f1'] else 0
        analysis = []
        run_time = os.path.basename(self.current_run_dir).replace('run_', '')
        analysis.append(f"=== OVLAFND Training Report ({run_time}) ===")
        analysis.append(f"Current Epoch: {self.history['epoch'][-1]}")
        analysis.append(f"Best Validation F1: {best_f1:.4f}")
        analysis.append("-" * 30)
        if curr_loss0 < 0.001:
            analysis.append("[WARNING] Primary Loss is extremely low (<0.001). Potential Overfitting to Training Data!")
            analysis.append(
                "Recommendation: Increase Dropout or Weight Decay, or check if Validation Score is plateauing.")
        if curr_logic_neg < 0.01:
            analysis.append(
                "[WARNING] Logic Negative Loss is near zero. Model might be collapsing (not learning contrast).")
        elif curr_logic_neg > 0.5:
            analysis.append(
                "[INFO] Logic Negative Loss is high. The model is struggling to distinguish Hard Negatives (Good for learning).")
        else:
            analysis.append("[OK] Logic Module is working within expected range.")

        analysis.append("-" * 30)
        analysis.append("History Data (Last 5 Epochs):")
        for i in range(max(0, len(self.history['epoch']) - 5), len(self.history['epoch'])):
            analysis.append(
                f"Epoch {self.history['epoch'][i]}: Total={self.history['total_loss'][i]:.4f} | Val F1={self.history['val_f1'][i]:.4f}")

        with open(report_path, 'w') as f:
            f.write('\n'.join(analysis))

        with open(os.path.join(self.current_run_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f)

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = clipdata2gpu(batch)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                outputs = self.model(**batch_data)
                batch_label_pred = outputs[0]
                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        metric_res = metricsTrueFalse(label, pred, category, self.category_dict)
        return metric_res
