import  torch
from torchvision.models import resnet18
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn
from torch.autograd import Function

# model/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F



class TopicPrototypeRouting(torch.nn.Module):
    def __init__(self, input_dim, num_topics=16, dropout=0.1):
        super(TopicPrototypeRouting, self).__init__()
        self.num_topics = num_topics

        # 降维并提取主题特征
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 可学习的主题原型 (Topic Prototypes)
        self.prototypes = nn.Parameter(torch.randn(num_topics, input_dim // 2))
        nn.init.xavier_uniform_(self.prototypes)

        self.tau = 0.1  # 温度系数，让分配更锐利

    def forward(self, x):
        feat = self.feature_proj(x)  # [batch, dim]
        feat_norm = F.normalize(feat, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)

        # 计算余弦相似度并缩放
        sim = torch.matmul(feat_norm, proto_norm.transpose(0, 1)) / self.tau

        # 输出软分配的主题概率分布 [batch, num_topics]
        topic_distribution = F.softmax(sim, dim=-1)
        return topic_distribution

class OpenTopicMemory(nn.Module):
    def __init__(self, feature_dim=320, num_prototypes=16, tau=0.1):
        super(OpenTopicMemory, self).__init__()
        self.tau = tau
        self.num_prototypes = num_prototypes

        # Initialize K learnable prototypes
        self.prototypes = nn.Parameter(torch.Tensor(num_prototypes, feature_dim))
        # Orthogonal initialization is CRITICAL here to prevent prototype collapse
        nn.init.orthogonal_(self.prototypes)

    def forward(self, x):
        # x shape: [batch, feature_dim]
        x_norm = F.normalize(x, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)

        # Calculate similarity and soft assignment
        sim = torch.matmul(x_norm, p_norm.T) / self.tau
        soft_weights = F.softmax(sim, dim=-1)

        # Reconstruct the continuous topic representation
        topic_rep = torch.matmul(soft_weights, p_norm)
        return topic_rep, soft_weights

class NegationAdapter(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=None, dropout=0.1):
        super(NegationAdapter, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 4

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        # 严格零初始化：保证 Epoch 1 时完全不干扰 CLIP 原始特征
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        delta = self.adapter(x)
        return self.norm(x + delta)



class LogicConsistencyModule(torch.nn.Module):
    def __init__(self, img_dim, text_dim, projection_dim=320, dropout=0.1):
        super(LogicConsistencyModule, self).__init__()

        # 将图文投影到同一维度 (带 LayerNorm 保证特征稳定)
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )

        # 经典 NLI 推理网络
        self.logic_reasoning = nn.Sequential(
            nn.Linear(projection_dim * 4, projection_dim),
            nn.LayerNorm(projection_dim),  # 必须加 LayerNorm 防止特征溢出
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, 1)
        )

        # 【核心秘诀】：极小值正态分布初始化
        # 这确保了初始的 logic_logit 都在 0 附近。
        # 因此 Epoch 1 的 Loss 会稳稳地落在 0.48 左右，然后慢慢下降！
        nn.init.normal_(self.logic_reasoning[-1].weight, std=0.01)
        nn.init.zeros_(self.logic_reasoning[-1].bias)

    def forward(self, img_emb, text_emb):
        i_vec = self.img_proj(img_emb)
        t_vec = self.text_proj(text_emb)

        # 构造交互特征 (NLI 标准做法)
        diff = torch.abs(i_vec - t_vec)
        mul = i_vec * t_vec

        cat_features = torch.cat([i_vec, t_vec, diff, mul], dim=-1)
        logic_score = self.logic_reasoning(cat_features)

        return logic_score, cat_features

class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class MLP(torch.nn.Module):
    def __init__(self,input_dim,embed_dims,dropout):
        super(MLP, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim,1))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)

class MLP_Mu(torch.nn.Module):
    def __init__(self,input_dim,embed_dims,dropout):
        super(MLP_Mu, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim,9))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)

class MLP_fusion(torch.nn.Module):
    def __init__(self,input_dim,out_dim,embed_dims,dropout):
        super(MLP_fusion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim,out_dim))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)
"""
class MLP_fusion_gate(torch.nn.Module):
    def __init__(self,input_dim,embed_dims,dropout):
        super(MLP_fusion_gate, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim,768))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)
"""
class clip_fuion(torch.nn.Module):
    def __init__(self,input_dim,out_dim,embed_dims,dropout):
        super(clip_fuion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim,embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim,out_dim))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self,x):
        return self.mlp(x)



class cnn_extractor(torch.nn.Module):
    def __init__(self,input_size,feature_kernel):
        super(cnn_extractor, self).__init__()
        self.convs =torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size,feature_num,kernel)
            for kernel,feature_num in feature_kernel.items()]
        )
    def forward(self,input_data):
        input_data = input_data.permute(0, 2, 1)
        feature = [conv(input_data)for conv in self.convs]
        feature = [torch.max_pool1d(f,f.shape[-1])for f in feature]
        feature = torch.cat(feature,dim = 1)
        feature = feature.view([-1,feature.shape[1]])
        return feature

class image_cnn_extractor(nn.Module):
    def __init__(self):
        super(image_cnn_extractor, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(197, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 24 * 96, 512)
        self.fc2 = nn.Linear(512, 320)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolutional layers with ReLU activation and pooling
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten the output for fully connected layers
        x = x.view(-1, 256 * 24 * 96)

        # Fully connected layers with ReLU activation
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class image_extractor(torch.nn.Module):
    def __init__(self,out_channels):
        super(image_extractor, self).__init__()
        self.img_backbone = resnet18(pretrained=True)
        self.img_model = torch.nn.ModuleList([
            self.img_backbone.conv1,
            self.img_backbone.bn1,
            self.img_backbone.relu,
            self.img_backbone.layer1,
            self.img_backbone.layer2,
            self.img_backbone.layer3,
            self.img_backbone.layer4
        ])
        self.img_model = torch.nn.Sequential(*self.img_model)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.img_fc = torch.nn.Linear(self.img_backbone.inplanes, out_channels)
    def forward(self,img):
        n_batch = img.size(0)
        img_out = self.img_model(img)
        img_out = self.avg_pool(img_out)#([64, 512, 1, 1])
        img_out = img_out.view(n_batch, -1)#([64, 512])
        img_out = self.img_fc(img_out)#([64, 320])
        img_out = F.normalize(img_out, p=2, dim=-1)
        return img_out

class classifier(torch.nn.Module):
    def __init__(self,out_dim=1):
        super(classifier, self).__init__()
        self.trim = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            # SimpleGate(),
            # nn.BatchNorm1d(64),
            # nn.Dropout(0.2),
        )
        self.classifier1 = nn.Sequential(
            nn.Linear(64, out_dim),
        )
    def forward(self,x):
        x = self.classifier1(self.trim(x))
        return x

class MaskAttention(torch.nn.Module):
    def __init__(self,input_dim):
        super(MaskAttention, self).__init__()
        self.Line = torch.nn.Linear(input_dim,1)

    def forward(self,input,mask):
        score = self.Line(input).view(-1,input.size(1))
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        score = torch.softmax(score, dim=-1).unsqueeze(1)
        output = torch.matmul(score,input).squeeze(1)
        return output

class TokenAttention(torch.nn.Module):
    """
    Compute attention layer
    """

    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
                            torch.nn.Linear(input_shape, input_shape),
                            nn.SiLU(),
                            #SimpleGate(dim=2),
                            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        #scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        scores = scores.unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        # scores = self.attention_layer(inputs)
        # outputs = scores*inputs
        return outputs, scores
class Attention(torch.nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
class MultiHeadedAttention(torch.nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = torch.nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn
class Resnet(torch.nn.Module):
    def __init__(self,out_channels):
        super(Resnet, self).__init__()
        self.img_backbone = resnet18(pretrained=True)
        self.img_model = torch.nn.ModuleList([
            self.img_backbone.conv1,
            self.img_backbone.bn1,
            self.img_backbone.relu,
            self.img_backbone.layer1,
            self.img_backbone.layer2,
            self.img_backbone.layer3,
            self.img_backbone.layer4
        ])
        self.img_model = torch.nn.Sequential(*self.img_model)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.img_fc = torch.nn.Linear(self.img_backbone.inplanes, out_channels)

    def forward(self, img):
        n_batch = img.size(0)
        img_out = self.img_model(img)
        img_out = self.avg_pool(img_out)
        img_out = img_out.view(n_batch, -1)
        img_out = self.img_fc(img_out)
        img_out = F.normalize(img_out, p=2, dim=-1)
        return img_out

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None