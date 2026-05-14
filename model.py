# 存放模型类
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import MultiheadAttention


# 定义DyT 动态双曲正切激活函数 替代 层归一化
class DyT(nn.Module):
    def __init__(self, C, init_alpha=1.0, eps=1e-3):     # C为输入维度 即 特征数
        super().__init__()
        # 初始化可学习参数
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)  # 缩放参数
        self.r = nn.Parameter(torch.ones(C))          # 缩放系数
        self.b = nn.Parameter(torch.zeros(C))         # 偏置项
        self.eps = eps #用于层归一化中 防止方差为0，除0异常的小值，因为直接在框架中修改，需要有这个参数，但不会用到

    def forward(self, x):
        # 应用 tanh 激活函数，并使用 α 进行缩放
        x = torch.tanh(self.alpha * x)
        # 应用仿射变换：γ * x + β
        return self.r * x + self.b
    


# 自定义Transformer编码器层
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__(d_model, nhead, dim_feedforward, dropout)

        # #在MultiheadAttention内部的计算：
        # Q = src × Wq  # Query矩阵
        # K = src × Wk  # Key矩阵  
        # V = src × Wv  # Value矩阵

        # # 计算注意力分数（未归一化）
        # attention_scores = Q × K^T / √d_k

        # # 归一化得到注意力权重
        # attn_weights = softmax(attention_scores)

        # # 加权求和得到输出
        # attn_output = attn_weights × V

        # 替换原始的MultiheadAttention为可返回注意力权重的版本
        self.self_attn = MultiheadAttention(
            embed_dim=d_model,  #输入和输出的维度
            num_heads=nhead,    #注意力头的数量
            dropout=dropout,    #dropout比例
            batch_first=True    #指定输入张量的第一个维度是batch
        )

        self.norm1 = DyT(d_model,init_alpha=1.0) # 替换为 层归一化 为 DyT 激活函数
        self.norm2 = DyT(d_model,init_alpha=1.0) # 替换为 层归一化 为 DyT 激活函数

    def forward(self, src, src_mask=None, src_key_padding_mask=None): #src 即输入的数据 x

        # 计算输出时返回注意力权重：attn_output输出，attn_weights为注意力权重
        attn_output, attn_weights = self.self_attn(
            src, src, src,       # 三个src输入后，分别乘以 query/key/value对应的矩阵(Wq,Wk,Wv)，得到Q/K/V
            attn_mask=src_mask,  # 用于屏蔽无效位置的布尔掩码
            key_padding_mask=src_key_padding_mask,  # 用于屏蔽padding位置的掩码
            need_weights=True,   # 显式要求返回注意力权重
            average_attn_weights=False  # 保留各注意力头的独立权重
        )

        src1 = src + self.dropout1(attn_output) #将注意力输出attn_output经过Dropout后与输入src相加，实现残差连接
        src1 = self.norm1(src1) #将残差连接后的结果归一化


        # 前馈网络部分也添加残差连接
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src1))))
        src3 = src1 + self.dropout2(src2)
        src3 = self.norm2(src3)

        return src3, attn_weights  # 返回 编码 和 注意力权重


# 自定义Transformer编码器
class CustomTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src  # 初始化输出为输入
        all_attn_weights = []  # 存储所有注意力权重
        for mod in self.layers:  # 遍历所有层
            output, attn_weights = mod( # 调用层的forward方法
                output,
                src_mask=mask, # 掩码为None 表示不使用掩码 只用编码器
                src_key_padding_mask=src_key_padding_mask # 掩码为None 表示不使用掩码
            )
            all_attn_weights.append(attn_weights)
        return output, all_attn_weights


# nhead 为注意力头的数量，同时并行做，消除Wq,Wk,Wv的权重矩阵的初始值的影响
# 定义模型
class MultiOmicsNet(nn.Module):
    def __init__(self, n_omics, d_model, nhead, dim_feedforward, dropout, input_dims):
        super().__init__()
#------------------------------------单个组学的特征提取------------------------------------
        # 为每个组学创建独立的特征投影层
        self.feature_projections_per_omics = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, d_model),  # 投影层
                # nn.LayerNorm(d_model),
                DyT(d_model,init_alpha=1.0),  # 替换为 DyT 激活函数
                nn.ReLU()   # 激活函数  学习非线性知识
            )
            for dim in input_dims   
        ])

        # 为每个组学创建独立的位置编码
        self.position_encodings_per_omics = nn.ParameterList([
            nn.Parameter(torch.randn(1, d_model))  # 形状为 (1, d_model) batchsize为1，加上位置编码时，会自动广播到实际的batch size
            for dim in input_dims                   # 可学习的位置编码 与传统的transformer不同
        ])

        # 初始化Transformer编码器层
        encoder_layer = CustomTransformerEncoderLayer(
                d_model=d_model, # 输入和输出的维度
                nhead=nhead,     # 注意力头的数量
                dim_feedforward=dim_feedforward, # 前馈神经网络的维度
                dropout=dropout, # dropout比例
                )
        # 为每个组学创建独立的Transformer编码器
        self.feature_encoders_per_omics = nn.ModuleList([
                # 使用自定义的Encoder层
                CustomTransformerEncoder(encoder_layer, 1) # 这个编码器层中只有一层layer即可
            for dim in input_dims  # input_dims应包含各个组学的特征维度
        ])
#------------------------------------全部组学的特征提取------------------------------------

        self.feature_projections_all_omics = nn.Sequential(
            nn.Linear(d_model, d_model),                                         
            # nn.LayerNorm(d_model), 
            DyT(d_model,init_alpha=1.0),
            nn.ReLU()
        )

        self.position_encodings_all_omics = nn.Parameter(torch.randn(1, n_omics, d_model))
        # 使用前面的encoder_layer即可
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=2)

        self.logits_predictor = nn.Sequential(
            nn.Linear(d_model * n_omics, 128),  #融合所有组学的编码 
            # nn.LayerNorm(128) , # 层归一化处理  消除数据偏移
            DyT(128,init_alpha=1.0),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_list):

        batch_size = x_list[0].shape[0]
        
#------------------------------------单个组学的特征提取------------------------------------

        # 分别处理每个组学特征
        encoded_features = []   # 存储每个组学编码后的特征
        attn_weights_list_per_omics = []  # 存储每个组学的单独的注意力权重
        for i, x in enumerate(x_list):

            # 将输入投影到 d_model 维度
            x_projected = self.feature_projections_per_omics[i](x)

            # 添加位置编码
            x_with_position = x_projected + self.position_encodings_per_omics[i]

            # 通过 Transformer 编码器
            encoded, all_attn_weights = self.feature_encoders_per_omics[i](x_with_position) #获得编码后的特征和注意力权重
            encoded_features.append(encoded)
            attn_weights_list_per_omics.append(all_attn_weights)


#------------------------------------全部组学的特征提取------------------------------------

        # 特征融合 将五个组学的数据，作为五个通道输入到transformer中  将batchsize放在第一维，通道放在第二维，特征放在第三维
        encoded_features = torch.stack(encoded_features, dim=1)
        
        feature_projections = self.feature_projections_all_omics(encoded_features)
        # 添加位置编码
        feature_projections_with_position = feature_projections + self.position_encodings_all_omics
        # 通过 Transformer 编码器
        features, attn_weights_list_all_omics = self.transformer_encoder(feature_projections_with_position)

        # 拼接所有组学的输出
        features = features.view(batch_size, -1) # 将所有组学的输出拼接成一个向量 即平滑操作  d_model*n_omics

        # 预测logits
        logits = self.logits_predictor(features)

        # 返回logits, 注意力权重 ,单个组学被编码器捕获的特征，融合被捕获的特征再被编码器捕获的特征
        return logits,attn_weights_list_all_omics,encoded_features,features

