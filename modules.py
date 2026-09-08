import math

import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
import torch.nn.functional as F

import torch
from einops import rearrange
from torch import nn


class SimplifiedLinearAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 focusing_factor=3, kernel_size=5):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.focusing_factor = focusing_factor
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        # self.dwc = nn.Sequential(nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=3,
        #                                    groups=head_dim, padding=1),
        #                          nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=3,
        #                                    groups=head_dim, padding=1)
        #                          )
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, window_size[0] * window_size[1], dim)))

        print('Linear Attention window{} f{} kernel{}'.
              format(window_size, focusing_factor, kernel_size))

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv.unbind(0)
        k = k + self.positional_encoding

        kernel_function = nn.ReLU()
        q = kernel_function(q)
        k = kernel_function(k)

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        with torch.cuda.amp.autocast(enabled=False):
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)

            z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
            if i * j * (c + d) > c * d * (i + j):
                kv = torch.einsum("b j c, b j d -> b c d", k, v)
                x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
            else:
                qk = torch.einsum("b i c, b j c -> b i j", q, k)
                x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)


        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map

        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def eval(self):
        super().eval()
        print('eval')

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, lstm_out):
        # lstm_out: [batch_size, seq_len, hidden_dim]
        attn_weights = F.softmax(self.linear(lstm_out), dim=1)  # 计算注意力权重
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_out)  # [batch_size, hidden_dim]
        return context.squeeze(1), attn_weights  # 返回上下文向量和注意力权重


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        # 如果输入和输出通道不一致，需要调整输入
        self.adjust_input = None
        if in_channels != out_channels:
            self.adjust_input = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 调整输入形状
        if self.adjust_input is not None:
            identity = self.adjust_input(identity)

        out += identity  # Adding the shortcut connection
        out = self.relu(out)
        return out


class BERTBiLSTMCRF(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, tagset_size,dropout=0.1,verbose=False):
        super(BERTBiLSTMCRF, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose=verbose
    def forward(self, evaluate=False, **batch):

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        label_ids = batch['labels']
        if label_ids is not None:
            label_ids = label_ids.to(self.device)
            # BERT 嵌入
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 详细打印
        if self.verbose:
            print("input_ids", input_ids, input_ids.size())  #[batch,max_len]
            print("label_ids", label_ids, label_ids.size())
            print(bert_output.last_hidden_state.shape)
        lstm_input = bert_output.last_hidden_state

        # LSTM
        lstm_out, _ = self.lstm(lstm_input)
        print(lstm_out.shape)
        # Dropout
        lstm_out = self.dropout(lstm_out)

        # 转换成标签空间
        emissions = self.hidden2tag(lstm_out)
        print(emissions.shape)
        if evaluate:
            loss = -self.crf(emissions, label_ids, reduction='mean')
            return self.crf.decode(emissions), loss
        else:
            if label_ids is not None:
                # 如果提供了标签，计算损失
                loss = -self.crf(emissions, label_ids, reduction='mean')
                return loss
            else:
                # 否则返回预测的标签
                return self.crf.decode(emissions)
class BERTBiLSTMCRF_Resnet(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, tagset_size, dropout=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(BERTBiLSTMCRF_Resnet, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        # LSTM定制
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim // 2, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=dropout)

        # ResNet结构
        self.resnet = ResNetBlock(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, evaluate=False, **batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        label_ids = batch['labels'].to(self.device) if 'labels' in batch else None

        # BERT 嵌入
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_input = bert_output.last_hidden_state

        # LSTM
        lstm_out, _ = self.lstm(lstm_input)

        # 使用 ResNet 处理 LSTM 输出
        lstm_out = lstm_out.permute(0, 2, 1)  # 需要调整形状以适配 ResNet
        lstm_out = self.resnet(lstm_out)  # 应用 ResNetBlock
        lstm_out = lstm_out.permute(0, 2, 1)  # 恢复形状回到 [batch, seq_len, hidden_dim]

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # 转换成标签空间
        emissions = self.hidden2tag(lstm_out)

        if evaluate:
            loss = -self.crf(emissions, label_ids, reduction='mean')
            return self.crf.decode(emissions), loss
        else:
            if label_ids is not None:
                # 计算损失
                loss = -self.crf(emissions, label_ids, reduction='mean')
                return loss
            else:
                # 返回预测的标签
                return self.crf.decode(emissions)

class BERTBiLSTMCRF_Resnet_Attention(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, tagset_size, dim=128, num_heads=4, dropout=0.1):
        super(BERTBiLSTMCRF_Resnet_Attention, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # LSTM定制
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim // 2, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=dropout)
        window_size=int(math.sqrt(hidden_dim // 2))
        print('windowsize',window_size)
        # 注意力层
        self.attention = SimplifiedLinearAttention(dim=dim, window_size=(window_size,window_size), num_heads=num_heads)

        # ResNet结构
        self.resnet = ResNetBlock(hidden_dim//2, hidden_dim//2)

        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, evaluate=False, **batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        label_ids = batch['labels'].to(self.device) if 'labels' in batch else None

        # BERT 嵌入
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_input = bert_output.last_hidden_state

        # LSTM
        lstm_out, _ = self.lstm(lstm_input)
        #print('lstm_out',lstm_out.shape)
        # 使用注意力机制
        att_result = self.attention(lstm_out)
        #print('att_result',att_result.shape)
        # ResNet处理上下文向量
        resnet_out = self.resnet(att_result)  # 应用 ResNetBlock
        #print('res_result', resnet_out.shape)
        # Dropout
        resnet_out = self.dropout(resnet_out)

        # 转换成标签空间
        emissions = self.hidden2tag(resnet_out)
        #print('emis',emissions.shape)
        # CRF损失和预测
        if evaluate:
            loss = -self.crf(emissions, label_ids, reduction='mean')
            return self.crf.decode(emissions), loss
        else:
            if label_ids is not None:
                loss = -self.crf(emissions, label_ids, reduction='mean')
                return loss
            else:
                return self.crf.decode(emissions)
