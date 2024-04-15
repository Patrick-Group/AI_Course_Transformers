from torch import nn
import torch
from torchinfo import summary


class Config():
    def __init__(self, embedding_size=16, img_height=28, img_width=28, patch_size=4,):
        self.embedding_size = embedding_size
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.patch_count = self.img_width // self.patch_size


config = Config()


class ViT(nn.Module):
    def __init__(self, config: Config = config):
        super().__init__()
        self.patch_size = config.patch_size
        self.patch_count = config.patch_count
        # 把图片切分成patch，并且每个patch分配16个out_channels.后续准备将每个patch转化为16个特征卷积结果，即16个卷积核得到16个值，这16个值作为一个向量来描述这个patch
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.patch_size ** 2, kernel_size=self.patch_size, padding=0,
                              stride=self.patch_size)
        # 把每个Patch转化成维度为embedding size的向量
        self.patch_emb = nn.Linear(in_features=self.patch_size ** 2, out_features=config.embedding_size)
        self.cls_token = nn.Parameter(torch.rand(1, 1, config.embedding_size))      # 1*1*16
        self.pos_embedding = nn.Parameter(
            torch.rand(1, self.patch_count ** 2 + 1, config.embedding_size)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.embedding_size, nhead=2, batch_first=True,), num_layers=3
        )
        self.cls_linear = nn.Linear(in_features=config.embedding_size, out_features=10)

    def forward(self, x):   # (batch_size,channel=1,width=28,height=28)
        x = self.conv(x)    # (batch_size,channel=16,width=7,height=7)

        x = x.view(x.size(0), x.size(1), self.patch_count ** 2)  # (batch_size,channel=16,seq_len=49)
        x = x.permute(0, 2, 1)      # (batch_size,seq_len=49,channel=16)

        x = self.patch_emb(x)       # (batch_size,seq_len=49,emb_size)

        cls_token = self.cls_token.expand(x.size(0), 1, x.size(2))      # (batch_size,1,emb_size)
        x = torch.cat((cls_token, x), dim=1)                     # add [cls] token
        x = self.pos_embedding + x

        y = self.transformer_encoder(x)
        return self.cls_linear(y[:, 0, :])                              # 对[CLS] token输出做分类


if __name__ == '__main__':
    vit = ViT(config)
    # vit = ViT()

    summary(vit, input_size=(5, 1, 28, 28))
