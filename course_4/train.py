import torch
from dataset import MNIST
from Vit import ViT
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

if __name__=='__main__':
    # 如果使用Mac并有MPS，这里改成MPS
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 选择对应的设备

    dataset = MNIST()           # 载入数据集
    model = ViT().to(DEVICE)    # 实例化模型

    try:  # 加载模型
        model.load_state_dict(torch.load('.model.pt'))
    except:
        pass

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # 选择优化器

    EPOCH = 30
    BATCH_SIZE = 64     # 从batch内选出10个不一样的数字

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, persistent_workers=True)      # 数据加载器

    iter_count = 0
    for epoch in range(EPOCH):
        for imgs, labels in dataloader:
            logits = model(imgs.to(DEVICE))

            loss = F.cross_entropy(logits, labels.to(DEVICE))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_count % 1000 == 0:
                print('epoch:{} iter:{},loss:{}'.format(epoch, iter_count, loss))
                torch.save(model.state_dict(), '.model.pt')
                os.replace('.model.pt', 'model.pt')
            iter_count += 1
