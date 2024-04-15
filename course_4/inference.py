from dataset import MNIST
import matplotlib.pyplot as plt
import torch
from Vit import ViT
import torch.nn.functional as F

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'   # 设备

dataset=MNIST() # 数据集

model=ViT().to(DEVICE) # 模型
model.load_state_dict(torch.load('.\\model.pt'))

model.eval()    # 预测模式

'''
对图片分类
'''
count = 1000
correct = 0
for i in range(count):
    image, label = dataset[i]

    plt.imshow(image.permute(1, 2, 0))
    # plt.show()
    logits = model(image.unsqueeze(0).to(DEVICE))
    print('预测分类:', logits.argmax(-1).item())
    print('正确分类:', label)
    if logits.argmax(-1).item()==label:
        correct += 1.0

print('正确率:', correct/count)