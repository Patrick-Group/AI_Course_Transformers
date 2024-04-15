from torch.utils.data import Dataset
from torchvision.transforms.v2 import PILToTensor, Compose
import torchvision


class MNIST(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        self.dataset = torchvision.datasets.MNIST('.\\mnist\\', train=is_train, download=True)
        self.img_convert = Compose([
            PILToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    # Normalization
    def __getitem__(self, item):
        img, label = self.dataset[item]
        return self.img_convert(img) / 255.0, label


# 手写数字
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = MNIST()
    img, label = data[1]
    print(label)
    plt.imshow(img.permute(1, 2, 0))
    # plt.imshow(img)
    plt.show()