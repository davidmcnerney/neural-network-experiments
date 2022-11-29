import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torchvision import datasets, transforms


# Reproducibility
torch.manual_seed(2147483647)


# Load training and test data
loading_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
training_dataset = datasets.MNIST(
    "/Users/dave/Temp/datasets/MNIST_TRAIN",
    download=True,
    train=True,
    transform=loading_transform,
)
test_dataset = datasets.MNIST(
    "/Users/dave/Temp/datasets/MNIST_TEST",
    download=True,
    train=False,
    transform=loading_transform,
)
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


# # Inspect
# images, labels = next(iter(training_loader))
# print(images.shape)
# print(labels.shape)
# image = images[0]
# plt.imshow(image.squeeze(), cmap='gray_r')
# plt.show()


