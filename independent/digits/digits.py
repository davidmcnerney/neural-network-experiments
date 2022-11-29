from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim
import torch.utils.data
from torchvision import datasets, transforms


# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627


# Hyperparameters
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
epochs = 15
learning_rate = 0.003
momentum = 0.9  # not sure what this is for

model_save_path = "/Users/dave/Temp/models/test7c.pt"


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


# Construct neural net
model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], output_size),
    nn.LogSoftmax(dim=1)
)


# Training
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
start_time = datetime.now()
for e in range(epochs):
    running_loss = 0
    for images, labels in training_loader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        #This is where the model learns by backpropagating
        loss.backward()

        #And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(training_loader)))
print("\nTraining time: ", (datetime.now() - start_time))


# Save model
torch.save(model, model_save_path)
