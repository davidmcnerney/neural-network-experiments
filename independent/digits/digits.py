from datetime import datetime

# import matplotlib.pyplot as plt
# import torch
from torch import nn
from torch.optim import SGD
import torch.utils.data
from torchvision import datasets, transforms


# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627


# Hyperparameters
do_training = True
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
epochs = 60
batch_size = 32
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
training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# # Inspect data
# images, labels = next(iter(training_loader))
# print(images.shape)
# print(labels.shape)
# image = images[0]
# plt.imshow(image.squeeze(), cmap='gray_r')
# plt.show()


if do_training:
    # Construct neural net
    model = nn.Sequential(
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], output_size),
        nn.LogSoftmax(dim=1)
    )

    # Train the model
    print("Training ...")
    criterion = nn.NLLLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    start_time = datetime.now()
    for epoch_num in range(epochs):
        # print(f"   starting epoch {epoch_num}")
        epoch_total_loss = 0
        for images, labels in training_loader:
            # print(f"      got {len(images)} images from training loader")
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_total_loss += loss.item()
        print(f"      epoch {epoch_num} training loss {epoch_total_loss / len(training_loader)}")
    print(f"Training time: {datetime.now() - start_time}")
    print("")

    # Save model
    torch.save(model, model_save_path)
else:
    # Load previously saved model
    model = torch.load(model_save_path)


# Check loss and accuracy on test data
print("Testing ...")
model.train(mode=False)
correct_count, all_count = 0, 0
for images, labels in test_loader:
    for i in range(len(labels)):
        image = images[i].view(1, 784)
        with torch.no_grad():
            output = model(image)
        probs = torch.exp(output)
        predicted_digit = probs.argmax().item()
        labelled_digit = labels[i].item()
        if(predicted_digit == labelled_digit):
            correct_count += 1
        all_count += 1
print(f"Tested {all_count} images, model accuracy {correct_count / all_count}.")
