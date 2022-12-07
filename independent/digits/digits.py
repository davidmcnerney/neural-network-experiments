import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# import matplotlib.pyplot as plt
# import torch
from torch import nn
from torch.optim import SGD
import torch.utils.data
from torchvision import datasets, io as torchvision_io, transforms
from torchvision.transforms import functional as transforms_functional


# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

argument_parser = argparse.ArgumentParser("Digits")
argument_parser.add_argument("--random-seed-offset", type=int, default=0, required=True)
args = argument_parser.parse_args()
random_seed_offset = args.random_seed_offset


# Run modes
do_training = True
evaluate_test_dataset = True
evaluate_additional = True

# Hyperparameters
epochs = 15
batch_size = 32
output_size = 10
learning_rate = 0.003
momentum = 0.9

random_seed = 2147483647 + random_seed_offset

dataset_save_folder = "/Users/dave/Temp/neural_net_training/datasets"
model_save_file = "/Users/dave/Temp/neural_net_training/models/digits_emnist.pt"


# Reproducibility
torch.manual_seed(random_seed)


# Load training and test data
loading_transform = transforms.Compose([
    lambda img: transforms_functional.rotate(img, -90),
    lambda img: transforms_functional.hflip(img),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
training_dataset = datasets.EMNIST(
    dataset_save_folder + "/EMNIST_TRAIN",
    split="digits",
    download=True,
    train=True,
    transform=loading_transform,
)
test_dataset = datasets.EMNIST(
    dataset_save_folder + "/EMNIST_TEST",
    split="digits",
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


# Setup used for both training and evaluation
loss_function = nn.NLLLoss()


if do_training:
    # Construct neural net
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),    # 1x28x28 -> 6x24x24
        nn.ReLU(),
        # nn.Dropout(p=dropout),
        nn.BatchNorm2d(num_features=6),
        nn.MaxPool2d(kernel_size=2),                                # 6x24x24 -> 6x12x12
        nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5),    # 6x12x12 -> 8x8x8
        nn.ReLU(),
        # nn.Dropout(p=dropout),
        nn.BatchNorm2d(num_features=8),
        nn.MaxPool2d(kernel_size=2),                                # 8x8x8 -> 8x4x4
        nn.Flatten(),                                               # 128
        nn.Linear(128, 64),                                         # 128
        nn.ReLU(),
        nn.Linear(64, 32),                                          # 64
        nn.ReLU(),
        nn.Linear(32, output_size),                                 # 10
        nn.LogSoftmax(dim=1)
    )

    # Train the model
    print("Training ...")
    print(f"   Random seed: {random_seed} (offset {random_seed_offset})")
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    start_time = datetime.now()
    for epoch_num in range(epochs):
        # print(f"   starting epoch {epoch_num}")
        epoch_total_loss = 0
        for images, labels in training_loader:
            # print(f"      got {len(images)} images from training loader")
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            epoch_total_loss += loss.item()

            # histogram_data = images[0].flatten().tolist()
            # plt.hist(histogram_data, density=True, bins=30)
            # plt.title("Training image values")
            # plt.show()
            # print("Showed histogram")

            # for image, label in zip(images, labels):
            #     if label == 9:
            #         plt.imshow(image.permute(1, 2, 0))
            #         plt.title(str(label.item()))
            #         plt.show()

        epoch_training_loss = round(epoch_total_loss / len(training_loader), 5)
        print(f"      epoch {epoch_num} training loss {epoch_training_loss}")
    print(f"Training time: {datetime.now() - start_time}")
    print("")

    # Save model
    torch.save(model, model_save_file)

    # turn off training mode for evaluation tasks below
    model.train(mode=False)
else:
    # Load previously saved model
    model = torch.load(model_save_file)


# Check loss and accuracy on test portion of dataset
test_loss: Optional[float] = None
if evaluate_test_dataset:
    print("Testing against test portion of dataset ...")
    total_loss, count_loss = 0., 0
    for images, labels in test_loader:
        with torch.no_grad():
            output = model(images)
            loss = loss_function(output, labels)
        total_loss += loss.item()
        count_loss += 1

        # image = images[0]
        # plt.imshow(image.permute(1, 2, 0))
        # plt.title(str(labelled_digit))
        # plt.show()

        # break  # temp!
    test_loss = round(total_loss / count_loss, 5)
    print(f"   loss in {count_loss} test data batches was {test_loss}")


# Try to recognize additional images from outside the dataset
# These additional images are generated by writing with a black pen on a mostly white
# paper, and come in with grayscale integer values from 0 to 255. The training
# dataset had background color exactly-1.0 and foreground color up to 1.0, so we need to
# flip the values and scale to -1.0, 1.0 range.
additional_images_accuracy: Optional[float] = None
if evaluate_additional:
    print("Testing additional images ...")

    # Load additional images
    additional_images_folder = Path(__file__).parent / Path("additional")
    additional_images: List[Tuple[int, str, torch.Tensor]] = []
    for path in additional_images_folder.glob("*.png"):
        # Load from .png file
        digit = int(path.name[0])
        image_raw = torchvision_io.read_image(
            path=str(path.resolve()),
            mode=torchvision_io.ImageReadMode.GRAY,
        ).float()

        # Reverse the sense of the colors to match our training data, which was
        # white images on black background
        image = 1.0 - image_raw / 255.

        # Normalize in the same way our dataset loader did
        image = transforms_functional.normalize(image, [0.5], [0.5])

        # Our training data is super clean, so try to clean it up here to match
        image[image < 0.] = -1.
        image[image >= 0.] = 1.

        additional_images.append((digit, path.name, image))

        # if digit == 9:
        #     import matplotlib.pyplot as plt
        #     histogram_data = image.flatten().tolist()
        #     plt.hist(histogram_data, density=True, bins=30)
        #     plt.title(f"Additional image {digit} values")
        #     plt.show()
        #     print("Showed histogram")

    # Calculate loss
    images = torch.stack([t[2] for t in additional_images], dim=0)
    labels = torch.tensor([t[0] for t in additional_images])
    with torch.no_grad():
        output = model(images)
        loss = loss_function(output, labels)

    # Calculate % correct digit predictions
    correct_count, all_count = 0, 0
    lowest_prob = 1.0
    for digit, filename, image in sorted(additional_images):
        input_ = torch.unsqueeze(image, 0)
        with torch.no_grad():
            output = model(input_)
        probs = torch.exp(output)
        predicted_digit = probs.argmax().item()
        did_pass = digit == predicted_digit
        if did_pass:
            correct_count += 1
        all_count += 1
        if probs.squeeze()[digit].item() < lowest_prob:
            lowest_prob = probs.squeeze()[digit].item()
        if not did_pass:
            print(f"   {filename} actual {digit} -> {predicted_digit} {'pass' if did_pass else 'FAIL'}")
            print(f"      probs: {[round(p, 4) for p in probs.squeeze().tolist()]}")
    additional_images_accuracy = round(100 * correct_count / all_count, 1)
    print(f"   Tested {all_count} additional images: loss {round(loss.item(), 5)}, model accuracy {additional_images_accuracy}%, lowest prob {round(lowest_prob, 4)}")


# Summary for git commit messages 
if test_loss is not None and additional_images_accuracy is not None:
    model_filename = Path(model_save_file).name
    print(f"{model_filename}: test loss {test_loss}, additional digit accuracy {additional_images_accuracy}%")

print("")
