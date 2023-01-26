import torch
import torch.utils.data

from gpt.model.gpt import GPT
from gpt.running.configuration import Configuration
from gpt.running import train


def test_train():
    config = Configuration.for_tests()
    config.count_epochs = 1
    config.batch_size = 2
    model = GPT(config=config)

    x = torch.tensor([
        [100, 200, 300, 400],
        [101, 201, 301, 401],
    ])
    y = torch.tensor([
        [10, 20, 30, 40],
        [11, 21, 31, 41],
    ])
    training_dataset = torch.utils.data.TensorDataset(x, y)

    x = torch.tensor([
        [500, 600, 700, 800],
        [501, 601, 701, 801],
    ])
    y = torch.tensor([
        [50, 60, 70, 80],
        [51, 61, 71, 81],
    ])
    validation_dataset = torch.utils.data.TensorDataset(x, y)

    train.train(model=model, training_dataset=training_dataset, validation_dataset=validation_dataset)


def test_static_learning_rate():
    config = Configuration.for_tests()
    config.learning_rate = 1e-4
    config.count_warmup_iterations = None
    config.count_decay_iterations = None
    assert train.get_dynamic_learning_rate(config, 1) == 1e-4
    assert train.get_dynamic_learning_rate(config, 1000) == 1e-4


def test_dynamic_learning_rate():
    config = Configuration.for_tests()
    config.learning_rate = 1e-3
    config.count_warmup_iterations = 30
    config.count_decay_iterations = 500
    config.decayed_learning_rate = 1e-4
    assert train.get_dynamic_learning_rate(config, 1) == 1e-3 * (1/30)
    assert train.get_dynamic_learning_rate(config, 15) == 1e-3 * (15/30)
    assert train.get_dynamic_learning_rate(config, 30) == 1e-3
    assert round(train.get_dynamic_learning_rate(config, 280), 2) == round((1e-3 - 1e-4) * 0.5 + 1e-4, 2)
    assert train.get_dynamic_learning_rate(config, 530) == 1e-4
    assert train.get_dynamic_learning_rate(config, 1000) == 1e-4
