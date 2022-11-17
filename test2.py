from typing import List, Tuple

from neurons import MultiLayerPerceptron
from value import Value

# First (tuple) is x input data, second (scalar) is y expected output.
training_data_sets = [
    ((2.0, 3.0, -1.0), 1.0),
    ((3.0, -1.0, 0.5), -1.0),
    ((0.5, 1.0, 1.0), -1.0),
    ((1.0, 1.0, -1.0), 1.0),
]
training_inputs: List[Tuple[float, ...]] = [training_data[0] for training_data in training_data_sets]
expected_outputs: List[Value] = [Value(training_data[1]) for training_data in training_data_sets]

mlp = MultiLayerPerceptron(3, [4, 4, 1])

for i in range(1, 20+1):
    # Forward pass
    actual_outputs: List[Value] = [
        mlp.execute(training_input)[0]
        for training_input in training_inputs
    ]
    loss: Value = sum((actual - expected)**2 for actual, expected in zip(actual_outputs, expected_outputs))
    # loss.print_debug_representation()
    print(f"Iteration {i} loss: {loss.data}")

    # Backward pass
    loss.zero_gradient()
    loss.grad = 1.0
    loss.back_propagate_gradient_to_inputs()
    for parameter in mlp.parameters():
        parameter.data += -0.05 * parameter.grad
