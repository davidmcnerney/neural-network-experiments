import random
from typing import Iterable, List, Union

from value import Value


class Neuron:
    def __init__(self, count_inputs: int):
        self.w: List[Value] = [Value(random.uniform(-1.0, 1.0), label=f"w{n+1}") for n in range(0, count_inputs)]
        self.b = Value(random.uniform(-1.0, 1.0), label="b")

    def execute(self, input_data: Iterable[Union[Value, int, float]]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, input_data)), self.b)
        return act.tanh()

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, count_inputs: int, count_outputs: int):
        self.neurons = [Neuron(count_inputs) for _ in range(count_outputs)]

    def execute(self, input_data: Iterable[Union[Value, int, float]]) -> List[Value]:
        return [n.execute(input_data) for n in self.neurons]

    def parameters(self) -> List[Value]:
        return [
            p
            for neuron in self.neurons
            for p in neuron.parameters()
        ]


class MultiLayerPerceptron:
    def __init__(self, count_inputs: int, count_outputs_lists: List[int]):
        sizes = [count_inputs] + count_outputs_lists
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def execute(self, input_data: Iterable[Union[Value, int, float]]) -> List[Value]:
        output_data = input_data
        for layer in self.layers:
            output_data = layer.execute(output_data)
        return output_data

    def parameters(self) -> List[Value]:
        return [
            p
            for layer in self.layers
            for p in layer.parameters()
        ]
