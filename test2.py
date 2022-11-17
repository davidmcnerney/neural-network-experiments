from neurons import MultiLayerPerceptron

training_data = [
    ((2.0, 3.0, -1.0), 1.0),
]

# Forward pass
mlp = MultiLayerPerceptron(3, [4, 4, 1])
o = mlp.execute(training_data[0][0])
o[0].print_debug_representation()
