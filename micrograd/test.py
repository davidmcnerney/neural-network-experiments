from value import Value


x1 = Value(2.0, "x1")
w1 = Value(-3.0, "w1")
x2 = Value(0.0, "x2")
w2 = Value(1.0, "w2")

b = Value(6.8813735870195432, "b")

o = (x1 * w1 + x2 * w2 + b).tanh()

o.grad = 1
o.back_propagate_gradient_to_inputs()
o.print_debug_representation()
