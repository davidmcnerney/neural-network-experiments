from value import Value


a = Value(2.0, "a")
b = Value(-3.0, "b")
c = Value(10.0, "c")
f = Value(-2.0, "f")

e = a * b; e.label = "e"
d = e + c; d.label = "d"
L = d * f; L.label = "L"

L.print_debug_representation()

L.grad = 1
L.back_propagate_gradient_to_inputs()
print(f"a.grad: {a.grad}")


