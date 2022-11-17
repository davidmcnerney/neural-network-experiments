from math import e
from typing import Callable, Optional, Tuple, Union


class Value:
    def __init__(
            self,
            data: Union[int, float],
            label: Optional[str] = None,
            _inputs: Tuple["Value", ...] = (),
            _input_operation: Optional[str] = None,
    ):
        self.data = data
        self.label = label
        self.grad = 0.0

        self.inputs = _inputs
        self.input_operation = _input_operation

        self._back_propagate_gradient_to_inputs = None

        if self.label is None:
            self.label = self._construct_default_label()

    def _construct_default_label(self) -> Optional[str]:
        if self.inputs is None or self.input_operation is None:
            return None

        input_labels = [input_value.label for input_value in self.inputs]
        if None in input_labels:
            return None

        if len(input_labels) > 1:
            return self.input_operation.join(input_labels)
        else:
            return f"{self.input_operation}({input_labels[0]})"

    def __repr__(self):
        if self.label is not None:
            return f"[{self.label}|data {self.data}|grad {self.grad}]"
        else:
            return f"[data {self.data}|grad {self.grad}]"

    #
    # Operations
    #

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out_value = Value(data=(self.data + other.data), _inputs=(self, other,), _input_operation="+")

        def backwards():
            self.grad += out_value.grad
            other.grad += out_value.grad
        out_value._back_propagate_gradient_to_inputs = backwards

        return out_value

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out_value = Value(data=(self.data - other.data), _inputs=(self, other,), _input_operation="-")

        def backwards():
            self.grad += out_value.grad
            other.grad += out_value.grad
        out_value._back_propagate_gradient_to_inputs = backwards

        return out_value

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out_value = Value(data=(self.data * other.data), _inputs=(self, other,), _input_operation="*")

        def backwards():
            self.grad += other.data * out_value.grad
            other.grad += self.data * out_value.grad
        out_value._back_propagate_gradient_to_inputs = backwards

        return out_value

    def __pow__(self, other):
        if not isinstance(other, int) or isinstance(other, float):
            raise Exception("Expected integer or float for ** power")
        out_value = Value(data=(self.data ** other), _inputs=(self,), _input_operation=f"**{other}")

        def backwards():
            self.grad += other * self.data ** (other - 1) * out_value.grad
        out_value._back_propagate_gradient_to_inputs = backwards

        return out_value

    def tanh(self):
        x = self.data
        out_data = (e ** (2.0 * x) - 1) / (e ** (2.0 * x) + 1)
        out_value = Value(out_data, _inputs=(self, ), _input_operation="tanh")

        def backwards():
            self.grad += 1.0 - out_data ** 2
        out_value._back_propagate_gradient_to_inputs = backwards

        return out_value

    #
    # Backprop
    #

    def zero_gradient(self):
        self.grad = 0
        for input_value in self.inputs:
            input_value.zero_gradient()

    def back_propagate_gradient_to_inputs(self):
        if self._back_propagate_gradient_to_inputs is not None:
            self._back_propagate_gradient_to_inputs()
            for input_value in self.inputs:
                input_value.back_propagate_gradient_to_inputs()
        else:
            if len(self.inputs) > 0:
                raise Exception(f"No back propagation closure available in {self}, but it has inputs. Unexpected.")


    #
    # Debug utilities
    #

    def debug_representation(self, indent: str = "") -> str:
        representation = f"{indent}{self}"

        if self.input_operation is not None:
            representation += f" {self.input_operation}:"

        representation += "\n"

        for input_value in self.inputs:
            representation += input_value.debug_representation(indent + "  ")

        return representation

    def print_debug_representation(self) -> None:
        print(self.debug_representation())
