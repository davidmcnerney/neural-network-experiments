from typing import Callable, Optional, Tuple, Union


class Value:
    def __init__(
            self,
            data: Union[int, float],
            label: Optional[str] = None,
            _inputs: Tuple["Value", ...] = (),
            _input_operation: Optional[str] = None,
            _back_propagate_to_inputs: Optional[Callable] = None
    ):
        self.data = data
        self.label = label
        self.grad = 0.0

        self.inputs = _inputs
        self.input_operation = _input_operation
        self._back_propagate_gradient_to_inputs = _back_propagate_to_inputs

    def __repr__(self):
        if self.label is not None:
            return f"[value {self.label}: {self.data}]"
        else:
            return f"[value: {self.data}]"

    #
    # Operations
    #

    def __add__(self, other):
        if not isinstance(other, Value):
            raise Exception(f"Cannot add {type(other)} to a Value")
        out_value = Value(data=(self.data + other.data), _inputs=(self, other,), _input_operation="+")

        def backwards():
            self.grad = out_value.grad
            other.grad = out_value.grad
        out_value._back_propagate_gradient_to_inputs = backwards

        return out_value

    def __mul__(self, other):
        if not isinstance(other, Value):
            raise Exception(f"Cannot add {type(other)} to a Value")
        out_value = Value(data=(self.data * other.data), _inputs=(self, other,), _input_operation="*")

        def backwards():
            self.grad = other.data * out_value.grad
            other.grad = self.data * out_value.grad
        out_value._back_propagate_gradient_to_inputs = backwards

        return out_value


    #
    # Backprop
    #

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
