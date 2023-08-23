from __future__ import annotations
from typing import Tuple
from teenygrad.helpers import dtypes
from teenygrad.ops import UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps
import numpy as np


class Device:
    DEFAULT = "CPU"
    _buffers = ["CPU"]

    def canonicalize(x):
        return "CPU"


def shape_to_axis(
    old_shape: Tuple[int, ...], new_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    assert len(old_shape) == len(new_shape), "reduce shapes must have same dimensions"
    return tuple(i for i, (a, b) in enumerate(zip(old_shape, new_shape)) if a != b)


class LazyBuffer:
    device = "CPU"
    dtype = dtypes.float32
    realized = None

    def __init__(self, initial_data):
        """
        Constructor for LazyBuffer class.

        Args:
            initial_data: Initial data to be stored in the buffer.
        """
        self._data = initial_data

    @property
    def shape(self):
        """
        Get the shape of the buffer.

        Returns:
            Tuple[int, ...]: Shape of the buffer's data.
        """
        return self._data.shape

    def contiguous(x):
        return x

    def realize(x):
        return x

    def const_like(self, value) -> LazyBuffer:
        return LazyBuffer(np.full_like(self._data, value))

    @staticmethod
    def fromCPU(x):
        return LazyBuffer(x)

    def toCPU(self):
        return self._data

    @staticmethod
    def loadop(op, shape, dtype, device, value=None, seed=1234, src=None) -> LazyBuffer:
        """
        Perform a load operation to create a LazyBuffer.

        Args:
            op: The load operation to perform (e.g., LoadOps.RAND, LoadOps.CONST).
            shape: Shape of the buffer to be created.
            dtype: Data type of the buffer's elements.
            device: Target device for the buffer (not used in this method).
            value: Value to be used in case of LoadOps.CONST operation.
            seed: Seed for random number generation (only used for LoadOps.RAND).
            src: Source buffer (not used in this method).

        Returns:
            LazyBuffer: A new LazyBuffer instance created based on the load operation.
        """
        if op == LoadOps.RAND:
            rng = np.random.default_rng(seed)
            return LazyBuffer(rng.random(size=shape, dtype=np.float32))
        elif op == LoadOps.CONST:
            return LazyBuffer(np.full(shape, value))
        else:
            raise NotImplementedError(op)

    # MovementOps
    def reshape(self, shape):
        """
        Reshape the buffer to a new shape.

        Args:
            arg: New shape.

        Returns:
            LazyBuffer: A new LazyBuffer with the reshaped data.
        """
        return LazyBuffer(self._data.reshape(shape))

    def expand(self, arg):
        """
        Expand the buffer's data to match a larger shape.

        Args:
            arg: Shape to be expanded to.

        Returns:
            LazyBuffer: A new LazyBuffer with the expanded data.
        """
        return LazyBuffer(np.broadcast_to(self._data, arg))

    def shrink(self, slice_ranges):
        """
        Shrink the buffer's data using slicing.

        Args:
            slice_ranges: List of slice ranges.

        Returns:
            LazyBuffer: A new LazyBuffer with the sliced data.
        """
        return LazyBuffer(self._data[tuple(slice(p[0], p[1], None) for p in slice_ranges)])

    def permute(self, permutation):
        """
        Permute the dimensions of the buffer.

        Args:
            arg: Permutation order.

        Returns:
            LazyBuffer: A new LazyBuffer with permuted dimensions.
        """
        return LazyBuffer(self._data.transpose(permutation))

    def pad(self, padding_config):
        """
        Pad the buffer's data with zeros.

        Args:
            arg: Padding configuration.

        Returns:
            LazyBuffer: A new LazyBuffer with padded data.
        """
        return LazyBuffer(np.pad(self._data, padding_config))

    def unary_op(self, op):
        """
        Apply a specified unary operation element-wise to the buffer's data.

        Args:
            op: The unary operation to be applied.

        Returns:
            LazyBuffer: A new LazyBuffer with the operation applied.
        """
        if op == UnaryOps.EXP2:
            return LazyBuffer(np.exp2(self._data))
        elif op == UnaryOps.LOG2:
            return LazyBuffer(np.log2(self._data))
        elif op == UnaryOps.SIN:
            return LazyBuffer(np.sin(self._data))
        elif op == UnaryOps.SQRT:
            return LazyBuffer(np.sqrt(self._data))
        else:
            raise NotImplementedError(op)

    def binary_op(self, op, y: LazyBuffer):
        """
        Apply a specified binary operation element-wise between this buffer and another.

        Args:
            operation: The binary operation to be applied.
            other_buffer: Another LazyBuffer for the operation.

        Returns:
            LazyBuffer: A new LazyBuffer with the operation applied.
        """

        if op == BinaryOps.MAX:
            return LazyBuffer(np.maximum(self._data, y._data))
        else:
            raise NotImplementedError(op)

    def ternary_op(self, op, y: LazyBuffer, z: LazyBuffer):
        if op == TernaryOps.WHERE:
            return LazyBuffer(np.where(self._data, y._data, z._data))
        else:
            raise NotImplementedError(op)

    def reduce_op(self, op, new_shape):
        if op == ReduceOps.SUM:
            return LazyBuffer(
                self._data.sum(shape_to_axis(self.shape, new_shape), keepdims=True)
            )
        elif op == ReduceOps.MAX:
            return LazyBuffer(
                self._data.max(shape_to_axis(self.shape, new_shape), keepdims=True)
            )
        else:
            raise NotImplementedError(op)

    def __add__(self, other: LazyBuffer):
        """
        Perform element-wise addition with another buffer.

        Args:
            other_buffer: The buffer to be added.

        Returns:
            LazyBuffer: A new LazyBuffer with the addition performed.
        """
        return LazyBuffer(self._data + other._data)

    def __sub__(self, other: LazyBuffer):
        """
        Perform element-wise subtraction with another buffer.

        Args:
            other_buffer: The buffer to be subtracted.

        Returns:
            LazyBuffer: A new LazyBuffer with the subtraction performed.
        """
        return LazyBuffer(self._data - other._data)

    def __mul__(self, x: LazyBuffer):
        """
        Perform element-wise multiplication with another buffer.

        Args:
            other_buffer: The buffer to be multiplied.

        Returns:
            LazyBuffer: A new LazyBuffer with the multiplication performed.
        """
        return LazyBuffer(self._data * x._data)

    def __truediv__(self, other: LazyBuffer):
        """
        Perform element-wise division with another buffer.

        Args:
            other_buffer: The buffer to be used as the divisor.

        Returns:
            LazyBuffer: A new LazyBuffer with the division performed.
        """
        return LazyBuffer(self._data / other._data)

    def __lt__(self, other: LazyBuffer):
        """
        Perform element-wise less-than comparison with another buffer.

        Args:
            other_buffer: The buffer to be compared against.

        Returns:
            LazyBuffer: A new LazyBuffer with the comparison results.
        """
        return LazyBuffer(self._data < other._data)

    def __neg__(self) -> LazyBuffer:
        """
        Negate the elements of the buffer.

        Returns:
            LazyBuffer: A new LazyBuffer with negated elements.
        """
        return self.const_like(0.0) - self
