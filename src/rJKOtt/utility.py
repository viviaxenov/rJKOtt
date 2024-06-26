import numpy as np
import teneva

from typing import Literal, Callable, Tuple, List, Union, TypeAlias


tt_vector: TypeAlias = List[np.ndarray]

def tt_sum_multi_axis(a: tt_vector, axis: Union[int, List[int]]=-1):
    """Sum TT-vector over specified axes.

    Note:
        Condtion marked by * in the source were misplaced in the original code; should I make pull request?

    Args:
        a (`tt_vector`) : Tensor Train for summation
        axis (`int` or `List[int]`) axes to sum along. Default value `-1` sums over all axes and returns a scalar

    Returns:
        `tt_vector` or `float` : Tensor Train, summed over the axes 
    """
    d = len(a)
    crs = teneva.act_one.copy(a)
    if isinstance(axis, int):
        if axis < 0:  # (*)
            axis = range(d)
        else:
            axis = [axis]
    axis = list(axis)[::-1]
    for ax in axis:
        crs[ax] = np.sum(crs[ax], axis=1)
        rleft, rright = crs[ax].shape
        if (rleft >= rright or rleft < rright and ax + 1 >= d) and ax > 0:
            crs[ax - 1] = np.tensordot(crs[ax - 1], crs[ax], axes=(2, 0))
        elif ax + 1 < d:
            crs[ax + 1] = np.tensordot(crs[ax], crs[ax + 1], axes=(1, 0))
        else:
            return np.sum(crs[ax])
        crs.pop(ax)
        d -= 1
    return crs


def tt_slice(X: tt_vector, slcs: List[slice]):
    return [X[_i][:, _sl, :] for _i, _sl in enumerate(slcs)]


