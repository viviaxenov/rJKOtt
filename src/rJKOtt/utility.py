import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm
from scipy.stats import multivariate_normal, norm

import matplotlib.pyplot as plt

import teneva

from functools import cache
from time import perf_counter
import traceback

import os.path

from typing import Literal, Callable, Tuple, List, Union, TypeAlias

from itertools import cycle

tt_vector: TypeAlias = List[np.ndarray]


def crop_to_cube(xs: np.ndarray, L: np.float64) -> np.ndarray:
    """foo
    """
    select_indices = np.all(xs <= L, axis=-1) & np.all(xs >= -L, axis=-1)
    xs = xs[select_indices, :]

    return xs

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


def div_KL(
    rho1: tt_vector, rho2: Union[np.ndarray, Callable], h_x: np.float64
) -> np.float64:
    dim = len(rho1)
    if isinstance(rho2, np.ndarray):
        if isinstance(rho1, List):
            rho1 = teneva.full(rho1)

        eps = 1.0 - np.sum(rho1) * h_x**dim
        try:
            V = (
                np.sum(
                    (np.log(np.where(rho1 > 1e-15, rho1, 1e-15)) - np.log(rho2)) * rho1
                )
                * h_x**dim
            )
            return V + eps
        except:
            return -1.0

    else:
        print("Estimating KL err", flush=True)
        log_quot = teneva.cross(
            lambda _S: np.log(
                np.maximum(teneva.act_one.get_many(rho1, _S), 1e-32)
                / np.maximum(rho2.eval_no_cache(_S), 1e-16)
            )
            * teneva.act_one.get_many(rho1, _S),
            teneva.tensors.const(teneva.props.shape(rho1), v=0.0),
            e=1e-5,
            m=int(1e6),
        )
        # log_quot = teneva.truncate(log_quot, e=1e-10, r=3)
        # print("", flush=True)
        # return teneva.mul_scalar(log_quot, rho1) * h_x**dim
        return (
            teneva.sum(
                log_quot,
            )
            * h_x**dim
        )

def KL_stupid(
    eta: tt_vector, hat_eta: tt_vector, beta: np.float64, h_x: np.float64
) -> np.float64:
    dim = len(eta)
    print("Estimating KL_stupid", flush=True)
    rho = teneva.mul(eta, hat_eta)
    rho = teneva.truncate(rho, r=20)
    normalization = 1.0 / (teneva.sum(rho) * h_x**dim)
    rho = teneva.mul(normalization, rho)

    # why tf dis works...
    eta = teneva.mul(normalization**0.5, eta)

    log_quot = teneva.cross(
        lambda _I: (
            -2.0 * beta * np.log(np.maximum(1e-16, teneva.act_one.get_many(eta, _I)))
        )
        * teneva.act_one.get_many(rho, _I),
        teneva.tensors.const(teneva.props.shape(eta), 0.0),
        m=int(1e6),
        e=1e-6,
    )
    return teneva.sum(log_quot) * h_x**dim


def div_L2(
    rho1: tt_vector, rho2: Union[np.ndarray, Callable], h_x: np.float64
) -> np.float64:
    dim = len(rho1)
    if isinstance(rho2, np.ndarray):
        rho1 = teneva.full(rho1)
        return np.linalg.norm((rho1 - rho2).ravel(), ord=2)
    else:
        print("Estimating L2 err", flush=True)
        diff = teneva.cross(
            lambda _S: (teneva.act_one.get_many(rho1, _S) - rho2.eval_no_cache(_S)),
            teneva.tensors.const(teneva.props.shape(rho1), v=0),
            e=1e-5,
            m=int(1e5),
        )
        diff = teneva.truncate(diff, r=15)
        dL2 = np.sqrt(teneva.mul_scalar(diff, diff) * h_x**dim)
        print(f"\tdone", flush=True)
        return dL2


def lp_err_test(
    x1: tt_vector,
    x2: Union[tt_vector, Callable],
    n_samples,
    seed=None,
    p=np.inf,
):
    test_idx = teneva.sample_rand(teneva.props.shape(x1), n_samples, seed)

    x1_on_test = teneva.act_one.get_many(x1, test_idx)
    x2_on_test = (
        teneva.act_one.get_many(x1, test_idx) if isinstance(x2, List) else x2(test_idx)
    )

    if np.isfinite(p):
        err = np.linalg.norm((x1_on_test - x2_on_test).ravel(), ord=p)
        err /= np.linalg.norm(x2_on_test.ravel(), ord=p)
    else:
        err = np.linalg.norm((x1_on_test - x2_on_test).ravel(), ord=p)

    return err


def tt_independent_gaussians(
    ms: List[np.float64], sigmas: List[np.float64], grids: Union[List[np.ndarray],np.ndarray]
) -> tt_vector:
    assert len(ms) == len(sigmas)
    d = len(ms)
    if isinstance(grids, np.ndarray):
        grids = [grids,]*d

    nodes = [
        norm.pdf(grids[i], loc=ms[i], scale=sigmas[i]).reshape(1, grids[i].shape[0], 1) for i in range(d)
    ]
    return nodes





def plot_1d_marginals(
    density: Union[tt_vector, np.ndarray, tuple],
    kind: Literal[
        "full",
        "TT",
        "gauss_mixture",
    ],
    grid: np.ndarray,
    dim: int,
    fig_and_axs=None,
    *plot_args,
    **plot_kwargs,
):
    h_x = grid[1] - grid[0]

    if fig_and_axs in [None, (None, None)]:
        fig, axs = plt.subplots(1, dim, sharey=True)
        fig.suptitle("Marginals")
    else:
        fig, axs = fig_and_axs

    for n_marginal in range(dim):
        density_marginal = get_marginal_on_grid(density, kind, grid, (n_marginal,))

        axs[n_marginal].plot(grid, density_marginal, *plot_args, **plot_kwargs)
        axs[n_marginal].grid(True)
        axs[n_marginal].set_xlabel(f"$x_{{ {n_marginal+1} }}$")

    axs[-1].legend()
    return (fig, axs)


def plot_2d_marginals(
    density: Union[tt_vector, np.ndarray],
    kind: Literal[
        "full",
        "TT",
        "gauss_mixture",
    ],
    grid: np.ndarray,
    marginals=(0, 1),
    fill=False,
    fig_and_axs=None,
    *plot_args,
    **plot_kwargs,
):
    h_x = grid[1] - grid[0]

    fig: plt.Figure
    axs: plt.Axes

    if fig_and_axs in [None, (None, None)]:
        fig, axs = plt.subplots(1, 1)
        fig.suptitle(
            f"Joint distribuition of $x_{marginals[0]}$ and $x_{marginals[1]}$"
        )

        axs.set_xlabel(f"$x_{{ {marginals[0]} }}$")
        axs.set_ylabel(f"$x_{{ {marginals[1]} }}$")

    else:
        fig, axs = fig_and_axs

    density_marginal = get_marginal_on_grid(density, kind, grid, marginals)

    # if marginals[0] > marginals[1]:
    #     density_marginal = density_marginal.T

    plot_fn = axs.contourf if fill else axs.contour

    contours = plot_fn(
        *np.meshgrid(grid, grid, indexing="ij"),
        density_marginal,
        10,
        *plot_args,
        **plot_kwargs,
    )
    return fig, axs


def plot_matrix_marginals(
    density: Union[tt_vector, np.ndarray],
    kind: Literal[
        "full",
        "TT",
        "gauss_mixture",
    ],
    grid: np.ndarray,
    dim: int,
    sample=None,
    fig_and_axs=None,
    cmap="Blues",
    sym=False,
    *plot_args,
    **plot_kwargs,
):
    # dim = len(density) if isinstance(density, tt_vector) else len(density.shape)
    h_x = grid[1] - grid[0]

    # fig, axs = plt.subplots(dim, dim, sharex='row', sharey='col')
    if fig_and_axs in [None, (None, None)]:
        fig, axs = plt.subplots(
            dim,
            dim,
        )
    else:
        fig, axs = fig_and_axs

    for _i in range(dim):
        axs[_i, 0].set_ylabel(f"$x_{{{_i + 1}}}$")
        for _j in range(_i + 1, dim):
            plot_2d_marginals(
                density,
                kind,
                grid,
                fig_and_axs=(fig, axs[_i, _j]),
                marginals=(_j, _i),
                cmap=cmap,
            )
            if sym:
                plot_2d_marginals(
                    density,
                    kind,
                    grid,
                    fig_and_axs=(fig, axs[_j, _i]),
                    marginals=(_i, _j),
                    cmap=cmap,
                    fill=True,
                )

    for _j in range(dim):
        axs[-1, _j].set_xlabel(f"$x_{{{_j + 1}}}$")

    color = plot_kwargs["color"]
    if sample is not None:
        for _i in range(dim):
            for _j in range(_i):
                axs[_i, _j].scatter(sample[:, _j], sample[:, _i], c=color, s=0.2)

    axs_diag = np.diag(axs)
    plot_1d_marginals(
        density, kind, grid, dim, (fig, axs_diag), *plot_args, **plot_kwargs
    )
    for _i in range(dim):
        axs[_i, _i].set_ylabel(f"$\\rho_{{{_i+1}}}$")
    for _i in range(1, dim):
        axs[_i, _i].sharey(axs[_i - 1, _i - 1])

    return fig, axs


def plot_convergence(
    Ts: List[np.float64],
    err_stats: dict,
    fig_path=None,
):
    time = np.cumsum(np.array([0.0] + Ts))
    fig, axs = plt.subplots(1, 1)
    markers = cycle(["*", "^", "s", "o"])

    for _stat in err_stats:
        axs.plot(time, err_stats[_stat], label=_stat, marker=next(markers))
    axs.set_yscale("log")
    axs.set_xlabel("Time")
    axs.set_title("Convergence")
    axs.legend()
    axs.grid()

    if fig_path is not None:
        fig.savefig(
            os.path.join(
                fig_path,
                "gf_covnergence.pdf",
            )
        )
    else:
        plt.show()

    return
