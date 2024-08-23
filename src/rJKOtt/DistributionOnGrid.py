from __future__ import annotations
import abc
from typing import Literal, Callable, Tuple, List, Union, TypeAlias, Dict, Iterable
from docstring_inheritance import GoogleDocstringInheritanceInitMeta

import warnings

import numpy as np
import teneva
from scipy.stats import multivariate_normal, multinomial, norm
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

import matplotlib.pyplot as plt

from .utility import tt_sum_multi_axis, tt_slice, tt_vector


class _Meta(abc.ABC, GoogleDocstringInheritanceInitMeta):
    pass


class Grid:
    """Represents uniform rectangular grid.

    Args:
        left : Left extent for the grid in each dimension
        right : Right extent for the grid in each dimension
        N_nodes : number of nodes in each dimension
        dim : dimension of the distribution

    Attributes :
        dim : dimension of the grid
    """

    def __init__(
        self,
        left: Union[float, List[float]],
        right: Union[float, List[float]],
        N_nodes: Union[int, List[int]],
        dim: int = 1,
    ):
        if isinstance(N_nodes, list):
            dim = len(N_nodes)
        else:
            N_nodes = [
                N_nodes,
            ] * dim

        if isinstance(left, float):
            left = [
                left,
            ] * dim

        if isinstance(right, float):
            right = [
                right,
            ] * dim

        assert len(left) == dim
        assert len(right) == dim

        self.dim = dim
        self.left = left
        self.right = right
        self.N_nodes = N_nodes
        self.hx = np.array(
            [(self.right[i] - self.left[i]) / self.N_nodes[i] for i in range(self.dim)]
        )

    def __iter__(
        self,
    ):
        """To use with teneva.ind_to_poi and teneva.poi_to_ind"""
        return (item for item in (self.left, self.right, self.N_nodes))

    def clip_sample(self, xs: np.ndarray) -> np.ndarray:
        """Given an array of points, delete those of them that are out of the grid's span

        Args:
            xs : the points; should have shape `(N_points, dim)`

        Returns:
            np.ndarray : fitting points
        """

        assert xs.shape[-1] == self.dim
        select_indices = np.all(xs <= self.right[np.newaxis, :], axis=-1) & np.all(
            xs >= self.left[np.newaxis, :], axis=-1
        )
        xs = xs[select_indices, :]

        return xs

    def get_1d_grid(self, i: int) -> np.ndarray:
        """Return grid points in i-th dimension

        Args:
            i : index of the dimension; `0 <= i < dim`

        Returns:
            array of shape `(grid.N[i])` --- grid points in i-th direction
        """
        return np.linspace(self.left[i], self.right[i], self.N_nodes[i], endpoint=True)

    def get_2d_grid(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return grid points in i-th dimension

        Args:
            i : index of the first dimension; `0 <= i < dim`
            j : index of the second dimension; `0 <= j < dim`

        Returns:
            Tuple[np.ndarray] : arrays of shape `(N_nodes[i], N_nodes[j])` (suitable for scatter plot)
        """
        x1 = np.linspace(self.left[i], self.right[i], self.N_nodes[i], endpoint=True)
        x2 = np.linspace(self.left[j], self.right[j], self.N_nodes[j], endpoint=True)
        return np.meshgrid(x1, x2, indexing="ij")


class DistributionOnGrid(metaclass=_Meta):
    """Convenience class for storing distributions and plotting.

    Args:
        grid : self-explanatory

    Attributes :
        dim : dimension of the distribution
    """

    def __init__(self, grid: Grid):
        self.grid = grid
        self._dim = grid.dim

    @property
    def dim(self) -> int:
        return self._dim

    def plot_1d_marginal(
        self,
        n_marginal: int,
        *plot_args,
        axs: plt.Axes = None,
        **plot_kwargs,
    ):
        """Plots marginal distributions of `x_i` for `i = 1..dim`

        Args:
            n_marginal : index of marginal to plot
            fig_and_axs (Tuple[plt.Figure, np.ndarray], optional)  : figure and axis to plot

        Returns:
            Tuple[plt.Figure, plt.Axes] : figure and axes with the plot
        """
        if axs is None:
            fig, axs = plt.subplots(1, 1, sharey=True)
            fig.suptitle("Marginals")
            axs.grid(True)
            axs.set_xlabel(f"$x_{{ {n_marginal+1} }}$")
        else:
            fig = axs.get_figure()

        grid = self.grid.get_1d_grid(n_marginal)
        density_marginal = self.get_marginal_on_grid(n_marginal)

        axs.plot(grid, density_marginal, *plot_args, **plot_kwargs)

        return (fig, axs)

    def plot_2d_marginal(
        self,
        marginals=Tuple[int, int],
        *contour_args,
        axs=None,
        fill=False,
        **contour_kwargs,
    ):
        """Plots distribution of 2-d marginals.

        Args:
            marginals : indices of marginals to plot
            axs : plt.Axes to plot; if `None`, will be initialized
            fill : if `True`, uses `plt.contourf`; default `plt.contour`
        """
        fig: plt.Figure
        axs: plt.Axes

        if axs is None:
            fig, axs = plt.subplots(1, 1)
            fig.suptitle(
                f"Joint distribuition of $x_{marginals[0]}$ and $x_{marginals[1]}$"
            )

            axs.set_xlabel(f"$x_{{{marginals[0]}}}$")
            axs.set_ylabel(f"$x_{{{marginals[1]}}}$")

        else:
            fig = axs.get_figure()

        density_marginal = self.get_marginal_on_grid(marginals)
        x_grid = self.grid.get_2d_grid(*marginals)

        plot_fn = axs.contourf if fill else axs.contour

        contours = plot_fn(
            *x_grid,
            density_marginal,
            *contour_args,
            **contour_kwargs,
        )
        return fig, axs

    def plot_matrix_marginals(
        self,
        param_subset: List[int] = None,
        sample: np.ndarray = None,
        axs: np.ndarray = None,
        sym: bool = False,
        w_ticks: bool = False,
        w_labels: bool = True,
        plot_args: List = [],
        plot_kwargs: Dict = {},
        contour_args: List = [],
        contour_kwargs: Dict = {},
        scatter_args: List = [],
        scatter_kwargs: Dict = {},
    ):
        """Plots 1- and 2-dimensional marginals on a grid of dim x dim plots.

        By default, the contorus are plotted on the upper-right triangle of the matrix.

        Args:
            param_subset : indices of parameters to plot. if `None`, will plot the marginals of all the parameters
            sample : an array of shape `(N_sample, dim)` to be plotted on the lower left submatrix
            axs : `(dim, dim)` array of `plt.Axes` to plot on; if `None`, will be initialized
            sym : if `True`, additionally plot a transpose contours on the bottom left matrix.
            w_ticks : whether to draw ticks on the axes; `False` generally better for higher dimension
            w_labels : whether to draw axis labels
            plot_args, plot_kwargs : will be passed to `plt.plot` for 1-d marginals
            contour_args, contour_kwargs : will be passed to `plt.contour` for 2-d marginals
            scatter_args, scatter_kwargs : will be passed to `plt.scatter` for 2-d sample marginals
        """

        if param_subset is None:
            param_subset = range(self.dim)
        param_subset = list(set(param_subset))
        N_plots = len(param_subset)

        # if axs are not given, prepare them
        if axs is None:
            fig, axs = plt.subplots(N_plots, N_plots)
            if w_labels:
                for i, param_i in enumerate(param_subset):
                    axs[-1, i].set_xlabel(f"$x_{{{param_i + 1}}}$")

                    ax = axs[i, -1]
                    # ax.yaxis.tick_right()
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(f"$x_{{{param_i + 1}}}$")

            if not w_ticks:
                for ax in axs.ravel():
                    ax.tick_params(
                        axis="both",
                        which="both",
                        bottom=False,
                        left=False,
                        labelbottom=False,
                        labelleft=False,
                    )
        else:
            fig = axs[0, 0].get_figure()

        # plot the contour lines on the upper right triangle
        for i, param_i in enumerate(param_subset):
            for shift, param_j in enumerate(param_subset[i + 1 :]):
                j = i + 1 + shift
                self.plot_2d_marginal(
                    (param_j, param_i),
                    *contour_args,
                    fill=False,
                    axs=axs[i, j],
                    **contour_kwargs,
                )
                # if sym, also plot filled contour lines in the bottom left triangle
                if sym:
                    self.plot_2d_marginal(
                        (param_i, param_j),
                        *contour_args,
                        fill=True,
                        axs=axs[j, i],
                        **contour_kwargs,
                    )

        if sample is not None:
            for i, param_i in enumerate(param_subset):
                for j, param_j in enumerate(param_subset[:i]):
                    axs[i, j].scatter(
                        sample[:, param_j],
                        sample[:, param_i],
                        *scatter_args,
                        **scatter_kwargs,
                    )

        axs_diag = np.diag(axs)
        for i, param_i in enumerate(param_subset):
            self.plot_1d_marginal(param_i, *plot_args, axs=axs[i, i], **plot_kwargs)
        for i in range(1, N_plots):
            axs[i, i].sharey(axs[i - 1, i - 1])

        return fig, axs

    @abc.abstractmethod
    def density(self, x: np.ndarray) -> np.ndarray:
        """
        Returns a value, proportional to the pdf of the distribution.

        Args:
            x (np.ndarray) : array of shape `(N_x, dim)` --- coordinates of N_x points

        Returns:
            np.ndarray : array of shape `(N_x)` --- value of density (up to normalization constant) at these points
        """
        pass

    @abc.abstractmethod
    def log_density(self, x: np.ndarray) -> np.ndarray:
        """
        Returns a value, equal to the log pdf of the distribution (up to an additive constant).

        Args:
            x : array of shape `(N_x, dim)` --- coordinates of N_x points

        Returns:
            np.ndarray : array of shape `(N_x)` --- value of log density (up to an additive constant) at these points
        """
        pass

    @abc.abstractmethod
    def get_marginal_on_grid(
        self, marginals: Union[int, Tuple[int], Tuple[int, int]]
    ) -> np.ndarray:
        """Returns a density of the marginal distribution of (x_i1 ... x_ik), discretized on the associated grid

        Note:
           Only 1 or 2 dimensional marginals supported. Order matters:

           >>> get_marginal_on_grid(i, j) == get_marginal_on_grid(j, i).T

        Args:
            marginals : indices of marginals to compute

        Returns:
            np.ndarray : array of shape `(grid.N[i])` or `(grid.N[i], grid.N[j])` --- density of the marginal distribution of x_i, discretized on the associated grid
        """
        pass


# TODO: move to utility?
def _get_correlated_gaussian_fn(
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    dim = mean.shape[0]

    precision = np.linalg.inv(covariance)
    scale = (2.0 * np.pi) ** (dim / 2.0) * np.linalg.det(covariance) ** 0.5

    def _dens(x: np.ndarray):
        x_centered = x - mean
        rho = (
            np.exp(
                -0.5 * np.einsum("...i,ij,...j->...", x_centered, precision, x_centered)
            )
            / scale
        )
        return rho

    return _dens


def _get_gaussian_mixture_fn(
    ms: List[np.ndarray],
    covs: List[np.ndarray],
    ws: np.ndarray,
):
    comps = [_get_correlated_gaussian_fn(mean, cov) for mean, cov in zip(ms, covs)]

    def _density(x: np.ndarray):
        comp_vals = np.stack([_comp(x) for _comp in comps])
        return np.einsum("i,i...", ws, comp_vals)

    return _density


def _gauss_mixture_process_args(
    means: Union[List[np.ndarray], np.ndarray],
    covariances: Union[List[Union[np.ndarray, np.float64]], np.ndarray, np.float64],
    weights: np.ndarray = None,
):
    if isinstance(means, np.ndarray) and len(means.shape) == 1:
        means = [means]

    dim = means[0].shape[-1]
    n_comp = len(means)

    assert all(_m.shape[0] == dim for _m in means)

    if not isinstance(covariances, List):
        covariances = [covariances]

    covariances_parsed = []

    for _cov in covariances:
        if isinstance(_cov, float):
            covariances_parsed.append(np.eye(dim) * _cov)
        elif isinstance(_cov, np.ndarray):
            if len(_cov.shape) == 1:
                _cov = np.diag(_cov)
            assert _cov.shape == (
                dim,
                dim,
            ), f"Shape of covariance matrix should be {dim, dim}, but got {_cov.shape}"
            covariances_parsed.append(_cov)
        else:
            raise ValueError

    if len(covariances_parsed) == 1:
        covariances_parsed *= n_comp

    assert len(covariances_parsed) == n_comp

    if weights is None:
        weights = np.ones(n_comp)

    assert weights.shape[0] == n_comp
    assert np.all(weights > 0.0)

    weights /= np.sum(weights)

    return means, covariances_parsed, weights


class GaussianMixture(DistributionOnGrid):
    """Gaussian Mixture distribution.

    Args:
        means (np.ndarray or List[np.ndarray]): means of each Gaussian components; either `np.ndarray` for one component or `List[np.ndarray]` of length equal to number of components
        covariances (float or List[float] or List[np.ndarray]): covariances of each Gaussian components; can be

            - A single scalar `c`. Then all the components have the same covariance matrix `c*np.eye(dim)`
            - A list of scalars `cs`. Then `i`-th component has covariance matrix `cs[i]*np.eye(dim)`
            - A list of `np.ndarray`. Then each of them should be an SPD matrix.

        weights (np.ndarray or None): positive weights for each component.
    """

    def __init__(
        self,
        grid: Grid,
        means: Union[List[np.ndarray], np.ndarray],
        covariances: Union[List[Union[np.ndarray, np.float64]], np.ndarray, np.float64],
        weights: np.ndarray = None,
    ):
        super().__init__(grid)
        self._means, self._covariances, self._weights = _gauss_mixture_process_args(
            means, covariances, weights
        )

        assert all(
            m.shape[0] == self.dim for m in self._means
        ), f"Dimension of the grid is {self._dim}, but the means and covariance have dimension {m.shape[0]}"

        self._comp_gens = [
            multivariate_normal(_mean, _cov)
            for _mean, _cov in zip(
                self._means,
                self._covariances,
            )
        ]
        self.density = _get_gaussian_mixture_fn(
            self._means, self._covariances, self._weights
        )

    def sample(self, n_samples: int):
        """Sample from the Gaussian Mixture distribution

        Args:
            n_samples : number of samples

        """
        nsamples_in_comp = multinomial.rvs(n_samples, self._weights)

        x = np.concatenate(
            [_gen.rvs(_n) for _n, _gen in zip(nsamples_in_comp, self._comp_gens)]
        )
        return x

    @property
    def means(self) -> List[np.ndarray]:
        return self._means

    @property
    def covariances(self) -> List[np.ndarray]:
        return self._covariances

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    # def density(self, x: np.ndarray) -> np.ndarray:
    #     comp_vals = np.stack([_comp(x) for _comp in self._comp_fns])
    #     return np.einsum("i,i...", self._weights, comp_vals)

    def log_density(self, x: np.ndarray) -> np.ndarray:
        return np.log(self.density(x))

    def get_marginal_on_grid(self, marginals):
        if isinstance(marginals, int):
            marginals = (marginals,)
        n_marg = len(marginals)
        assert n_marg in (1, 2)

        marg_idx = list(marginals)
        ms_marg = [_m[marg_idx] for _m in self._means]
        covs_marg = [_c[marg_idx, :][:, marg_idx] for _c in self._covariances]

        density_marginal = _get_gaussian_mixture_fn(ms_marg, covs_marg, self._weights)

        if n_marg == 2:
            X = np.stack(self.grid.get_2d_grid(*marginals), axis=-1)
        else:
            X = self.grid.get_1d_grid(*marginals).reshape((-1, 1))
        return density_marginal(X)


class DenseArrayDistribution(DistributionOnGrid):
    """Stores the density on the full grid. Run for dim <= 4, or face memory issues

    Args:
        rho (Callable) : function proportional to the density
    """

    def __init__(
        self,
        grid: Grid,
        rho: Callable,
    ):
        super().__init__(grid)

        self._rho = rho
        self._grids_1d = [
            np.linspace(
                self.grid.left[i],
                self.grid.right[i],
                self.grid.N_nodes[i],
                endpoint=True,
            )
            for i in range(self.dim)
        ]
        self._full_grid = None
        self._rho_on_grid = None
        self._normalization_const = 1.0

        if self._dim > 4:
            warnings.warn(
                f"Dimension {self._dim} > 4 can cause memory errors", ResourceWarning
            )

        else:
            self._set_full_grid

    def _set_full_grid(
        self,
    ):
        self._full_grid = np.stack(np.meshgrid(*self._grids_1d, indexing="ij"), axis=-1)
        old_shape = self._full_grid.shape[:-1]
        self._rho_on_grid = self._rho(self._full_grid.reshape(-1, self.dim)).reshape(
            old_shape
        )
        self._normalization_const = np.sum(self._rho_on_grid) * np.prod(self.grid.hx)
        self._rho_on_grid /= self._normalization_const

    def density(self, x: np.ndarray) -> np.ndarray:
        return self._rho(x) / self._normalization_const

    def log_density(self, x: np.ndarray) -> np.ndarray:
        return np.log(self.density(x))

    def get_marginal_on_grid(self, marginals):
        if self._rho_on_grid is None:
            self._set_full_grid()

        if isinstance(marginals, int):
            marginals = (marginals,)
        other_axes = set(range(self._dim))
        other_axes.difference_update(marginals)
        other_axes = list(other_axes)

        density_marginal = np.sum(self._rho_on_grid, axis=tuple(other_axes)) * np.prod(
            self.grid.hx[other_axes]
        )
        if len(marginals) == 2 and marginals[0] > marginals[1]:
            density_marginal = density_marginal.T

        return density_marginal

    @classmethod
    def get_nonconvex(cls, grid: Grid, a: np.ndarray) -> DenseArrayDistribution:
        """Get a test distribution with nonconvex potential

        .. math::

            \\rho_{\\text{NC}} \\propto \\exp(-V_{\\text{NC}})

            V_{ \\text{NC}}(x) =  \\left(  \\sum \\limits_{i = 1}^3  \\sqrt{|x_i - a_i|}  \\right)^2

        Args:
            grid : grid
            a : the shift of the distribution

        Returns:
            DenseArrayDistribution : the distribution
        """
        assert grid.dim == a.shape[0]

        def _log_prob_nonconvex(_x: np.ndarray) -> np.ndarray:
            return -np.linalg.norm(_x - a[np.newaxis, :], ord=0.5, axis=-1)

        distr = cls(grid, lambda _x: np.exp(_log_prob_nonconvex(_x)))
        distr.log_density = _log_prob_nonconvex

        return distr

    @classmethod
    def get_double_moon(cls, grid: Grid, a: np.float64) -> DenseArrayDistribution:
        """Get a test bimodal distribution of the form

        .. math::

             \\rho_{ \\text{DM}}(x)  \\propto  \\exp \\left( -2( \\|x \\|_2 - a)^2 \\right)  \\left(  \\exp(-2(x_1 -a)^2) +  \\exp(-2(x_1 +a)^2)  \\right)

        Args:
            grid : grid
            a : the shift of the distribution (int the first dimension)

        Returns:
            DenseArrayDistribution : the distribution

        """

        def _prob_double_moon(_x: np.ndarray) -> np.ndarray:
            nx = np.linalg.norm(_x, ord=2, axis=-1)
            x1 = _x[..., 0]
            return np.exp(-2.0 * (nx - a) ** 2) * (
                np.exp(-2.0 * (x1 - a) ** 2) + np.exp(-2.0 * (x1 + a) ** 2)
            )

        return cls(grid, _prob_double_moon)


class TensorTrainDistribution(DistributionOnGrid):
    """Distribution with density on grid stored in the compressed TT format

    Args:
        rho_tt (tt_vector) : Tensor Train decomposition of the density on grid
    """

    def __init__(
        self,
        grid: Grid,
        rho_tt: tt_vector,
    ):
        super().__init__(grid)
        self.rho_tt = rho_tt
        self._normalization_const = teneva.sum(self.rho_tt) * np.prod(self.grid.hx)
        self.rho_tt = teneva.mul(self.rho_tt, 1.0 / self._normalization_const)
        self._rho_getter = teneva.act_one.getter(self.rho_tt)

    def density(self, x: np.ndarray) -> np.ndarray:
        """
        Note:
            Returns a nearest interpolation; Will implement linear interpolation in the future...
        """
        I = teneva.poi_to_ind(x, *self.grid)
        return teneva.act_one.get_many(self.rho_tt, I)

    def log_density(self, x: np.ndarray) -> np.ndarray:
        return np.log(self.density(x))

    def get_marginal_on_grid(self, marginals):
        if isinstance(marginals, int):
            marginals = (marginals,)
        other_axes = set(range(self._dim))
        other_axes.difference_update(marginals)
        other_axes = list(other_axes)

        density_marginal = tt_sum_multi_axis(self.rho_tt, axis=tuple(other_axes))
        density_marginal = teneva.full(density_marginal) * np.prod(
            self.grid.hx[other_axes]
        )
        if len(marginals) == 2 and marginals[0] > marginals[1]:
            density_marginal = density_marginal.T

        return density_marginal

    def get_credible_interval(self, i: int, prob: float):
        rho_1d_grid = self.get_marginal_on_grid(i)
        x_1d = self.grid.get_1d_grid(i)

        # rho_1d_grid[rho_1d_grid < 0.0] = 0.0
        # rho_1d_grid += 1e-30
        # rho_1d_grid /= np.sum(rho_1d_grid) * self.grid.hx[i]

        rho_1d = lambda x: np.interp(x, x_1d, rho_1d_grid)

        event = lambda t, y: y[0] - prob
        event.terminal = True

        def _interval_len(left: float):
            ode_result = solve_ivp(
                lambda t, y: (rho_1d(t),),
                [
                    left,
                    self.grid.right[i],
                ],
                (0.0,),
                events=event,
                max_step=self.grid.hx[i]/2.,
            )
            right = ode_result.t_events[-1]
            if len(right) > 0:
                return right[-1] - left
            return 1e30

        opt_res = minimize_scalar(
            _interval_len,
            bracket=(x_1d[0], x_1d[-1]),
            method="Golden",
        )
        left = opt_res.x
        right = np.minimum(opt_res.x + opt_res.fun, self.grid.right[i])
        return (left, right)

    @classmethod
    def rank1_fx(
        cls, grid: Grid, fns: Union[List[Callable], Callable]
    ) -> TensorTrainDistribution:
        """Convenience function to create a rank-1 TT with components

        .. math::

            A^i_{1k1} = f^i(x_{i,k}),\\ i = \\overline{1,\\ d}

        where :math:`x_{i,k},\\ k = \\overline{1, N_i}` is the unidimensional grid in i-th direction

        Args:
            grid: the grid to discretize on
            fns: functions :math:`f^i`

        Returns:
            TensorTrainDistribution : the distribution
        """
        if not isinstance(fns, Iterable):
            fns = [
                fns,
            ] * grid.dim
        assert len(fns) == grid.dim

        tt_nodes = [
            f(grid.get_1d_grid(i)).reshape((1, -1, 1)) for i, f in enumerate(fns)
        ]
        return cls(grid, tt_nodes)

    @classmethod
    def gaussian(
        cls,
        grid: Grid,
        ms: Union[float, List[float], np.ndarray] = 0.0,
        sigmas: Union[float, List[float], np.ndarray] = 1.0,
    ) -> TensorTrainDistribution:
        """A TT approximation of the density of the distribution with each parameter being independent and distributed as

        .. math::

            x_i \sim \\mathcal{N}(m_i, \\sigma_i),\\ i = \\overline{1,\\ d}

        Args:
            grid: the grid to discretize on
            ms: means :math:`m_i` of each parameter
            sigmas: standard deviations :math:`sigma_i` of each parameter

        Returns:
            TensorTrainDistribution : the distribution
        """
        if not isinstance(ms, Iterable):
            ms = [
                ms,
            ] * grid.dim
        if not isinstance(sigmas, Iterable):
            sigmas = [
                sigmas,
            ] * grid.dim

        assert len(ms) == grid.dim
        assert len(sigmas) == grid.dim

        fns = [norm(loc=m, scale=sigma).pdf for m, sigma in zip(ms, sigmas)]

        return cls.rank1_fx(grid, fns)
