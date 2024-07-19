from typing import Literal, Callable, Tuple, List, Union, TypeAlias, Dict
from dataclasses import dataclass
from functools import cache
from docstring_inheritance import GoogleDocstringInheritanceInitMeta
import warnings

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import expm, eigsh
from scipy.integrate import RK45
import teneva

import matplotlib.pyplot as plt

from time import perf_counter

from .utility import tt_sum_multi_axis, tt_slice, tt_vector
from .DistributionOnGrid import Grid, TensorTrainDistribution


@cache
def _get_dxx_matrix(N: int, hx: np.float64) -> np.ndarray:
    """Computes a matrix of a one-dimensional second-order finite difference Laplacian on a uniform grid

    Args:
        N (int) : number of nodes in the grid
        hx (float) : stepsize of the grid
    """
    sub_diag = np.full((N - 1,), 1.0)
    main_diag = np.concatenate(
        (
            (-1.0,),
            np.full((N - 2,), -2.0),
            (-1.0,),
        )
    )

    Lap = (
        diags(
            [sub_diag, main_diag, sub_diag],
            offsets=(-1, 0, 1),
        )
        / hx**2
    )

    return Lap


@cache
def _get_heat_equation_matrix(N: int, hx: np.float64, s: np.float64) -> np.ndarray:
    """Computes matrix of the operator  :math:`e^{s\\Delta}`, which is a discrete solution of the heat equation on the grid.

    Args:
        N (int) : number of nodes in the grid
        hx (float) : stepsize of the grid
        s (float) : effective timestep
    """
    Lap = _get_dxx_matrix(N, hx)
    expLap = expm(s * Lap)
    expLap = expLap.todense()
    return expLap


def _solve_heat_TT(
    eta: tt_vector,
    beta: np.float64,
    T: np.float64,
    grid: Grid,
) -> tt_vector:
    """Solves the heat equation for initial data represented in TT format

    Args:
       eta: initial data
    """
    mat_exps = [
        _get_heat_equation_matrix(grid.N_nodes[i], grid.hx[i], beta * T)
        for i in range(grid.dim)
    ]

    return [np.einsum("ij,kjl->kil", U, v) for U, v in zip(mat_exps, eta)]


def _fixed_point_picard(
    x_cur: tt_vector,
    g_cur: tt_vector,
    *args,
    relaxation: np.float64 = 1.0,
    max_rank: int = 10,
    trunc_tol: float = 1e-16,
) -> Tuple[tt_vector, np.float64]:
    x_new = teneva.truncate(
        teneva.add(
            teneva.mul((1.0 - relaxation), x_cur), teneva.mul(relaxation, g_cur)
        ),
        e=trunc_tol,
        r=max_rank,
    )

    return x_new, relaxation


def _fixed_point_aitken(
    x_cur: tt_vector,
    g_cur: tt_vector,
    x_prev: tt_vector,
    g_prev: tt_vector,
    relaxation: np.float64 = 1.0,
    max_rank: int = 10,
    trunc_tol: float = 1e-16,
) -> Tuple[tt_vector, np.float64]:
    f_prev = teneva.act_two.sub(g_prev, x_prev)
    f_cur = teneva.act_two.sub(g_cur, x_cur)

    f_diff = teneva.act_two.sub(f_cur, f_prev)

    nom = teneva.mul_scalar(f_prev, f_diff)
    denom = teneva.mul_scalar(f_diff, f_diff)

    relaxation_new = -relaxation * nom / denom if denom >= 1e-16 else relaxation
    relaxation_new = np.clip(relaxation_new, 1e-16, 1.0)

    x_new, _ = fixed_point_picard(
        x_cur, g_cur, relaxation=relaxation_new, max_rank=max_rank
    )

    return x_new, relaxation_new


def _fixed_point_2_anderson(
    x_cur: tt_vector,
    g_cur: tt_vector,
    x_prev: tt_vector,
    g_prev: tt_vector,
    relaxation: np.float64 = 0.9,
    max_rank: int = 10,
    trunc_tol: float = 1e-16,
) -> Tuple[tt_vector, np.float64]:
    f_prev = teneva.act_two.sub(g_prev, x_prev)
    f_cur = teneva.act_two.sub(g_cur, x_cur)

    f_diff = teneva.act_two.sub(f_cur, f_prev)

    nom = -teneva.mul_scalar(f_prev, f_diff)
    denom = teneva.mul_scalar(f_diff, f_diff)
    if denom <= 1e-16:
        x_new, _ = _fixed_point_picard(
            x_cur, g_cur, relaxation=relaxation, max_rank=max_rank
        )
        return x_new, relaxation
    alpha = nom / denom
    alpha = np.clip(alpha, 0.0, 2.0)

    print(f"\tAnderson step with {alpha=:.3e}", flush=True)

    # TODO: ugly! write conveniece teneva lincomb?
    x_new = teneva.add_many(
        [
            teneva.mul(alpha * relaxation, g_cur),
            teneva.mul(alpha * (1.0 - relaxation), x_cur),
            teneva.mul((1.0 - alpha) * relaxation, g_prev),
            teneva.mul((1.0 - alpha) * (1.0 - relaxation), x_cur),
        ],
        r=max_rank,
        e=trunc_tol,
    )

    return x_new, relaxation


@dataclass
class TensorTrainSolverParams:
    """Store the solver params such as tolerances and number of iterations. Default values provided.

    These parameters can be changed between or possibly even during the fixed-point iterations.

    """

    cross_nfev_with_posterior: int = int(4e5)
    """Number of function evaluation in TT-cross when real posterior calls are required (terminal condition and KL)"""
    cross_nfev_no_posterior: int = int(1e6)
    """Number of function evaluation in TT-cross when real posterior calls are not required (initial condition and KL estimation)"""
    cross_rel_diff: float = 1e-6
    """Cross stopping criterion: if the solution relative change is less, cross stops"""
    cross_use_validation: bool = False
    """If to use error on validation subset stopping criterion during the cross approximation iteration"""
    cross_validation_rtol: float = 1e-6
    """Cross stopping criterion: if error on validation subset is less, cross stops """
    cross_n_validation: int = 1000
    """Number of indices in the validation subset"""

    max_rank_eta: int = 5
    """Maximal TT-rank when representing the entropic potential :math:`\\eta`"""
    max_rank_hat_eta: int = 5
    """Maximal TT-rank when representing the entropic potential :math:`\\hat\\eta`"""
    max_rank_density: int = 20
    """Maximal TT-rank when representing the next distribution"""

    trunc_tol_hat_eta: float = 1e-13
    """TT truncation tolerance for potential :math:`\\hat\\eta` (i.e. in the initial condition)"""
    trunc_tol_eta: float = 1e-13
    """TT truncation tolerance for potential :math:`\\eta` (i.e. in the terminal condition)"""
    trunc_tol_density: float = 1e-13
    """TT truncation tolerance when computing the next density"""

    fp_method: Literal["picard", "2_anderson", "aitken"] = "2_anderson"
    """Method of solving the fixed-point problem at a step.

        - `picard` : The simplest method, next iterate is a linear combination of current iterate and the operator at current value with fixed relaxation factor
        - `aitken` : Same as `picard`, but the relaxation is selected adaptively. Supposed to be more robust
        - `2_anderson` : Saves the history of `2` previous iterates and function values and generates a next iterate based on a minimization subproblem.
    """
    fp_relaxation: float = 0.9
    """Relaxation for the fixed-point method """
    fp_stopping_rtol: float = 1e-8
    """Relative tolerance of the fixed-point iteration; if :math:`\\frac{\\|x_k  - G(x_k)\\|_2}{\\|x_k\\|_2}` is smaller than this value, the iteration terminates"""
    fp_max_iter: int = 100
    """Fixed-point solution will terminate after doing this amount of iterations regardless of the convergence"""
    zero_threshold: float = 1e-16
    """Safeguard when computing small values that must be positive, but could go below zero due to numerical erros"""

    sampling_ode_rtol: float = 1e-3
    """Tolerance of solving the sampling ODE"""
    sampling_ode_atol: float = 1e-6
    """Tolerance of solving the sampling ODE"""
    sampling_sde_fraction: float = 5e-3
    """Fraction of each timestep to be solved with the SDE dynamic"""
    sampling_n_euler_maruyama_steps: int = 50
    """Number of Euler-Maruyama steps used in the SDE solution"""

    @property
    def max_rank(self) -> int:
        return max(self.max_rank_eta, self.max_rank_hat_eta, self.max_rank_density)

    @max_rank.setter
    def max_rank(self, r: int):
        self.max_rank_eta = r
        self.max_rank_hat_eta = r
        self.max_rank_density = r

    @property
    def trunc_tol(self) -> int:
        return max(self.trunc_tol_eta, self.trunc_tol_hat_eta, self.trunc_tol_density)

    @trunc_tol.setter
    def trunc_tol(self, tol: int):
        self.trunc_tol_eta = tol
        self.trunc_tol_hat_eta = tol
        self.trunc_tol_density = tol


class TensorTrainSolver(metaclass=GoogleDocstringInheritanceInitMeta):
    """Approximate a given distribution `rho_infty` by entropy-regularized JKO scheme with finite-difference spatial discretization and Tensor-Train decomposition.

    Args:
        rho_infty (Callable) : function, proportional to the probability density of the posterior. Should have signature `(N_samples, dim) -> (dim, )`
        rho_start (TensorTrainDistribution) : starting distribution, discretized on grid. The posterior will be approximated on the same grid.
        posterior_cache_size (int): maximal size of the cache
        solver_params (TensorTrainSolverParams) : self-explanatory; if not given, uses defaults defined in `TensorTrainSolverParams` class

    Attributes:
        grid (Grid) : self-explanatory
        n_calls (int) : tracks the amount of real calls to the posterior durnig the solve
        n_cache (int) : tracks the amount of posterior calls loaded from cache during the solve
        Ts (List[float]) : timesteps taken
        betas (List[float]) : regularization factors at steps taken
    """

    def __init__(
        self,
        rho_infty: Callable,
        rho_start: TensorTrainDistribution,
        solver_params: TensorTrainSolverParams = None,
        posterior_cache_size: int = int(1e6),
    ):
        if solver_params is None:
            solver_params = TensorTrainSolverParams()  # default params
        self._rho_infty = rho_infty
        self.grid = rho_start.grid
        self.params = solver_params

        # Cache-related stuff
        self.posterior_cache_max_size = posterior_cache_size
        self.n_calls = 0
        self.n_cache = 0
        self._posterior_cache = {}

        # Stored data for sampling:
        self.Ts = []
        self.betas = []
        self._etas_t1: List[tt_vector] = []
        self._hat_etas_t0: List[tt_vector] = []

        # Other history data
        self.KLs = []
        self.KLs_est = []

        # Initialize variables for step
        self._rho_cur, self._eta_cur, self._hat_eta_cur = self._init_potentials(
            rho_start.rho_tt
        )

    def _init_potentials(
        self,
        rho_0: tt_vector,
    ) -> Tuple[tt_vector, tt_vector, tt_vector,]:
        eta = teneva.cross(
            lambda _I: np.sqrt(teneva.act_one.get_many(rho_0, _I)),
            teneva.tensors.const(teneva.props.shape(rho_0), 1.0),
            m=self.params.cross_nfev_no_posterior,
        )

        hat_eta = teneva.truncate(
            eta, self.params.trunc_tol_hat_eta, self.params.max_rank_hat_eta
        )
        eta = teneva.truncate(eta, self.params.trunc_tol_eta, self.params.max_rank_eta)

        rho_0 = teneva.truncate(
            rho_0, self.params.trunc_tol_density, self.params.max_rank_density
        )

        return rho_0, eta, hat_eta

    def _eval_posterior_cached(
        self,
        _I: Union[Tuple[int], np.ndarray[int]],
        update=True,
    ):
        """Maintains the posterior cache. Cache is filled up to `self.posterior_cache_max_size`, and, for indicies already in cache, the call to real posterior is not performed

        Args:
            _I: array of shape `(N_samples, dim)` of integer indices to compute the posterior at.

        """
        I_and_idx_new = [
            (i, idx)
            for idx, i in enumerate(_I)
            if tuple(i) not in self._posterior_cache
        ]

        I_new = np.array([_v[0] for _v in I_and_idx_new])
        idx_new = [_v[1] for _v in I_and_idx_new]

        idx_old = [_i for _i in range(_I.shape[0]) if _i not in idx_new]
        I_old = _I[idx_old, :]

        Y_old = np.array([self._posterior_cache[tuple(_i)] for _i in I_old])

        Y_return = np.zeros(_I.shape[0])
        Y_return[idx_old] = Y_old

        if len(idx_new) > 0:
            x_new = teneva.ind_to_poi(I_new, *self.grid)

            Y_new = self._rho_infty(x_new)
            Y_return[idx_new] = Y_new

        if update:
            n_add_to_cache = min(
                len(idx_new),
                self.posterior_cache_max_size - len(self._posterior_cache),
            )

            for k, i in enumerate(I_new[:n_add_to_cache]):
                self._posterior_cache[tuple(i)] = float(Y_new[k])

            self.n_cache += len(idx_old)
            self.n_calls += len(idx_new)

        return Y_return

    def _solve_initial_condition(
        self,
        eta: tt_vector,
        hat_eta_initial: tt_vector = None,
    ) -> tt_vector:
        """Computes the potential hat_eta such that :math:`\\hat\\eta(0, x)\\cdot \\eta(0, x) = \\rho^k(x)`

        Args:
            eta : other potential, at t = 0 and at current fixed-point iteration
            hat_eta_initial : initial guess for the cross approximation
        """
        rhs_fn = lambda _I: teneva.act_one.get_many(self._rho_cur, _I) / np.maximum(
            teneva.act_one.get_many(eta, _I), self.params.zero_threshold
        )
        if self.params.cross_use_validation:
            I_test = teneva.sample_rand(
                teneva.props.shape(eta), self.params.cross_n_validation
            )
            Y_test = rhs_fn(I_test)
            e_test = (
                teneva.sum(hat_eta_initial)
                / np.prod(self.grid.N_nodes)
                * self.params.cross_validation_rtol
            )
            # old behavior; leave for reproducibility
            # norm_test = np.linalg.norm(Y_test.ravel(), ord=np.inf)
        else:
            I_test = None
            Y_test = None
            e_test = None

        info = {}

        print("\tSolving initial condition  ", end="", flush=True)
        t = perf_counter()
        hat_eta = teneva.cross(
            rhs_fn,
            Y0=hat_eta_initial,
            m=self.params.cross_nfev_no_posterior,
            e=self.params.cross_rel_diff,
            info=info,
            I_vld=I_test,
            y_vld=Y_test,
            e_vld=e_test,
        )
        dt = perf_counter() - t

        # Some diagnostic printing
        rank_str = " ".join([f"{_r:2d}" for _r in teneva.ranks(hat_eta)])
        ncalls = info["m"]
        stop_condition = info["stop"]
        ncalls_field_len = len(
            str(
                max(
                    self.params.cross_nfev_no_posterior,
                    self.params.cross_nfev_with_posterior,
                )
            )
        )
        ncalls_str = f"{ncalls:{ncalls_field_len}d}"
        print(
            f"ranks= [{rank_str}] {dt=:.2e} n_calls= {ncalls_str} {stop_condition=}",
            flush=True,
        )

        # old behavior
        # err_before_round = lp_err_test(hat_eta, rhs_fn, 1000)
        # if err_before_round >= 1e-1:
        #     raise RuntimeError("Cross diverged")

        print("\tRounding                   ", end="", flush=True)
        t = perf_counter()
        hat_eta = teneva.truncate(
            hat_eta, self.params.trunc_tol_hat_eta, self.params.max_rank_hat_eta
        )
        dt = perf_counter() - t
        rank_str = " ".join([f"{_r:2d}" for _r in teneva.ranks(hat_eta)])

        print(
            f"ranks= [{rank_str}] {dt=:.2e}",
            flush=True,
        )

        # Some further diagnostic printing
        rank_str = " ".join([f"{_r:2d}" for _r in teneva.ranks(hat_eta)])
        # err_after_round = lp_err_test(hat_eta, rhs_fn, 1000)

        return hat_eta

    def _solve_terminal_condition(
        self,
        hat_eta: tt_vector,
        beta: np.float64,
        eta_initial: tt_vector = None,
    ) -> tt_vector:
        """Computes the potential eta such that the terminal condition for the minimization of :math:`\\operatorname{KL}` divergence holds, i.e.

        .. math::

            \\eta(T, x)\\cdot = \\left( \\frac{\\rho^\\infty(x)}{\\hat\\eta(T, x)}\\right)^{\\frac{1}{1 + 2 \\beta}}

        Args:
            hat_eta : other potential, at t = T and at current fixed-point iteration
            eta_initial : initial guess for the cross approximation
        """
        # I decided not to update cache in the test calls to estimate the influence only of the indices, selected by maxvol inside tt_cross; maybe should turn update=True in real application

        if self.params.cross_use_validation:
            rhs_fn = lambda _I: (
                self._eval_posterior_cached(_I, update=False)
                / np.maximum(
                    teneva.act_one.get_many(hat_eta, _I), self.params.zero_threshold
                )
            ) ** (1.0 / (1.0 + 2.0 * beta))
            I_test = teneva.sample_rand(
                teneva.props.shape(eta), self.params.cross_n_validation
            )
            Y_test = rhs_fn(I_test)
            e_test = (
                teneva.sum(hat_eta_initial)
                / np.prod(self.grid.N_nodes)
                * self.params.cross_validation_rtol
            )
            # old behavior; leave for reproducibility
            # norm_test = np.linalg.norm(Y_test.ravel(), ord=np.inf)
        else:
            I_test = None
            Y_test = None
            e_test = None

        rhs_fn_cached = lambda _I: (
            self._eval_posterior_cached(_I)
            / np.maximum(
                teneva.act_one.get_many(hat_eta, _I), self.params.zero_threshold
            )
        ) ** (1.0 / (1.0 + 2.0 * beta))
        info = {}

        print("\tSolving terminal condition ", end="", flush=True)
        t = perf_counter()
        eta = teneva.cross(
            rhs_fn_cached,
            eta_initial,
            e=self.params.cross_rel_diff,
            m=self.params.cross_nfev_with_posterior,
            info=info,
            I_vld=I_test,
            y_vld=Y_test,
            e_vld=e_test,
        )
        dt = perf_counter() - t
        # Some diagnostic printing
        rank_str = " ".join([f"{_r:2d}" for _r in teneva.ranks(eta)])
        ncalls = info["m"]
        stop_condition = info["stop"]
        ncalls_field_len = len(
            str(
                max(
                    self.params.cross_nfev_no_posterior,
                    self.params.cross_nfev_with_posterior,
                )
            )
        )
        ncalls_str = f"{ncalls:{ncalls_field_len}d}"
        print(
            f"ranks= [{rank_str}] {dt=:.2e} n_calls= {ncalls_str} {stop_condition=}",
            flush=True,
        )
        print("\tRounding                   ", end="", flush=True)
        t = perf_counter()
        eta = teneva.truncate(
            eta, e=self.params.trunc_tol_eta, r=self.params.max_rank_eta
        )
        dt = perf_counter() - t
        # Some further diagnostic printing
        rank_str = " ".join([f"{_r:2d}" for _r in teneva.ranks(eta)])
        print(
            f"ranks= [{rank_str}] {dt=:.2e}",
            flush=True,
        )
        return eta

    def _fixed_point_inner_cycle(
        self,
        eta: tt_vector,
        beta: np.float64,
        T: np.float64,
        start_value_init: tt_vector = None,
        start_value_term: tt_vector = None,
    ):
        """The operator for the fixed point iteration.

        Args:
            eta:

        """
        eta_t0 = _solve_heat_TT(eta, -beta, -T, self.grid)  # (4.5.2)

        hat_eta_t0 = self._solve_initial_condition(
            eta_t0,
            hat_eta_initial=start_value_init,
        )  # (4.5.3)
        # TODO: A. proper linear solver

        hat_eta_next = _solve_heat_TT(hat_eta_t0, beta, T, self.grid)  # (4.5.1)
        eta_next = self._solve_terminal_condition(
            hat_eta_next,
            beta,
            eta_initial=start_value_term,
        )

        n_eta = teneva.sum(eta_next)
        n_hat_eta = teneva.sum(hat_eta_next)

        C = np.sqrt(n_hat_eta / n_eta)
        eta_next = teneva.mul(C, eta_next)
        hat_eta_next = teneva.mul(1.0 / C, hat_eta_next)

        return eta_next, hat_eta_next, eta_t0, hat_eta_t0

    def _KL(
        self,
    ) -> np.float64:
        """Estimates :math:`\\operatorname{KL}(\\rho^k|\\rho^\\infty)` for the current iteration. Relies on the updated value of `_rho_cur`.

        Note:
            Uses actual posterior calls, thus can be slow.

        Returns :
            float : the value of the divergence.
        """
        # TODO: this is not working properly if the state hasn't been set properly! fix.
        print("Computing KL err", flush=True)
        rhs_fn = lambda _S: np.log(
            np.maximum(
                teneva.act_one.get_many(self._rho_cur, _S),
                self.params.zero_threshold,
            )
            / np.maximum(
                self._eval_posterior_cached(
                    _S,
                    update=False,
                ),
                self.params.zero_threshold,
            )
        ) * teneva.act_one.get_many(self._rho_cur, _S)

        log_quot = teneva.cross(
            rhs_fn,
            teneva.tensors.const(
                teneva.props.shape(self._rho_cur), v=self.params.zero_threshold
            ),
            e=self.params.cross_rel_diff,
            m=self.params.cross_nfev_with_posterior,
        )
        return teneva.sum(
            log_quot,
        ) * np.prod(self.grid.hx)

    def _KL_est(
        self,
        beta,
    ) -> np.float64:
        """Estimates :math:`\\operatorname{KL}(\\rho^k|\\rho^\\infty)` for the current iteration with the formula :math:`-2\\beta\\int\\log \\eta \\rho^k`. Relies on the updated values of `_rho_cur`, `_eta_cur`.

        Note:
            Doesn't use posterior calls.

        Returns :
            float : the value of the divergence.
        """
        print("Computing KL estimate", flush=True)

        log_quot = teneva.cross(
            lambda _I: (
                -2.0
                * beta
                * np.log(
                    np.maximum(
                        self.params.zero_threshold,
                        teneva.act_one.get_many(self._eta_cur, _I),
                    )
                )
            )
            * teneva.act_one.get_many(self._rho_cur, _I),
            teneva.tensors.const(
                teneva.props.shape(self._eta_cur), self.params.zero_threshold
            ),
            m=self.params.cross_nfev_no_posterior,
            e=self.params.cross_rel_diff,
        )
        return teneva.sum(log_quot) * np.prod(self.grid.hx)

    def step(
        self,
        beta: np.float64,
        T: np.float64,
        save_history: bool = False,
    ) -> Tuple[tt_vector, tt_vector]:
        """Perform a regularized JKO step.

        TODO implement more detailed description, with math?!

        Args:
            beta : regularization factor
            T : timestep
            save_history : if `True`, returns the intermediate ranks, errors etc during the fixed-point iterations
        """

        fp_relaxation = self.params.fp_relaxation

        start_value_init = self._hat_eta_cur
        start_value_term = self._eta_cur

        eta_cur = self._eta_cur
        try:
            print(f"Initializing FP with method={self.params.fp_method}")
            (
                tilde_eta_cur,
                hat_eta_cur,
                eta_t0,
                hat_eta_t0,
            ) = self._fixed_point_inner_cycle(
                eta_cur,
                beta,
                T,
                start_value_init=start_value_init,
                start_value_term=start_value_term,
            )
        except (RuntimeWarning, RuntimeError, FloatingPointError) as e:
            print(
                "".join(traceback.TracebackException.from_exception(e).format()),
                flush=True,
            )
            # TODO: save the correct state!
            raise RuntimeError(eta_cur, hat_eta_cur)

        # TODO: copy here???
        eta_prev = eta_cur
        tilde_eta_prev = tilde_eta_cur

        abs_errors = []
        rel_errors = []
        relaxations = []
        ranks = []

        fp_err_old = np.inf

        for _i in range(self.params.fp_max_iter):
            print(
                f"Starting FP step {_i + 1} rel_err {fp_err_old:.2e}",
                flush=True,
            )
            try:
                (
                    tilde_eta_cur,
                    hat_eta_cur,
                    eta_t0,
                    hat_eta_t0,
                ) = self._fixed_point_inner_cycle(
                    eta_cur,
                    beta,
                    T,
                    start_value_init=start_value_init,
                    start_value_term=start_value_term,
                )
            except (RuntimeWarning, RuntimeError, FloatingPointError) as e:
                print(
                    "".join(traceback.TracebackException.from_exception(e).format()),
                    flush=True,
                )

                if save_history:
                    raise RuntimeError(
                        eta_cur, hat_eta_cur, abs_errors, rel_errors, relaxations, ranks
                    )
                else:
                    raise RuntimeError(eta_cur, hat_eta_cur)

            start_value_init = hat_eta_t0
            start_value_term = eta_cur

            abs_err = teneva.norm(teneva.sub(tilde_eta_cur, eta_cur))
            fp_rel_err = abs_err / teneva.norm(eta_cur)
            rk = max(teneva.props.ranks(eta_cur))

            abs_errors.append(abs_err)
            rel_errors.append(fp_rel_err)
            relaxations.append(fp_relaxation)
            ranks.append(rk)

            if (
                fp_rel_err <= self.params.fp_stopping_rtol
            ):  # or fp_rel_err >= 1.2 * fp_err_old:
                break

            fp_err_old = fp_rel_err

            # TODO: a proper class for the update, i.e, for anderson > 2 managing the history
            if _i > 1:
                if self.params.fp_method == "aitken":
                    fixed_point_update = _fixed_point_aitken
                elif self.params.fp_method == "2_anderson":
                    fixed_point_update = _fixed_point_2_anderson
                else:
                    fixed_point_update = _fixed_point_picard
            else:
                fixed_point_update = _fixed_point_picard

            eta_next, fp_relaxation = fixed_point_update(
                eta_cur,
                tilde_eta_cur,
                eta_prev,
                tilde_eta_prev,
                relaxation=fp_relaxation,
                max_rank=self.params.max_rank_eta,
                trunc_tol=self.params.trunc_tol_eta,
            )

            eta_prev = eta_cur.copy()
            tilde_eta_prev = tilde_eta_cur.copy()

            eta_cur = eta_next

        self._etas_t1.append(eta_cur)
        self._hat_etas_t0.append(hat_eta_t0)

        self.Ts.append(T)
        self.betas.append(beta)

        self._eta_cur = eta_cur
        self._hat_eta_cur = hat_eta_cur

        rho_cur = teneva.truncate(
            teneva.mul(eta_cur, hat_eta_cur),
            e=self.params.trunc_tol_density,
            r=self.params.max_rank_density,
        )
        Z_const = teneva.sum(rho_cur) * np.prod(self.grid.hx)
        rho_cur = teneva.mul(rho_cur, 1.0 / Z_const)

        self._rho_cur = rho_cur

        self.KLs.append(self._KL())
        self.KLs_est.append(self._KL_est(beta))

        if save_history:
            return (
                abs_errors,
                rel_errors,
                relaxations,
                ranks,
            )
        else:
            return

    def _get_drift_terms_fn(
        self,
        eta_t1: tt_vector,
        hat_eta_t0: tt_vector,
        T: np.float64,
        beta: np.float64,
    ) -> Callable:
        """Given potentials :math:`\\eta(T),\\ \\hat\\eta(0)` (from the converged solution for one step), compute the drift terms for the ODE and SDE, describing the interpolating dynamics.

        .. math::

            \\dot x(t) = \\tilde v_t(x, t)

            \\tilde v_t(x, t) = \\beta \\nabla\\left(\\log\\eta(t, x) - \\log\\hat\\eta(t, x)\\right)

            \\mathrm{d}X_t = v(t, X)\\mathrm{d}t + \\sqrt{\\beta} \\mathrm{d}W_t

            v(X, t) = 2\\beta\\nabla{\\log\\eta(X_t, t)}

        The derivatives are computed with a 2-nd order finite difference formula.

        Args:
            eta_t1 : potential :math:`\\eta(T)`
            hat_eta_t0 : :math:`\\hat\\eta(0)`
            T : timestep
            beta : regularization parameter

        Returns:
            Callable: a function such that :math:`(t, x) \\mapsto (\\tilde v(t, x),\\ v(t, x))`
        """
        dim = self.grid.dim
        shift = np.eye(self.grid.dim, dtype=int)
        ode_potential_fn = lambda _eta, _hat_eta: np.log(
            np.maximum(_eta / _hat_eta, self.params.zero_threshold)
        )
        sde_potential_fn = lambda _eta: np.log(
            np.maximum(_eta, self.params.zero_threshold)
        )

        def _v(t: np.float64, x: np.ndarray):
            assert t >= 0.0 and t <= T
            Tau_fwd = t
            Tau_bwd = T - t
            hat_eta = _solve_heat_TT(hat_eta_t0, beta, Tau_fwd, self.grid)
            eta = _solve_heat_TT(eta_t1, -beta, -Tau_bwd, self.grid)

            ind_x = teneva.poi_to_ind(x, *self.grid)
            eta_hat_0 = teneva.act_one.get_many(hat_eta, ind_x)
            eta_0 = teneva.act_one.get_many(eta, ind_x)

            # evaluate indices of grid points k_i + 1, i = 1, ... d
            ind_xp = ind_x[..., np.newaxis, :] + shift  # xp_ijk = x_ki + h_x*delta_ij
            ind_xp = np.where(
                ind_xp < np.array(self.grid.N_nodes)[np.newaxis, :], ind_xp, ind_xp - 1
            )

            # evaluate indices of grid points k_i - 1, i = 1, ... d
            ind_xm = ind_x[..., np.newaxis, :] - shift  # xm_ijk = x_ki + h_x*delta_ij
            ind_xm = np.where(ind_xm > -1, ind_xm, ind_xm + 1)  # "padding"

            eta_p = teneva.act_one.get_many(eta, ind_xp.reshape((-1, dim))).reshape(
                (-1, dim)
            )
            eta_hat_p = teneva.act_one.get_many(
                hat_eta, ind_xp.reshape((-1, dim))
            ).reshape((-1, dim))
            eta_m = teneva.act_one.get_many(eta, ind_xm.reshape((-1, dim))).reshape(
                (-1, dim)
            )
            eta_hat_m = teneva.act_one.get_many(
                hat_eta, ind_xm.reshape((-1, dim))
            ).reshape((-1, dim))

            # evaluate  \beta(\eta - )
            ode_potential_p = beta * ode_potential_fn(eta_p, eta_hat_p)
            ode_potential_0 = beta * ode_potential_fn(eta_0, eta_hat_0)
            ode_potential_m = beta * ode_potential_fn(eta_m, eta_hat_m)

            sde_potential_p = 2.0 * beta * sde_potential_fn(eta_p)
            sde_potential_0 = 2.0 * beta * sde_potential_fn(eta_0)
            sde_potential_m = 2.0 * beta * sde_potential_fn(eta_m)

            # get a 2-nd order finite difference expansion coefs
            c_x = (x - ind_x * self.grid.hx[np.newaxis, :]) / self.grid.hx[
                np.newaxis, :
            ]
            _u = np.ones_like(ode_potential_p)
            _M = np.array(
                [
                    [_u, _u, _u],
                    [-(1.0 + c_x), -c_x, (1.0 - c_x)],
                    [(1.0 + c_x) ** 2, c_x**2, (1.0 - c_x)],
                ]
            )
            _M = np.moveaxis(_M, (2, 3), (0, 1))
            _rhs = np.array([0.0, 1.0, 0.0])
            A = np.linalg.solve(_M, _rhs[np.newaxis, np.newaxis])

            ode_grad = (
                ode_potential_m * A[:, :, 0]
                + ode_potential_0[:, np.newaxis] * A[:, :, 1]
                + ode_potential_p * A[:, :, 2]
            ) / self.grid.hx[np.newaxis, :]

            sde_grad = (
                sde_potential_m * A[:, :, 0]
                + sde_potential_0[:, np.newaxis] * A[:, :, 1]
                + sde_potential_p * A[:, :, 2]
            ) / self.grid.hx[np.newaxis, :]

            return ode_grad, sde_grad

        return _v

    def sample(
        self,
        sample_x0: np.ndarray,
    ) -> np.ndarray:
        """Starting from the sample from the initial distribution, propagate it through the fitted dynamics and return a sample from the distribution on the last step.

        Args:
            sample_x0 : sample from the initial distribution, should have shape `(n_samples, dim)`

        Returns:
            np.ndarray : sample from the distribution of the last step
        """
        x_cur = sample_x0.copy()
        x_cur = self.grid.clip_sample(x_cur)
        old_shape = x_cur.shape
        dim = sample_x0.shape[-1]
        n_em = (
            self.params.sampling_n_euler_maruyama_steps
            if self.params.sampling_sde_fraction > 0
            else 0
        )
        eps_split = self.params.sampling_sde_fraction

        for T_cur, beta_cur, eta_t1, hat_eta_t0 in zip(
            self.Ts, self.betas, self._etas_t1, self._hat_etas_t0
        ):
            drifts = self._get_drift_terms_fn(eta_t1, hat_eta_t0, T_cur, beta_cur)
            ode_drift = lambda _t, _x: drifts(_t, _x)[0]
            rhs_ode = lambda _t, _x: ode_drift(_t, _x.reshape(old_shape)).reshape(-1)
            sde_drift = lambda _t, _x: drifts(_t, _x)[1]

            init_val_ode = x_cur.reshape(-1)
            t_start_ode = 0.0
            t_stop_ode = (1.0 - eps_split) * T_cur
            ode_integrator = RK45(
                rhs_ode,
                t_start_ode,
                init_val_ode,
                t_stop_ode,
                first_step=T_cur * 1e-4,
                rtol=self.params.sampling_ode_rtol,
                atol=self.params.sampling_ode_atol,
            )
            with np.errstate(under="ignore"):
                while ode_integrator.status == "running":
                    ode_integrator.step()

            x_cur = ode_integrator.y.reshape(old_shape)
            # Do Euler-Maruyama steps
            t_cur = t_stop_ode
            tau_em = eps_split * T_cur / max(n_em, 1)
            for _ in range(n_em):
                x_cur += tau_em * sde_drift(t_cur, x_cur) + np.sqrt(
                    2.0 * beta_cur * tau_em
                ) * np.random.randn(*x_cur.shape)
                t_cur += tau_em
            x_cur = self.grid.clip_sample(x_cur)
        return x_cur

    def get_current_distribution(
        self,
    ) -> TensorTrainDistribution:
        return TensorTrainDistribution(self.grid, self._rho_cur)
