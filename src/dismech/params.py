import dataclasses
import typing


@dataclasses.dataclass
class GeomParams:
    rod_r0: float
    shell_h: float
    axs: typing.Optional[float] = None
    jxs: typing.Optional[float] = None
    ixs1: typing.Optional[float] = None
    ixs2: typing.Optional[float] = None


@dataclasses.dataclass
class Material:
    density: float
    youngs_rod: float
    youngs_shell: float
    poisson_rod: float
    poisson_shell: float


@dataclasses.dataclass
class SimParams:
    static_sim: bool
    two_d_sim: bool
    use_mid_edge: bool
    use_line_search: bool
    log_data: bool
    log_step: int
    show_floor: bool
    dt: float
    max_iter: int
    total_time: float
    plot_step: int
    tol: float
    ftol: float
    dtol: float
    solver: str = 'np'
    sparse: bool = False
    line_search_iters: int = 20
    use_predictor_corrector_for_ground_contact: bool = False
    ground_level_for_predictor_corrector: float = None
    min_contact_edge_gap: int = 3
    n_edges_beam_A: int = 0  # >0 enables cross-beam contact pair construction
    # If True (and n_edges_beam_A > 0), skip ALL within-beam contact pairs;
    # only cross-beam (beam A vs beam B) pairs are constructed. Use when the
    # simulated motion cannot bring the same beam into contact with itself.
    cross_beam_only_contact: bool = False
