I'll locate and read your claude.md preference files, then propose a refined, elegant architecture for a simulation/learning factory aimed at image denoising (including MNIST) via time-reversal trajectories and parameter learning.
Restating the goal in your terms
Target: A factory-based simulation & learning framework that, given any 
N
×
N
N×N image (and MNIST), can:
Treat the image as a noisy finite-time sample of a thermodynamic system.
Infer latent parameters and/or latent clean states by effectively running the most probable backward trajectory (time-reversal dynamics / MAP reconstruction).
Constraints: Elegant, type-safe, model-first design, using factories and clean abstraction layers, aligned with your CLAUDE.md standards.
Core conceptual design
State space & dynamics
Represent images as lattice systems: 
x
∈
R
N
×
N
x∈R 
N×N
  (or 
{
−
1
,
1
}
N
×
N
{−1,1} 
N×N
  for Ising-style).
Use an overdamped Langevin SDE:
d
x
t
=
f
θ
(
x
t
)
d
t
+
2
D
d
W
t
dx 
t
​
 =f 
θ
​
 (x 
t
​
 )dt+ 
2D
​
 dW 
t
​
 
For image models, 
f
θ
f 
θ
​
  comes from an energy:
U
θ
(
x
)
=
∑
(
i
,
j
)
(
h
i
j
x
i
j
+
∑
(
k
,
l
)
∈
N
(
i
,
j
)
J
(
i
,
j
)
,
(
k
,
l
)
x
i
j
x
k
l
)
+
(optional higher-order/conv filters)
U 
θ
​
 (x)= 
(i,j)
∑
​
 (h 
ij
​
 x 
ij
​
 + 
(k,l)∈N(i,j)
∑
​
 J 
(i,j),(k,l)
​
 x 
ij
​
 x 
kl
​
 )+(optional higher-order/conv filters)
Time reversal / denoising
Model the observed noisy image 
x
obs
x 
obs
​
  as 
x
(
t
k
)
x(t 
k
​
 ) after forward dynamics from latent 
x
0
x 
0
​
  (or given 
θ
θ).
Most probable backward trajectory:
Either:
Optimize \(x_0, \theta\) such that forward simulation 
x
(
t
k
;
x
0
,
θ
)
x(t 
k
​
 ;x 
0
​
 ,θ) matches 
x
obs
x 
obs
​
 , or
Use analytic time-reversed drift for the SDE (when tractable) and integrate backward as an ODE approximate MAP path.
In practice, for elegance and extensibility:
Implement a generic “inverse problem” layer that defines:
L
(
x
0
,
θ
)
=
∥
x
(
t
k
;
x
0
,
θ
)
−
x
obs
∥
2
+
priors/regularization
L(x 
0
​
 ,θ)=∥x(t 
k
​
 ;x 
0
​
 ,θ)−x 
obs
​
 ∥ 
2
 +priors/regularization
and optimizes it via gradient descent / variational learning.
Pydantic model layer (configuration & contracts)
System configs
LangevinSystemConfig:
n: int, dim: int | tuple[int, int], beta: float, noise_level: float, dt: float, t_max: float.
energy_type: Literal["quadratic", "ising", "conv"].
ImageLatticeConfig(LangevinSystemConfig):
height: int, width: int, boundary: Literal["periodic","reflect","zero"].
MnistDenoisingConfig(ImageLatticeConfig):
digit_classes: list[int] | None, noise_schedule: list[float], etc.
TrainingConfig:
learning_rate: float, n_epochs: int, batch_size: int, optimizer: Literal["sgd","adam"], loss_type: Literal["reconstruction","contrastive","time_reversal"], etc.
Task/experiment models
DenoisingTaskConfig:
dataset: Literal["image","mnist"], system_config: ImageLatticeConfig | MnistDenoisingConfig, training: TrainingConfig.
All external configuration flows through these models; no raw dicts.
Core abstractions & factories
Thermodynamic system abstraction
Define a protocol/ABC:
ThermoSystem with:
state_shape: tuple[int, ...]
parameters: ThermoParameters (itself a Pydantic model: fields like J, h, possibly as np.ndarray wrappers).
Methods:
drift(x: np.ndarray) -> np.ndarray
step(x: np.ndarray, dt: float, rng: np.random.Generator) -> np.ndarray (Euler–Maruyama)
run_forward(x0, t_max, dt, observe_times) -> Trajectory (batch-aware)
Concrete implementations:
ImageLatticeSystem(ThermoSystem)
MnistLatticeSystem(ThermoSystem) (extends image lattice but with dataset-specific priors/initialization).
Simulation factory
SimulationFactory with a single, elegant entrypoint:
create_system(config: LangevinSystemConfig) -> ThermoSystem
Internally dispatches on config.energy_type and config type:
ImageLatticeConfig → ImageLatticeSystem
MnistDenoisingConfig → MnistLatticeSystem
Extensible to future systems (graphs, continuous fields).
Learning / inverse problem engine
Trajectory statistics & objectives
Trajectory dataclass:
states: np.ndarray # [batch, time, *state_shape]
times: np.ndarray
Utility functions:
compute_local_stats(trajectory) -> LocalStats (means, correlations).
Learning engines
BaseLearningEngine protocol:
fit(system: ThermoSystem, data: ImageBatch, config: TrainingConfig) -> TrainedSystem
Implementations:
ContrastiveLearningEngine (implements correlator-based updates like in the paper).
TimeReversalMAPEngine:
For each observed image 
x
obs
x 
obs
​
 :
Initialize latent x0 (and optionally θ).
Run forward simulation to time t_k.
Compute reconstruction loss 
∥
x
(
t
k
;
x
0
,
θ
)
−
x
obs
∥
2
∥x(t 
k
​
 ;x 
0
​
 ,θ)−x 
obs
​
 ∥ 
2
 .
Use gradient-based optimization over latent x0 and possibly θ.
Encapsulate optimization loop cleanly (no side-channel state), with hooks for logging / diagnostics.
Factory for learning
LearningFactory:
create_engine(task: DenoisingTaskConfig) -> BaseLearningEngine
Enables:
engine = LearningFactory.create_engine(task_cfg)
trained = engine.fit(system, dataset, task_cfg.training)
Data layer (images & MNIST)
Generic image adapter
ImageDataset abstraction (Pydantic model for config, class for runtime):
Handles:
Loading arbitrary 
N
×
N
N×N images (e.g. from np.ndarray or files).
Normalization to 
[
−
1
,
1
]
[−1,1] or 
[
0
,
1
]
[0,1].
Optional noise injection to create synthetic forward samples.
MNIST adapter
Minimal, dependency-light:
Either:
Use sklearn.datasets.fetch_openml("mnist_784"), or
Introduce a small optional dependency if you prefer (we can add later).
Provide:
Batches as (images, labels) aligned with MnistDenoisingConfig.
These are surfaced via a dataset factory:
DatasetFactory.create(task_cfg) -> ImageDataset | MnistDataset.
Directory & module structure (within physics)
Proposed layout (Python package inside this repo):
mwp_sim/
__init__.py
config.py – Pydantic configs (LangevinSystemConfig, ImageLatticeConfig, DenoisingTaskConfig, etc.)
systems.py – ThermoSystem ABC/protocol, ImageLatticeSystem, MnistLatticeSystem.
factory.py – SimulationFactory, LearningFactory, DatasetFactory.
dynamics.py – SDE integrators, time-reversal utilities, trajectory stats.
learning.py – BaseLearningEngine, ContrastiveLearningEngine, TimeReversalMAPEngine.
data.py – ImageDataset, MnistDataset.
experiments/
image_denoising.py – demo CLI / script for arbitrary 
N
×
N
N×N.
mnist_denoising.py – demo for MNIST.
tests/
test_systems.py, test_learning.py, etc.
All entrypoints use config models and factories, so a typical experiment looks like:
from mwp_sim.config import DenoisingTaskConfig, ImageLatticeConfig, TrainingConfigfrom mwp_sim.factory import SimulationFactory, LearningFactory, DatasetFactorytask_cfg = DenoisingTaskConfig(    dataset="image",    system_config=ImageLatticeConfig(...),    training=TrainingConfig(...))system = SimulationFactory.create_system(task_cfg.system_config)dataset = DatasetFactory.create(task_cfg)engine = LearningFactory.create_engine(task_cfg)trained_system = engine.fit(system, dataset, task_cfg.training)
Concrete milestones
Milestone 1 – Core math & abstractions
Implement ThermoSystem + ImageLatticeSystem + basic Euler–Maruyama + SimulationFactory.
Milestone 2 – Denoising / inverse engine
Implement TimeReversalMAPEngine for a single image; support arbitrary 
N
×
N
N×N.
Milestone 3 – MNIST integration
Add MnistDataset + MnistLatticeSystem; run end-to-end denoising experiment.
Milestone 4 – Refinement & elegance
Add contrastive learning, cleaner time-reversal utilities, diagnostics, and a small, beautiful CLI/API to run experiments from config files.
If this architecture matches your intent, I’ll next start Milestone 1 inside this repo (mwp_sim/ package) and wire up the first end-to-end “simulate + reconstruct a single noisy 
N
×
N
N×N image” path.