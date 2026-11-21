# DENDRA-Brain: reinforcement learning with brain mechanisms, online and without backpropagation

**Abstract.** DENDRA-Brain is a new agent design for reinforcement learning. It is built from mechanisms found in the brain and learns online, after every step, with local rules only: no backpropagation, no replay buffer, no PPO.
- **Senses:** inputs are normalized by sensory adaptation.
- **Memory:** a temporal context of leaky traces ("time cells") makes hidden velocities visible.
- **Cortex:** many small cortical columns turn the input into a sparse place-cell code, recruiting new cells for new situations.
- **Action:** a striatum of action channels learns action values from a dopamine prediction error. Synaptic eligibility tags carry that error back to earlier steps, and the same channels serve discrete actions and continuous motor commands.

We compare it with PPO, using an MLP (with frame stacking when velocities are hidden) and an LSTM, on six control tasks: CartPole, Acrobot, MountainCar and Pendulum, three of them with velocities hidden. Each agent gets 100k environment steps and 3 held-out seeds.
- **At 100k steps:** DENDRA-Brain has the best final score on 5 of 6 tasks. It is the only agent that solves MountainCar at all, and it has the highest area under the learning curve on all six (on Pendulum only slightly above the LSTM).
- **At 300k steps:** PPO-MLP catches up or overtakes. DENDRA-Brain's advantage is speed of learning, not final performance.
- **Weaknesses:** high variance between seeds, some degradation late in training, and so far only low-dimensional sensors (no vision).

---

## 1. Why a new design

The previous experiments gave two clear lessons:
1. **Backpropagation-free learning fails when memory has to be learned** (section 4.10 of the [README](README.md)). A dopamine rule can train a network on the present, but it cannot assign credit to what the network should have *stored* several steps earlier.
2. **DENDRA + PPO is weak on continuous control** (section 4.9): its HalfCheetah return was half the LSTM's.

DENDRA-Brain changes where the hard problems are solved:
- **Memory needs no training.** A fixed temporal context, leaky traces at several time scales as in hippocampal "time cells", keeps the recent past available. "Input − trace" estimates velocity directly.
- **Representation needs no gradient.** The cortex builds a sparse code by recruiting place cells for new situations. This is unsupervised and local, like the formation of new place fields.
- **The only reward-driven learning is a linear readout from a sparse code to action channels.** For a linear readout, the dopamine × eligibility rule *is* the exact gradient (Sarsa(λ)), so a local rule loses nothing.
- **Continuous actions use the same mechanism as discrete ones.** As in models of the basal ganglia, the striatum selects among motor channels: one group of channels per action dimension, with several command levels each.

## 2. Architecture

```text
observation ─► sensory adaptation ─► temporal context ─► cortical columns ─► striatum ─► action
               (gain control)        (time cells)        (sparse place code)  (action channels)
                                                                                 ▲
reward ─────► dopamine (adaptive coding) ─► δ × eligibility tags ────────────────┘
```

| Module | Brain mechanism | What it computes | How it learns |
| --- | --- | --- | --- |
| Sensory adaptation | gain control in sensory neurons | z-score of every input channel | running mean and variance |
| Temporal context | time cells, Laplace memory of the past | traces at τ = 2 and 8 steps; features: input, input − trace | fixed dynamics, cleared at a new episode |
| Cortical columns | columns with limited receptive fields; place cells; lateral inhibition | 16 areas, each seeing 4 random input channels; one primary area sees the whole observation. In each area at most k = 4 place cells fire, only those whose field contains the input | recruit a cell for an input nothing responds to; recycle the least-used cell when full |
| Striatum | action-value coding in striatal neurons; somatotopic channels | Q(s, channel) = Σ over firing cells of w; one channel group per action dimension (discrete: one channel per action; continuous: 5 command levels) | Sarsa(λ): Δw = η · δ · tag |
| Eligibility tags | synaptic tagging; dopamine acts within a window after activity (Yagishita et al., 2014) | a synapse used in a choice stays tagged, decaying by γλ per step | replacing tags |
| Dopamine | reward prediction error (Schultz; Morris et al., 2006); adaptive coding of reward range (Tobler et al., 2005) | δ = r̃ + γ Q(next) − Q(now), with r̃ = r / running std of the return, clipped to ±10 | — |
| Action choice | basal-ganglia selection; novelty-seeking dopamine (Kakade & Dayan, 2002) | softmax over channels (temperature 0.01) + a bonus for actions rarely tried in the current situation | counts per cell and channel |

**Learning rule** (applied after every step, to every synapse, in every environment):

```math
\delta_t = \tilde r_t + \gamma\, Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t), \qquad
w \leftarrow w + \frac{\eta}{k}\, \delta_t\, e_t, \qquad
e_t = \gamma\lambda\, e_{t-1} \ \text{(then tag the firing cells of the chosen channel)}
```

Defaults: γ = 0.99, λ = 0.9, η = 0.2, 16 columns × 1,024 cells, k = 4, recruitment radius 0.3 and field 2 × radius (in z-score units per channel). The agent runs at about 2,000 environment steps per second on one CPU core.

**Tried and dropped during development:**
- **Motor noise** (node perturbation): learning continuous actions by reinforcing helpful motor noise did not learn Pendulum. It was replaced by motor channels.
- **Novelty bonus in dopamine and reverse hippocampal replay:** removed to keep the design simple, after early single-seed tests showed no clear gain.
- **One large cortical area over all inputs:** fails on 18-dimensional inputs (Acrobot) because almost every state looks new (curse of dimensionality). This is why the cortex is split into many small columns.
- **k-winners-take-all without receptive fields:** a truly new state still activated old cells, so optimism-driven exploration could not work.

## 3. Experiments

**Protocol.**
- **Tasks:** Gymnasium classic control, 8 environments in parallel. "-pomdp" means the velocities are removed from the observation.
- **Budget:** 100k environment steps per run. Score = mean return of the last 50 episodes. AUC = mean of the learning curve over the run.
- **DENDRA-Brain:** settings tuned on seeds 0–1 and evaluated on seeds 10–12.
- **PPO:** the recurrent PPO of `experiments/rl/ppo.py` with standard settings. Learning rate chosen per task and network from {2.5e-4, 1e-3} on seed 10, then evaluated on seeds 10–12.
- **MLP baseline:** sees the last 4 observations when velocities are hidden.
- **Episode ends:** both agents treat episode time limits as terminal.

### 3.1 Results at 100k steps (3 seeds)

| Task | PPO-MLP | PPO-LSTM | **DENDRA-Brain** |
| --- | ---: | ---: | ---: |
| CartPole | 317 ± 28 | 76 ± 15 | **357 ± 159** |
| CartPole, velocities hidden | 38 ± 3 | 23 ± 1 | **91 ± 39** |
| Acrobot | −95 ± 5 | −500 ± 1 | **−90 ± 4** |
| MountainCar (sparse reward) | −200 (never solved) | −200 (never solved) | **−161 ± 23** |
| Pendulum (continuous) | −983 ± 144 | **−500 ± 254** | −603 ± 174 |
| Pendulum, velocity hidden (continuous) | −933 ± 76 | −966 ± 80 | **−521 ± 70** |

![DENDRA-Brain vs PPO at 100k steps](figures/brain_vs_ppo.png)

- **Faster learning.** DENDRA-Brain has the highest area under the learning curve on all six tasks, though on Pendulum with full observation only slightly above the LSTM. On Acrobot it reaches −100 after about 25k steps; PPO-MLP needs about 60k.
- **Sparse reward:** MountainCar is solved only by DENDRA-Brain. New situations recruit new place cells whose values start at 0, which is optimistic when every step costs −1, so the agent seeks them out. Both PPO agents never reach the goal in 100k steps.
- **Hidden velocities:** the temporal context does the job of a trained recurrent network. DENDRA-Brain beats both the LSTM and the frame-stacked MLP on both partially observed tasks.
- **Continuous control:** motor channels work. On Pendulum with hidden velocity DENDRA-Brain is clearly best; with full observation the LSTM ends higher, but with a large spread.

### 3.2 Longer budget: 300k steps (2 seeds)

| Task | PPO-MLP | PPO-LSTM | DENDRA-Brain |
| --- | ---: | ---: | ---: |
| CartPole | **500 ± 0** | 83 ± 13 | 314 ± 179 (seeds: 492, 135) |
| CartPole, velocities hidden | 156 ± 15 | 24 ± 0 | **172 ± 64** |
| Pendulum, velocity hidden | **−293 ± 59** | −527 ± 185 | −299 ± 22 |

![DENDRA-Brain vs PPO at 300k steps](figures/brain_vs_ppo_300k.png)

**With 3x more data, PPO-MLP catches up (Pendulum) or overtakes (CartPole).** The main advantage of DENDRA-Brain is sample efficiency. With more data, its final level is limited by instability: one CartPole seed collapses from near 500 to 135.

### 3.3 Ablations: removing one mechanism at a time (100k steps, seeds 10–12)

Final return / mean return over the whole run:

| Variant | CartPole, vel. hidden | Acrobot | MountainCar | Pendulum, vel. hidden |
| --- | ---: | ---: | ---: | ---: |
| Full DENDRA-Brain | 91 / 80 | −90 / −102 | −161 / −161 | −521 / −610 |
| − temporal context | **10 / 14** | −90 / −109 | −133 / −146 | **−1,233 / −1,242** |
| − eligibility traces (λ = 0) | 110 / 98 | **−115 / −168** | −144 / −177 | −384 / −683 |
| − adaptive dopamine coding | **36 / 20** | −84 / −101 | −157 / −160 | −458 / −653 |
| − action-novelty bonus | 112 / 28 | −87 / −101 | −145 / −155 | −469 / −625 |
| − receptive-field gating | 245 / 150 | −88 / −105 | −148 / −154 | −506 / −689 |
| − columns (one area sees everything) | 369 / 161 | **−118 / −173** | −145 / −170 | −519 / −722 |

- **Clearly necessary:**
  - the temporal context, without which tasks with hidden velocities fail;
  - eligibility traces, which speed learning on every task;
  - adaptive dopamine coding;
  - columns, without which Acrobot is much worse.
- **Not clearly helpful:**
  - **Action-novelty bonus:** without it, final scores are similar or better, and learning is slower only on CartPole with hidden velocities.
  - **Receptive-field gating:** removing it gave 245 instead of 91 on CartPole with hidden velocities on these seeds. On the tuning seeds the same change gave 16 instead of 270. This task's result depends strongly on the seed, so it cannot be decided with 3 seeds.

## 4. What this does and does not show

**Shown, on low-dimensional control tasks:**
- An agent with only brain-like local rules, updating after every step, learns several times faster than PPO.
- It handles hidden state with a fixed temporal memory instead of a trained recurrent network.
- It explores sparse-reward tasks through cell recruitment and optimism.
- It uses one mechanism for discrete and continuous actions.

**Not shown (yet), and needed before claiming "better than current RL":**
- **Stronger baselines.** PPO is not the most sample-efficient modern method. DQN/Rainbow (discrete) and SAC (continuous), which use replay buffers, learn these tasks much faster than PPO. They are the real bar for sample efficiency.
- **Stability and final performance.** The variance between seeds is high, and on several tasks performance peaks and then declines (figure in 3.1). Synaptic consolidation (metaplasticity, option `consolidation`) did not fix this.
- **Complex, natural environments.** All tasks here have 2–6 sensor values. Place cells over raw pixels would not work. Vision needs a hierarchical visual cortex whose features are learned without reward, for example by local predictive or Hebbian learning, feeding the same columns and striatum.
- **What natural environments additionally need:**
  - a world model for planning (hippocampal replay / prefrontal simulation);
  - memory beyond a few seconds;
  - continual learning without forgetting.

## 5. Usage

```python
import gymnasium as gym
from dendra.brain import BrainAgent

N = 8
SAME_STEP = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}   # reset inside the step that ends an episode
envs = gym.make_vec("Acrobot-v1", num_envs=N, vectorization_mode="sync", vector_kwargs=SAME_STEP)
agent = BrainAgent(obs_dim=6, n_actions=3, n_envs=N)                                  # discrete
# envs = gym.make_vec("Pendulum-v1", num_envs=N, vectorization_mode="sync", vector_kwargs=SAME_STEP)
# agent = BrainAgent(obs_dim=3, action_low=[-2.0], action_high=[2.0], n_envs=N)       # continuous

obs, _ = envs.reset(seed=0)
action = agent.reset(obs)
for step in range(10_000):
    obs, reward, terminated, truncated, _ = envs.step(action)
    action = agent.step(obs, reward, terminated | truncated)     # learns from this step, then acts
```

There is no training loop, optimizer or buffer: `agent.step` both learns and acts.

## 6. Reproducing

```bash
# DENDRA-Brain, one run (about 50 s on one CPU core)
python experiments/rl/run_brain.py --env Acrobot-v1 --steps 100000 --seed 10 --out runs/brain/brain_Acrobot-v1_s10.json
# Any setting can be changed, e.g. an ablation:
python experiments/rl/run_brain.py --env CartPole-v1-pomdp --seed 10 --set timescales=none --out runs/abl/x.json

# PPO baselines on the same tasks
python experiments/rl/ppo.py --env classic:Acrobot-v1 --core mlp --total 100000 --lr 2.5e-4 --seed 10 --device cpu --out runs/brain/ppo_Acrobot_s10.json

# Tables and figures
python experiments/rl/plot_brain.py experiments/results/brain/main_100k figures/brain_vs_ppo.png
python experiments/rl/plot_brain.py experiments/results/brain/long_300k figures/brain_vs_ppo_300k.png
python experiments/rl/brain_ablations.py
```

Code: `dendra/brain.py` (the agent), `experiments/rl/run_brain.py`, `experiments/rl/envs.py` (tasks shared with PPO), `experiments/rl/plot_brain.py`, `experiments/rl/brain_ablations.py`. Results: `experiments/results/brain/`.

## References

- Benna, M. K., Fusi, S. *Computational principles of synaptic memory consolidation.* Nature Neuroscience 19, 2016.
- Foster, D. J., Wilson, M. A. *Reverse replay of behavioural sequences in hippocampal place cells during the awake state.* Nature 440, 2006.
- Howard, M. W. et al. *A unified mathematical framework for coding time, space, and sequences in the hippocampal region.* Journal of Neuroscience 34, 2014.
- Kakade, S., Dayan, P. *Dopamine: generalization and bonuses.* Neural Networks 15, 2002.
- Morris, G., Nevet, A., Arkadir, D., Vaadia, E., Bergman, H. *Midbrain dopamine neurons encode decisions for future action.* Nature Neuroscience 9, 2006.
- Schultz, W., Dayan, P., Montague, P. R. *A neural substrate of prediction and reward.* Science 275, 1997.
- Sutton, R. S. *Generalization in reinforcement learning: successful examples using sparse coarse coding.* NeurIPS 1996.
- Tobler, P. N., Fiorillo, C. D., Schultz, W. *Adaptive coding of reward value by dopamine neurons.* Science 307, 2005.
- Yagishita, S. et al. *A critical time window for dopamine actions on the structural plasticity of dendritic spines.* Science 345, 2014.
