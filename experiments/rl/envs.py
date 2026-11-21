"""
Environments shared by the RL experiments: vectorized, reset inside the step that ends an episode
(so done[i] = 1 means obs[i] is already the first observation of the next episode).

Names
  CartPole-v1, Acrobot-v1, MountainCar-v0, Pendulum-v1     Gymnasium classic control
  ...-pomdp                                                 velocities hidden (positions/angles only)
  minatar:breakout (or another MinAtar game)                10x10 Atari-like game
  atari:Breakout (or another ALE game)                      real Atari 2600, 84x84 grayscale, frame skip 4,
                                                            sticky actions (ALE v5 default), one frame per step
"""
import gymnasium as gym
import numpy as np

# Observation entries kept in the "-pomdp" versions (the velocities are removed)
VISIBLE = {
    "CartPole-v1": [0, 2],            # cart position, pole angle
    "Acrobot-v1": [0, 1, 2, 3],       # cos/sin of both joint angles
    "MountainCar-v0": [0],            # car position
    "Pendulum-v1": [0, 1],            # cos/sin of the angle
}


class KeepEntries(gym.ObservationWrapper):
    def __init__(self, env, keep):
        super().__init__(env)
        self.keep = keep
        low, high = env.observation_space.low[keep], env.observation_space.high[keep]
        self.observation_space = gym.spaces.Box(low, high, (len(keep),), np.float32)

    def observation(self, obs):
        return obs[self.keep].astype(np.float32)


def make_classic(name, frame_stack=1):
    hidden = name.endswith("-pomdp")
    base = name.replace("-pomdp", "")
    env = gym.make(base)
    if hidden:
        env = KeepEntries(env, VISIBLE[base])
    if frame_stack > 1:
        env = gym.wrappers.FlattenObservation(gym.wrappers.FrameStackObservation(env, frame_stack))
    return env


def make_atari(game, frame_stack=1):
    import ale_py
    gym.register_envs(ale_py)
    env = gym.make(f"ALE/{game}-v5", frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, noop_max=30)
    if frame_stack > 1:
        env = gym.wrappers.FrameStackObservation(env, frame_stack)
    return env


class VecEnv:
    """n copies of a classic-control or MinAtar environment.
    step(actions) -> obs [n, ...], rewards [n], dones [n] (bool)."""

    def __init__(self, name, n, seed, frame_stack=1):
        self.minatar = name.startswith("minatar:")
        if self.minatar:
            from minatar import Environment
            self.envs = [Environment(name.split(":")[1]) for _ in range(n)]
            for i, env in enumerate(self.envs):
                env.seed(seed * 1000 + i)
                env.reset()
            self.n_actions = self.envs[0].num_actions()
            self.obs_shape = (self.envs[0].state_shape()[2], 10, 10)
            self.continuous = False
            return
        make = (lambda: make_atari(name.split(":")[1], frame_stack)) if name.startswith("atari:") else \
               (lambda: make_classic(name, frame_stack))
        self.vec = gym.vector.SyncVectorEnv([make for _ in range(n)],
                                            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        self.seed = seed
        space = self.vec.single_action_space
        self.continuous = not hasattr(space, "n")
        self.n_actions = space.shape[0] if self.continuous else space.n
        if self.continuous:
            self.low, self.high = space.low, space.high
        self.obs_shape = self.vec.single_observation_space.shape

    def _minatar_obs(self):
        return np.stack([env.state().transpose(2, 0, 1) for env in self.envs]).astype(np.float32)

    def reset(self):
        if self.minatar:
            return self._minatar_obs()
        return self.vec.reset(seed=self.seed)[0].astype(np.float32)

    def step(self, actions):
        if self.minatar:
            rewards, dones = np.zeros(len(self.envs), np.float32), np.zeros(len(self.envs), bool)
            for i, (env, a) in enumerate(zip(self.envs, actions)):
                rewards[i], dones[i] = env.act(int(a))
                if dones[i]:
                    env.reset()
            return self._minatar_obs(), rewards, dones
        if self.continuous:
            actions = np.clip(actions, self.low, self.high)
        obs, rewards, terminated, truncated, _ = self.vec.step(actions)
        return obs.astype(np.float32), rewards.astype(np.float32), terminated | truncated
