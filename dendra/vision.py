"""
Visual cortex for DENDRA-Brain: features learned without reward, online, with local rules.

    frame (84x84) ─► retina ─────► V1 ──────────► V1 complex ─► V2 ──────────► V2 complex ─► sparse code
                     luminance +    simple cells    max-pool 2x2  conjunctions    max-pool 2x2
                     transient      8x8 patches     (position     of V1 cells     (position
                     (motion)       one winner per  tolerance)    3x3, winner     tolerance)
                                    location                      per location

  Retina      two channels: luminance and its change since the previous frame (transient / motion cells).
  V1          at every location (8x8 patch, stride 4) the simple cells compete; only the best-matching cell
              fires, and only if the patch has contrast. Cells learn by competitive Hebbian learning (the
              winner moves its preferred patch toward the input), with a homeostatic "conscience" so that
              every cell gets used.
  V1 complex  max-pooling over 2x2 neighbouring locations: the same feature, slightly shifted.
  V2          cells that respond to 3x3 arrangements of V1 complex features, learned the same way.
  Critical period  visual plasticity fades linearly to zero over the first `critical_period` steps, like the
              developmental critical period of visual cortex, so that the code the striatum reads stays stable.
  Output      indices of the active cells (sparse binary code): V1 complex (where + what) and V2 complex.

Weight sharing across locations (the same cell types at every location) stands for the many columns of
the same type that tile the visual field; it is the main non-biological simplification here.
"""
import torch
import torch.nn.functional as F


class CompetitiveLayer:
    """Cells with preferred unit-norm patterns. At each location the most similar cell fires (if the
    similarity exceeds `threshold`). Learning: the winner moves toward the input (competitive Hebbian rule);
    a conscience term penalizes cells that win too often, so all cells take part."""

    def __init__(self, in_dim, n_cells, lr=0.02, threshold=0.0, conscience=0.5, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.w = F.normalize(torch.randn(n_cells, in_dim, generator=g), dim=1)
        self.lr, self.threshold, self.conscience = lr, threshold, conscience
        self.win_rate = torch.full((n_cells,), 1.0 / n_cells)
        self.n_cells = n_cells

    def __call__(self, x, valid, plasticity=1.0):
        """x [M, in_dim] unit-norm inputs, valid [M] bool. Returns the firing cell per input (-1 = silent).
        The best-matching cell fires; the homeostatic bias only decides which cell learns."""
        sim = x @ self.w.T                                                       # [M, cells]
        winner = torch.argmax(sim, dim=1)
        best = sim.gather(1, winner[:, None])[:, 0]
        fires = valid & (best > self.threshold)
        if plasticity > 0 and valid.any():              # every valid input trains a cell, firing or not
            bias = self.conscience * (1.0 / self.n_cells - self.win_rate)       # favours rarely winning cells
            learner = torch.argmax(sim + bias[None], dim=1)
            xs, ws = x[valid], learner[valid]
            delta = torch.zeros_like(self.w).index_add_(0, ws, xs - self.w[ws])
            counts = torch.zeros(self.n_cells).index_add_(0, ws, torch.ones(len(ws)))
            self.w = F.normalize(self.w + plasticity * self.lr * delta / counts.clamp(min=1)[:, None], dim=1)
            self.win_rate = 0.999 * self.win_rate + 0.001 * counts / counts.sum()
        return torch.where(fires, winner, torch.full_like(winner, -1))


class VisualCortex:
    """
    frames [N, 84, 84] (uint8 or float) -> list of N tensors of active feature indices (sparse binary code).
    Feature index layout: [V1 complex: 10*10*n_v1][V2 complex: 4*4*n_v2].
    """

    def __init__(self, n_envs, n_v1=64, n_v2=128, contrast=0.02, lr=0.02, critical_period=3000, seed=0):
        self.N, self.n_v1, self.n_v2, self.contrast = n_envs, n_v1, n_v2, contrast
        self.v1 = CompetitiveLayer(2 * 8 * 8, n_v1, lr=lr, seed=seed)
        self.v2 = CompetitiveLayer(3 * 3 * n_v1, n_v2, lr=lr, threshold=0.3, seed=seed + 1)
        self.prev = None
        self.critical_period, self.age = critical_period, 0      # plasticity fades to 0 over this many steps
        self.v1_size = 10 * 10 * n_v1
        self.n_features = self.v1_size + 4 * 4 * n_v2

    def retina(self, frames, new_episode):
        lum = torch.as_tensor(frames, dtype=torch.float32).reshape(self.N, 84, 84) / 255.0
        if self.prev is None:
            self.prev = lum.clone()
        self.prev[new_episode] = lum[new_episode]                 # no motion signal on the first frame
        transient = lum - self.prev
        self.prev = lum
        return torch.stack([lum, transient], 1)                    # [N, 2, 84, 84]

    def __call__(self, frames, new_episode, learn=True):
        image = self.retina(frames, new_episode)
        plasticity = 0.0
        if learn:
            self.age += 1
            plasticity = 1.0 if self.critical_period <= 0 else max(0.0, 1.0 - self.age / self.critical_period)
        # V1: 8x8 patches with stride 4 -> 20x20 locations
        patches = F.unfold(image, 8, stride=4).transpose(1, 2)    # [N, 400, 128]
        patches = patches - patches.mean(-1, keepdim=True)        # local contrast (centre-surround)
        norm = patches.norm(dim=-1)
        valid = norm > self.contrast * 128 ** 0.5
        cells = self.v1(F.normalize(patches, dim=-1).reshape(-1, 128), valid.reshape(-1), plasticity)
        v1 = torch.zeros(self.N * 400, self.n_v1)
        on = cells >= 0
        v1[on.nonzero()[:, 0], cells[on]] = 1.0
        v1 = v1.reshape(self.N, 20, 20, self.n_v1).permute(0, 3, 1, 2)          # [N, n_v1, 20, 20]
        v1c = F.max_pool2d(v1, 2)                                                # [N, n_v1, 10, 10]
        # V2: 3x3 arrangements of V1 complex cells -> 8x8 locations
        p2 = F.unfold(v1c, 3).transpose(1, 2)                                   # [N, 64, 9*n_v1]
        valid2 = p2.sum(-1) >= 2                                                 # at least two V1 features
        cells2 = self.v2(F.normalize(p2, dim=-1).reshape(-1, p2.shape[-1]), valid2.reshape(-1), plasticity)
        v2 = torch.zeros(self.N * 64, self.n_v2)
        on2 = cells2 >= 0
        v2[on2.nonzero()[:, 0], cells2[on2]] = 1.0
        v2c = F.max_pool2d(v2.reshape(self.N, 8, 8, self.n_v2).permute(0, 3, 1, 2), 2)   # [N, n_v2, 4, 4]
        code = torch.cat([v1c.permute(0, 2, 3, 1).reshape(self.N, -1),          # index = (y*10+x)*n_v1 + cell
                          v2c.permute(0, 2, 3, 1).reshape(self.N, -1)], 1)
        return [row.nonzero()[:, 0] for row in code], v1c
