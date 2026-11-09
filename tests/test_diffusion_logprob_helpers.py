import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from copyright_pipeline.inference_common import (  # noqa: E402
    _ddpm_transition_params_from_eps,
    _sample_logprob_diag_gaussian,
    build_descending_timestep_path,
)


class DiffusionLogprobHelpersTest(unittest.TestCase):
    def test_build_descending_timestep_path_includes_start_and_zero(self):
        path = build_descending_timestep_path(start_timestep=200, trajectory_steps=10)
        self.assertGreaterEqual(len(path), 2)
        self.assertEqual(path[0], 200)
        self.assertEqual(path[-1], 0)
        for idx in range(len(path) - 1):
            self.assertGreater(path[idx], path[idx + 1])

    def test_ddpm_transition_and_logprob_shapes(self):
        x_t = torch.randn(3, 4, 8, 8)
        eps = torch.randn_like(x_t)
        mean, var = _ddpm_transition_params_from_eps(
            x_t,
            eps,
            alpha_bar_t=0.5,
            alpha_bar_prev=0.6,
        )
        self.assertEqual(tuple(mean.shape), tuple(x_t.shape))
        self.assertEqual(tuple(var.shape), tuple(x_t.shape))
        self.assertTrue(torch.all(var > 0).item())

        x_next = mean + torch.sqrt(var) * torch.randn_like(mean)
        logprob = _sample_logprob_diag_gaussian(x_next, mean, var)
        self.assertEqual(tuple(logprob.shape), (x_t.shape[0],))
        self.assertTrue(torch.isfinite(logprob).all().item())
        self.assertLess(float(logprob.max().item()), 1.0e9)
        self.assertGreater(float(logprob.min().item()), -1.0e12)
        self.assertTrue(math.isfinite(float(logprob.mean().item())))


if __name__ == "__main__":
    unittest.main()
