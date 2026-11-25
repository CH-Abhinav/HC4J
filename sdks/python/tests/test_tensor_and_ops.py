# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Live GPU tests for hc4py engine telemetry, Tensor lifecycle invariants, NumPy/stdlib IO, ops, and FusedExpr."""

import math
import unittest

import hc4py
from hc4py import DType, FusedExpr, Hc4xError, InvalidParamError, Tensor, linalg, nn
from hc4py import math as hmath
from hc4py.dtype import has_numpy

if has_numpy():
    import numpy as np
else:
    np = None  # type: ignore[assignment]


class TensorAndOpsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not hc4py.is_gpu_available():
            raise unittest.SkipTest("No WebGPU adapter available")

    def test_engine_telemetry_and_batch(self) -> None:
        info = hc4py.adapter_info()
        self.assertTrue(len(info.name) > 0)
        self.assertIn(info.backend, {"Vulkan", "Dx12", "Metal", "Gl", "WebGPU", "Noop"})

        stats_before = hc4py.engine_stats()
        with hc4py.batch():
            with Tensor.from_list([1.0, 2.0, 3.0, 4.0], (2, 2)) as t:
                with (t + 5.0) as out:
                    self.assertEqual(out.to_list(), [[6.0, 7.0], [8.0, 9.0]])
        stats_after = hc4py.engine_stats()
        mstats = hc4py.mem_stats()
        self.assertGreaterEqual(stats_after.batches, stats_before.batches + 1)
        self.assertGreater(mstats.vram_budget, 0)

    def test_refcount_contiguous_close_invariants(self) -> None:
        base = Tensor.from_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        base_id = base.vram_id

        # Invariant 2: contiguous() on contiguous zero-offset returns retained same-buffer view
        c_view = base.contiguous()
        self.assertEqual(c_view.vram_id, base_id)

        # Transposed view shares handle, while contiguous() on transposed view allocates new buffer
        t_view = base.T
        self.assertEqual(t_view.vram_id, base_id)
        t_contig = t_view.contiguous()
        self.assertNotEqual(t_contig.vram_id, base_id)
        self.assertEqual(t_contig.to_list(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

        # Closing base leaves retained views alive (Invariant 1)
        base.close()
        base.close()  # Invariant 3: idempotent close
        with self.assertRaises(Hc4xError):
            _ = base.vram_id

        self.assertEqual(c_view.to_flat_list(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        c_view.close()
        t_view.close()
        t_contig.close()

    def test_layout_mismatched_inplace_aliasing_guard(self) -> None:
        # Invariant 4: a.add_(a.T) must materialize a.T before mutating a
        with Tensor.from_list([[1.0, 2.0], [3.0, 4.0]]) as a:
            with a.T as at:
                a.add_(at)
            # [[1+1, 2+3], [3+2, 4+4]] = [[2, 5], [5, 8]]
            self.assertEqual(a.to_list(), [[2.0, 5.0], [5.0, 8.0]])

    def test_numpy_zero_copy_and_stdlib_fallback(self) -> None:
        # Stdlib f16 & i32 & f32 round-trip
        with Tensor.from_list([10, -20, 30], dtype=DType.i32) as ti:
            with (-ti) as neg_ti:
                self.assertEqual(neg_ti.to_flat_list(), [-10, 20, -30])
            with ti.cast(DType.f32) as tf:
                self.assertEqual(tf.to_flat_list(), [10.0, -20.0, 30.0])

        with Tensor.from_list([1.5, -2.5], dtype=DType.f16) as tf16:
            vals = tf16.to_flat_list()
            self.assertAlmostEqual(vals[0], 1.5, places=3)
            self.assertAlmostEqual(vals[1], -2.5, places=3)

        if np is not None:
            src = np.arange(12, dtype=np.float32).reshape(3, 4)
            with Tensor.from_numpy(src) as t:
                with (t * 2.0 + 1.0) as res:
                    np.testing.assert_allclose(res.numpy(), src * 2.0 + 1.0, rtol=1e-5, atol=1e-5)

    def test_math_linalg_nn_and_fused_expr(self) -> None:
        # Broadcasted math + where + compare
        with (
            Tensor.from_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) as a,
            Tensor.from_list([10.0, 20.0, 30.0]) as b,
        ):
            with (a + b) as s, hmath.sin(b) as sb, hmath.exp(Tensor.from_list([0.0, 1.0])) as eb:
                self.assertEqual(s.to_list(), [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])
                sb_vals = sb.to_flat_list()
                self.assertAlmostEqual(sb_vals[0], math.sin(10.0), places=4)
                eb_vals = eb.to_flat_list()
                self.assertAlmostEqual(eb_vals[1], math.e, places=4)

            with (a > 3.5) as mask, Tensor.full((2, 3), -1.0) as neg1:
                with hmath.where(mask, a, neg1) as chosen:
                    self.assertEqual(chosen.to_list(), [[-1.0, -1.0, -1.0], [4.0, 5.0, 6.0]])

        # Linalg matmul, matmul_transposed_b, f16 weight, batched matmul, plan
        plan = linalg.matmul_plan(64, 64, 64)
        self.assertIsNotNone(plan.kernel)
        with (
            Tensor.from_list([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) as x,
            Tensor.from_list([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]]) as w_t,
        ):
            with w_t.cast(DType.f16) as w_f16:
                with linalg.matmul_transposed_b(x, w_f16) as y:
                    # [[1-3, 2+2], [4-6, 8+5]] = [[-2, 4], [-2, 13]]
                    y_vals = y.to_list()
                    self.assertAlmostEqual(y_vals[0][0], -2.0, places=3)
                    self.assertAlmostEqual(y_vals[1][1], 13.0, places=3)

        # NN: softmax, rms_norm, cross_entropy, embedding, rope, FusedExpr
        with Tensor.from_list([[1.0, 2.0, 3.0]]) as logits, Tensor.from_list([2], dtype=DType.i32) as target:
            with nn.softmax(logits) as probs, nn.cross_entropy(logits, target) as ce:
                self.assertAlmostEqual(sum(probs.to_flat_list()), 1.0, places=5)
                expected_ce = -math.log(math.exp(3.0) / (math.exp(1.0) + math.exp(2.0) + math.exp(3.0)))
                self.assertAlmostEqual(ce.item(), expected_ce, places=4)

        with Tensor.from_list([-2.0, -0.5, 1.0, 3.0]) as x, Tensor.from_list([1.0, 1.0, 1.0, 1.0]) as y:
            with ((x.lazy() * 2.0) + y).relu().eval() as fused:
                # relu([-3.0, 0.0, 3.0, 7.0]) = [0.0, 0.0, 3.0, 7.0]
                self.assertEqual(fused.to_flat_list(), [0.0, 0.0, 3.0, 7.0])

        with self.assertRaises(InvalidParamError):
            _ = (Tensor.ones((2, 2)).lazy() / 0.0) if False else FusedExpr.of(Tensor.ones((2, 2))).add(float("nan"))


if __name__ == "__main__":
    unittest.main()
