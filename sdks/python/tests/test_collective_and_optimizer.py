# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Live tests for hc4py Collective runtime, Exact Theorems 1-4 Optimizer, and Split-KV merge."""

import math
import unittest

import hc4py
from hc4py import (
    HOST_CPU_RANK,
    Collective,
    DeviceProfile,
    DeviceRole,
    HostCpuProfile,
    Link,
    OutOfMemoryError,
    SplitKvSpec,
    Transport,
    WorkloadSpec,
    merge_split_kv,
    optimizer,
)


def _gpu(rank: int, vram_gb: int, gflops: float, dtype: str = "Discrete") -> DeviceProfile:
    return DeviceProfile(rank, f"GPU-{rank}", dtype, 0x10DE, 0x2684 + rank, vram_gb << 30, gflops)


def _mesh(n: int, bw: float, lat_us: float, tr: Transport) -> list[Link]:
    links: list[Link] = []
    for s in range(n):
        for d in range(n):
            links.append(Link(s, d, 25.0 if s == d else bw, Transport.PEER_TO_PEER if s == d else tr, s == d, 0.0 if s == d else lat_us))
    return links


class CollectiveAndOptimizerTest(unittest.TestCase):
    def test_optimizer_straggler_and_zero_degradation(self) -> None:
        devices = [_gpu(r, 32, 100_000.0) for r in range(10)]
        devices.append(_gpu(10, 4, 250.0, "Integrated"))

        links: list[Link] = []
        for s in range(11):
            for d in range(11):
                if s == d:
                    links.append(Link(s, d, 25.0, Transport.PEER_TO_PEER, True, 0.0))
                elif s < 10 and d < 10:
                    links.append(Link(s, d, 28.0, Transport.PEER_TO_PEER, False, 3.0))
                else:
                    links.append(Link(s, d, 1.0, Transport.HOST_STAGED, False, 150.0))

        spec = WorkloadSpec(total_units=10_240, flops_per_unit=1.5e9, bytes_per_unit=4096, sync_bytes=64 << 20)
        plan = optimizer.plan_work(devices, links, spec)
        self.assertTrue(plan.allocations[10].excluded_as_straggler)
        self.assertNotIn(10, plan.active_ranks)
        self.assertEqual(sum(a.units for a in plan.allocations), 10_240)

        fast_only = optimizer.plan_work(devices[:10], [l for l in links if l.src < 10 and l.dst < 10], spec)
        self.assertLessEqual(plan.estimated_makespan_ms, fast_only.estimated_makespan_ms + 1e-6)

    def test_optimizer_three_tier_spilling_and_cpu_and_oom(self) -> None:
        devs = [_gpu(0, 1, 20_000.0)]
        cpu = HostCpuProfile("Ryzen-9", ram_bytes=8 << 30, compute_gflops=500.0, ram_bandwidth_gbps=80.0, aggregate_pcie_gbps=16.0)
        spec = WorkloadSpec(
            total_units=3200,
            flops_per_unit=5e8,
            bytes_per_unit=1 << 20,
            disk_bandwidth_gbps=4.0,
            disk_capacity_bytes=16 << 30,
            host_cpu=cpu,
        )
        plan = optimizer.plan_work(devs, [], spec)
        self.assertEqual(sum(a.units for a in plan.allocations), 3200)
        self.assertTrue(any(a.host_stream_units > 0 for a in plan.allocations))
        self.assertTrue(any(a.rank == HOST_CPU_RANK and a.cpu_in_place_units > 0 for a in plan.allocations))

        with self.assertRaises(OutOfMemoryError) as ctx:
            optimizer.plan_work(devs, [], WorkloadSpec(total_units=5000, flops_per_unit=1e8, bytes_per_unit=1 << 20))
        self.assertIn("out of memory", str(ctx.exception).lower())

    def test_optimizer_plan_roles_short_vs_long_context(self) -> None:
        devs = [_gpu(0, 24, 90_000.0), _gpu(1, 16, 400.0, "Integrated")]
        links = _mesh(2, 2.0, 80.0, Transport.HOST_STAGED)
        tp_spec = WorkloadSpec(total_units=4096, flops_per_unit=2e9, bytes_per_unit=4096, sync_bytes=64 << 20)

        short_role = optimizer.plan_roles(
            devs, links, tp_spec, SplitKvSpec(128, 512 << 10, 1e6, 64 << 10, 2 << 30, 0.85)
        )
        self.assertEqual(short_role.active_regime, "STRAGGLER_PRUNED")

        long_role = optimizer.plan_roles(
            devs, links, tp_spec, SplitKvSpec(32_768, 512 << 10, 2e7, 64 << 10, 2 << 30, 0.85)
        )
        self.assertEqual(long_role.active_regime, "SPLIT_KV_RECLAIMED")
        self.assertEqual(long_role.reclaimed_gpu_count, 1)
        self.assertEqual(long_role.split_kv_assignments[1].role, DeviceRole.SPLIT_KV_RECLAIMED)

    def test_merge_split_kv_extreme_logits(self) -> None:
        # Worker 0: m=-50, l=1, o=[10, -10]; Worker 1: m=+50, l=1, o=[-3, 7]
        # Numerically e^(-100) vanishes cleanly so merged == Worker 1 output
        merged = merge_split_kv(
            num_heads=1,
            head_dim=2,
            partials_o=[[10.0, -10.0], [-3.0, 7.0]],
            partials_stats=[[-50.0, 1.0], [50.0, 1.0]],
        )
        self.assertTrue(all(math.isfinite(x) for x in merged))
        self.assertAlmostEqual(merged[0], -3.0, places=5)
        self.assertAlmostEqual(merged[1], 7.0, places=5)

    def test_live_vulkan_collective_runtime(self) -> None:
        if not hc4py.is_gpu_available():
            self.skipTest("No GPU runtime available")
        with Collective(2) as coll:
            self.assertEqual(coll.ranks, 2)
            self.assertTrue(len(coll.backend_name()) > 0)
            profiles = coll.device_profiles()
            self.assertEqual(len(profiles), 2)
            links = coll.topology()
            self.assertEqual(len(links), 2)
            ring = coll.ring_order()
            self.assertEqual(sorted(ring), [0, 1])
            islands = coll.islands()
            self.assertGreaterEqual(len(islands), 1)

            with coll.allocate(0, 4) as b0, coll.allocate(1, 4) as b1:
                b0.write([1.0, 2.0, 3.0, 4.0])
                b1.write([10.0, 20.0, 30.0, 40.0])
                coll.allreduce_sum([b0, b1])
                self.assertEqual(b0.read_list(), [11.0, 22.0, 33.0, 44.0])
                self.assertEqual(b1.read_list(), [11.0, 22.0, 33.0, 44.0])


if __name__ == "__main__":
    unittest.main()
