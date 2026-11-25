# Copyright 2026 Abhinav Chinnala (chinnala.abhinav5@gmail.com)
# SPDX-License-Identifier: Apache-2.0

"""Verifies 1-to-1 ctypes struct layouts, 107 C-ABI symbol bindings, and hc4j_last_error."""

import ctypes
import unittest

from hc4py._ffi import (
    BOUND_SYMBOLS,
    AdapterInfoC,
    DeviceProfileC,
    EngineStats,
    HostCpuProfileC,
    InvalidParamError,
    IslandHeaderC,
    LinkInfo,
    MemStats,
    NotFoundError,
    RolePlanHeaderC,
    ShardAllocationC,
    SplitKvAssignmentC,
    SplitKvSpecC,
    WorkloadPlanHeaderC,
    WorkloadSpecC,
    check_status,
    get_library,
)


class FfiLayoutsTest(unittest.TestCase):
    def test_all_107_symbols_bound_and_resolvable(self) -> None:
        lib = get_library()
        self.assertEqual(len(BOUND_SYMBOLS), 107)
        self.assertEqual(len(set(BOUND_SYMBOLS)), 107)
        for name in BOUND_SYMBOLS:
            fn = getattr(lib, name, None)
            self.assertIsNotNone(fn, f"Missing symbol {name}")

    def test_all_13_struct_sizes_and_key_offsets(self) -> None:
        expected_sizes = {
            MemStats: 168,
            EngineStats: 104,
            AdapterInfoC: 668,
            LinkInfo: 32,
            DeviceProfileC: 32,
            IslandHeaderC: 16,
            HostCpuProfileC: 32,
            WorkloadSpecC: 120,
            ShardAllocationC: 72,
            WorkloadPlanHeaderC: 40,
            SplitKvSpecC: 48,
            SplitKvAssignmentC: 48,
            RolePlanHeaderC: 80,
        }
        self.assertEqual(len(expected_sizes), 13)
        for cls, sz in expected_sizes.items():
            self.assertEqual(ctypes.sizeof(cls), sz, f"Size mismatch for {cls.__name__}")

        self.assertEqual(WorkloadSpecC.allow_spilling.offset, 84)
        self.assertEqual(WorkloadSpecC.host_cpu.offset, 88)
        self.assertEqual(ShardAllocationC.units.offset, 8)
        self.assertEqual(ShardAllocationC.cpu_in_place_units.offset, 64)
        self.assertEqual(WorkloadPlanHeaderC.allocation_count.offset, 32)
        self.assertEqual(RolePlanHeaderC.estimated_tp_ms.offset, 40)
        self.assertEqual(RolePlanHeaderC.regime_code.offset, 76)

    def test_last_error_enriches_invalid_param_exception(self) -> None:
        lib = get_library()
        hdr = WorkloadPlanHeaderC()
        status = lib.hc4j_optimizer_plan_work(0, None, 0, None, None, ctypes.byref(hdr), None, 0, None, 0)
        with self.assertRaises(InvalidParamError) as ctx:
            check_status(status, "test_plan_work_null", lib)
        msg = str(ctx.exception)
        self.assertIn("[", msg)
        self.assertIn("null", msg.lower())

    def test_status_mapping_types(self) -> None:
        with self.assertRaises(NotFoundError):
            check_status(-1, "test_not_found")


if __name__ == "__main__":
    unittest.main()
