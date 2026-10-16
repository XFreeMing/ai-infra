from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab_tools.media_resources import ResourceMonitor, _parse_stat, statistics


class ResourceMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.proc = self.root / "proc"
        self.sys = self.root / "sys"
        self.output = self.root / "output"
        self.now = 10.0
        self.console = []
        self.host(cpu="100 0 100 800 0 0 0 0 0 0", available=800)
        self.process(42, ppid=1, cpu=100, rss=10, read=1000, write=2000)
        self.gpu()

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(text), encoding="utf-8")

    def host(self, *, cpu, available):
        self.write(self.proc / "stat", f"cpu {cpu}\ncpu0 0 0 0 0\n")
        self.write(self.proc / "meminfo", f"MemTotal: 1000 kB\nMemAvailable: {available} kB\nSwapTotal: 100 kB\nSwapFree: 70 kB\n")

    def process(self, pid, *, ppid, cpu, rss, read=0, write=0, born=500):
        fields = ["0"] * 22
        fields[0] = "S"
        fields[1] = str(ppid)
        fields[11] = str(cpu)
        fields[12] = "0"
        fields[19] = str(born)
        fields[21] = str(rss)
        self.write(self.proc / str(pid) / "stat", f"{pid} (worker (with) spaces) " + " ".join(fields))
        self.write(self.proc / str(pid) / "io", f"rchar: 999\nread_bytes: {read}\nwrite_bytes: {write}\n")

    def gpu(self, index=0):
        pci = f"0000:c{index + 5}:00.0"
        device = self.sys / "devices" / pci
        self.write(device / "vendor", "0x1002")
        self.write(device / "uevent", f"DRIVER=amdgpu\nPCI_SLOT_NAME={pci}\n")
        for name, value in {
            "gpu_busy_percent": 5,
            "mem_info_vram_used": 100,
            "mem_info_vram_total": 1000,
            "mem_info_gtt_used": 200,
            "mem_info_gtt_total": 2000,
        }.items():
            self.write(device / name, value)
        hwmon = device / "hwmon" / f"hwmon{index}"
        self.write(hwmon / "name", "amdgpu")
        self.write(hwmon / "power1_average", "13500000")
        self.write(hwmon / "temp1_input", "38500")
        self.write(hwmon / "temp1_label", "edge")
        for name in (f"card{index + 1}", f"renderD{128 + index}"):
            directory = self.sys / "class/drm" / name
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "device").symlink_to(device, target_is_directory=True)
        node = self.sys / "class/kfd/kfd/topology/nodes" / str(index + 1)
        self.write(node / "gpu_id", str(100 + index))
        self.write(node / "properties", f"vendor_id 4098\nsimd_count 80\ndrm_render_minor {128 + index}\n")
        return device

    def monitor(self, **kwargs):
        return ResourceMonitor(
            self.output,
            proc_root=self.proc,
            sys_root=self.sys,
            monotonic=lambda: self.now,
            utc_now=lambda: "2026-10-16T00:00:00+00:00",
            printer=self.console.append,
            clock_ticks=100,
            page_size=4096,
            **kwargs,
        )

    def phase(self, value):
        self.write(self.output / "runtime-performance.json", json.dumps({"active_phase": value}))

    def test_valid_statistics_exclude_missing_nan_and_infinity(self):
        self.assertEqual(statistics([1, None, float("nan"), 3, float("inf"), False]), {"count": 2, "min": 1, "max": 3, "mean": 2.0})
        self.assertEqual(statistics([None, float("nan")]), {"count": 0, "min": None, "max": None, "mean": None})

    def test_proc_stat_accepts_spaces_and_parentheses_in_comm(self):
        result = _parse_stat((self.proc / "42/stat").read_text())
        self.assertEqual(result, {"ppid": 1, "cpu_ticks": 100, "start_ticks": 500, "rss_pages": 10})

    def test_initial_sample_has_no_fabricated_cpu_usage_and_flushes_jsonl(self):
        monitor = self.monitor()
        monitor.start(42)
        sample = json.loads((self.output / "resources.jsonl").read_text())
        self.assertIsNone(sample["metrics"]["process_cpu_percent"])
        self.assertIsNone(sample["metrics"]["host_cpu_percent"])
        self.assertEqual(sample["metrics"]["host_memory_used_bytes"], 200 * 1024)
        self.assertEqual(sample["metrics"]["host_swap_used_bytes"], 30 * 1024)
        self.assertEqual(sample["metrics"]["process_rss_bytes"], 10 * 4096)
        self.assertEqual(sample["phase"], "runtime_startup")
        self.assertIn("CPU 进程 N/A", self.console[0])
        monitor.finish()

    def test_sampling_is_rate_limited_and_cpu_can_exceed_one_core(self):
        monitor = self.monitor()
        monitor.start(42)
        self.now = 11.9
        self.assertFalse(monitor.poll())
        self.assertEqual(len(monitor.samples), 1)
        self.now = 12.0
        self.host(cpu="200 0 200 1000 0 0 0 0 500 0", available=600)
        self.process(42, ppid=1, cpu=500, rss=20, read=1600, write=2800)
        self.assertTrue(monitor.poll())
        values = monitor.samples[-1]["metrics"]
        self.assertEqual(values["host_cpu_percent"], 50)
        self.assertEqual(values["process_cpu_percent"], 200)
        self.assertEqual(values["process_read_bytes"], 600)
        self.assertEqual(values["process_write_bytes"], 800)
        summary = monitor.finish()
        self.assertEqual(summary["metrics"]["process_cpu_percent"]["count"], 1)
        self.assertEqual(summary["metrics"]["process_cpu_percent"]["mean"], 200)

    def test_process_tree_includes_new_children_and_retains_reparented_child(self):
        monitor = self.monitor()
        monitor.start(42)
        self.now += 2
        self.process(42, ppid=1, cpu=150, rss=10)
        self.process(43, ppid=42, cpu=50, rss=5, born=600)
        monitor.poll()
        self.assertEqual(monitor.samples[-1]["metrics"]["process_count"], 2)
        self.assertEqual(monitor.samples[-1]["metrics"]["process_rss_bytes"], 15 * 4096)
        self.assertEqual(monitor.samples[-1]["metrics"]["process_cpu_percent"], 50)
        (self.proc / "42/stat").unlink()
        self.process(43, ppid=1, cpu=150, rss=4, born=600)
        self.now += 2
        monitor.poll()
        self.assertEqual(monitor.samples[-1]["metrics"]["process_count"], 1)
        self.assertEqual(monitor.samples[-1]["metrics"]["process_cpu_percent"], 50)
        monitor.finish()

    def test_pid_reuse_does_not_include_an_unrelated_process(self):
        monitor = self.monitor()
        monitor.start(42)
        self.process(42, ppid=1, cpu=400, rss=200, born=900)
        self.now += 2
        monitor.poll()
        self.assertEqual(monitor.samples[-1]["metrics"]["process_count"], 0)
        self.assertIsNone(monitor.samples[-1]["metrics"]["process_rss_bytes"])
        monitor.finish()

    def test_gpu_is_matched_through_kfd_render_minor_and_uses_soc_units(self):
        monitor = self.monitor()
        monitor.start(42)
        gpu = monitor.samples[0]["gpus"][0]
        self.assertEqual(gpu["drm_card"], "card1")
        self.assertEqual(gpu["render_minor"], 128)
        self.assertEqual(gpu["pci_address"], "0000:c5:00.0")
        self.assertEqual(gpu["hip_index"], 0)
        self.assertEqual(gpu["metrics"]["soc_power_watts"], 13.5)
        self.assertEqual(gpu["metrics"]["temperature_celsius"], 38.5)
        self.assertIn("SoC 13.5W", self.console[0])
        summary = monitor.finish()
        self.assertEqual(summary["gpus"][0]["metrics"]["gtt_used_bytes"]["max"], 200)
        self.assertIn("includes CPU", summary["metadata"]["units"]["gpu"]["soc_power_watts"])
        self.assertTrue(any("must not be added" in item for item in summary["metadata"]["limitations"]))

    def test_no_silent_multi_gpu_ordinal_guess(self):
        self.gpu(1)
        monitor = self.monitor(devices="0,1")
        monitor.start(42)
        self.assertEqual(monitor.samples[0]["gpus"], [])
        summary = monitor.finish()
        self.assertTrue(any("cannot be uniquely mapped" in item["message"] for item in summary["errors"]))

    def test_multi_gpu_explicit_pci_mapping_keeps_metrics_separate(self):
        second = self.gpu(1)
        self.write(second / "gpu_busy_percent", 80)
        monitor = self.monitor(devices="0,1", hip_pci_bus_ids={0: "0000:c6:00.0", 1: "0000:c5:00.0"})
        monitor.start(42)
        readings = monitor.samples[0]["gpus"]
        self.assertEqual([item["metrics"]["busy_percent"] for item in readings], [80, 5])
        self.assertEqual([item["hip_index"] for item in readings], [0, 1])
        summary = monitor.finish()
        self.assertEqual(summary["gpus"][0]["metrics"]["busy_percent"]["mean"], 80)
        self.assertEqual(summary["gpus"][1]["metrics"]["busy_percent"]["mean"], 5)

    def test_invalid_or_missing_device_does_not_stop_process_monitoring(self):
        for selection in ("1", "0,0", "-1", "GPU-unknown"):
            with self.subTest(selection=selection):
                monitor = self.monitor(devices=selection)
                monitor.start(42)
                self.assertEqual(monitor.samples[0]["gpus"], [])
                self.assertEqual(monitor.samples[0]["metrics"]["process_rss_bytes"], 40960)
                self.assertTrue(monitor.finish()["errors"])

    def test_missing_invalid_and_unreadable_measurements_are_null(self):
        device = self.sys / "devices/0000:c5:00.0"
        (device / "mem_info_gtt_used").unlink()
        self.write(device / "gpu_busy_percent", "NaN")
        self.write(device / "mem_info_vram_used", "-7")
        original = Path.read_text
        def read_text(path, *args, **kwargs):
            if path.name == "power1_average":
                raise PermissionError("test")
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", read_text):
            monitor = self.monitor()
            monitor.start(42)
            summary = monitor.finish()
        values = monitor.samples[0]["gpus"][0]["metrics"]
        for key in ("busy_percent", "vram_used_bytes", "gtt_used_bytes", "soc_power_watts"):
            self.assertIsNone(values[key])
            self.assertEqual(summary["gpus"][0]["metrics"][key]["count"], 0)
        self.assertTrue(any("PermissionError" in item["message"] for item in summary["errors"]))
        text = (self.output / "resources.json").read_text()
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)

    def test_gpu_busy_percent_out_of_range_is_not_reported_as_real(self):
        self.write(self.sys / "devices/0000:c5:00.0/gpu_busy_percent", 101)
        monitor = self.monitor()
        monitor.start(42)
        self.assertIsNone(monitor.samples[0]["gpus"][0]["metrics"]["busy_percent"])
        self.assertTrue(any("out of range" in item["message"] for item in monitor.finish()["errors"]))

    def test_phase_summaries_and_final_sample_are_written_with_compact_return(self):
        monitor = self.monitor()
        monitor.start(42)
        self.phase("model_load")
        self.now += 2
        self.process(42, ppid=1, cpu=200, rss=20)
        monitor.poll()
        self.phase("generation")
        self.now += 2
        self.process(42, ppid=1, cpu=400, rss=30)
        monitor.poll()
        self.phase(None)
        self.now += 0.5
        summary = monitor.finish()
        self.assertEqual(summary["sample_count"], 4)
        self.assertEqual(summary["elapsed_seconds"], 4.5)
        self.assertEqual(list(summary["by_phase"]), ["runtime_startup", "model_load", "generation", "worker_finishing"])
        self.assertEqual(summary["by_phase"]["model_load"]["metrics"]["process_rss_bytes"]["max"], 20 * 4096)
        self.assertEqual(summary["by_phase"]["generation"]["metrics"]["process_cpu_percent"]["mean"], 100)
        self.assertNotIn("samples", summary)
        document = json.loads((self.output / "resources.json").read_text())
        self.assertEqual(document["summary"], summary)
        self.assertEqual(document["samples"], monitor.samples)
        self.assertEqual(len((self.output / "resources.jsonl").read_text().splitlines()), 4)
        self.assertTrue(any("最小 / 最大 / 平均" in line for line in self.console))
        self.assertTrue(any("资源汇总" in line for line in self.console))

    def test_finish_is_idempotent_and_poll_stops(self):
        monitor = self.monitor()
        self.assertFalse(monitor.poll())
        monitor.start(42)
        first = monitor.finish()
        lines = len(self.console)
        self.now += 9
        self.assertFalse(monitor.poll())
        self.assertIs(monitor.finish(), first)
        self.assertEqual(len(self.console), lines)

    def test_running_worker_without_active_phase_is_overhead(self):
        self.write(self.output / "runtime-performance.json", json.dumps({"status": "running", "active_phase": None}))
        monitor = self.monitor()
        monitor.start(42)
        self.assertEqual(monitor.samples[0]["phase"], "worker_overhead")
        monitor.finish()

    def test_unreadable_proc_files_produce_errors_without_breaking_generation(self):
        (self.proc / "stat").unlink()
        self.write(self.proc / "meminfo", "MemTotal: not-a-number kB\nMemAvailable: 999999 kB\n")
        monitor = self.monitor()
        monitor.start(42)
        summary = monitor.finish()
        self.assertIsNone(summary["metrics"]["host_cpu_percent"]["mean"])
        self.assertIsNone(summary["metrics"]["host_memory_total_bytes"]["mean"])
        self.assertEqual(summary["metrics"]["process_rss_bytes"]["mean"], 40960)
        self.assertGreaterEqual(len(summary["errors"]), 2)

    def test_io_permission_failure_does_not_become_zero_or_mask_other_metrics(self):
        original = Path.read_text
        def read_text(path, *args, **kwargs):
            if path == self.proc / "42/io":
                raise PermissionError("test")
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", read_text):
            monitor = self.monitor()
            monitor.start(42)
            summary = monitor.finish()
        self.assertIsNone(summary["metrics"]["process_read_bytes"]["mean"])
        self.assertEqual(summary["metrics"]["process_rss_bytes"]["mean"], 40960)

    def test_invalid_interval_rejected(self):
        for interval in (0, -1, float("nan"), float("inf")):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                self.monitor(interval=interval)

    def test_console_failure_is_recorded_and_data_still_saved(self):
        monitor = self.monitor()
        monitor.printer = lambda line: (_ for _ in ()).throw(BrokenPipeError("test"))
        monitor.start(42)
        summary = monitor.finish()
        self.assertTrue((self.output / "resources.json").is_file())
        self.assertTrue(any("display resource progress" in item["message"] for item in summary["errors"]))


if __name__ == "__main__":
    unittest.main()
