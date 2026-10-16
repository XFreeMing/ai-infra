"""Best-effort, standard-library resource sampling for media hello workers.

CPU and RAM come from Linux procfs.  AMD readings come from the selected DRM
device's sysfs files; they describe the whole device, not just this worker.
Monitoring errors must never turn a successful inference into a failed one.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Callable, Mapping, TextIO


HOST_PROCESS_UNITS = {
    "host_cpu_percent": "% of all logical CPUs",
    "host_memory_total_bytes": "bytes",
    "host_memory_available_bytes": "bytes",
    "host_memory_used_bytes": "bytes",
    "host_swap_total_bytes": "bytes",
    "host_swap_used_bytes": "bytes",
    "process_cpu_percent": "% (100% = one logical CPU)",
    "process_rss_bytes": "bytes",
    "process_count": "processes",
    "process_read_bytes": "bytes since monitoring began (observed deltas)",
    "process_write_bytes": "bytes since monitoring began (observed deltas)",
}
GPU_UNITS = {
    "busy_percent": "%",
    "vram_used_bytes": "bytes",
    "vram_total_bytes": "bytes",
    "gtt_used_bytes": "bytes",
    "gtt_total_bytes": "bytes",
    "soc_power_watts": "watts (SoC; includes CPU on an APU)",
    "temperature_celsius": "degrees Celsius",
}
LIMITATIONS = [
    "CPU percentages are interval measurements; process CPU 100% means one logical CPU, while host CPU 100% means all logical CPUs.",
    "Process metrics include the worker and observed descendants. Short-lived processes between samples may be missed; shared pages can be counted more than once in summed RSS.",
    "GPU readings are device-wide and may include other processes. VRAM, GTT and process RSS can overlap on unified memory and must not be added together.",
    "AMD power1 readings describe SoC power; on this APU they include CPU power, not isolated GPU power.",
    "Statistics use available finite samples only. The mean is an arithmetic sample mean, not a time-weighted mean; sampled extrema can miss brief peaks.",
    "Missing or unreadable measurements are null, not zero. Disk I/O is observed process-tree counter growth since monitoring began, not storage-device throughput.",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def statistics(values: list[object]) -> dict[str, int | float | None]:
    """A missing reading is not a zero and must not bias a summary."""
    valid = [value for value in values if _finite(value)]
    return {
        "count": len(valid),
        "min": min(valid) if valid else None,
        "max": max(valid) if valid else None,
        "mean": math.fsum(valid) / len(valid) if valid else None,
    }


def _clean_json(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    return value


def _parse_stat(text: str) -> dict[str, int]:
    # comm may contain spaces and ')' characters; split at the final ')'.
    fields = text[text.rfind(")") + 2 :].split()
    return {
        "ppid": int(fields[1]),
        "cpu_ticks": int(fields[11]) + int(fields[12]),
        "start_ticks": int(fields[19]),
        "rss_pages": max(0, int(fields[21])),
    }


class ResourceMonitor:
    """Sample a worker without dependencies or a background sampling thread.

    ``poll`` returns whether a new sample was printed. ``finish`` is idempotent,
    prints sampled extrema/means and returns a compact summary with file paths.
    Tests may inject procfs/sysfs roots, a clock and a console printer. For
    multiple GPUs, explicit HIP ordinal -> PCI address mappings are required:
    Linux KFD node order alone does not prove the HIP runtime's device order.
    """

    def __init__(
        self,
        directory: Path,
        interval: float = 2.0,
        devices: str = "0",
        *,
        proc_root: Path = Path("/proc"),
        sys_root: Path = Path("/sys"),
        monotonic: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], str] = _utc_now,
        printer: Callable[[str], object] | None = None,
        hip_pci_bus_ids: Mapping[int, str] | None = None,
        clock_ticks: int | None = None,
        page_size: int | None = None,
    ) -> None:
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("resource sampling interval must be positive and finite")
        self.directory = Path(directory)
        self.interval = interval
        self.devices = devices
        self.proc_root = Path(proc_root)
        self.sys_root = Path(sys_root)
        self.monotonic = monotonic
        self.utc_now = utc_now
        self.printer = printer or (lambda line: print(line, flush=True))
        self.hip_pci_bus_ids = dict(hip_pci_bus_ids or {})
        self.clock_ticks = clock_ticks or int(os.sysconf("SC_CLK_TCK"))
        self.page_size = page_size or int(os.sysconf("SC_PAGE_SIZE"))
        self.samples: list[dict] = []
        self.errors: Counter[str] = Counter()
        self.limitations = list(LIMITATIONS)
        self._sample_errors: list[str] = []
        self._jsonl: TextIO | None = None
        self._pid: int | None = None
        self._started: float | None = None
        self._last_at: float | None = None
        self._next_at = 0.0
        self._started_at: str | None = None
        self._host_previous: tuple[int, int] | None = None
        self._process_previous: dict[tuple[int, int], dict] = {}
        self._known_processes: set[tuple[int, int]] = set()
        self._root_identity: tuple[int, int] | None = None
        self._io_totals = {"read_bytes": 0, "write_bytes": 0}
        self._gpu_devices: list[dict] = []
        self._summary: dict | None = None

    def _error(self, message: str) -> None:
        self.errors[message] += 1
        self._sample_errors.append(message)

    def _read(self, path: Path, *, optional: bool = False) -> str | None:
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            if not optional:
                self._error(f"missing: {path}")
        except (OSError, UnicodeError) as error:
            self._error(f"cannot read {path}: {type(error).__name__}")
        return None

    def _number(self, path: Path, *, scale: float = 1, nonnegative: bool = True) -> int | float | None:
        text = self._read(path)
        if text is None:
            return None
        try:
            value = int(text)
            if nonnegative and value < 0:
                raise ValueError("negative counter")
            result = value if scale == 1 else value / scale
            if not math.isfinite(result):
                raise ValueError("non-finite counter")
            return result
        except (ValueError, OverflowError):
            self._error(f"invalid numeric reading: {path}")
            return None

    def _print(self, line: str) -> None:
        try:
            self.printer(line)
        except Exception as error:
            self._error(f"cannot display resource progress: {type(error).__name__}")

    def _discover_gpus(self) -> None:
        try:
            ordinals = [int(item.strip()) for item in self.devices.split(",")]
            if not ordinals or any(item < 0 for item in ordinals) or len(set(ordinals)) != len(ordinals):
                raise ValueError("invalid or repeated ordinal")
        except ValueError:
            self._error(f"unsupported HIP_VISIBLE_DEVICES selection: {self.devices!r}")
            self.limitations.append("GPU selection could not be resolved; no GPU counters were attributed to this worker.")
            return

        drm = self.sys_root / "class/drm"
        cards: dict[Path, dict] = {}
        try:
            card_paths = sorted(path for path in drm.glob("card*") if re.fullmatch(r"card\d+", path.name))
        except OSError as error:
            self._error(f"cannot enumerate DRM devices: {type(error).__name__}")
            card_paths = []
        for card in card_paths:
            device = card / "device"
            vendor = self._read(device / "vendor", optional=True)
            if vendor is None or vendor.lower() != "0x1002":
                continue
            uevent = self._read(device / "uevent", optional=True) or ""
            entries = dict(line.split("=", 1) for line in uevent.splitlines() if "=" in line)
            cards[device.resolve()] = {
                "device_path": device,
                "drm_card": card.name,
                "pci_address": entries.get("PCI_SLOT_NAME"),
            }

        kfd_gpus: list[dict] = []
        nodes = self.sys_root / "class/kfd/kfd/topology/nodes"
        try:
            node_paths = sorted((path for path in nodes.glob("*") if path.name.isdecimal()), key=lambda path: int(path.name))
        except OSError as error:
            self._error(f"cannot enumerate KFD topology: {type(error).__name__}")
            node_paths = []
        for node in node_paths:
            gpu_id = self._read(node / "gpu_id", optional=True)
            if gpu_id in (None, "0"):
                continue
            properties_text = self._read(node / "properties")
            if properties_text is None:
                continue
            try:
                props = {key: int(value, 0) for key, value in (line.split(None, 1) for line in properties_text.splitlines())}
            except ValueError:
                self._error(f"invalid KFD properties: {node / 'properties'}")
                continue
            if props.get("vendor_id") != 0x1002 or props.get("simd_count", 0) <= 0:
                continue
            render_minor = props.get("drm_render_minor")
            render_device = (drm / f"renderD{render_minor}" / "device").resolve()
            match = cards.get(render_device)
            if match is None:
                self._error(f"KFD GPU {node.name} renderD{render_minor} has no matching AMD DRM card")
                continue
            kfd_gpus.append({**match, "kfd_node": node.name, "kfd_gpu_id": gpu_id, "render_minor": render_minor})

        selected: list[tuple[int, dict, str]] = []
        if self.hip_pci_bus_ids:
            for ordinal in ordinals:
                pci = self.hip_pci_bus_ids.get(ordinal)
                matches = [gpu for gpu in kfd_gpus if gpu["pci_address"] == pci] if pci else []
                if len(matches) == 1:
                    selected.append((ordinal, matches[0], "explicit HIP ordinal to PCI mapping, confirmed against KFD/DRM"))
                else:
                    self._error(f"HIP device {ordinal} PCI mapping is missing or does not uniquely match KFD/DRM")
        elif len(kfd_gpus) == 1 and len(cards) == 1 and ordinals == [0]:
            selected.append((0, kfd_gpus[0], "only AMD KFD GPU, matched to DRM via render minor"))
        elif not kfd_gpus and len(cards) == 1 and ordinals == [0] and not node_paths:
            selected.append((0, next(iter(cards.values())), "only AMD DRM GPU; KFD topology unavailable"))
            self.limitations.append("KFD topology was unavailable; GPU 0 was matched to the only AMD DRM GPU.")
        else:
            self._error("HIP device ordinals cannot be uniquely mapped to AMD DRM devices")
            self.limitations.append("Multi-GPU HIP order is not inferred from KFD node order. An explicit HIP ordinal to PCI mapping is required; unresolved devices are not sampled.")

        for ordinal, device, resolution in selected:
            path = device["device_path"]
            hwmon_paths = sorted((path / "hwmon").glob("hwmon*"))
            power = None
            temperature = None
            temperature_label = None
            for hwmon in hwmon_paths:
                if (self._read(hwmon / "name", optional=True) or "") != "amdgpu":
                    continue
                for filename in ("power1_average", "power1_input"):
                    if (hwmon / filename).exists():
                        power = hwmon / filename
                        break
                temperatures = sorted(hwmon.glob("temp*_input"))
                if temperatures:
                    temperature = temperatures[0]
                    for candidate in temperatures:
                        label = self._read(candidate.with_name(candidate.name.replace("_input", "_label")), optional=True)
                        if label == "edge":
                            temperature = candidate
                            break
                    temperature_label = self._read(temperature.with_name(temperature.name.replace("_input", "_label")), optional=True)
                break
            metadata = {key: value for key, value in device.items() if key != "device_path"}
            metadata.update({
                "id": device.get("pci_address") or device["drm_card"],
                "hip_index": ordinal,
                "selection": resolution,
                "scope": "whole device, including other processes",
                "temperature_sensor": temperature_label,
                "soc_power_sensor": str(power) if power else None,
            })
            if power is None:
                self.limitations.append(f"{metadata['id']}: SoC power sensor is unavailable.")
            if temperature is None:
                self.limitations.append(f"{metadata['id']}: temperature sensor is unavailable.")
            self._gpu_devices.append({"metadata": metadata, "path": path, "power": power, "temperature": temperature})

    def start(self, pid: int) -> None:
        if self._started is not None:
            raise RuntimeError("resource monitor has already started")
        self._pid = pid
        self._started = self.monotonic()
        self._started_at = self.utc_now()
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._jsonl = (self.directory / "resources.jsonl").open("w", encoding="utf-8")
        except OSError as error:
            self._error(f"cannot open resource sample log: {type(error).__name__}")
        self._discover_gpus()
        self._sample(self._started)

    def _host(self) -> dict:
        result = {key: None for key in HOST_PROCESS_UNITS if key.startswith("host_")}
        stat = self._read(self.proc_root / "stat")
        if stat:
            try:
                cpu_line = next(line for line in stat.splitlines() if line.startswith("cpu "))
                # guest and guest_nice are already included in user/nice.
                values = [int(value) for value in cpu_line.split()[1:9]]
                if len(values) < 4 or any(value < 0 for value in values):
                    raise ValueError("invalid CPU counters")
                total = sum(values)
                idle = values[3] + (values[4] if len(values) > 4 else 0)
                if self._host_previous:
                    previous_total, previous_idle = self._host_previous
                    delta = total - previous_total
                    idle_delta = idle - previous_idle
                    if delta > 0 and 0 <= idle_delta <= delta:
                        result["host_cpu_percent"] = 100.0 * (delta - idle_delta) / delta
                self._host_previous = (total, idle)
            except (ValueError, StopIteration):
                self._error(f"invalid CPU counters: {self.proc_root / 'stat'}")
        text = self._read(self.proc_root / "meminfo")
        if text:
            memory: dict[str, int] = {}
            for line in text.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                if key not in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                    continue
                try:
                    fields = value.split()
                    amount = int(fields[0])
                    if amount < 0 or (len(fields) > 1 and fields[1] != "kB"):
                        raise ValueError("invalid memory value")
                    memory[key] = amount * 1024
                except (ValueError, IndexError):
                    self._error(f"invalid {key} reading: {self.proc_root / 'meminfo'}")
            result["host_memory_total_bytes"] = memory.get("MemTotal")
            result["host_memory_available_bytes"] = memory.get("MemAvailable")
            result["host_swap_total_bytes"] = memory.get("SwapTotal")
            for total, available, output in (
                ("MemTotal", "MemAvailable", "host_memory_used_bytes"),
                ("SwapTotal", "SwapFree", "host_swap_used_bytes"),
            ):
                if total in memory and available in memory:
                    if memory[total] >= memory[available]:
                        result[output] = memory[total] - memory[available]
                    else:
                        self._error(f"inconsistent {total}/{available}: {self.proc_root / 'meminfo'}")
        return result

    def _processes(self, now: float) -> dict:
        result = {key: None for key in HOST_PROCESS_UNITS if key.startswith("process_")}
        processes: dict[int, dict] = {}
        try:
            entries = list(self.proc_root.iterdir())
        except OSError as error:
            self._error(f"cannot enumerate processes: {type(error).__name__}")
            return result
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            try:
                data = _parse_stat((entry / "stat").read_text(encoding="utf-8"))
                processes[int(entry.name)] = data
            except (FileNotFoundError, ProcessLookupError):
                continue  # Expected when a short-lived process exits.
            except (OSError, ValueError, IndexError, UnicodeError) as error:
                if int(entry.name) == self._pid or any(pid == int(entry.name) for pid, _ in self._known_processes):
                    self._error(f"cannot read process stat {entry.name}: {type(error).__name__}")

        root = processes.get(self._pid)
        if root and self._root_identity is None:
            self._root_identity = (self._pid, root["start_ticks"])
        selected: set[int] = {
            pid for pid, data in processes.items()
            if (pid, data["start_ticks"]) in self._known_processes
        }
        if root and (self._pid, root["start_ticks"]) == self._root_identity:
            selected.add(self._pid)
        while True:
            children = {pid for pid, data in processes.items() if data["ppid"] in selected}
            expanded = selected | children
            if expanded == selected:
                break
            selected = expanded
        result["process_count"] = len(selected)
        if not selected:
            return result

        rss = 0
        cpu_delta = 0
        io_valid = {"read_bytes": True, "write_bytes": True}
        current: dict[tuple[int, int], dict] = {}
        for pid in sorted(selected):
            data = processes[pid]
            identity = (pid, data["start_ticks"])
            previous = self._process_previous.get(identity)
            rss += data["rss_pages"] * self.page_size
            if self._last_at is not None:
                cpu_delta += max(0, data["cpu_ticks"] - (previous["cpu_ticks"] if previous else 0))
            text = self._read(self.proc_root / str(pid) / "io", optional=True)
            io: dict[str, int] = {}
            if text is not None:
                try:
                    io = {key.strip(): int(value.strip()) for key, value in (line.split(":", 1) for line in text.splitlines() if ":" in line)}
                except ValueError:
                    self._error(f"invalid process I/O counters: {self.proc_root / str(pid) / 'io'}")
            for name in io_valid:
                value = io.get(name)
                if value is None or value < 0:
                    io_valid[name] = False
                    continue
                old = previous.get(name) if previous else None
                if self._last_at is not None and (previous is None or old is not None):
                    self._io_totals[name] += max(0, value - (old or 0))
            current[identity] = {**data, **{key: io.get(key) for key in io_valid}}
        self._known_processes.update(current)
        self._process_previous.update(current)
        result["process_rss_bytes"] = rss
        if self._last_at is not None and now > self._last_at:
            result["process_cpu_percent"] = 100.0 * cpu_delta / self.clock_ticks / (now - self._last_at)
        for name, valid in io_valid.items():
            if valid:
                result[f"process_{name}"] = self._io_totals[name]
        return result

    def _gpus(self) -> list[dict]:
        readings = []
        for gpu in self._gpu_devices:
            path = gpu["path"]
            metrics = {
                "busy_percent": self._number(path / "gpu_busy_percent"),
                "vram_used_bytes": self._number(path / "mem_info_vram_used"),
                "vram_total_bytes": self._number(path / "mem_info_vram_total"),
                "gtt_used_bytes": self._number(path / "mem_info_gtt_used"),
                "gtt_total_bytes": self._number(path / "mem_info_gtt_total"),
                "soc_power_watts": self._number(gpu["power"], scale=1_000_000) if gpu["power"] else None,
                "temperature_celsius": self._number(gpu["temperature"], scale=1000, nonnegative=False) if gpu["temperature"] else None,
            }
            busy = metrics["busy_percent"]
            if busy is not None and busy > 100:
                self._error(f"GPU busy percent out of range: {path / 'gpu_busy_percent'}")
                metrics["busy_percent"] = None
            readings.append({**gpu["metadata"], "metrics": metrics})
        return readings

    def _phase(self) -> str:
        try:
            document = json.loads((self.directory / "runtime-performance.json").read_text(encoding="utf-8"))
            phase = document.get("active_phase")
            if isinstance(phase, str) and phase:
                return phase
            return "worker_overhead" if document.get("status") == "running" else "worker_finishing"
        except (OSError, ValueError, AttributeError):
            return "runtime_startup"

    def _sample(self, now: float) -> None:
        sample = {
            "timestamp": self.utc_now(),
            "elapsed_seconds": max(0.0, now - self._started),
            "phase": self._phase(),
            "metrics": {**self._host(), **self._processes(now)},
            "gpus": self._gpus(),
            "errors": list(dict.fromkeys(self._sample_errors)),
        }
        self._sample_errors.clear()
        sample = _clean_json(sample)
        self.samples.append(sample)
        if self._jsonl:
            try:
                self._jsonl.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                self._jsonl.flush()
            except (OSError, ValueError) as error:
                self._error(f"cannot write resource sample log: {type(error).__name__}")
        self._last_at = now
        self._next_at = now + self.interval
        self._print(self._format_sample(sample))

    @staticmethod
    def _value(value: object, *, divisor: float = 1, suffix: str = "", digits: int = 1) -> str:
        return f"{value / divisor:.{digits}f}{suffix}" if _finite(value) else "N/A"

    def _format_sample(self, sample: dict) -> str:
        metrics = sample["metrics"]
        value = self._value
        gib = 1024 ** 3
        pieces = [
            f"资源｜{sample['phase']}｜已用 {sample['elapsed_seconds']:.1f}s",
            f"CPU 进程 {value(metrics['process_cpu_percent'], suffix='%')} / 主机 {value(metrics['host_cpu_percent'], suffix='%')}",
            f"RSS {value(metrics['process_rss_bytes'], divisor=gib, suffix=' GiB')}",
            f"RAM {value(metrics['host_memory_used_bytes'], divisor=gib)} / {value(metrics['host_memory_total_bytes'], divisor=gib, suffix=' GiB')}",
        ]
        for gpu in sample["gpus"]:
            data = gpu["metrics"]
            pieces.append(
                f"GPU{gpu['hip_index']} {value(data['busy_percent'], suffix='%')}"
                f" VRAM {value(data['vram_used_bytes'], divisor=gib)} / {value(data['vram_total_bytes'], divisor=gib, suffix=' GiB')}"
                f" GTT {value(data['gtt_used_bytes'], divisor=gib)} / {value(data['gtt_total_bytes'], divisor=gib, suffix=' GiB')}"
                f" {value(data['temperature_celsius'], suffix='℃')} SoC {value(data['soc_power_watts'], suffix='W')}"
            )
        if not sample["gpus"]:
            pieces.append("GPU N/A（设备映射或计数器不可用）")
        return "｜".join(pieces)

    def poll(self) -> bool:
        if self._started is None or self._summary is not None:
            return False
        now = self.monotonic()
        if now < self._next_at:
            return False
        self._sample(now)
        return True

    def _aggregate(self, samples: list[dict]) -> dict:
        result = {
            "sample_count": len(samples),
            "metrics": {key: statistics([sample["metrics"].get(key) for sample in samples]) for key in HOST_PROCESS_UNITS},
            "gpus": [],
        }
        for device in self._gpu_devices:
            metadata = device["metadata"]
            readings = [gpu["metrics"] for sample in samples for gpu in sample["gpus"] if gpu["id"] == metadata["id"]]
            result["gpus"].append({
                **metadata,
                "metrics": {key: statistics([reading.get(key) for reading in readings]) for key in GPU_UNITS},
            })
        return result

    def _print_summary(self, summary: dict) -> None:
        self._print(f"资源汇总｜{summary['sample_count']} 次采样｜监测 {summary['elapsed_seconds']:.1f}s｜以下为最小 / 最大 / 平均（有效采样值）")
        def line(label: str, data: dict, divisor: float = 1, suffix: str = "") -> str:
            return f"{label} " + " / ".join(self._value(data[key], divisor=divisor, suffix=suffix) for key in ("min", "max", "mean")) + f"（n={data['count']}）"
        metrics = summary["metrics"]
        gib = 1024 ** 3
        self._print("｜".join((line("CPU 进程", metrics["process_cpu_percent"], suffix="%"), line("CPU 主机", metrics["host_cpu_percent"], suffix="%"))))
        self._print("｜".join((line("RSS", metrics["process_rss_bytes"], gib, " GiB"), line("RAM 已用", metrics["host_memory_used_bytes"], gib, " GiB"))))
        for gpu in summary["gpus"]:
            data = gpu["metrics"]
            self._print(f"GPU{gpu['hip_index']} ({gpu['id']})｜" + "｜".join((line("忙碌", data["busy_percent"], suffix="%"), line("VRAM", data["vram_used_bytes"], gib, " GiB"), line("GTT", data["gtt_used_bytes"], gib, " GiB"))))
            self._print("｜".join((line("温度", data["temperature_celsius"], suffix="℃"), line("SoC 功耗", data["soc_power_watts"], suffix="W"))))
        if summary["errors"]:
            self._print(f"资源采集说明：{len(summary['errors'])} 类读数不可用或采集错误，详见 resources.json；缺失值记为 null。")
        self._print("统计口径：GPU 为整设备占用；统一内存的 RSS、VRAM、GTT 不相加；SoC 功耗包含 APU CPU。")

    def finish(self) -> dict:
        if self._summary is not None:
            return self._summary
        if self._started is None:
            raise RuntimeError("resource monitor has not started")
        now = self.monotonic()
        if self._last_at is None or now > self._last_at:
            self._sample(now)
        if self._jsonl:
            try:
                self._jsonl.close()
            except OSError as error:
                self._error(f"cannot close resource sample log: {type(error).__name__}")
            self._jsonl = None
        metadata = {
            "pid": self._pid,
            "interval_seconds": self.interval,
            "requested_hip_visible_devices": self.devices,
            "units": {**HOST_PROCESS_UNITS, "gpu": GPU_UNITS},
            "scope": {"host": "whole host", "process": "worker and observed descendant processes", "gpu": "whole selected device, including other processes"},
            "limitations": list(dict.fromkeys(self.limitations)),
            "gpu_devices": [device["metadata"] for device in self._gpu_devices],
        }
        summary = {
            "schema_version": 1,
            "started_at": self._started_at,
            "finished_at": self.utc_now(),
            "elapsed_seconds": max(0.0, now - self._started),
            **self._aggregate(self.samples),
            "by_phase": {phase: self._aggregate([sample for sample in self.samples if sample["phase"] == phase]) for phase in dict.fromkeys(sample["phase"] for sample in self.samples)},
            "metadata": metadata,
            "errors": [{"message": message, "count": count} for message, count in self.errors.items()],
            "artifacts": {"json": str(self.directory / "resources.json"), "jsonl": str(self.directory / "resources.jsonl")},
        }
        self._print_summary(summary)
        summary["errors"] = [{"message": message, "count": count} for message, count in self.errors.items()]
        summary = _clean_json(summary)
        document = {"schema_version": 1, "metadata": metadata, "samples": self.samples, "summary": summary}
        try:
            (self.directory / "resources.json").write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        except OSError as error:
            self._error(f"cannot write resource report: {type(error).__name__}")
            summary["errors"] = [{"message": message, "count": count} for message, count in self.errors.items()]
        self._summary = summary
        return summary
