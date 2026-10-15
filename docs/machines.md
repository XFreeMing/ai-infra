# AI 100 / 300 / 700 使用说明

最后更新：2026-10-15。机器编号是配置选择，不是性能分数。

300 新增 Ideogram4/IndexTTS2 运行入口及独立 ModelScope 下载，默认模型根目录改为 `/home/ops396/models`；700 下载根目录为 `/data/models`。详见 [生图和配音命令](media.md)。2026-10-15 的设备访问及 HIP 小探针已通过；下面的 2026-10-14 权限记录是历史观测。

## 机器清单

以下来自用户提供的机器盘点；`doctor` 的输出才是运行时观测，可用内存、网络连接与驱动版本可能变化。

| 配置 | AI 100 mini | AI 300 AMD395 | AI 700 PGX GB10 |
| --- | --- | --- | --- |
| 整机 | 联想 mini，主板 BYNUCA8PAB V1.0 | Lenovo Baiying NUC AI Max+ 395 | Lenovo ThinkStation PGX，30KM0003CH |
| CPU | Ryzen 7 8745HS，8C/16T，基础约 3.8 GHz | Ryzen AI Max+ 395，16C/32T，最高约 5.19 GHz，L3 64 MB，AVX-512/AMD-V | ARM64 20C/20T：10×Cortex-X925 约 4.0 GHz + 10×Cortex-A725 约 2.86 GHz |
| GPU | Radeon 780M iGPU，无独显 | Radeon 8060S iGPU，amdgpu 已加载；gfx1151 | NVIDIA GB10，驱动 580.95.05，CUDA 13.0；sm_121a |
| 内存 | 32 GB DDR5-5600，2×16 GB；盘点时可用约 20.6 GB | 128 GB，系统可见 124 GiB，盘点时可用约 119 GiB；8 GiB swap 未使用 | 128 GB，系统可用总量 119 GiB；15 GiB swap 未使用 |
| GPU 内存描述 | 共享系统内存 | UMA；512 MiB 固定显存、约 96 GiB GTT 窗口 | UMA，不显示独立显存 |
| NPU | 未纳入盘点 | AMD XDNA，amdxdna 已加载 | 未纳入盘点 |
| 磁盘 | Lexar NQ7A1 1TB，系统约 953 GiB | Crucial P310 2TB，约 1.8 TiB | ESL04TBTLCZ-27J4-TYN 4TB，约 3.6 TiB |
| 网络 | 未纳入盘点 | RTL8125 2.5GbE 未插线；MediaTek Wi-Fi 已连接 | 4×ConnectX-7、RTL8127、MT7925 Wi-Fi 已连接 |
| OS | Windows 11 Pro 64 位，Build 26200 | Ubuntu 24.04.4，Linux 7.0.0，x86_64 | Ubuntu 24.04.3，Linux 6.11 NVIDIA，AArch64 |
| BIOS | AMI 1.06，2025-07-07 | AMI AMSTXH04.012.20260108 | S0QKT09A，2025-10-14 |
| 其他 | 1920×1080，无电池 | 本流程不使用 XDNA NPU | 单路，每核单线程 |

UMA、GTT 窗口和物理内存不能相加作为模型容量。应记录实际可分配 GPU 内存、权重驻留、KV cache、workspace 与系统余量。

## 统一命令

`uv run 100|300|700` 表示选择一个数字，并非直接输入竖线。

```sh
uv run 300 --help
uv run 300 doctor
uv run 300 init                 # 默认 ds4 + llama
uv run 300 init llama           # 只构建 llama.cpp HIP
uv run 300 init --engines llama # 等价写法
uv run 300 init all             # ds4 + llama；只选择本机已实现配方
uv run 300 init ds4 llama --dry-run
uv run 300 init llama --check-only
uv run 300 llama --help
uv run 300 llama 2222 --model /data/models/model.gguf --context 4096
uv run 300 ds4 --model /data/models/model.gguf --devices 0
uv run 300 llama --model /data/models/model.gguf -- --reasoning off
uv run 300 new study rocm-first-smoke
uv run 300 run smoke
```

`doctor` 是盘点，退出 0 不代表 GPU 兼容性通过。`init --dry-run` 只写计划；`--check-only` 只盘点，退出 2 表示尚未准备。`run` 仍要求干净且已提交的快照；100 的 Windows 支持准备和服务启动，现有实验记录器的 `run` 仍限 Linux/macOS。

100/300 的初始化不依赖 R0 历史 DeepSeek 巨型模型。可以用 `init ... --model <本地GGUF>` 冻结准备时的模型路径及 SHA256，随后服务默认使用它；未指定时服务默认 `models/model.gguf`，建议启动时显式传 `--model`。`--models-dir` 用于盘点候选文件，不自动选择、不下载。100 默认盘点项目 `models/`，300 默认 `/data/models`。SHA256 只证明产物身份；格式头部检查不证明模型架构能被引擎加载。

100/300 服务支持 `--dry-run`，未准备或未下载时也可查看计划；`--server-help` 要求已构建服务，不加载模型。`--preparation-dir artifacts/init/<编号>/<run-id>` 可固定准备，但仍校验当前引擎源码 commit。700 保留各 `serve_*.py` 顶部的模型、context、端口和路径设置，具体参数见 README 的 CUDA 章节。

## 300 ROCm 准备

```sh
uv run 300 init ds4 llama --rocm-home /opt/rocm --jobs 4 --build-timeout 900
# 本地子模块及全部构建依赖已齐备时才使用离线模式
uv run 300 init llama --offline
```

ROCm 根目录按 `--rocm-home`、`ROCM_PATH`、`HIP_PATH`、PATH 中的 hipcc、`/opt/rocm` 顺序选择。准备过程先确认 Linux x86_64、hipcc、`/dev/kfd` 读写权限、rocminfo 中的 gfx1151，再编译执行一个 HIP kernel（预期 42），然后在独立 clone 构建所选引擎。ds4 使用 `make strix-halo ROCM_ARCH=gfx1151`，服务显式传 `--rocm`；llama.cpp 使用 `GGML_HIP=ON`、`GPU_TARGETS=gfx1151`，并检查 `llama-server --list-devices` 中的 ROCm 设备。服务前再次执行已记录 SHA256 的 HIP probe。

100/300 默认墙钟预算 900 秒，包括 300 等待初始化锁的时间；`--build-timeout 0` 可显式取消总预算。HIP 编译/执行及服务帮助另有短步骤预算。失败、超时、中断保留目录、状态与日志，退出 2/130；新建准备不会改动已有服务使用的 clone。100/300 暂不支持 `--resume`，700 的原有恢复流程保留。300 初始化在本 workspace 内串行，性能实验仍需单独约束并发。

本机实际只完成环境盘点：`/opt/rocm` 指向 `/opt/rocm-7.2.0`，hipconfig 输出 `7.2.26015-fc0010cf6a`；`rocminfo` 退出 1，提示 `/dev/kfd` Permission denied、用户不在 render 组。先让管理员核对 `render` / `video` 组和设备 ACL，重新登录后再运行 `uv run 300 doctor`。本轮未更改系统权限或安装驱动，也没有 GPU 推理结果。

AMD 的版本化 [ROCm 7.2.1 Ryzen Linux 支持矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)列出 Ryzen AI Max+ 395 / gfx1151。此矩阵与本机 ROCm 7.2.0、Linux 7.0.0-38 是不同版本信息，不能据此宣称本机组合经过官方或本项目验证。不设置 `HSA_OVERRIDE_GFX_VERSION` 来伪装设备架构。

## 100 Windows mini 准备

安装 Git、uv/Python 3.12+、Visual Studio 2022 的 Desktop development with C++ 与 CMake；从 C++ Developer PowerShell 进入项目。

```powershell
uv run 100 doctor
uv run 100 init llama --jobs 4 --build-timeout 900
uv run 100 llama --model C:/models/small-model.gguf --context 2048
```

本轮 100 明确选择 CPU，关闭 CUDA、HIP 和 Vulkan；启动 `--gpu-layers 0`。Windows 多配置构建使用 `--config Release`，记录实际找到的 `.exe` 和 DLL。准备命令继承 Developer PowerShell 的编译器环境。当前 Linux 机器只验证 100 的命令路由与计划，没有 Windows 编译、780M GPU 或性能实测。

## 700 与原生命令

```sh
uv run 700 init ds4 llama
uv run 700 init sglang vllm freetoken
uv run 700 init all --dry-run
uv run 700 freetoken 6666 --dry-run
uv run 700 sglang 3333
uv run 700 vllm 5555
```

700 保留现有 CUDA 构建、冻结 P0、resume、模型配置与服务实现。旧 `uv run init` / `uv run llama` 等别名也继续使用这套配置，不根据宿主机自动切到 AMD。新 700 命令在真正构建/启动前要求 Linux ARM64，计划可在其他机器查看。

300 对 SGLang/vLLM/FreeToken、100 对 ds4/原生 Python 框架尚无本机配方，显式请求时退出 2，不安装 CUDA wheel。不据此断言上游不支持 AMD；只是本仓库未实现这些框架在当前机器和固定源码版本上的准备与正确性流程。

## 来源与证据

构建源码优先读取干净子模块的实际 HEAD，未初始化时读取 parent index gitlink；不会自动追 main。`--update-sources` 是显式更新选项，会按原有流程更新所有干净子模块，随后冻结实际 commits；不能和 dry-run/check-only/offline/resume 混用。

本轮配方的源代码依据：

- ds4 commit `fc80bd695da76ee14bcb86820bf900a5cdfda806`：[Makefile 的 `strix-halo`、`ROCM_ARCH`、`ROCM_CFLAGS`](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/Makefile#L282)，[ds4_server.c 的 `parse_options` / `--rocm`](https://github.com/antirez/ds4/blob/fc80bd695da76ee14bcb86820bf900a5cdfda806/ds4_server.c#L14961)。这些是上游接口，本机执行结果另记。
- llama.cpp commit `091b6a7fc561b78da90a0ed2d524ec995458c657`：[docs/build.md HIP 构建及 GPU isolation](https://github.com/ggml-org/llama.cpp/blob/091b6a7fc561b78da90a0ed2d524ec995458c657/docs/build.md#L339)，[Windows CPU 构建要求](https://github.com/ggml-org/llama.cpp/blob/091b6a7fc561b78da90a0ed2d524ec995458c657/docs/build.md#L63)。
- 数字命令通过标准 `[project.scripts]` 注册：[uv 项目入口文档](https://docs.astral.sh/uv/concepts/projects/config/#command-line-interfaces)。

100/300 的准备写入 `artifacts/init/<编号>/<run-id>/manifest.json`，服务写入 `artifacts/servers/<编号>/<launch-id>/launch.json`。700 保留原来的目录，新增 `machine_id=700` / `backend=cuda`。旧清单缺少这两个字段时只按 700 解释。准备和启动选择校验机器、后端、源码、嵌套子模块和二进制/动态库 SHA256；启动记录包含有效 argv、环境覆盖和本地模型身份。实验 `run` 在 `artifacts/runs/<编号>/` 记录 machine ID，子进程得到 `AI_INFRA_MACHINE_ID`。

工具验证：`uv run --offline python -m unittest discover -s tests -v` 共 115 项通过；12 项实际 CLI 检查通过，包括数字入口、300 权限阻塞、700/旧入口计划。原始测试日志在本机 `artifacts/machine-tests.log`；命令、退出码、日志与 SHA256 清单在 `artifacts/machine-validation-20261014/summary.json`。这些是工具与环境盘点证据，不是 Windows/GB10/ROCm 模型推理结果。原有服务引用的旧 ModelKey 成员同时改为当前共享目录的四段式成员，保持已有模型身份与路径。

已理解：三个编号明确路由到各自后端；300 的 GPU 使用需要进程权限和 HIP kernel 证据。待确认：Windows 构建、当前内核/ROCm/gfx1151 实际兼容性、各模型的加载与输出质量。下一步：修复当前用户设备访问后，在独立干净快照执行 `300 init llama`，选本地小 GGUF 做 32-token 生成与正确性检查，再增加模型/context 与测量预算。
