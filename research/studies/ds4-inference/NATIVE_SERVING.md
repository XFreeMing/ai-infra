# DeepSeek V4.1 NVFP4 原生框架入口

日期：2026-10-13。证据类型：远端文件元数据观测、固定源码阅读和脚本测试；未进行此模型的 GB10 推理。

用户指定 [ModelScope 仓库](https://www.modelscope.cn/models/nv-community/DeepSeek-V4.1-Flash-NVFP4)。以 [固定 revision 文件 API](https://modelscope.cn/api/v1/models/nv-community/DeepSeek-V4.1-Flash-NVFP4/repo/files?Revision=962b2d0e8b06e475b77ab32e694b5df51f7949fb&Recursive=True) 返回的 blob 文件清单生成 [NATIVE_MODEL.json](NATIVE_MODEL.json)。该 revision 的清单与读取时 master 一致；config.json 的 SHA256 为 `6f53b8f185a587f8e4b9ebd859c158ba92459cc119d11243e0c8291a02a0e265`。只读取元数据/config/model card，未下载权重。

元数据观测：`model_type=deepseek_v41`、`architectures=[DeepseekV41ForCausalLM]`，text_config 的 context 上限 1,048,576。量化配置同时包含 `quant_method=fp8`、`quant_algo=MIXED_PRECISION`、`moe_quant_algo=NVFP4`、group size 16；不将普通 FP8 或全模型 NVFP4 参数强加到混合精度 checkpoint。48 个 safetensors 文件合计 527,293,384,576 bytes（491.1 GiB）。

上游报告：仓库模型卡说明 routed experts 使用 NVFP4 W4A4，其他模块保留源精度，并给出四张 GB300 的 SGLang/vLLM 部署例子。SGLang 报告版本为 `da64c5cbb8cf6bfd39be19da43573fdfd484c43a`，vLLM 例子使用 `deepseekv41-flash-0909` 镜像标签。它们均不同于本工作台 gitlink，不能推定本机已兼容。

固定源码依据：

- SGLang `f006fb49ecd93322802ddebff80821f8e6296893`：[python/pyproject.toml](https://github.com/sgl-project/sglang/blob/f006fb49ecd93322802ddebff80821f8e6296893/python/pyproject.toml) 的 build/runtime torch 2.14.1；[kernels/aot/pyproject.toml](https://github.com/sgl-project/sglang/blob/f006fb49ecd93322802ddebff80821f8e6296893/python/sglang/kernels/aot/pyproject.toml) 的 sglang-kernel 0.4.9/source build；[CMakeLists.txt](https://github.com/sgl-project/sglang/blob/f006fb49ecd93322802ddebff80821f8e6296893/python/sglang/kernels/aot/CMakeLists.txt) 的 ARM64/CUDA 13 sm_121a gencode；[ModelConfig](https://github.com/sgl-project/sglang/blob/f006fb49ecd93322802ddebff80821f8e6296893/python/sglang/srt/configs/model_config.py#L684) 的 hybrid NVFP4 识别及 `nvfp4_moe_meta`。
- vLLM `2534a96fcf165202f5358335f8b0705175159fd0`：[requirements/build/cuda.txt](https://github.com/vllm-project/vllm/blob/2534a96fcf165202f5358335f8b0705175159fd0/requirements/build/cuda.txt) 的 torch 2.13.0/source build 依赖；[setup.py](https://github.com/vllm-project/vllm/blob/2534a96fcf165202f5358335f8b0705175159fd0/setup.py) 的 CUDA/native build；[DeepseekV4FP8Config（V4.1 模块）](https://github.com/vllm-project/vllm/blob/2534a96fcf165202f5358335f8b0705175159fd0/vllm/models/deepseek_v41/quant_config.py#L111) 的 `_is_nvfp4_expert_layer` / `_get_nvfp4_config` 专用处理；[envs.py](https://github.com/vllm-project/vllm/blob/2534a96fcf165202f5358335f8b0705175159fd0/vllm/envs.py#L667) 的预编译开关读取，包含 wheel-location 覆盖，因此准备脚本也清空该变量。
- ModelScope SDK 1.40.2：[snapshot_download](https://github.com/modelscope/modelscope/blob/v1.40.2/modelscope/hub/snapshot_download.py#L60) 接收 `model_id`、`revision`、`local_dir`，下载入口独立于 init。脚本固定 revision 后逐文件核对发布元数据的 SHA256。

AI 推断：当前单 GB10 的约 119 GiB 可见总内存小于该 checkpoint 的原始权重文件大小；依照这两个启动入口的完整模型驻留方式，降低 context 或 CPU offload 不能满足容量门槛。本次没有实现 SSD expert streaming、多机切分或二次量化。入口针对后续具备足够容量的运行环境，模型/TP/可见设备均可在脚本顶部修改。

## 2026-10-14 初始化失败排查（本机观测）

环境：Linux ARM64、NVIDIA GB10、驱动 595.91.07、本机 CUDA toolkit 13.4.92、Python 3.12.3、uv 0.12.15；父提交 `3fa0bc06b988b02f219e1dae5c9bf83843738325`，修复中的 tooling 为未提交修改，其 SHA256 写入恢复 manifest。这里是构建恢复检查，不是模型推理或性能实验。

- SGLang run `20261013T090231Z-native-cuda-tamtzok9`：原 `07-sglang-build-deps.log` 显示下载 `nvidia-cudnn-cu13==9.24.0.43` 时等待 distribution cache lock 超过 uv 的 300 秒默认限制，退出 2。另一套 vLLM 初始化同时下载共享依赖。增加单 workspace native init 锁，将 uv 单次锁等待设为 3600 秒，仅对明确的缓存锁超时重试，默认总预算仍不限时。原目录续跑后的 `11-sglang-build-deps.log` 安装成功，耗时约 5.3 秒；没有下载模型。
- vLLM run `20261013T090239Z-native-cuda-7r7lp7fq`：原 `08-vllm-framework-source-build.log:35652` 在 DeepGEMM 编译中缺少 `elfutils/libdwfl.h`。系统已安装 libdw/libelf runtime 0.194-4，缺少对应 dev headers。`init` 现在先读取 runtime 版本，下载同版本 `libdw-dev`、`libelf-dev` 的 arm64 Debian 包，解压到本地 sysroot，将 include 路径通过 CPATH 传给 C++ 编译器；不修改系统安装。包 SHA256 和头文件检查命令保存到 manifest。独立修复检查 `artifacts/build-deps/elfutils-recovery/09-deepgemm-extension-build.log` 已成功编译 `_C_extension.abi3.so`，耗时约 58.5 秒。

源码证据：固定 vLLM 的 [tools/build_deepgemm_C.py](https://github.com/vllm-project/vllm/blob/2534a96fcf165202f5358335f8b0705175159fd0/tools/build_deepgemm_C.py) 调用 `.deps/deepgemm-src/setup.py build_ext --inplace`。实际 `.deps/deepgemm-src` 提交为 `1e1842a833699298f7afc02eefb2ed168fff6938`；其 nested `third-party/deep_jit/include/deep_jit/utils/exception.hpp:3` 引用该头文件，`:41` 在运行时 dlopen `libdw.so.1`。SGLang 固定 [CMakeLists.txt](https://github.com/sgl-project/sglang/blob/f006fb49ecd93322802ddebff80821f8e6296893/python/sglang/kernels/aot/CMakeLists.txt#L161) 使用 `SGL_KERNEL_COMPILE_THREADS`（默认 32），因此后续准备通过 scikit-build 配置明确设为 1；普通 `NVCC_THREADS` 对此选项不生效。

恢复会保留旧日志和 `manifest-before-resume-*.json`，复用失败环境的固定源码与 venv，拒绝恢复 prepared 环境。后续 vLLM 初始化用外部 setuptools 自定义 build_ext 配置保留 CMake/object cache；SGLang 使用独立的 scikit-build cache。完整框架安装及 CUDA/import probe 的结果以各 run 的最新 manifest 为准；DeepGEMM 单独编译成功不能替代完整初始化检查。

SGLang 的 `12-sglang-kernel-source-build.log` 已成功安装本地 `sglang-kernel==0.4.9`，耗时 984.4 秒。下一步 `13-sglang-framework-source-build.log` 暴露新的准备脚本问题：全局 `--no-build-isolation` 同时影响第三方 `cuda-tile==1.6.0rc5`，导致其 `wheel_stub.buildapi` 不可导入。改为 `--no-build-isolation-package <目标框架>`，让第三方包按自身 build-system 安装构建依赖。独立 venv 验证见 `artifacts/build-deps/cuda-tile-isolation-probe/01-cuda-tile-build-isolation.log`，正常构建并安装成功，耗时 2.2 秒；此修复也适用于 vLLM/FreeToken 后续初始化。

缓存配置验证另发现：setuptools editable 重初始化会按自定义 class name 读取选项，setuptools-rust 将 vLLM 的 `cmake_build_ext` 包装成 `build_ext_rust_extension`。第一版 `[build_ext]` 配置在实际恢复中未生效，当前 vLLM run 仍在临时目录编译；后续配置改用这两个 class section，避免影响第三方普通 build_ext。在相同 vLLM venv 的独立 Distribution 对象上验证，两个自定义 class 均在重初始化后保留指定 build_temp，普通 build_ext 仍使用临时目录。完整运行的实际构建目录以 CMakeCache/日志为准。

两个环境的 PyTorch CUDA 小探针均通过：SGLang 为 `2.14.1+cu130`，vLLM 为 `2.13.0+cu130`，GPU 为 NVIDIA GB10、capability `[12, 1]`，`torch.arange(8, device='cuda').sum()` 均得到 28 并完成 CUDA synchronize。日志位于 `artifacts/build-deps/cuda-recovery-probe/`。这只验证 CUDA Tensor 路径，没有加载框架服务或模型。当前构建依赖阶段尚未安装 NumPy，import 会警告缺少 NumPy；后续框架依赖安装将包含它，这不是本次失败原因。

SGLang 自有 AOT RMSNorm CUDA kernel 另已实际执行并通过：`torch.ops.sgl_kernel.rmsnorm.default`，4×128 BF16 输入、固定 seed 7、eps=1e-6、PDL 关闭，与 FP32 Torch 参考计算转回 BF16 比较（rtol=0.02、atol=0.01），最大绝对误差 0.0。Torch 为 `2.14.1+cu130`，sglang-kernel 为 0.4.9，实际 import 来自该失败目录的 venv。命令、预算 60 秒、环境和原始输出见 `artifacts/build-deps/sglang-kernel-cuda-probe/00-native-rmsnorm-cuda.log` 及 manifest；没有加载模型或测量吞吐。API 来源为固定 [elementwise.py::_rmsnorm_internal](https://github.com/sgl-project/sglang/blob/f006fb49ecd93322802ddebff80821f8e6296893/python/sglang/kernels/aot/python/sgl_kernel/elementwise.py#L16)。

已理解：V4.1 NVFP4 和 P0 的 Flash 0731 Q2 GGUF 是不同 checkpoint/格式，必须分开记录；两框架有独立依赖和实际执行路径，脚本只在用户显式下载入口下载权重。原初始化的两个失败分别来自缓存锁等待和缺少开发头文件。

未解决：完整框架初始化及 CUDA/import probe 仍待恢复流程完成；sm_121 NVFP4 的全部模型 kernel 和实际加载/输出正确性尚未验证，模型卡的 GB300 测试不能替代 GB10 观测。

下一步：完成本次串行恢复，检查最新 manifest 状态、CUDA/import provenance 与扩展 SHA256；按实际硬件容量决定模型执行环境，手动下载并校验权重，最后串行执行 32-token smoke，保存失败与实际 GPU kernel 日志。
