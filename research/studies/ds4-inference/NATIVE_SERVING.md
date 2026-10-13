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

已理解：V4.1 NVFP4 和 P0 的 Flash 0731 Q2 GGUF 是不同 checkpoint/格式，必须分开记录；两框架有独立依赖和实际执行路径，脚本只在用户显式下载入口下载权重。

未解决：ARM64 依赖 wheel、源码编译、sm_121 NVFP4 的所有模型 kernel 和实际加载/输出正确性尚未验证；模型卡的 GB300 测试不能替代 GB10 观测。

下一步：先审阅 `uv run init --engines sglang vllm --dry-run`，再按实际硬件容量决定执行环境；执行 init 后检查 CUDA/import provenance，手动下载并校验权重，最后串行执行 32-token smoke，保存失败与实际 GPU kernel 日志。
