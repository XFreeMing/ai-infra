# Mac 研究与 PGX GB10 实验流程

最后更新：2026-10-12

目标是把一个研究问题交接成可复现的实验，同时允许 Mac 继续阅读和提出新问题。首版采用手动 Git 同步；GPU 兼容性、后台任务管理和实验结果需要在实际 PGX 环境中验证。

## 1. Mac：从一手资料形成问题

优先选择一条官方发布、PR、Issue 或维护者讨论，创建研究记录：

```sh
python main.py new study prefix-cache-boundary
```

填写 `research/studies/prefix-cache-boundary/README.md`：原始链接、参与者、来源日期、源码提交、文件与符号、自己的理解、可检验假设。用 GitHub 的 `/blob/<commit>/<path>#L<line>` 固定源码引用；分支链接可能随上游改变。

让 AI 先定位入口、调用关系与相关测试，再解释机制。将原作者主张、AI 或自己的推断、已经完成的验证分别记录。结束时写明下一次阅读的入口。

## 2. Mac：建立实验定义并固定版本

```sh
python main.py new experiment prefix-cache-boundary
```

填写 `experiments/prefix-cache-boundary/README.md` 和 `experiment.json`。先定义假设、基线、单个主要变量、质量检查、指标、输入和模型版本、资源预算及停止条件，再选择运行命令。

每个实验可以有多次运行。修改参数或代码后创建新的运行记录，保留先前数据。为正式运行预留一次小规模试跑，用它检查输出是否正确并估算耗时。

确认所需源码版本：

```sh
git submodule status --recursive
git -C sources/vllm rev-parse HEAD
git -C sources/vllm status --short
git status --short
```

这里只用 vLLM 举例，使用其他框架时替换路径。主仓库的提交保存 submodule 的提交引用，不保存子仓库未提交的修改。修改上游源码后，先将子仓库提交推送至 PGX 能访问的仓库，再提交主仓库的新引用；使用 fork 时同步维护 `.gitmodules` 中可获取的地址。

将已审阅的研究和实验定义提交、推送后，记下主仓库的完整提交 SHA：

```sh
git rev-parse HEAD
```

如果调整源码版本，在研究卡中记录旧版本、新版本和调整原因。不要把“最新源码”作为实验版本。

## 3. PGX：独立 clone 并检出实验提交

每次长实验使用独立 clone 与依赖环境。不要在正在执行实验的目录中 `git pull`、切换提交、更新 submodule 或修改依赖。首次使用时配置好 PGX 的 GitHub 访问权限。

以下命令中的 SHA 必须换成上一步记录的完整提交值：

```sh
mkdir -p ~/ai-infra-runs
cd ~/ai-infra-runs
git clone --no-checkout git@github.com:XFreeMing/ai-infra.git prefix-cache-001
cd prefix-cache-001
git checkout --detach <MAIN_REPO_COMMIT_SHA>
git submodule update --init --recursive sources/vllm
python3 --version
python3 main.py doctor
```

如使用 uv，可将 `python3 main.py` 替换为 `uv run --offline python main.py`，前提是本机已有 Python 3.12+。

按实验需要初始化 `sources/sglang` 或 `sources/llama.cpp`；无需为每次运行下载所有框架。独立 clone 较占磁盘，优点是每次运行的 Git 和 submodule 工作目录清晰独立。模型权重与下载缓存可以使用明确配置的外部目录复用，但需要固定模型版本和输入，记录路径及校验值。

在快照目录的 `.venv` 中创建运行专用的 Python 环境，或使用独立容器，并依据**当前所固定提交**的官方构建文档安装框架。不同框架与版本的 GB10 支持和编译需求需要现场检查，不能从 Mac 上的检查结果推导。依赖准备必须在正式运行前完成。

## 4. PGX：确认真正执行的源码

固定 submodule 只是第一步。运行脚本可能导入其他环境里安装的同名包，或调用之前编译的可执行文件。运行前必须记录：

- 主仓库 SHA、所有参与实验的 submodule SHA、工作目录是否有修改。
- Python / 编译器 / CUDA / 驱动版本、硬件信息、构建命令及依赖版本。
- 实际导入模块的 `__file__` 路径；使用二进制时记录绝对路径、版本和构建信息。
- 实际加载代码与目标 submodule 的关系；若使用 wheel 或容器，记录构建来源和镜像 digest。
- 模型 revision、精度、输入数据版本、配置和完整启动命令。

例如，在安装好目标框架后，可以检查其实际加载位置：

```sh
python -c 'import vllm; print(vllm.__file__)'
```

这个命令只检查路径，不证明 GPU 可用，也不证明构建成功。路径不属于目标源码构建时，先解决环境问题，再进行性能测量。

## 5. PGX 执行，Mac 继续研究

先做短时试跑，再运行正式对照组。性能实验默认一次只运行一个主要任务，避免同机负载干扰。记录开始时间、最新进度、结束时间、退出码和终止原因。

实验配置完成、已提交且工作区干净后，执行：

```sh
python3 main.py run prefix-cache-boundary
```

执行器读取 `experiments/prefix-cache-boundary/experiment.json`，在主仓库根目录运行配置中的 argv 命令。默认写入 `artifacts/runs/` 下的唯一运行目录，包含运行清单和标准输出、错误日志。执行器不负责安装框架；真正的测量与质量检查由本实验脚本实现。配置字段见[实验约定](../experiments/README.md)。

长任务使用 PGX 上已经安装并检查过的后台运行工具，例如 `tmux`；主仓库的执行入口本身不等于远程调度或断连保护。测试断开连接后任务仍可观察，再启动长实验。

Mac 可以继续新研究、新配置或源码升级，并形成新的提交。新提交不会自动进入 PGX 已经固定的实验目录。需要修改正在验证的方案时，先保留当前结果，再建立新的运行。

## 6. 结果回流到 Mac

每次运行使用唯一编号，例如 `prefix-cache-20261012-001`。小型原始数据、运行清单、汇总表与报告保存在 `reports/<experiment>/<run-id>/`；大日志、trace、模型权重和缓存保存在 Git 以外的位置。仓库中记录大产物位置、校验值、大小与获取方法。不要将带凭据或私人数据的日志发布到仓库。

运行结束后，在独立的结果整理 clone 中创建分支，避免修改运行快照：

```sh
git clone git@github.com:XFreeMing/ai-infra.git ~/ai-infra-results
cd ~/ai-infra-results
git switch -c codex/results/prefix-cache-20261012-001
```

将需要保存的小型结果复制到对应报告目录，填写运行使用的原始主仓库 SHA；结果提交 SHA 与实验提交 SHA 是两个不同的标识。检查将要提交的内容后：

```sh
git add reports/prefix-cache-boundary/prefix-cache-20261012-001
git diff --cached --stat
git diff --cached
git commit -m "Record prefix cache experiment results"
git push -u origin codex/results/prefix-cache-20261012-001
```

Mac 在工作区干净时取回并检查：

```sh
git fetch origin
git diff main...origin/codex/results/prefix-cache-20261012-001 -- reports/
git switch main
git merge --no-ff origin/codex/results/prefix-cache-20261012-001
```

合并前确认数据对应的源码、环境、输入与配置，再用实验分析脚本生成图表。报告写明支持假设的证据、没有支持假设的证据、未测试范围和下一步。需要公开的大型证据应放到读者可访问的位置。

## 每周 5–8 小时的节奏

| 工作 | 建议时间 | 留下的产物 |
| --- | --- | --- |
| 官方资料、作者讨论与英文精读 | 2 小时 | 研究卡、英文复述 |
| 源码定位与机制分析 | 2 小时 | 固定版本调用路径、疑问 |
| 实验设计、启动与检查 | 1–2 小时 | 实验定义、试跑或正式运行 |
| 结果分析和整理 | 0–2 小时 | 图表、结论、后续问题 |

GB10 等待时间不计入主动学习时间。先保持一个主要研究问题和一个主要实验，连续完成两轮流程后再决定需要哪些自动化。

相关文档：[研究约定](../research/README.md)、[实验约定](../experiments/README.md)、[成果标准](../reports/README.md)、[源码来源](../sources/README.md)。
