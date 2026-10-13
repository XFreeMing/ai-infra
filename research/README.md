# 研究记录

最后更新：2026-10-13

围绕技术问题保存阅读过程，把一手资讯、作者讨论、固定版本源码和实验串起来。

## 从基础开始学习

想先动手，可用 [Mac 小模型实验](../experiments/tiny-inference/README.md)：约 61 MB 的英文故事模型已经在本机跑通，可以逐行调试。

[后端开发者的 AI Infra 学习路线](learning/README.md)从单条推理请求开始，逐步学习模型数据、attention/KV、GPU、量化与服务调度。当前先读[第 1 课：一次请求如何变成模型输出](learning/01-request-to-tokens.md)，再用一个循环练习检查理解；不要求先读完整源码调研。

## 已建立的专题

- [ds4 / DwarfStar：源码、性能、效果与横向调研](studies/ds4-inference/README.md)：静态调研已完成；兼容性、性能及任务效果尚待实际运行验证。
- [antirez / Salvatore Sanfilippo](people/antirez.md)：从具体提交追踪其专用推理、状态恢复和质量评测工作。

## 目录约定

- `studies/<slug>/README.md`：一次可持续推进的研究问题，由 `python main.py new study <slug>` 创建。
- `people/<github-handle>.md`：按需复制[人物模板](../templates/person.md)，记录公开技术工作及与研究问题的关系。
- `learning/`：按先修关系组织的源码导读、概念解释和练习；备课状态与学习者掌握情况分开记录。
- 专题汇总放在对应研究目录中，链接原始证据，避免复制多份事实。

研究卡采用[阅读模板](../templates/reading.md)。每条关键结论要能找到来源、源码或实验：

| 记录 | 必备依据 |
| --- | --- |
| 原文主张 | 原始 URL、作者、发布日期、引用位置 |
| 源码解释 | submodule 路径、完整 commit、文件与符号、永久链接 |
| AI 或自己的推断 | 推断过程、依赖证据、仍不确定的部分 |
| 验证结果 | 实验定义、运行编号、原始数据、环境与限制 |

PR 状态分别记录提议、合并、发布；“已合并”不自动等于所用版本包含该功能。关注维护者时保留角色依据和核对时间。对英文讨论，先写自己的短复述，再使用 AI 校对。

每周选择一个主问题，优先深读最相关的一条讨论与一条源码路径。结束时记录下次入口，让 Mac 可以在 GB10 运行期间继续研究。

来源入口见 [sources/README.md](../sources/README.md)，双机交接见 [docs/workflow.md](../docs/workflow.md)。
