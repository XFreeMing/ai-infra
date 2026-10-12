# 记录流程检查

这个实验只检查本地执行、版本记录、日志和结果清单，不运行模型，也不测量 GPU 性能。

先提交项目变更，然后在项目根目录执行：

```sh
python main.py run smoke
```

配置中的 `{python}` 使用当前 CLI 的 Python 解释器。结果写入独立的
`artifacts/runs/<run_id>/` 目录，包括 `manifest.json`、`stdout.log` 和
`stderr.log` 和 `result.json`。脚本从 `AI_INFRA_RUN_DIR` 获取本次结果目录，
从 `AI_INFRA_RUN_ID` 获取运行编号。成功时退出码为 0，清单中的 `status` 为 `succeeded`；标准输出包含
输入文本的 SHA-256 和 `success: true`。

这证明记录流程可用，不能作为模型质量、速度、GPU 可用性或跨设备一致性的证据。
