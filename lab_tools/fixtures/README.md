# IndexTTS2 hello 参考音频

`indextts2-voice01.wav` 是上游公开演示样本的原始文件，用于本地功能验证。`hello` 不联网获取参考音频。

- 仓库：https://github.com/index-tts/index-tts
- 固定提交：`830f6f8f94a51fea23ab1d639027a86200075a4e`
- 上游路径：`examples/voice_01.wav`
- 原文件：https://media.githubusercontent.com/media/index-tts/index-tts/830f6f8f94a51fea23ab1d639027a86200075a4e/examples/voice_01.wav
- 大小：478,050 bytes
- SHA256：`e33e6ee0107a1dd58e1d66dd90c13df3d55a8683047cc3d7ea206dad84ed3fc8`
- 仓库许可证副本：[INDEX-TTS-LICENSE](INDEX-TTS-LICENSE)，bilibili Model Use License Agreement。上游未提供单独的音频许可证。

原文件为 IEEE float WAV，保留原始字节；使用 prepared 环境中的 soundfile 读取。可通过 `uv run 300 hello Indextts2 --reference /path/to/your.wav` 使用自己的参考音频。
