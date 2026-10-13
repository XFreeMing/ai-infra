# /// script
# requires-python = ">=3.12"
# dependencies = ["modelscope==1.40.2"]
# ///
"""手动执行：uv run scripts/download_deepseek_v41_flash_nvfp4.py。"""
from pathlib import Path

# 用户配置：仅此脚本下载模型；init/启动命令不会调用本脚本。
MODEL_ID = 'nv-community/DeepSeek-V4.1-Flash-NVFP4'
MODEL_REVISION = '962b2d0e8b06e475b77ab32e694b5df51f7949fb'
LOCAL_DIR = Path('/data/models/DeepSeek-V4.1-Flash-NVFP4')

import argparse
from datetime import datetime, timezone
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ai_infra.cli import WorkflowError
from ai_infra.native_model import profile, verify_download


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='手动从 ModelScope 下载固定 revision，并核对所有文件 SHA256。')
    parser.add_argument('--dry-run', action='store_true', help='只显示配置；不导入 SDK、不访问网络。')
    parser.add_argument('--verify-only', action='store_true', help='只校验已有文件，不访问网络。')
    parser.add_argument('--local-dir', type=Path, default=LOCAL_DIR)
    args = parser.parse_args(argv)
    expected = profile(ROOT)
    try:
        if (MODEL_ID, MODEL_REVISION) != (expected['model_id'], expected['revision']):
            raise WorkflowError('MODEL_ID/revision 已更改；请同时更新 NATIVE_MODEL.json 的文件身份。')
        print(json.dumps({'model_id': MODEL_ID, 'revision': MODEL_REVISION,
                          'local_dir': str(args.local_dir), 'weight_bytes': expected['weight_bytes'],
                          'download_requested': not (args.dry_run or args.verify_only)}, indent=2))
        if args.dry_run:
            return 0
        if not args.verify_only:
            existing = args.local_dir.resolve()
            while not existing.exists():
                existing = existing.parent
            remaining = sum(item['size_bytes'] for item in expected['files']
                            if not (args.local_dir / item['path']).is_file() or
                            (args.local_dir / item['path']).stat().st_size != item['size_bytes'])
            if shutil.disk_usage(existing).free < remaining + 5 * 1024**3:
                raise WorkflowError(f'磁盘不足：缺失文件约 {remaining / 1024**3:.1f} GiB，另需下载缓存空间。')
            from modelscope import snapshot_download
            snapshot_download(model_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(args.local_dir))
        print('逐文件核对 SHA256（约 491 GiB 权重，需要完整读盘）……', flush=True)
        receipt = verify_download(args.local_dir.resolve(), expected)
        receipt['verified_at'] = datetime.now(timezone.utc).isoformat()
        (args.local_dir / 'download-identity.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('下载身份校验完成。')
        return 0
    except (WorkflowError, OSError, ValueError) as error:
        print(f'下载/校验未完成：{error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
