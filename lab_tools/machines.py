"""Machine-specific uv entry points. Hardware inventory is user supplied, not a probe."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from lab_tools.cli import WorkflowError
from lab_tools.media_models import APPS as MEDIA_APPS, models_root

PROFILES = {
    '100': {'name': 'mini', 'system': 'Windows', 'architecture': 'AMD64', 'backend': 'cpu',
            'cpu': 'AMD Ryzen 7 8745HS (8C/16T)', 'gpu': 'Radeon 780M',
            'memory_gb': 32, 'disk': 'Lexar NQ7A1 1TB', 'engines': ['llama'],
            'reported_os': 'Windows 11 Pro Build 26200', 'bios': 'AMI 1.06 (2025-07-07)'},
    '300': {'name': 'AMD395', 'system': 'Linux', 'architecture': 'x86_64', 'backend': 'rocm',
            'cpu': 'AMD Ryzen AI Max+ 395 (16C/32T)', 'gpu': 'Radeon 8060S', 'gpu_arch': 'gfx1151',
            'memory_gb': 128, 'disk': 'Crucial P310 2TB', 'engines': ['ds4', 'llama', 'ideogram4', 'indextts2'],
            'reported_os': 'Ubuntu 24.04.4 / Linux 7.0.0', 'bios': 'AMSTXH04.012.20260108',
            'npu': 'AMD XDNA (本流程不使用)', 'reported_gtt_gib': 96},
    '700': {'name': 'PGX GB10', 'system': 'Linux', 'architecture': 'aarch64', 'backend': 'cuda',
            'cpu': '10 Cortex-X925 + 10 Cortex-A725 (20C/20T)', 'gpu': 'NVIDIA GB10',
            'gpu_arch': 'sm_121a', 'memory_gb': 128, 'disk': 'ESL04TBTLCZ-27J4-TYN 4TB',
            'engines': ['ds4', 'llama', 'sglang', 'vllm', 'freetoken'],
            'reported_os': 'Ubuntu 24.04.3 / Linux 6.11 NVIDIA', 'bios': 'S0QKT09A (2025-10-14)',
            'reported_driver': '580.95.05', 'reported_cuda': '13.0'},
}
ALL_ENGINES = (*PROFILES['700']['engines'], *MEDIA_APPS)


def engine_name(value: str) -> str:
    return value.casefold()


def host_matches(machine: str) -> bool:
    spec = PROFILES[machine]
    aliases = {'AMD64': {'amd64', 'x86_64'}, 'x86_64': {'amd64', 'x86_64'},
               'aarch64': {'aarch64', 'arm64'}}
    return platform.system() == spec['system'] and platform.machine().lower() in aliases[spec['architecture']]


def require_host(machine: str) -> None:
    if not host_matches(machine):
        spec = PROFILES[machine]
        raise WorkflowError(f"AI {machine} 需要 {spec['system']} {spec['architecture']}；"
                            f"实际为 {platform.system()} {platform.machine()}。可用 --dry-run 查看计划。")


def rocm_home(explicit: Path | None = None) -> Path:
    configured = explicit or os.environ.get('ROCM_PATH') or os.environ.get('HIP_PATH')
    if configured:
        return Path(configured).resolve()
    hipcc = shutil.which('hipcc')
    return Path(hipcc).resolve().parents[1] if hipcc else Path('/opt/rocm').resolve()


def probe(argv: list[str]) -> dict:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=10, check=False)
        return {'argv': argv, 'exit_code': result.returncode,
                'output': result.stdout.strip(), 'error': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'argv': argv, 'exit_code': None, 'error': str(error)}


def rocm_report(home: Path | None = None) -> dict:
    home = home or rocm_home()
    info = home / 'bin/rocminfo'
    return {'home': str(home), 'hipcc_available': (home / 'bin/hipcc').is_file(),
            'hip_version': probe([str(home / 'bin/hipconfig'), '--version']),
            'kfd': {'exists': Path('/dev/kfd').exists(),
                    'read_write': os.access('/dev/kfd', os.R_OK | os.W_OK)},
            'rocminfo': probe([str(info)])}


def require_rocm(report: dict) -> None:
    if not report['hipcc_available']:
        raise WorkflowError('缺少 hipcc；安装 ROCm toolkit 或设置 --rocm-home。')
    if not report['kfd']['read_write']:
        raise WorkflowError('当前用户无法读写 /dev/kfd；核对 render/video 组与设备权限，重新登录后运行 300 doctor。')
    info = report['rocminfo']
    if info['exit_code'] != 0 or 'gfx1151' not in info.get('output', ''):
        raise WorkflowError('rocminfo 未确认可访问的 gfx1151；查看 doctor 原始输出。')


def add_machine_init_arguments(parser: argparse.ArgumentParser, root: Path, machine: str) -> None:
    from lab_tools.prepare import optional_timeout, positive_int
    parser.add_argument('--engines', nargs='+', type=engine_name, choices=(*ALL_ENGINES, 'all'),
                        help=f"本机已实现：{', '.join(PROFILES[machine]['engines'])}；all 选择这些引擎。")
    parser.add_argument('--dry-run', action='store_true', help='只写计划，不下载、编译或运行 GPU。')
    parser.add_argument('--models-dir', type=Path, default=root / 'models' if machine == '100' else models_root(machine),
                        help='盘点本地 GGUF 文件，不自动选择或下载。')
    parser.add_argument('--model', type=Path, help='可选：冻结本地 GGUF 路径与 SHA256，供启动复用。')
    parser.add_argument('--jobs', type=positive_int, default=4)
    parser.add_argument('--build-timeout', type=optional_timeout, default=900, help='墙钟预算秒数，默认 900；0 表示不限时。')
    parser.add_argument('--update-sources', action='store_true', help='先更新所有干净子模块，再冻结实际 commits。')
    parser.add_argument('--offline', action='store_true', help='禁止源码/依赖网络下载；需要完整本地子模块与工具链。')
    parser.add_argument('--check-only', action='store_true', help='只盘点；退出 2 表示尚未构建。')
    # Reject legacy-only flags with an actionable error, without advertising them as supported.
    parser.add_argument('--resume', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--cuda-home', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--frozen-p0', action='store_true', help=argparse.SUPPRESS)
    parser.set_defaults(download_timeout=None)


def main(machine: str, argv: list[str] | None = None, *, root: Path | None = None) -> int:
    from lab_tools import cli
    from lab_tools.prepare import add_init_arguments, prepare
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    parser = argparse.ArgumentParser(prog=machine, description=f"AI {machine} / {PROFILES[machine]['name']} 命令集")
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('doctor', help='查看实际环境、机器配置和设备访问。')
    init = sub.add_parser('init', help='在新目录准备本机后端；不下载模型。')
    if machine == '700':
        add_init_arguments(init)
    else:
        add_machine_init_arguments(init, root, machine)
    init.add_argument('targets', nargs='*', type=engine_name, choices=(*ALL_ENGINES, 'all'), metavar='ENGINE')
    init.add_argument('--rocm-home', type=Path, help='ROCm 根目录；默认探测环境变量、hipcc 和 /opt/rocm。' if machine == '300' else argparse.SUPPRESS)
    init.set_defaults(engines=None)
    sub.add_parser('run', help='记录干净快照中的实验（Linux/macOS）。').add_argument('arguments', nargs=argparse.REMAINDER)
    sub.add_parser('new', help='创建研究/实验模板。').add_argument('arguments', nargs=argparse.REMAINDER)
    sub.add_parser('download', add_help=False, help='独立下载模型；init 不调用。').add_argument('arguments', nargs=argparse.REMAINDER)
    sub.add_parser('hello', add_help=False, help='AI 300 生图/配音快速生成验证。').add_argument('arguments', nargs=argparse.REMAINDER)
    for engine in ALL_ENGINES:
        # Preserve the upstream launcher's own flags and -- separator.
        supported = engine in PROFILES[machine]['engines']
        sub.add_parser(engine, add_help=False, help=f'启动 {engine} 服务。' if supported else '尚无本机后端配方。').add_argument('arguments', nargs=argparse.REMAINDER)
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw:
        raw[0] = raw[0].casefold()
    # argparse REMAINDER alone rejects leading --dry-run / --help.
    if raw and raw[0] in (*ALL_ENGINES, 'download', 'hello'):
        args = argparse.Namespace(action=raw[0], arguments=raw[1:])
    else:
        args = parser.parse_args(raw)
    try:
        if args.action == 'doctor':
            report = cli.doctor(root)
            report.update(machine_id=machine, declared_profile=PROFILES[machine],
                          profile_source='user supplied; not measured', host_matches_profile=host_matches(machine))
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.action == 'init':
            if args.targets and args.engines:
                raise WorkflowError('ENGINE 与 --engines 二选一。')
            engines = args.targets or args.engines or PROFILES[machine]['engines'][:2]
            args.engines = PROFILES[machine]['engines'][:] if 'all' in engines else engines
            unsupported = set(args.engines) - set(PROFILES[machine]['engines'])
            if unsupported:
                raise WorkflowError(f"AI {machine} 尚无 {', '.join(sorted(unsupported))} 的本机后端构建配方；现有原生配方仅适用于 700 CUDA。")
            if machine == '700':
                if args.rocm_home:
                    raise WorkflowError('700 使用 --cuda-home；--rocm-home 属于 300。')
                if not args.dry_run and not args.check_only:
                    require_host(machine)
                directory, code = prepare(root, args)
            else:
                from lab_tools.machine_prepare import prepare_machine
                media = [name for name in args.engines if name in MEDIA_APPS]
                if media:
                    from lab_tools.media_prepare import prepare_media
                    from copy import copy
                    legacy = [name for name in args.engines if name not in MEDIA_APPS]
                    if legacy:
                        old = copy(args)
                        old.engines = legacy
                        directory, code = prepare_machine(root, machine, old)
                        print(directory)
                        if code:
                            return code
                    for name in media:
                        directory, code = prepare_media(root, name, args)
                        print(directory)
                        if code:
                            return code
                    return 0
                directory, code = prepare_machine(root, machine, args)
            print(directory)
            return code
        if args.action == 'download':
            from lab_tools.media_download import main as download
            return download([*args.arguments, '--machine', machine], root=root)
        if args.action == 'hello':
            if machine != '300':
                raise WorkflowError('生图/配音 hello 当前仅支持 AI 300。')
            from lab_tools.media_hello import main as hello
            return hello(args.arguments, root=root)
        if args.action in ('run', 'new'):
            if args.action == 'new':
                return cli.main(['new', *args.arguments], root=root)
            run = argparse.ArgumentParser(prog=f'{machine} run')
            run.add_argument('slug')
            run.add_argument('--output-dir', type=Path, default=root / 'artifacts/runs' / machine)
            options = run.parse_args(args.arguments)
            require_host(machine)
            directory, code = cli.run_experiment(root, options.slug, options.output_dir, machine_id=machine)
            print(directory)
            return code
        if args.action not in PROFILES[machine]['engines']:
            raise WorkflowError(f'AI {machine} 尚未实现 {args.action} 的本机后端；使用 {machine} --help 查看命令。')
        if machine == '700':
            if '--dry-run' not in args.arguments and '--help' not in args.arguments:
                require_host(machine)
            return importlib.import_module(f'lab_tools.serve_{args.action}').main(args.arguments)
        if args.action in MEDIA_APPS:
            from lab_tools.media_launcher import launch_media
            return launch_media(root, args.action, args.arguments)
        from lab_tools.machine_launcher import launch_machine
        return launch_machine(root, machine, args.action, args.arguments)
    except (WorkflowError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f'AI {machine}：{error}', file=sys.stderr)
        return 2


def main_100() -> int:
    return main('100')


def main_300() -> int:
    return main('300')


def main_700() -> int:
    return main('700')
