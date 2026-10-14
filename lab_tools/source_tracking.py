"""Track upstream branches while freezing each build to an actual commit."""
from pathlib import Path
import subprocess

from lab_tools.cli import WorkflowError, _git, _reject_hidden_changes


def source_revision(root: Path, name: str, git=_git) -> tuple[str, str]:
    path = root / 'sources' / name
    if (path / '.git').exists():
        _reject_hidden_changes(path)
        if git(path, 'status', '--porcelain', '--untracked-files=no'):
            raise WorkflowError(f'{path} 存在 tracked 修改；请先保存修改。')
        return git(path, 'rev-parse', 'HEAD'), str(path)
    ref = f':sources/{name}'
    return git(root, 'rev-parse', ref), ref


def update_sources(root: Path) -> None:
    """Fast-forward clean submodules; leave existing build clones untouched."""
    entries = _git(root, 'config', '-f', '.gitmodules', '--get-regexp', r'^submodule\..*\.path$')
    paths = [line.split(maxsplit=1)[1] for line in entries.splitlines()]
    for name in paths:
        path = root / name
        if (path / '.git').exists():
            _reject_hidden_changes(path)
            if _git(path, 'status', '--porcelain'):
                raise WorkflowError(f'{name} 有本地修改，更新已停止；不会覆盖。')
        branch = _git(root, 'config', '-f', '.gitmodules', '--get', f'submodule.{name}.branch')
        if branch not in ('main', 'master'):
            raise WorkflowError(f'{name} 未配置 main/master。')
    for name in paths:
        path = root / name
        branch = _git(root, 'config', '-f', '.gitmodules', '--get', f'submodule.{name}.branch')
        def run(*args):
            print(' '.join(args), flush=True)
            try:
                subprocess.run(args, cwd=root, check=True)
            except subprocess.CalledProcessError as error:
                raise WorkflowError(f'{name} 更新失败（退出 {error.returncode}）；原分支及修改保留，请核对日志。') from error
        if not (path / '.git').exists():
            run('git', 'submodule', 'update', '--init', '--', name)
        run('git', '-C', name, 'fetch', 'origin', branch)
        exists = subprocess.run(['git', '-C', name, 'show-ref', '--verify', '--quiet',
                                 f'refs/heads/{branch}'], cwd=root).returncode == 0
        if exists:
            run('git', '-C', name, 'switch', branch)
            run('git', '-C', name, 'merge', '--ff-only', f'origin/{branch}')
        else:
            run('git', '-C', name, 'switch', '--create', branch, '--track', f'origin/{branch}')
        print(f'{name}: {branch} {_git(path, "rev-parse", "HEAD")}', flush=True)
