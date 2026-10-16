"""Pinned ROCm wheels from NJU; ordinary dependencies keep the project's PyPI index."""
from urllib.parse import quote

from lab_tools.media_models import APPS, TORCH_VERSION

MIRROR = 'https://mirror.nju.edu.cn/pytorch/whl'
# SHA256 from the official rocm7.2 torch/torchaudio indexes, checked against NJU.
# The official triton-rocm index does not publish full-wheel SHA256 fragments.
TORCH_SHA256 = {
    '3.11': 'e86e64dc070eebe40bd7d7d397b418f3e16a528ed91495efcfbbe430ba3bb45d',
    '3.12': '8f02a36213288ab5921e277e13f5954cc62afe7cfa899b79d739d4ef2be6510b',
}
AUDIO_SHA256 = {
    '3.11': 'db4ae9ac6c3d38459b88941028c1a2296fa968015c3bdc9602c199e62eb71029',
    '3.12': 'e0316d938346cfa4ff04b91405a91ecc50bd18781023eca5763b79f0f7b9bc8c',
}


def wheel_sources(app: str) -> list[dict]:
    version = APPS[app]['python']
    abi = 'cp' + version.replace('.', '')
    # A version bump must explicitly update the filenames and their upstream hashes.
    if TORCH_VERSION != '2.11.0+rocm7.2':
        raise ValueError('更新 torch 时需要同时更新 ROCm wheel 下载身份。')
    wheels = [('torch', 'rocm7.2/torch-2.11.0+rocm7.2-' + abi + '-' + abi + '-manylinux_2_28_x86_64.whl', TORCH_SHA256[version]),
              ('triton-rocm', 'triton_rocm-3.6.0-' + abi + '-' + abi + '-linux_x86_64.whl', None)]
    if app == 'indextts2':
        wheels.append(('torchaudio', 'rocm7.2/torchaudio-2.11.0+rocm7.2-' + abi + '-' + abi + '-manylinux_2_28_x86_64.whl', AUDIO_SHA256[version]))
    return [{'package': package, 'url': MIRROR + '/' + quote(path), 'sha256': digest,
             'upstream_url': 'https://download.pytorch.org/whl/' + quote(path)}
            for package, path, digest in wheels]


def wheel_requirements(wheels: list[dict]) -> list[str]:
    return [item['package'] + ' @ ' + item['url'] + ('#sha256=' + item['sha256'] if item['sha256'] else '')
            for item in wheels]
