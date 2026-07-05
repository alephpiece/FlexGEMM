from typing import *
import hashlib
import json
import torch
import triton


def get_gpu_name():
    return torch.cuda.get_device_name()


def get_gpu_arch():
    if not torch.cuda.is_available():
        return 'unknown'

    props = torch.cuda.get_device_properties("cuda")
    for attr in ('gcnArchName', 'gcn_arch_name'):
        arch = getattr(props, attr, None)
        if arch:
            return str(arch).split(':', 1)[0].lower()

    major = getattr(props, 'major', None)
    minor = getattr(props, 'minor', None)
    if major is not None and minor is not None:
        return f'sm{major}{minor}'
    return 'unknown'


def get_platform_name():
    if torch.cuda.is_available():
        if getattr(torch.version, 'hip', None) is not None:
            return 'hip'
        return 'cuda'
    return 'unknown'
    

def get_num_sm():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def _json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def config_signature(configs: List[triton.Config]) -> int:
    payload = []
    for cfg in configs:
        kwargs = getattr(cfg, 'kwargs', {})
        payload.append({
            'kwargs': sorted((str(key), _json_value(value)) for key, value in kwargs.items()),
            'num_warps': getattr(cfg, 'num_warps', None),
            'num_stages': getattr(cfg, 'num_stages', None),
        })
    payload.sort(key=lambda elem: json.dumps(elem, sort_keys=True))
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.blake2s(encoded, digest_size=4).digest(), 'big')
    

def get_autotune_config(
    default: List[triton.Config] = None,
    platform: Dict[str, List[triton.Config]] = None,
    device: Dict[str, List[triton.Config]] = None,
    arch: Dict[str, List[triton.Config]] = None,
) -> List[triton.Config]:
    """
    Get the autotune configuration for the current platform and device.
    """
    if device is not None:
        gpu_name = get_gpu_name()
        for key, value in device.items():
            if key.lower() in gpu_name.lower():
                return value

    if arch is not None:
        gpu_arch = get_gpu_arch()
        for key, value in arch.items():
            if key.lower() == gpu_arch:
                return value
    
    if platform is not None:
        platform_name = get_platform_name()
        for key, value in platform.items():
            if key.lower() in platform_name.lower():
                return value
    
    if default is None:
        raise ValueError("No autotune configuration found for the current platform and device.")
    return default
