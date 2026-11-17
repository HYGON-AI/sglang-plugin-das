# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig

import torch


@functools.lru_cache(maxsize=1)
def load_dense_mqa_out():
    import fcntl
    from torch.utils.cpp_extension import COMMON_HIP_FLAGS, COMMON_HIPCC_FLAGS

    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(':')[0]
    if arch != 'gfx936':
        raise RuntimeError(f'dense MQA output workspace requires gfx936, got {arch}')
    source = Path(__file__).resolve().parents[4] / 'jit_overlay/csrc/dsv41_mqa_out'
    files = ['fp8_fp4_mqa_logits.cu', 'fp8_fp4_mqa_logits.h', 'utils.h', 'lds_even_wrap.hpp']
    compiler = shutil.which('hipcc')
    if compiler is None:
        raise RuntimeError('dense MQA output workspace requires hipcc')
    sdk = Path('/opt/dtk')
    torch_root = Path(torch.__file__).resolve().parent
    identity = dict(source={name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files}, torch_distribution=importlib.metadata.version('torch'), torch_root=str(torch_root), compiler=str(Path(compiler).resolve()), compiler_version=subprocess.check_output([compiler, '--version'], timeout=30).decode(), arch=arch, cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ.get('TORCH_EXTENSIONS_DIR', str(Path.home()/'.cache/hcu_sglang'))) / ('dsv41_mqa_out_'+digest)
    cache.mkdir(parents=True, exist_ok=True)
    name = 'hcu_mqa_logits_out01'
    binary = cache / (name+'.so')
    with (cache/'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.is_file():
            temporary = cache / (name+'.'+str(os.getpid())+'.so')
            command = [compiler,'-O3','-std=c++17','-shared','-fPIC','--offload-arch='+arch,'-mllvm','-enable-num-vgprs-512=true','-DTORCH_EXTENSION_NAME='+name,'-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
            for key in ['_PYBIND11_COMPILER_TYPE','_PYBIND11_STDLIB','_PYBIND11_BUILD_ABI']:
                value = getattr(torch._C,key,None)
                if isinstance(value,str):
                    command.append('-D'+key[1:]+'="'+value+'"')
            command += list(COMMON_HIP_FLAGS)+list(COMMON_HIPCC_FLAGS)
            command += [str(source/files[0]),'-I'+str(source),'-I'+str(torch_root/'include'),'-I'+str(torch_root/'include/torch/csrc/api/include'),'-I'+sysconfig.get_paths()['include'],'-I'+str(sdk/'include'),'-L'+str(torch_root/'lib'),'-Wl,-rpath,'+str(torch_root/'lib'),'-L'+str(sdk/'lib'),'-Wl,-rpath,'+str(sdk/'lib')]
            command += ['-l'+name for name in ['torch','torch_cpu','torch_python','c10','torch_hip','c10_hip','amdhip64']]
            command += ['-o',str(temporary)]
            (cache/'build.json').write_text(json.dumps(dict(identity=identity,command=command),indent=2))
            with (cache/'compiler.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
            temporary.replace(binary)
            (cache/'binary.sha256').write_text(hashlib.sha256(binary.read_bytes()).hexdigest())
        expected = (cache/'binary.sha256').read_text().strip()
        if hashlib.sha256(binary.read_bytes()).hexdigest() != expected:
            raise RuntimeError('dense MQA output binary digest mismatch')
    spec = importlib.util.spec_from_file_location(name,binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fp8_fp4_mqa_logits_out


class DenseMqaOutputWorkspace:
    def __init__(self, device, capacity_bytes):
        self.capacity_bytes = capacity_bytes//4*4
        self.operation = load_dense_mqa_out()
        self.storage = torch.empty(self.capacity_bytes//4,device=device,dtype=torch.float32)
        self.stream = None

    def view(self, rows, width):
        if rows*width*4 > self.capacity_bytes:
            return None
        stream = torch.cuda.current_stream(self.storage.device).cuda_stream
        if self.stream is None:
            self.stream = stream
        elif self.stream != stream:
            return None
        return self.storage[:rows*width].view(rows,width)


def dense_mqa_workspace_capacity(total, fraction, mem_fraction_static):
    guard = int(total*.3)
    if mem_fraction_static is not None:
        guard = min(guard,int(total*max(0.,1.-mem_fraction_static)*fraction))
    if guard > 1024**3:
        return 0
    return max(8000000*4,guard)//4*4


def make_dense_mqa_output_workspace(device, mem_fraction_static):
    name = 'SGLANG_DSV41_PAGED_MQA_LOGITS_FREE_MEM_FRACTION'
    raw = os.environ.get(name)
    if raw is None:
        return None
    fraction = float(raw)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError(f'{name} must be finite and in (0, 1], got {raw!r}')
    props = torch.cuda.get_device_properties(device)
    if props.gcnArchName.split(':')[0] != 'gfx936':
        return None
    capacity = dense_mqa_workspace_capacity(props.total_memory,fraction,mem_fraction_static)
    return DenseMqaOutputWorkspace(device,capacity) if capacity else None
