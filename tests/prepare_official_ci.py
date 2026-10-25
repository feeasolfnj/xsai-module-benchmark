"""Prepare pinned upstream sources for CPU-only CI, without CUDA/TensorRT.

The fixture disables plugin initialization only; tested DSP/codec functions remain
the upstream SRK-patched implementation. Not installed in users' working trees.
"""
import pathlib
import shutil
import subprocess
import sys

srk, oai = map(pathlib.Path, sys.argv[1:])
patch = (srk / 'patches/openairinterface5g.patch').read_text().replace('\r', '')
# Some upstream patches contain empty deleted files with no ---/+++ header.
blocks = patch.split('diff --git ')[1:]
clean = ''.join('diff --git ' + b for b in blocks if '\n--- ' in b and '\n+++ ' in b)
patch_path = oai / 'srk-ci.patch'
patch_path.write_text(clean)
subprocess.run(['git', 'apply', '--whitespace=nowarn', str(patch_path.resolve())], cwd=oai, check=True)
shutil.copytree(srk / 'plugins', oai / 'plugins', symlinks=True)
fixture = oai / 'benchmark_cpu_plugins.c'
fixture.write_text('''
#include "plugins/common/src/plugins.h"
#include "plugins/common/src/mac_plugins.h"
receiver_interface_t receiver_interface = {0};
demapper_interface_t demapper_interface = {0};
link_adaptation_interface_t link_adaptation_interface = {0};
void init_plugins(const NR_DL_FRAME_PARMS *fp){(void)fp;}
void free_plugins(void){}
void worker_thread_plugin_init(void){}
void init_mac_plugins(void){}
void free_mac_plugins(void){}
int is_channel_emulation_enabled(void){return 0;}
const void *channel_emulator_cir_read_and_apply(void){return 0;}
''')
with (oai / 'CMakeLists.txt').open('a') as f:
    f.write('\n# CPU-only benchmark CI fixture (no GPU plugin initialization)\n')
    f.write('target_sources(PHY_NR PRIVATE ${CMAKE_SOURCE_DIR}/benchmark_cpu_plugins.c)\n')
