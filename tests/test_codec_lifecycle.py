"""Exercise the real linker wrappers, including matched plugin cleanup."""
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('cc'), 'Linux C linker required')
class CodecLifecycleTests(unittest.TestCase):
    def test_cleanup_requires_matching_init(self):
        with tempfile.TemporaryDirectory() as directory:
            p = pathlib.Path(directory)
            (p / 'PHY').mkdir()
            (p / 'PHY/defs_nr_common.h').write_text('typedef struct frame NR_DL_FRAME_PARMS;\n')
            (p / 'real.c').write_text('int initialized, freed; void init_plugins(void *p){initialized++;} void free_plugins(void){freed++;}\n')
            (p / 'main.c').write_text('''
#include <assert.h>
extern int initialized, freed;
void init_plugins(void *); void free_plugins(void);
int main(void) {
  free_plugins(); assert(freed == 0);
  init_plugins(0); assert(initialized == 1);
  free_plugins(); assert(freed == 1);
  free_plugins(); assert(freed == 1);
  init_plugins(0); free_plugins(); assert(freed == 2);
  return 0;
}
''')
            source = pathlib.Path(__file__).resolve().parents[1] / 'src/codec_plugin_lifecycle.c'
            subprocess.run(['cc', '-I'+str(p), str(source), str(p/'real.c'), str(p/'main.c'),
                            '-Wl,--wrap=init_plugins', '-Wl,--wrap=free_plugins', '-o', str(p/'check')], check=True)
            subprocess.run([str(p/'check')], check=True)
