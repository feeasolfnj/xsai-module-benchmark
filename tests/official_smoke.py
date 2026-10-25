"""Real CPU module smoke tests; hosted runners have no NVIDIA GPU."""
import pathlib
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[1]
for metric in ('estimation', 'measurement', 'symbol', 'llr', 'UL', 'DL', 'dft', 'idft'):
    for threads in (1, 8):
        args = [sys.executable, str(root / 'bench.py'), 'run', metric,
                '--threads', str(threads), '--warmup', '1']
        if metric in ('UL', 'DL'):
            args += ['--trials', '2']
        elif metric in ('dft', 'idft'):
            args += ['--iterations', '20', '--batches', '2']
        else:
            args += ['--seconds', '0.05']
        subprocess.run(args, cwd=root, check=True)
