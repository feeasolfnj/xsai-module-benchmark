"""Real CPU module smoke tests; hosted runners have no NVIDIA GPU."""
import pathlib
import os
import signal
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
        child = subprocess.Popen(args, cwd=root, start_new_session=True)
        try:
            code = child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, args)
