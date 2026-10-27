"""Real CPU module smoke tests; hosted runners have no NVIDIA GPU."""
import pathlib
import os
import signal
import subprocess
import sys
import json

root = pathlib.Path(__file__).resolve().parents[1]
cycle_status = {}


def run_case(args, expected=0):
    child = subprocess.Popen(args, cwd=root, start_new_session=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        output, _ = child.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.communicate()
        raise
    print(output, end='', flush=True)
    if child.returncode != expected:
        raise RuntimeError(f'Expected {expected}, got {child.returncode}: {args}')
    if expected:
        return None
    result_dir = pathlib.Path(output.rsplit('Results:', 1)[1].strip())
    if 'UL' in args:
        q = json.loads((result_dir / 'summary.json').read_text())['decoder_quality']
        assert q['measured_trials'] == 2 and q['succeeded'] == 2 and not q['false_positives'], q
    cycles = json.loads((result_dir / 'cycles.json').read_text())
    assert cycles['status'] in ('available', 'unavailable', 'invalid', 'disabled'), cycles
    if cycles['status'] == 'available':
        assert cycles['cycles'] > 0 and cycles['cycles_per_unit'] > 0, cycles
    else:
        assert cycles['cycles'] is None and cycles['cycles_per_unit'] is None, cycles
    assert json.loads((result_dir / 'metadata.json').read_text())['hardware_cycles'] == cycles
    return cycles['status']


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
        cycle_status[metric, threads] = run_case(args)

# Both a PUSCH module and a transform must honor runtime policy.
for metric, extra in [('llr', ['--seconds', '0.05']), ('dft', ['--iterations', '20', '--batches', '2'])]:
    base = [sys.executable, str(root / 'bench.py'), 'run', metric, '--threads', '1', '--warmup', '1', *extra]
    assert run_case([*base, '--cycles', 'off']) == 'disabled'
    run_case([*base, '--cycles', 'required'], expected=0 if cycle_status[metric, 1] == 'available' else 2)
