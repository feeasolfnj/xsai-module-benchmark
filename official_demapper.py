"""Run the user's unmodified upstream SRK demapper accuracy tests."""
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import uuid
import xml.etree.ElementTree as ET

from bench_support import BenchError, benchmark_lock


def junit_status(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    failed = sum(c.find('failure') is not None or c.find('error') is not None for c in cases)
    skipped = sum(c.find('skipped') is not None for c in cases)
    return dict(tests=len(cases), failed=failed, skipped=skipped,
                passed=len(cases)-failed-skipped,
                status='passed' if cases and not failed and not skipped else 'failed_or_skipped')


def run_official(a, root):
    srk = pathlib.Path(a.srk).expanduser().resolve()
    test = srk / 'plugins/neural_demapper/tests/unit/test_demappers.py'
    if not test.is_file():
        raise BenchError(f'Official test not found: {test}; --srk must point to the SRK root')
    out = pathlib.Path(a.out).resolve() if a.out else root / 'results' / (
        datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-official-demapper-'+uuid.uuid4().hex[:6])
    with benchmark_lock():
        out.mkdir(parents=True, exist_ok=False)
        cmd = [a.python, '-m', 'pytest', str(test), '-v', '-rs', '--junitxml='+str(out/'junit.xml')]
        meta = dict(status='running', command=cmd, source=str(test),
                    source_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
                    scope='upstream_sionna_reference_accuracy_not_performance',
                    env={k:os.environ[k] for k in ('SRK_PLATFORM', 'PLAN_POSTFIX') if k in os.environ})
        try:
            with (out/'run.log').open('w') as log:
                p = subprocess.Popen(cmd, cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                try:
                    for line in p.stdout:
                        print(line, end='', flush=True); log.write(line)
                    meta['returncode'] = p.wait()
                finally:
                    if p.poll() is None:
                        p.terminate(); p.wait(timeout=10)
            if (out/'junit.xml').exists():
                meta.update(junit_status(out/'junit.xml'))
            if meta['returncode'] or meta.get('status') != 'passed':
                raise BenchError(f'Official demapper tests failed, skipped, or collected no cases; see {out}/run.log')
        except BaseException as exc:
            meta.update(status='failed', error=str(exc)); raise
        finally:
            (out/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print('Results:', out)
