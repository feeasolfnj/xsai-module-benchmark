"""Validation and result handling shared by the CLI and its regression tests."""
import contextlib
import csv
import os
import pathlib
import re
import statistics


class BenchError(Exception):
    pass


def cpu_list(value):
    result = set()
    try:
        for part in value.split(','):
            ends = part.split('-')
            lo, hi = (int(ends[0]), int(ends[-1]))
            if len(ends) > 2 or lo < 0 or hi < lo:
                raise ValueError()
            result.update(range(lo, hi + 1))
    except ValueError:
        raise BenchError('Invalid CPU list; use e.g. 0-7 or 0,2,4,6') from None
    if not result:
        raise BenchError('CPU list is empty')
    return sorted(result)


def select_cpus(value, threads, available=None):
    available = set(os.sched_getaffinity(0) if available is None else available)
    selected = cpu_list(value) if value else sorted(available)[:threads]
    missing = set(selected) - available
    if missing:
        raise BenchError(f'CPUs {sorted(missing)} are unavailable; allowed CPUs: {sorted(available)}')
    if not selected:
        raise BenchError('No available CPUs')
    return selected


@contextlib.contextmanager
def benchmark_lock(path='/tmp/k3-module-benchmark.lock', wait=False):
    import fcntl
    # flock does not require a writable file on Linux. Open an existing lock
    # read-only so an earlier sudo run does not break later unprivileged runs.
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        try:
            fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            fd = os.open(path, flags)
    with os.fdopen(fd, 'r') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            raise BenchError('Another benchmark is running; retry later or use --wait-lock') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def records(log, warmup=0):
    raw, measured = [], []
    tags = ('RESULT', 'CODING_UL', 'CODING_DL', 'PMU_CYCLES')
    for line in log.splitlines():
        tag = next((t for t in tags if line.startswith('[' + t + ']')), None)
        if tag is None:
            continue
        row = dict(re.findall(r'(\w+)=([^\s]+)', line))
        row['record_type'] = tag
        row['phase'] = 'warmup' if tag.startswith('CODING_') and int(row['trial']) < warmup else 'measurement'
        raw.append(row)
        if row['phase'] == 'measurement':
            measured.append(row)
    return raw, measured


def write_csv(path, rows):
    if not rows:
        return
    with pathlib.Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)


def summary(rows):
    result = {}
    for tag in sorted({r['record_type'] for r in rows}):
        group = [r for r in rows if r['record_type'] == tag]
        item = {'records': len(group)}
        for key in ('wall_us', 'mean_us', 'cores', 'median_batch_mean_us'):
            values = [float(r[key]) for r in group if key in r]
            if values:
                item[key] = statistics.mean(values)
        if tag.startswith('CODING_'):
            pairs = [(float(r['cpu_us']), float(r['wall_us'])) for r in group if 'cpu_us' in r]
            if pairs and sum(w for _, w in pairs):
                item['cores'] = sum(c for c, _ in pairs) / sum(w for _, w in pairs)
        result[tag] = item
    return result


def frequency_snapshot(cpus, root=pathlib.Path('/sys/devices/system/cpu')):
    policies = {}
    for cpu in cpus:
        path = root / f'cpu{cpu}' / 'cpufreq'
        if not path.exists():
            continue
        path = path.resolve()
        if str(path) not in policies:
            policies[str(path)] = {name: (path / name).read_text().strip()
                                   for name in ('scaling_cur_freq', 'scaling_governor', 'scaling_available_frequencies')
                                   if (path / name).exists()}
    return policies


@contextlib.contextmanager
def fixed_frequency(cpus, khz, root=pathlib.Path('/sys/devices/system/cpu')):
    snapshot = frequency_snapshot(cpus, root)
    originals = []
    if khz is not None:
        if not snapshot:
            raise BenchError('CPU frequency controls are unavailable; omit --freq')
        for name, data in snapshot.items():
            path = pathlib.Path(name)
            if data.get('scaling_governor') != 'userspace':
                raise BenchError(f'{path}: --freq requires userspace governor; configure it explicitly or omit --freq')
            if str(khz) not in data.get('scaling_available_frequencies', '').split():
                raise BenchError(f'{path}: unsupported frequency {khz}; available: {data.get("scaling_available_frequencies", "unknown")}')
            setting = path / 'scaling_setspeed'
            if not setting.exists() or not os.access(setting, os.W_OK):
                raise BenchError(f'{setting}: not writable; omit --freq or use an authorized account')
            originals.append((setting, setting.read_text().strip()))
    changed = []
    try:
        for setting, original in originals:
            setting.write_text(str(khz) + '\n')
            changed.append((setting, original))
        if khz is not None:
            check_frequency(cpus, khz, root)
        yield snapshot
    finally:
        for setting, original in reversed(changed):
            setting.write_text(original + '\n')


def check_frequency(cpus, khz, root=pathlib.Path('/sys/devices/system/cpu')):
    for name, data in frequency_snapshot(cpus, root).items():
        if data.get('scaling_cur_freq') != str(khz):
            raise BenchError(f'{name}: frequency differs from requested {khz}: {data.get("scaling_cur_freq")}')
