"""Read the local OAI build configuration; never guess foreign ISA flags."""
import json
import hashlib
import pathlib
import platform
import re
import shlex
import shutil

from bench_support import BenchError

P = pathlib.Path


def adapt_statistics(text, header):
    """Remove only known XSAI instrumentation absent from official gNB headers."""
    for field in ('ulsch_channel_measurement_stats', 'ulsch_resource_extraction_stats',
                  'ulsch_channel_compensation_stats'):
        if not re.search(r'\b' + field + r'\b', header):
            pattern = r'\b(?:start_meas|stop_meas|merge_meas)\(\s*&gNB->' + field + r'\b[^;\n]*\);'
            text = re.sub(pattern, '(void)0;', text)
    return text


def source_root(value):
    root = P(value).expanduser().resolve()
    if (root / 'ext/openairinterface5g/CMakeLists.txt').is_file():
        root /= 'ext/openairinterface5g'
    if not (root / 'openair1/PHY/defs_gNB.h').is_file():
        raise BenchError(f'Not an OAI source tree or prepared sionna-rk root: {root}')
    return root


def cache_value(build, name):
    cache = P(build) / 'CMakeCache.txt'
    if cache.exists():
        for line in cache.read_text().splitlines():
            if line.startswith(name + ':'):
                return line.split('=', 1)[1]


def compiler(build, override=None):
    result = override or cache_value(build, 'CMAKE_C_COMPILER') or shutil.which('cc')
    if not result or not shutil.which(result):
        raise BenchError(f'C compiler unavailable: {result}; use --cc or configure OAI first')
    return result


def host_arch(machine=None):
    machine = (machine or platform.machine()).lower()
    aliases = {'amd64': 'x86_64', 'arm64': 'aarch64'}
    machine = aliases.get(machine, machine)
    if machine not in ('x86_64', 'aarch64', 'riscv64'):
        raise BenchError(f'Unsupported host architecture: {machine}; use 64-bit Linux')
    return machine


def check_library_arch(path, arch):
    data = P(path).read_bytes()[:20]
    if len(data) < 20 or data[:4] != b'\x7fELF':
        raise BenchError(f'Not an ELF shared library: {path}')
    machine = int.from_bytes(data[18:20], 'little' if data[5] == 1 else 'big')
    if machine != {'x86_64': 62, 'aarch64': 183, 'riscv64': 243}[arch]:
        raise BenchError(f'{path}: ELF machine {machine} does not match host {arch}')


def compile_flags(build, target):
    """Works with Makefiles and Ninja compile databases, including ccache."""
    database = P(build) / 'compile_commands.json'
    if not database.exists():
        path = P(build) / f'CMakeFiles/{target}.dir/flags.make'
        if path.exists():
            values = dict(line.split(' = ', 1) for line in path.read_text().splitlines() if ' = ' in line)
            return sum((shlex.split(values.get(k, '')) for k in ('C_DEFINES', 'C_INCLUDES', 'C_FLAGS')), [])
        raise BenchError('Official profile requires flags.make or CMAKE_EXPORT_COMPILE_COMMANDS=ON; reconfigure the existing OAI build with that option')
    for entry in json.loads(database.read_text()):
        args = entry.get('arguments') or shlex.split(entry['command'])
        if not any(f'CMakeFiles/{target}.dir/' in token for token in args):
            continue
        result = []
        skip = False
        # Find the compiler even when CMake enabled a compiler launcher.
        start = 2 if P(args[0]).name in ('ccache', 'sccache') else 1
        for token in args[start:]:
            if skip:
                skip = False
                continue
            if token in ('-o', '-MF', '-MT', '-MQ'):
                skip = True
                continue
            if token in ('-c', '-MD', '-MMD', '-MP') or token == entry['file']:
                continue
            if token.startswith('-I') and len(token) > 2 and not P(token[2:]).is_absolute():
                token = '-I' + str(P(entry['directory']) / token[2:])
            result.append(token)
        return result
    raise BenchError(f'No compile command for {target}; build the OAI simulator targets first')


def link_command(build, target):
    path = P(build) / f'CMakeFiles/{target}.dir/link.txt'
    if path.exists():
        return shlex.split(path.read_text())
    # CMake Ninja's tool prints the real linker command, with transitive archives.
    import subprocess
    ninja = cache_value(build, 'CMAKE_MAKE_PROGRAM')
    if ninja and 'ninja' in P(ninja).name:
        result = subprocess.run([ninja, '-t', 'commands', target], cwd=build,
                                capture_output=True, text=True)
        if result.returncode:
            raise BenchError(f'Cannot read Ninja linker recipe for {target}: {result.stderr.strip()}')
        for line in reversed(result.stdout.splitlines()):
            args = shlex.split(line)
            if '-o' not in args or P(args[args.index('-o') + 1]).name != target:
                continue
            if args[:2] == [':', '&&']:
                args = args[2:]
            if '&&' in args:
                args = args[:args.index('&&')]
            return args
    raise BenchError(f'No linker recipe for {target}; configure/build nr_ulsim and nr_dlsim first')


def module_link(original, obj, output):
    if not original or '-o' not in original:
        raise BenchError('Invalid OAI linker recipe; reconfigure the original CMake build')
    result = []
    skip = False
    for token in original:
        if skip:
            skip = False
            continue
        if token == '-o':
            skip = True
            continue
        # Replace the simulator main and its glue; retain other target helpers.
        mains = ('ulsim.c.o', 'dlsim.c.o', 'nr_ulschsim.c.o', 'nr_dlschsim.c.o',
                 'softmodem-common.c.o', 'rrc_nsa.c.o', 'nfapi.c.o', 'gnb_ind_vars.c.o', 'queue_t.c.o')
        if P(token).name in mains and any(f'CMakeFiles/{name}.dir/' in token for name in ('nr_ulsim', 'nr_dlsim')):
            continue
        if token == '-rdynamic':
            continue
        if token == '-lgcc_s_asneeded':
            token = '-lgcc_s'
        result.append(token)
    result[1:1] = [str(obj)]
    return result + ['-o', str(output), '-Wl,--gc-sections',
                     '-Wl,--export-dynamic-symbol=T_*', '-Wl,--export-dynamic-symbol=exit_function',
                     '-Wl,--export-dynamic-symbol=g_log', '-Wl,--export-dynamic-symbol=logRecord_mt',
                     '-Wl,--export-dynamic-symbol=cpu_meas_enabled',
                     '-Wl,--export-dynamic-symbol=write_file_matlab', '-ldl',
                     '-Wl,--no-as-needed', '-lm', '-Wl,--as-needed']


def build_official(root, xsai, build, out, cc, keys, execute, pmu=False):
    """Reuse the installed tree's flags and complete CMake linker dependency graph."""
    from bench import NAMES
    generated = out / 'generated'
    generated.mkdir(exist_ok=True)
    resolved = {}

    def artifact(token):
        path = P(token) if P(token).is_absolute() else build / token
        if path.exists() or not token.endswith(('.o', '.a')):
            return token
        if token in resolved:
            return resolved[token]
        if token.endswith('.o') and 'CMakeFiles/' in token and '.dir/' in token:
            target, relative = token.split('CMakeFiles/', 1)[1].split('.dir/', 1)
            source = xsai / relative[:-2]
            if not source.is_file():
                raise BenchError(f'Missing generated helper {path}; complete the OAI build first')
            obj = out / ('helper-' + target + '-' + source.name + '.o')
            execute([cc, *compile_flags(build, target), '-I' + str(source.parent), '-c', str(source), '-o', str(obj)])
            resolved[token] = str(obj)
        elif token.endswith('.a'):
            target = path.name.removeprefix('lib').removesuffix('.a')
            recipe = build / f'CMakeFiles/{target}.dir/link.txt'
            if not recipe.is_file():
                raise BenchError(f'Missing {path}; run cmake --build {build} --target {target}')
            inputs = [x for x in shlex.split(recipe.read_text().splitlines()[0]) if x.endswith('.o')]
            if not inputs:
                raise BenchError(f'Cannot reconstruct {target}; complete the OAI build first')
            archive = out / path.name
            manifest = out / (path.name + '.inputs.json')
            fingerprint = {'compiler': cc, 'flags': compile_flags(build, target), 'inputs': {}}
            for item in inputs:
                candidate = build / item
                if not candidate.is_file():
                    candidate = xsai / item.split('.dir/', 1)[1][:-2]
                fingerprint['inputs'][str(candidate)] = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if not archive.exists() or not manifest.exists() or json.loads(manifest.read_text()) != fingerprint:
                objects = [artifact(x) for x in inputs]
                # Recreate rather than append, so removed members cannot survive a rebuild.
                if archive.exists():
                    archive.unlink()
                execute(['ar', 'rcs', str(archive), *objects])
                manifest.write_text(json.dumps(fingerprint, indent=2))
            resolved[token] = str(archive)
        else:
            raise BenchError(f'Missing OAI build artifact: {path}')
        return resolved[token]

    def link(obj, output, target):
        command = module_link(link_command(build, target), obj, output)
        plugins = build / 'plugins/libplugins_plugins.a'
        if plugins.exists() and not any('libplugins_plugins.a' in x for x in command):
            command.insert(2, str(plugins))
        # The UL benchmark's unitary helpers call the queue API directly.
        if output.name == 'nr_ulschsim':
            command.insert(2, artifact('CMakeFiles/nr_ulsim.dir/openair2/PHY_INTERFACE/queue_t.c.o'))
        if output.name in ('nr_ulschsim', 'nr_dlschsim'):
            command.append('-rdynamic')
        command.insert(1, '-Wl,--start-group')
        command.append('-Wl,--end-group')
        return [artifact(x) for x in command]
    (out / 'pmu_disabled.h').write_text('static inline void pmu_begin(void){}\nstatic inline void pmu_end(unsigned n){(void)n;}\n')
    gnb_header = (xsai / 'openair1/PHY/defs_gNB.h').read_text()
    for key in keys:
        target = 'nr_dlsim' if key == 'DL' else 'nr_ulsim'
        flag_target = target if key in ('UL', 'DL') else 'PHY_NR'
        flags = ['-D_GNU_SOURCE', '-I' + str(root / 'src'), *compile_flags(build, flag_target)]
        flags += ['-I' + str(root / 'src'), '-I' + str(xsai / 'openair1/PHY/NR_TRANSPORT'),
                  '-ffunction-sections', '-fdata-sections', '-UNDEBUG', '-DTB_KIND=' + str({'measurement': 1, 'symbol': 2}.get(key, 0))]
        header = root / 'src/pmu_cycles.h' if pmu else out / 'pmu_disabled.h'
        if key == 'dft':
            objects = []
            for name in ('bench.c', 'adapter.c'):
                obj = out / (name + '.o')
                instrumentation = ['-include', str(header)] if name == 'bench.c' else []
                execute([cc, *flags, *instrumentation, '-DBENCH_NATIVE_DFT=1', '-c', str(root / 'src' / name), '-o', str(obj)])
                objects.append(obj)
            command = link(objects[0], out / NAMES[key], target)
            command.insert(2, str(objects[1]))
            execute(command)
            continue
        name = {'estimation': 'standalone.c', 'measurement': 'standalone.c', 'symbol': 'standalone.c',
                'llr': 'random_llr.c', 'neural': 'random_neural.c', 'UL': 'nr_ulschsim.c', 'DL': 'nr_dlschsim.c'}[key]
        text = (root / 'src' / name).read_text()
        text = adapt_statistics(text, gnb_header)
        source = generated / name
        source.write_text(text)
        obj = out / (NAMES[key] + '.o')
        execute([cc, *flags, '-include', str(header), '-c', str(source), '-o', str(obj)])
        execute(link(obj, out / NAMES[key], target))
