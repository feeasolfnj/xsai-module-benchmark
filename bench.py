#!/usr/bin/env python3
"""Independent, relocatable XSAI module benchmark launcher (64-bit Linux: RISC-V, x86-64, ARM64)."""
import argparse,csv,datetime,math,hashlib,json,os,pathlib,re,shlex,shutil,subprocess,sys,uuid,statistics
from platform_build import source_root, compiler, host_arch, check_library_arch, build_official, cache_value
from bench_support import BenchError, select_cpus, benchmark_lock, fixed_frequency, frequency_snapshot, check_frequency, records, write_csv, summary
P=pathlib.Path;ROOT=P(__file__).resolve().parent
NAMES={'estimation':'test_pusch_estimation','measurement':'test_pusch_measurement','symbol':'test_pusch_symbol','llr':'test_llr_region','UL':'nr_ulschsim','DL':'nr_dlschsim','neural':'test_neural','dft':'bench-coop','idft':'bench-coop'}
PAIRS={'estimation':(0,1),'measurement':(2,3),'symbol':(4,5),'neural':(6,7),'llr':(8,9),'UL':(10,11),'DL':(12,13),'dft':(14,16)}
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def check_neural_log(log,backend):
 if backend=='tensorrt' and re.search(r'failed to deserialize engine|no engine available|execution context unavailable|Error Code\s*:?\s*[1-9]|CUDA error',log,re.I):
  raise BenchError('TensorRT reported an engine/context/inference failure; this timing is invalid. Check run.log and rebuild the engine for this GPU/TensorRT version.')
def shell(cmd,cwd):
 q=subprocess.run(cmd,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 print(q.stdout,end='');return q
def config():
 if not (ROOT/'config.local.json').exists():raise BenchError('Run build --xsai /path/to/XSAI first')
 return json.loads((ROOT/'config.local.json').read_text())
def flags(path):
 if not path.exists():
  database=path.parents[2]/'compile_commands.json';target=path.parent.name
  if not database.exists():raise BenchError(f'Missing {path}; enable CMAKE_EXPORT_COMPILE_COMMANDS=ON in your XSAI CMake build to rebuild missing helpers')
  for entry in json.loads(database.read_text()):
   args=entry.get('arguments') or shlex.split(entry['command'])
   if not any(target+'/' in token for token in args):continue
   result=[];skip=False
   for token in args[1:]:
    if skip:skip=False;continue
    if token=='-o':skip=True;continue
    if token=='-c' or token==entry['file']:continue
    if token.startswith('-I') and not P(token[2:]).is_absolute():token='-I'+str(P(entry['directory'])/token[2:])
    result.append(token)
   return result
  raise BenchError(f'{database}: no compile command found for {target}')
 d={}
 for line in path.read_text().splitlines():
  if ' = ' in line:k,v=line.split(' = ',1);d[k]=shlex.split(v)
 return sum([d[k] for k in ['C_DEFINES','C_INCLUDES','C_FLAGS']],[])
def build(a):
 xsai=source_root(a.xsai);b=P(a.build).resolve() if a.build else xsai/'cmake_targets/ran_build/build';out=ROOT/'build';out.mkdir(exist_ok=True)
 if not (b/'libdfts.so').exists():raise BenchError('Missing libdfts.so; run doctor --xsai '+str(xsai))
 a.cc=compiler(b,a.cc);arch=host_arch();profile=a.profile if a.profile!='auto' else ('xsai' if arch=='riscv64' else 'official')
 check_library_arch(b/'libdfts.so',arch)
 if profile=='xsai' and arch!='riscv64':raise BenchError('The xsai profile requires a RISC-V host; use --profile official')
 keys=list(PAIRS) if a.only=='all' else list(dict.fromkeys('dft' if x=='idft' else x for x in a.only.split(',')))
 if set(keys)-set(PAIRS):raise BenchError('Unknown build metric')
 ready_file=out/'ready.json';ready=json.loads(ready_file.read_text()) if ready_file.exists() else {}
 identity=dict(_xsai=str(xsai),_build=str(b),_profile=profile,_cc=a.cc,_pmu=a.pmu,_libdfts=sha(b/'libdfts.so'))
 if any(ready.get(k)!=v for k,v in identity.items()):ready={}
 ready.update(identity)
 for key in keys:ready.pop(key,None)
 save(ready_file,ready)
 previous=json.loads((ROOT/'config.local.json').read_text()) if (ROOT/'config.local.json').exists() else {}
 c=dict(xsai=str(xsai),build=str(b),src=str(ROOT/'src'),out=str(out),cc=a.cc,architecture=arch,profile=profile,dft_backend='rvv-cooperative' if profile=='xsai' else 'local-native-serial',pmu=a.pmu,metric_pmu=previous.get('metric_pmu',{}) if previous.get('xsai')==str(xsai) else {})
 if not a.pmu:(out/'pmu_disabled.h').write_text('static inline void pmu_begin(void){}\nstatic inline void pmu_end(unsigned long n){(void)n;}\n')
 recipes=json.loads((ROOT/'templates/commands.json').read_text());log=[]
 def execute(cmd):
  log.append(cmd);q=shell(cmd,b);save(out/'build-commands.json',log)
  if q.returncode:raise BenchError('Build failed; run doctor for missing targets. This build is not marked ready.')
 def expand(cmd):
  result=[]
  for arg in cmd:
   if '{dependency}' in arg:
    name=arg.rsplit('/',1)[1];matches=list(b.rglob(name))
    if not matches:
     sources=list(xsai.rglob(name[:-2]))
     if len(sources)!=1:raise BenchError(f'Cannot unambiguously locate source for {name}')
     dep=out/name;execute([a.cc,*flags(b/'CMakeFiles/PHY_NR.dir/flags.make'),'-I'+str(sources[0].parent),'-c',str(sources[0]),'-o',str(dep)]);matches=[dep]
    arg=str(matches[0])
   for key in ['xsai','build','src','out']:arg=arg.replace('{'+key+'}',c[key])
   if arg.endswith('/pmu_cycles.h') and not a.pmu:arg=str(out/'pmu_disabled.h')
   if arg.endswith('/libMAC_UE_NR.a') and not P(arg).exists():
    if not (b/'CMakeFiles/MAC_UE_NR.dir/link.txt').exists():raise BenchError('Missing MAC archive; run '+shlex.join(['cmake','--build',str(b),'--target','MAC_UE_NR','-j','8']))
    archive=out/'libMAC_UE_NR.a';manifest=out/'mac-cache.json'
    recipe=shlex.split((b/'CMakeFiles/MAC_UE_NR.dir/link.txt').read_text().splitlines()[0])
    mac_flags=b/'CMakeFiles/MAC_UE_NR.dir/flags.make'
    inputs=[mac_flags if mac_flags.exists() else b/'compile_commands.json',b/'CMakeFiles/MAC_UE_NR.dir/link.txt']
    for token in recipe:
     if token.endswith('.o'):
      inputs.append(b/token if (b/token).exists() else xsai/token.split('.dir/',1)[1][:-2])
    fingerprint={str(p.resolve()):[p.stat().st_size,p.stat().st_mtime_ns] for p in inputs};fingerprint['compiler']=a.cc
    if not archive.exists() or not manifest.exists() or json.loads(manifest.read_text())!=fingerprint:
     objects=[]
     for token in recipe:
      if not token.endswith('.o'):continue
      obj=b/token
      if not obj.exists():
       rel=token.split('.dir/',1)[1][:-2];source=xsai/rel;obj=out/('mac-'+source.name+'.o');execute([a.cc,*flags(b/'CMakeFiles/MAC_UE_NR.dir/flags.make'),'-I'+str(source.parent),'-c',str(source),'-o',str(obj)])
      objects.append(str(obj))
     execute(['ar','rcs',str(archive),*objects]);save(manifest,fingerprint)
    arg=str(archive)
   result.append(arg)
  if '-c' in result:result.append('-UNDEBUG')
  result[0]=a.cc if '-c' in result else ('c++' if '++' in result[0] else result[0])
  return result
 if profile=='official':
  build_official(ROOT,xsai,b,out,a.cc,keys,execute,a.pmu)
  for key in keys:
   c['metric_pmu'][key]=a.pmu if key!='dft' else False
   ready[key]=source_signature(key)
  save(ready_file,ready);save(ROOT/'config.local.json',c)
  save(out/'provenance.json',dict(config=c,libdfts_sha256=sha(b/'libdfts.so'),sources={p.name:sha(p) for p in (ROOT/'src').iterdir() if p.is_file()}))
  return
 # validated above
 for key in dict.fromkeys(keys):
  if key=='dft':
   cc=expand(recipes[14]);i=cc.index('-o');cc[i+1]=str(out/'parallel_fft.o');cc=[str(ROOT/'src/parallel_fft.c') if x==str(ROOT/'src/adapter.c') else x for x in cc];execute(cc)
   execute(expand(recipes[15]))
  first,last=PAIRS[key];execute(expand(recipes[first]));link=expand(recipes[last])
  if key=='dft':link+=['-Wl,--no-as-needed','-lm','-Wl,--as-needed']
  execute(link)
  c['metric_pmu'][key]=a.pmu if key!='dft' else False
  if key=='dft':c['metric_pmu']['idft']=False
  ready[key]=source_signature(key);save(ready_file,ready);save(ROOT/'config.local.json',c)
  if key=='DL':print('DL uses your local libldpc.so; no private adaptive library is bundled.')
 save(ROOT/'config.local.json',c);save(out/'provenance.json',dict(config=c,libdfts_sha256=sha(b/'libdfts.so'),sources={p.name:sha(p) for p in (ROOT/'src').iterdir() if p.is_file()}))

def doctor(a):
 xsai=source_root(a.xsai);b=P(a.build).resolve() if a.build else xsai/'cmake_targets/ran_build/build';missing=[]
 a.cc=compiler(b,a.cc)
 arch=host_arch();profile=a.profile if a.profile!='auto' else ('xsai' if arch=='riscv64' else 'official')
 print('Architecture:',arch);print('Profile:',profile);print('Source:',xsai);print('Build:',b);print('Compiler:',shutil.which(a.cc))
 if (b/'libdfts.so').is_file():check_library_arch(b/'libdfts.so',arch)
 if not shutil.which(a.cc):missing.append(a.cc)
 for name,target in {'libdfts.so':'dfts','libMAC_UE_NR.a':'MAC_UE_NR','libSCHED_NR_UE_LIB.a':'SCHED_NR_UE_LIB','libL2_NR.a':'L2_NR','libnr_ue_phy_meas.a':'nr_ue_phy_meas','libplugins_plugins.a':'plugins_plugins'}.items():
  if profile=='official' and name=='libplugins_plugins.a' and cache_value(b,'ENABLE_PLUGINS') in ('OFF','0','FALSE'):
   print('NOTE: local CMake explicitly disables plugins; no GPU plugin archive required for CPU modules');continue
  found=(b/name).is_file() or any(b.glob('**/'+name));print(('OK ' if found else 'MISSING ')+name)
  if not found and name=='libMAC_UE_NR.a' and (b/'CMakeFiles/MAC_UE_NR.dir/link.txt').exists():print('  will construct a private archive inside benchmark/build');continue
  if not found:missing.append(name);print('  repair:',shlex.join(['cmake','--build',str(b),'--target',target,'-j','8']))
 if not (b/'CMakeFiles/PHY_NR.dir/flags.make').exists():print('NOTE: no flags.make; missing helper rebuilds require compile_commands.json (CMAKE_EXPORT_COMPILE_COMMANDS=ON). Complete Ninja builds may already contain the helpers.')
 if hasattr(os,'sched_getaffinity'):print('Allowed CPUs:',sorted(os.sched_getaffinity(0)))
 print('ASN.1: apt asn1c 0.9.28 lacks -gen-APER; follow your XSAI setup. Build required NR targets if unrelated LTE/SSE targets fail.')
 print('Neural requires an external ABI-compatible plugin plus native weights or a TensorRT engine; hardware requirements depend on that plugin.')
 if missing:raise BenchError('Preflight incomplete: '+', '.join(missing))
 print('Preflight passed; actual source/API compatibility is checked by build.')


def source_signature(key):
 sources={'estimation':['standalone.c'],'measurement':['standalone.c'],'symbol':['standalone.c'],'llr':['random_llr.c'],'neural':['random_neural.c'],'UL':['nr_ulschsim.c'],'DL':['nr_dlschsim.c'],'dft':['bench.c','adapter.c','parallel_fft.c']}
 files=[ROOT/'src'/name for name in sources[key]]+list((ROOT/'src').glob('*.h'))+[ROOT/'platform_build.py',ROOT/'bench.py',ROOT/'templates/commands.json']
 return {p.name:sha(p) for p in files}

def run(a):
 if not hasattr(os,'sched_getaffinity'):raise BenchError('Running requires 64-bit Linux')
 if a.seconds is not None and (not math.isfinite(a.seconds) or a.seconds<=0 or a.metric in ('UL','DL','dft','idft')):raise BenchError('--seconds is only for PUSCH/LLR/neural; use --trials for UL/DL, --iterations for DFT')
 if a.input_bank<1 or a.input_bank>16:raise BenchError('--input-bank must be 1..16')
 if a.metric in ('UL','DL') and a.input_bank!=1:raise BenchError('Input banks are not supported for UL/DL')
 if a.metric not in ('UL','DL') and any(v is not None for v in (a.rb,a.carrier_rb,a.mcs,a.symbols,a.snr,a.trials)):raise BenchError('Radio/trial parameters apply only to UL/DL')
 if a.metric not in ('dft','idft') and (a.verify!='strict' or a.iterations!=10000 or a.batches!=7):raise BenchError('Verification/iteration/batch options apply only to DFT/IDFT')
 for value,name,lo,hi in [(a.rb,'rb',1,106),(a.carrier_rb,'carrier-rb',1,106),(a.mcs,'mcs',0,28),(a.symbols,'symbols',1,13),(a.trials,'trials',1,100000),(a.iterations,'iterations',1,10000000),(a.batches,'batches',1,1000)]:
  if value is not None and not lo<=value<=hi:raise BenchError(f'--{name} must be in [{lo}, {hi}]')
 if (a.rb or 50)>(a.carrier_rb or 106):raise BenchError('--rb cannot exceed --carrier-rb')
 if not 0<=a.seed<=4294967295:raise BenchError('--seed must fit uint32')
 if a.snr is not None and not math.isfinite(a.snr):raise BenchError('--snr must be finite')
 c=config();b=P(c['build'])
 if c.get('architecture',host_arch())!=host_arch():raise BenchError('Build architecture differs from this host; rebuild locally')
 allowed=select_cpus(a.cpus,a.threads);cpus=','.join(map(str,allowed))
 binary=ROOT/'build'/NAMES[a.metric];key='dft' if a.metric=='idft' else a.metric
 ready_file=ROOT/'build/ready.json';ready=json.loads(ready_file.read_text()) if ready_file.exists() else {}
 if ready.get('_profile')!=c.get('profile') or not binary.exists() or any(ready.get(key,{}).get(name)!=digest for name,digest in source_signature(key).items()):raise BenchError('Missing/stale/failed binary; rebuild --only '+key)
 notes=[]
 serial=a.metric in ('estimation','measurement','llr') or (a.metric in ('dft','idft') and c.get('dft_backend')=='local-native-serial')
 if serial and a.threads>1:notes.append('This module/backend is serial; eight requested threads do not create eight-way parallelism.')
 if len(allowed)<a.threads and not serial:notes.append('Oversubscribed workers may reduce performance.')
 if a.metric=='neural':notes.append('Plugin controls parallelism; --threads does not parallelize a serial plugin.')
 for note in notes:print('NOTE:',note)
 n=0 if a.threads==1 or serial else 8
 warmup=a.warmup if a.warmup is not None else (10 if a.metric in ('UL','DL') else (10000 if a.metric in ('dft','idft') else (1000 if a.metric=='llr' else 20)))
 if warmup<0:raise BenchError('--warmup must be >= 0')
 env=os.environ.copy();env.update(LD_LIBRARY_PATH=str(b)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),RANDOM_SEED=str(a.seed),OAI_RNGSEED=str(a.seed),BENCH_CPU_LIST=cpus,RANDOM_CPU_START=str(min(allowed)),RANDOM_CPU_COUNT=str(len(allowed)),BENCH_INPUT_BANK=str(a.input_bank),BENCH_WARMUP=str(warmup))
 if a.metric in ('estimation','measurement','symbol'):args=[str(a.seed),str(n),str(a.seconds or 1.5)]
 elif a.metric=='llr':args=[str(a.seed),str(a.seconds or 1.5)]
 elif a.metric in ('UL','DL'):
  total=warmup+(a.trials or 100);snr=20. if a.snr is None else a.snr
  env.update(CODING_WORKERS=str(n),CODING_REPEATS=str(total),CODING_WARMUP=str(warmup))
  args=['-n',str(total) if a.metric=='UL' else '3','-s',str(snr),'-S',str(snr+0.01),'-r',str(a.rb or 50),'-R',str(a.carrier_rb or 106),'-m',str(a.mcs if a.mcs is not None else 14),'-l',str(a.symbols or 13)]
  if a.metric=='DL':args+=['-X','n' if n==0 else ','.join(['-1']*8)]
 elif a.metric=='neural':
  model=a.engine if a.neural_backend=='tensorrt' else a.weights
  if not a.library or not model:raise BenchError('Provide --library and '+('--engine for tensorrt' if a.neural_backend=='tensorrt' else '--weights for native'))
  lib=str(P(a.library).resolve());weights=str(P(model).resolve())
  if not P(lib).is_file() or not P(weights).is_file():raise BenchError('Plugin or weight file does not exist')
  env.update(NATIVE_TEST_LIB=lib,OAI_NRX_WEIGHTS=weights,OAI_NRX_OUT_SCALE='4',OAI_NRX_RX_SCALE='1',XSAI_RECEIVER_NATIVE_WEIGHTS=weights,XSAI_RECEIVER_LLR_GAIN='4',XSAI_RECEIVER_H_INTERP='1');args=[str(a.seed),str(a.seconds or 1.5)]
 else:
  env.update(COOP_EIGHT='1' if n else '0',BENCH_VERIFY=a.verify,BENCH_ITERATIONS=str(a.iterations),BENCH_BATCHES=str(a.batches));args=[a.metric,str(a.seed)]
 cmd=['taskset','-c',cpus,str(binary),*args]
 if a.launcher:
  if not P(a.launcher).is_file():raise BenchError('Launcher does not exist')
  cmd.insert(0,str(P(a.launcher).resolve()))
 changes={k:v for k,v in env.items() if os.environ.get(k)!=v}
 if a.print_command:print(shlex.join(cmd));print(json.dumps(changes,indent=2));return
 with benchmark_lock(wait=a.wait_lock),fixed_frequency(allowed,a.freq) as before:
  out=P(a.out).resolve() if a.out else ROOT/'results'/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+a.metric+'-'+uuid.uuid4().hex[:6]);out.mkdir(parents=True,exist_ok=False)
  meta=dict(status='running',metric=a.metric,threads_requested=a.threads,workers_configured=n,cpus=cpus,seed=a.seed,config=c,argv=cmd,env=changes,notes=notes,frequency_before=before,pmu=c.get('metric_pmu',{}).get(a.metric,False),kernel=subprocess.check_output(['uname','-a'],text=True).strip(),libdfts_sha256=sha(b/'libdfts.so'),warmup_calls=warmup,input_bank=a.input_bank,cache_protocol='fixed-input warm-cache steady state' if a.input_bank==1 else 'cycling pregenerated bank; not guaranteed cold-cache')
  if a.metric=='neural':
   meta.update(plugin_sha256=sha(a.library),model_sha256=sha(weights),neural_backend=a.neural_backend)
   meta['engine_sha256' if a.neural_backend=='tensorrt' else 'weights_sha256']=sha(weights)
   if a.neural_backend=='tensorrt':
    cfg=out/'plugins/neural_receiver/config';cfg.mkdir(parents=True)
    (cfg/'receiver_trt.config').write_text(weights+'\n')
    meta['notes'].append('Official TensorRT callback includes its stream synchronization. CPU cores exclude GPU work; --threads only constrains host CPUs.')
  save(out/'metadata.json',meta);p=None
  try:
   with (out/'run.log').open('w') as f:
    p=subprocess.Popen(cmd,cwd=out,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in p.stdout:print(line,end='',flush=True);f.write(line)
    rc=p.wait()
   meta['returncode']=rc
   if rc:raise BenchError(f'Benchmark failed ({rc}), see {out}/run.log')
   if a.metric=='neural':check_neural_log((out/'run.log').read_text(),a.neural_backend)
   if a.freq:check_frequency(allowed,a.freq)
   raw,parsed=records((out/'run.log').read_text(),warmup)
   if not parsed:raise BenchError('Missing result records')
   for row in raw:
    if row['record_type']=='RESULT' and 'verification_scope' not in row:row['verification_scope']='serial_same_implementation' if a.metric in ('estimation','measurement','symbol') else 'repeatability_same_implementation'
    elif row['record_type']=='CODING_UL':row['verification_scope']='random_llr_not_decode_correctness'
    elif row['record_type']=='CODING_DL':row['verification_scope']='repeatability_same_payload'
   save(out/'results.raw.json',raw);write_csv(out/'results.raw.csv',raw);save(out/'results.json',parsed);write_csv(out/'results.csv',parsed)
   report=summary(parsed);save(out/'summary.json',report);print(json.dumps(report,indent=2));meta['status']='passed'
  except BaseException as exc:meta['status']='failed';meta['error']=str(exc);raise
  finally:
   if p and p.poll() is None:p.terminate();p.wait(timeout=10)
   meta['frequency_after']=frequency_snapshot(allowed);save(out/'metadata.json',meta)
 print('Results:',out)



def sweep(a):
 specs=json.loads(P(a.file).read_text())
 if not isinstance(specs,list) or not specs:raise BenchError('Sweep file must be a nonempty JSON list')
 # Validate CLI syntax for every entry before executing any benchmark.
 parser=make_parser();commands=[]
 for spec in specs:
  if not isinstance(spec,dict) or 'metric' not in spec or 'threads' not in spec:raise BenchError('Each sweep entry needs metric and threads')
  argv=['run',str(spec['metric'])]
  for key,value in spec.items():
   if key=='metric':continue
   flag='--'+key.replace('_','-')
   if isinstance(value,bool):
    if value:argv.append(flag)
   elif value is not None:argv.extend([flag,str(value)])
  commands.append(parser.parse_args(argv))
 for args in commands:run(args)

def locked_build(a):
 with benchmark_lock():build(a)

def make_parser():
 ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='command',required=True)
 for name in ('doctor','build'):
  b=sp.add_parser(name);b.add_argument('--xsai','--srk','--source',dest='xsai',required=True);b.add_argument('--build');b.add_argument('--cc');b.add_argument('--profile',choices=['auto','xsai','official'],default='auto')
  if name=='build':b.add_argument('--only',default='all');b.add_argument('--pmu',action='store_true')
 r=sp.add_parser('run');r.add_argument('metric',choices=NAMES);r.add_argument('--threads',type=int,choices=[1,8],required=True);r.add_argument('--cpus');r.add_argument('--freq',type=int);r.add_argument('--seed',type=int,default=20261022);r.add_argument('--seconds',type=float);r.add_argument('--library');r.add_argument('--weights');r.add_argument('--engine');r.add_argument('--neural-backend',choices=['native','tensorrt'],default='native');r.add_argument('--launcher');r.add_argument('--out');r.add_argument('--print-command',action='store_true');r.add_argument('--wait-lock',action='store_true')
 for opt in ('warmup','rb','carrier-rb','mcs','symbols','trials'):r.add_argument('--'+opt,type=int)
 r.add_argument('--snr',type=float);r.add_argument('--input-bank',type=int,default=1);r.add_argument('--iterations',type=int,default=10000);r.add_argument('--batches',type=int,default=7)
 r.add_argument('--verify',choices=['strict','report','skip'],default='strict');r.add_argument('--skip-verify',dest='verify',action='store_const',const='skip')
 sw=sp.add_parser('sweep');sw.add_argument('file',help='JSON list of run argument dictionaries')
 return ap

def main():
 a=make_parser().parse_args()
 try: {'doctor':doctor,'build':locked_build,'run':run,'sweep':sweep}[a.command](a)
 except (BenchError,AssertionError,OSError,ValueError,KeyError) as exc:print('ERROR:',exc,file=sys.stderr);return 2
 return 0
if __name__=='__main__':sys.exit(main())
