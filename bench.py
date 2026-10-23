#!/usr/bin/env python3
"""Independent, relocatable XSAI module benchmark launcher (Linux/RISC-V)."""
import argparse,csv,datetime,fcntl,hashlib,json,os,pathlib,re,shlex,shutil,subprocess,sys,uuid,statistics
P=pathlib.Path;ROOT=P(__file__).resolve().parent
NAMES={'estimation':'test_pusch_estimation','measurement':'test_pusch_measurement','symbol':'test_pusch_symbol','llr':'test_llr_region','UL':'nr_ulschsim','DL':'nr_dlschsim','neural':'test_neural','dft':'bench-coop','idft':'bench-coop'}
PAIRS={'estimation':(0,1),'measurement':(2,3),'symbol':(4,5),'neural':(6,7),'llr':(8,9),'UL':(10,11),'DL':(12,13),'dft':(14,16)}
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def shell(cmd,cwd):
 q=subprocess.run(cmd,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 print(q.stdout,end='');return q
def config():return json.loads((ROOT/'config.local.json').read_text())
def flags(path):
 d={}
 for line in path.read_text().splitlines():
  if ' = ' in line:k,v=line.split(' = ',1);d[k]=shlex.split(v)
 return sum([d[k] for k in ['C_DEFINES','C_INCLUDES','C_FLAGS']],[])
def build(a):
 xsai=P(a.xsai).resolve();b=P(a.build).resolve() if a.build else xsai/'cmake_targets/ran_build/build';out=ROOT/'build';out.mkdir(exist_ok=True)
 assert (b/'libdfts.so').exists(),'Build local XSAI first; see README'
 previous=json.loads((ROOT/'config.local.json').read_text()) if (ROOT/'config.local.json').exists() else {}
 c=dict(xsai=str(xsai),build=str(b),src=str(ROOT/'src'),out=str(out),cc=a.cc,pmu=a.pmu,metric_pmu=previous.get('metric_pmu',{}) if previous.get('xsai')==str(xsai) else {})
 if not a.pmu:(out/'pmu_disabled.h').write_text('static inline void pmu_begin(void){}\nstatic inline void pmu_end(unsigned long n){(void)n;}\n')
 recipes=json.loads((ROOT/'templates/commands.json').read_text());log=[]
 def execute(cmd):
  log.append(cmd);q=shell(cmd,b);save(out/'build-commands.json',log);assert q.returncode==0,'Build failed; do not run stale binaries'
 def expand(cmd):
  result=[]
  for arg in cmd:
   if '{dependency}' in arg:
    name=arg.rsplit('/',1)[1];matches=list(b.rglob(name))
    if not matches:
     sources=list(xsai.rglob(name[:-2]));assert len(sources)==1,f'Cannot locate source for {name}'
     dep=out/name;execute([a.cc,*flags(b/'CMakeFiles/PHY_NR.dir/flags.make'),'-I'+str(sources[0].parent),'-c',str(sources[0]),'-o',str(dep)]);matches=[dep]
    arg=str(matches[0])
   for key in ['xsai','build','src','out']:arg=arg.replace('{'+key+'}',c[key])
   if arg.endswith('/pmu_cycles.h') and not a.pmu:arg=str(out/'pmu_disabled.h')
   if arg.endswith('/libMAC_UE_NR.a') and not P(arg).exists():
    archive=out/'libMAC_UE_NR.a'
    if not archive.exists():
     recipe=shlex.split((b/'CMakeFiles/MAC_UE_NR.dir/link.txt').read_text().splitlines()[0]);objects=[]
     for token in recipe:
      if not token.endswith('.o'):continue
      obj=b/token
      if not obj.exists():
       rel=token.split('.dir/',1)[1][:-2];source=xsai/rel;obj=out/('mac-'+source.name+'.o');execute([a.cc,*flags(b/'CMakeFiles/MAC_UE_NR.dir/flags.make'),'-I'+str(source.parent),'-c',str(source),'-o',str(obj)])
      objects.append(str(obj))
     execute(['ar','rcs',str(archive),*objects])
    arg=str(archive)
   result.append(arg)
  result[0]=a.cc if '-c' in result else ('c++' if '++' in result[0] else result[0])
  return result
 keys=list(PAIRS) if a.only=='all' else [x if x!='idft' else 'dft' for x in a.only.split(',')]
 for key in dict.fromkeys(keys):
  if key=='dft':
   cc=expand(recipes[14]);i=cc.index('-o');cc[i+1]=str(out/'parallel_fft.o');cc=[str(ROOT/'src/parallel_fft.c') if x==str(ROOT/'src/adapter.c') else x for x in cc];execute(cc)
   execute(expand(recipes[15]))
  first,last=PAIRS[key];execute(expand(recipes[first]));link=expand(recipes[last])
  if key=='dft':link+=['-Wl,--no-as-needed','-lm','-Wl,--as-needed']
  execute(link)
  c['metric_pmu'][key]=a.pmu if key!='dft' else False
  if key=='dft':c['metric_pmu']['idft']=False
  if key=='DL':print('DL uses your local libldpc.so; no private adaptive library is bundled.')
 save(ROOT/'config.local.json',c);save(out/'provenance.json',dict(config=c,libdfts_sha256=sha(b/'libdfts.so'),sources={p.name:sha(p) for p in (ROOT/'src').iterdir() if p.is_file()}))
def run(a):
 c=config();b=P(c['build']);cpus=a.cpus or ('7' if a.threads==1 else '0-7');allowed=[]
 for part in cpus.split(','):
  if '-' in part:lo,hi=map(int,part.split('-'));allowed.extend(range(lo,hi+1))
  else:allowed.append(int(part))
 assert len(set(allowed))>=a.threads,'CPU set too small';assert sorted(allowed)==list(range(min(allowed),max(allowed)+1)),'Use a contiguous CPU range'
 binary=ROOT/'build'/NAMES[a.metric];assert binary.exists(),'Build this metric first'
 env=os.environ.copy();env.update(LD_LIBRARY_PATH=str(b),RANDOM_SEED=str(a.seed),OAI_RNGSEED=str(a.seed),RANDOM_CPU_START=str(min(allowed)),RANDOM_CPU_COUNT=str(len(allowed)))
 n=0 if a.threads==1 else 8
 if a.metric in ['estimation','measurement','symbol']:args=[str(a.seed),str(n),str(a.seconds)]
 elif a.metric=='llr':args=[str(a.seed),str(a.seconds)]
 elif a.metric in ['UL','DL']:
  env.update(CODING_WORKERS=str(n),CODING_REPEATS='110');args=['-n','110' if a.metric=='UL' else '3','-s','20','-S','20.01','-r','50','-R','106','-m','14','-l','13']
  if a.metric=='DL':args+=['-X','n' if n==0 else ','.join(['-1']*8)]
 elif a.metric=='neural':
  assert a.library and a.weights,'Provide --library and --weights; they are not distributed';lib=str(P(a.library).resolve());weights=str(P(a.weights).resolve());assert P(lib).is_file() and P(weights).is_file()
  env.update(NATIVE_TEST_LIB=lib,OAI_NRX_WEIGHTS=weights,OAI_NRX_OUT_SCALE='4',OAI_NRX_RX_SCALE='1',XSAI_RECEIVER_NATIVE_WEIGHTS=weights,XSAI_RECEIVER_LLR_GAIN='4',XSAI_RECEIVER_H_INTERP='1');args=[str(a.seed),str(a.seconds)]
 else:env['COOP_EIGHT']='1' if n else '0';args=[a.metric,str(a.seed)]
 cmd=['taskset','-c',cpus,str(binary),*args]
 if a.launcher:cmd.insert(0,str(P(a.launcher).resolve()))
 if a.print_command:print(shlex.join(cmd));print(json.dumps({k:v for k,v in env.items() if os.environ.get(k)!=v},indent=2));return
 out=P(a.out).resolve() if a.out else ROOT/'results'/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+a.metric+'-'+uuid.uuid4().hex[:6]);out.mkdir(parents=True,exist_ok=False)
 lock=open('/tmp/k3-module-benchmark.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 policy=P(f'/sys/devices/system/cpu/cpu{min(allowed)}/cpufreq');original=None;p=None
 meta=dict(metric=a.metric,threads=a.threads,cpus=cpus,seed=a.seed,config=c,argv=cmd,env={k:v for k,v in env.items() if os.environ.get(k)!=v},frequency_before_khz=(policy/'scaling_cur_freq').read_text().strip() if policy.exists() else None,pmu=c.get('metric_pmu',{}).get(a.metric,False),kernel=subprocess.check_output(['uname','-a'],text=True).strip(),libdfts_sha256=sha(b/'libdfts.so'))
 try:
  if a.freq:
   assert str(a.freq) in (policy/'scaling_available_frequencies').read_text().split(),'Unsupported frequency';original=(policy/'scaling_setspeed').read_text().strip();(policy/'scaling_setspeed').write_text(str(a.freq)+'\n');assert (policy/'scaling_cur_freq').read_text().strip()==str(a.freq)
  save(out/'metadata.json',meta)
  with (out/'run.log').open('w') as f:
   p=subprocess.Popen(cmd,cwd=out,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   for line in p.stdout:print(line,end='',flush=True);f.write(line)
   rc=p.wait()
  assert rc==0,f'Benchmark failed ({rc}), see run.log'
  if a.freq:assert (policy/'scaling_cur_freq').read_text().strip()==str(a.freq),'Frequency changed during measurement'
  text=(out/'run.log').read_text();parsed=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in text.splitlines() if line.startswith(('[RESULT]','[CODING_UL]','[CODING_DL]','[PMU_CYCLES]'))];assert parsed,'Missing result records'
  save(out/'results.json',parsed)
  with (out/'results.csv').open('w',newline='') as f:
   columns=sorted(set().union(*(z.keys() for z in parsed)));w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(parsed)
 finally:
  if p and p.poll() is None:p.terminate();p.wait(timeout=10)
  if original is not None:(policy/'scaling_setspeed').write_text(original+'\n')
  meta['frequency_after_khz']=(policy/'scaling_cur_freq').read_text().strip() if policy.exists() else None;save(out/'metadata.json',meta)
 print('Results:',out)
def main():
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='command',required=True)
 b=sp.add_parser('build');b.add_argument('--xsai',required=True);b.add_argument('--build');b.add_argument('--cc',default='gcc-16');b.add_argument('--only',default='all');b.add_argument('--pmu',action='store_true')
 r=sp.add_parser('run');r.add_argument('metric',choices=NAMES);r.add_argument('--threads',type=int,choices=[1,8],required=True);r.add_argument('--cpus');r.add_argument('--freq',type=int);r.add_argument('--seed',type=int,default=20261022);r.add_argument('--seconds',type=float,default=1.5);r.add_argument('--library');r.add_argument('--weights');r.add_argument('--launcher');r.add_argument('--out');r.add_argument('--print-command',action='store_true')
 a=ap.parse_args();build(a) if a.command=='build' else run(a)
if __name__=='__main__':main()
