"""Optional on-device regression; requires a successful build and local XSAI."""
import argparse
import json
import os
import signal
import pathlib
import subprocess
import sys

root=pathlib.Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--library-single')
parser.add_argument('--library-eight')
parser.add_argument('--weights')
args=parser.parse_args()
outcomes=[]

def check(label, argv, expected=0):
    child=subprocess.Popen([sys.executable,str(root/'bench.py'),*argv],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    try:
        stdout,stderr=child.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL)
        child.communicate()
        raise
    proc=subprocess.CompletedProcess(child.args,child.returncode,stdout,stderr)
    record={'case':label,'returncode':proc.returncode,'expected':expected}
    if proc.returncode!=expected:
        print(proc.stdout[-4000:]);print(proc.stderr[-2000:]);raise RuntimeError(record)
    if expected==0 and 'Results:' in proc.stdout:
        output=pathlib.Path(proc.stdout.rsplit('Results:',1)[1].strip())
        record['result_dir']=str(output)
        rows=json.loads((output/'results.json').read_text())
        if any(row['phase']=='warmup' for row in rows):raise RuntimeError('Warmup leaked into measured results')
        if argv[1] in ('UL','DL') and len(rows)!=3:raise RuntimeError(f'Expected 3 measured codec records, got {len(rows)}')
        if argv[1] in ('dft','idft') and rows[0]['verification']!='passed':raise RuntimeError('Strict DFT check did not pass')
    outcomes.append(record);print('PASS',label,flush=True)

for metric in ('estimation','measurement','symbol','llr','UL','DL','dft','idft','neural'):
    for threads in (1,8):
        command=['run',metric,'--threads',str(threads)]
        if metric in ('UL','DL'):command+=['--trials','3','--warmup','2']
        elif metric in ('dft','idft'):command+=['--iterations','200','--batches','2','--warmup','20','--input-bank','2']
        else:command+=['--seconds','0.12','--input-bank','2','--warmup','2']
        if metric=='neural':
            lib=args.library_single if threads==1 else args.library_eight
            if not lib or not args.weights:print('SKIP neural: provide compatible plugin and weights');continue
            command+=['--library',lib,'--weights',args.weights]
        check(metric+'-'+str(threads),command)
check('sparse-cpus',['run','symbol','--threads','8','--cpus','0,2,4,6','--seconds','0.12'])
check('unavailable-cpu',['run','llr','--threads','1','--cpus','999'],2)
check('invalid-duration',['run','UL','--threads','1','--seconds','1'],2)
check('invalid-frequency',['run','llr','--threads','1','--freq','123'],2)
(root/'smoke-results.json').write_text(json.dumps(outcomes,indent=2))
print('All',len(outcomes),'checks passed')
