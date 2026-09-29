"""Reproduce the full study or verify the included exact experiment outputs."""
import argparse,subprocess,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--workers',type=int,default=8)
 p.add_argument('--verify-only',action='store_true',help='Check tests, exact certificates, and seeded sampling.')
 a=p.parse_args()
 for item in json.loads((ROOT/'data/sources.json').read_text()):
  path=ROOT/'data'/(item['name']+'.zip')
  if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError(f'Dataset checksum mismatch: {path.name}')
 commands=[['-m','unittest','discover','-s','tests','-v']]
 if not a.verify_only:commands += [['run_experiments.py','--workers',str(a.workers),'--force'],['evaluate.py'],['robustness.py']]
 commands += [['validate.py']]
 for cmd in commands:
  print('Running',' '.join(cmd),flush=True);subprocess.run([sys.executable,*cmd],cwd=ROOT,check=True)
 print('Reproduction checks passed.',flush=True)
if __name__=='__main__':main()
