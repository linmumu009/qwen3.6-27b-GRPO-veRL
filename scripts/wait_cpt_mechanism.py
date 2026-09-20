"""Wait for device availability without occupying or terminating other work."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cpt_stage_b import device_idle


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();status=a.out/'waiting.safe.json'
    def save(value):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(status)
    with (a.out/'wait.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.time()
        while time.time()-started<21600:
            result=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
            idle,used=device_idle(result.stdout)
            if result.returncode==0 and idle:
                save(dict(state='handing_to_fixed_coordinator',checked_at=time.time(),waiter_pid=os.getpid()))
                code=a.package/'scripts/run_cpt_mechanism.py'
                result=subprocess.run([sys.executable,str(code),'--package',str(a.package),'--out',str(a.out)])
                save(dict(state='coordinator_returned',returncode=result.returncode,checked_at=time.time()))
                raise SystemExit(result.returncode)
            save(dict(state='waiting_for_devices',checked_at=time.time(),memory_mb=used,waiter_pid=os.getpid(),
                      deadline=started+21600,training_started=False,other_processes_terminated=False))
            time.sleep(60)
        save(dict(state='device_wait_timeout',checked_at=time.time(),training_started=False))


if __name__=='__main__':main()
