"""Wait for device availability without occupying or terminating other work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cpt_stage_b import device_idle


def stable_idle(previous_since, now, idle):
    """Require three spaced checks spanning >=120s; busy resets the window."""
    since=(now if previous_since is None else previous_since) if idle else None
    return since, since is not None and now-since>=120


def main():
    import fcntl
    p=argparse.ArgumentParser()
    p.add_argument('--package',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();status=a.out/'waiting.safe.json'
    def save(value):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(status)
    with (a.out/'wait.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.time();idle_since=None
        while time.time()-started<21600:
            result=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
            idle,used=device_idle(result.stdout)
            idle_since,ready=stable_idle(idle_since,time.time(),result.returncode==0 and idle)
            if ready:
                save(dict(state='handing_to_fixed_coordinator',checked_at=time.time(),waiter_pid=os.getpid()))
                code=a.package/'scripts/run_cpt_mechanism.py'
                result=subprocess.run([sys.executable,str(code),'--package',str(a.package),'--out',str(a.out)])
                save(dict(state='coordinator_returned',returncode=result.returncode,checked_at=time.time()))
                raise SystemExit(result.returncode)
            save(dict(state='waiting_for_devices',checked_at=time.time(),memory_mb=used,waiter_pid=os.getpid(),
                      deadline=started+21600,idle_since=idle_since,required_idle_seconds=120,
                      training_started=False,other_processes_terminated=False))
            time.sleep(60)
        save(dict(state='device_wait_timeout',checked_at=time.time(),training_started=False))


if __name__=='__main__':main()
