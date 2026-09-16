"""Run a frozen coverage packet once, after a named draft job releases the devices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def predecessor_ready(state):
    if state.get('status')=='failed':raise RuntimeError('predecessor failed; inspect before continuing')
    return state.get('status')=='drafted_pending_independent_audit'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predecessor',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--sha',required=True)
    p.add_argument('--expected-cases',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--runner',type=Path,required=True);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--timeout-seconds',type=int,default=14400);a=p.parse_args();os.umask(0o077)
    assert hashlib.sha256(a.cases.read_bytes()).hexdigest()==a.sha
    assert not a.out.exists() and not a.state.exists()
    def save(status,**kw):
        a.state.write_text(json.dumps(dict(status=status,training_running=False,output=str(a.out),**kw),indent=2)+'\n')
    save('waiting_for_draft_completion',predecessor=str(a.predecessor))
    deadline=time.monotonic()+a.timeout_seconds
    try:
        while time.monotonic()<deadline:
            try:state=json.loads(a.predecessor.read_text())
            except (FileNotFoundError,json.JSONDecodeError):state={}
            if predecessor_ready(state):
                info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True).stdout
                if info.count('No running processes found in NPU')==8:break
                save('waiting_for_idle_devices')
            time.sleep(30)
        else:raise TimeoutError('coverage queue deadline reached without idle devices')
        save('starting_coverage_probe',devices_idle_before_launch=True)
        child=subprocess.Popen(['bash',str(a.runner),'--cases',str(a.cases),'--sha',a.sha,
            '--expected-cases',str(a.expected_cases),'--out',str(a.out)])
        save('coverage_probe_running',pid=child.pid)
        code=child.wait()
        if code:raise RuntimeError('coverage probe exited '+str(code))
        save('coverage_completed_pending_verification')
    except BaseException as e:
        save('queue_failed',error=type(e).__name__,detail=str(e));raise


if __name__=='__main__':main()
