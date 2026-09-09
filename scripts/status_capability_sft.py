"""Read-only compact SFT status; distinguish profiler warnings from progress."""
import json
from pathlib import Path
import re

out=Path('/opt/llin-capability-sft-cpt-20260909-01')
logs=list((out/'torchrun_logs').glob('*/attempt_0/*/stdout.log'))
lines=[line for path in logs for line in path.read_text(errors='replace').splitlines()]
progress=[line for line in lines if re.search(r'step:|train/loss|global_step|Train epoch',line) and len(line)<2500]
print(json.dumps(dict(status=(out/'status.txt').read_text().strip(),progress=progress[-3:],profiler_range_warnings=sum('range_end failed' in line for line in lines),tracebacks=sum('Traceback (most recent call last)' in line for line in lines)),indent=2))
