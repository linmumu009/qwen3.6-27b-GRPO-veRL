"""Private judge-reason review and safe per-condition agreement diagnostic."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(x) for x in (a.out/'grading/reviews.private.jsonl').read_text().splitlines()]
    for r in rows:
        print(json.dumps(dict(id=r['id'],repeat=r['repeat'],mapping=r['mapping'],verdict=r.get('verdict')),ensure_ascii=False))

if __name__=='__main__':main()
