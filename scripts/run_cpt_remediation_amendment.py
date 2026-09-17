"""Fixed twenty-call amendment audit; preserve both versions rather than retry answers."""
import argparse
from pathlib import Path
from run_cpt_remediation_review import run, MODELS

AMENDMENT_SHA='c314bb5afed8bdaf71d774b89c58a9649459e7e1cfc2f8000715ed4642564a6d'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--model-manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--model-label',choices=list(MODELS),required=True)
    a=p.parse_args();run(a.cases,a.model_manifest,a.out,a.model_label,AMENDMENT_SHA,4)
