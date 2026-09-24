"""Dispatch one native detector or utility evaluator. No model is loaded for this help."""
import argparse,importlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROUTES={k:'wmmd_bench.methods.'+k+'.evaluate' for k in ['ctcc','evertracer','llmprint','huref','awm']}
ROUTES.update({k:'wmmd_bench.methods.'+k for k in ['pnfp','pnfp_cross','iseal','scw','scw_generate','reef','zeroprint']})
ROUTES['utility']='wmmd_bench.evaluation.utility'
def main():
    # Parse only the selector here, so METHOD --help reaches the method parser.
    p=argparse.ArgumentParser(description=__doc__,epilog='Pass METHOD --help for its frozen-asset interface; see docs/methods.md.',add_help=False)
    if len(sys.argv)==1 or sys.argv[1] in ('-h','--help'):
        p.add_argument('method',choices=sorted(ROUTES));p.print_help();return
    p.add_argument('method',choices=sorted(ROUTES))
    a,args=p.parse_known_args()
    sys.argv=[sys.argv[0]]+args
    importlib.import_module(ROUTES[a.method]).main()
if __name__=='__main__':main()
