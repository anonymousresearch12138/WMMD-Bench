"""Standard-library checks of release data, source syntax and local links."""
import ast,csv,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    checks=[]
    config=json.loads((ROOT/'configs/benchmark.json').read_text())
    assert config['main_3b']['training_seed']==42
    assert config['transformation']['seed']==config['7b']['seed']==42
    assert len(config['methods'])==10
    checks.append('ten methods and single training seed 42')
    with (ROOT/'results/ownership_3b_seed42.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==10
    expected={'PN-FP':('98','72','110'),'LLMPrint':('0.82','0.785','0.86')}
    for row in rows:
        if row['method'] in expected:assert tuple(row[k] for k in ['same_lineage_ba','same_lineage_bb','same_lineage_bc'])==expected[row['method']]
    checks.append('ownership schema, historical counts and native units')
    tr=json.loads((ROOT/'results/active_trajectories_seed42.json').read_text())
    assert len(tr)==5 and sum(map(len,tr.values()))==55
    for values in tr.values():
        assert [r['step'] for r in values]==[0,25,50,100,250,500,1000,2000,4000,6000,7500]
        assert all(r.get('seed',42)==42 for r in values)
    assert tr['pnfp'][-1]['detected']==99
    checks.append('55 observations; historical Ba and reconstructed trajectory remain distinct')
    seven=json.loads((ROOT/'results/extension_7b_seed42.json').read_text())['rows']
    assert [r['hits'] for r in seven if r['method']=='PN-FP']==[804,53,71,144]
    checks.append('7B representative result scope')
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    forbidden=re.compile(r'(?:training[_ -]?seeds?\s*[=:]\s*\[?\{?42\s*,\s*43|three[- ]seed|mean\s*±\s*std|seed[_ -](?:43|44)\b)',re.I)
    for p in files:
        if p.suffix=='.py':ast.parse(p.read_text('utf-8-sig'),filename=str(p.relative_to(ROOT)))
        if p.suffix in ['.json','.csv','.md','.yaml','.yml']:
            text=p.read_text('utf-8-sig')
            assert not forbidden.search(text),p.relative_to(ROOT)
        if p.suffix=='.md':
            for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^)]+)\)',p.read_text('utf-8')):
                if '://' in target or target.startswith('#'):continue
                assert (p.parent/target.split('#')[0]).exists(),(p.relative_to(ROOT),target)
    checks.append('Python syntax, local documentation links, seed-claim scan')
    print(json.dumps({'status':'PASS','checks':checks,'scientific_execution':'NOT_RUN'},indent=2))
if __name__=='__main__':main()
