"""Table 8 manuscript protocol: 10 independently generated random instances per scale."""
from pathlib import Path
import csv, statistics, copy
from src import table5_exact_validation as exact
from src import tqalns_benchmark_core as core

OUT=Path(__file__).resolve().parents[1]/'results'/'table8_robustness'
OUT.mkdir(parents=True,exist_ok=True)
INSTANCE_SEEDS=list(range(1,11))
SEARCH_SEEDS=list(range(1001,1011))

def _mean(xs): return statistics.mean(xs) if xs else None
def _sd(xs): return statistics.stdev(xs) if len(xs)>=2 else (0.0 if xs else None)

def _configure(scale):
    if scale=='medium':
        core.N_CANDIDATE_PLATFORMS=15; core.N_MONITORING_POINTS=50; core.N_HAZARDS=3; core.N_NOFLY_ZONES=3
        core.MIN_OPEN_PLATFORMS=4; core.MAX_OPEN_PLATFORMS=4; core.MAX_TOTAL_UAVS=10
    elif scale=='large':
        core.N_CANDIDATE_PLATFORMS=18; core.N_MONITORING_POINTS=300; core.N_HAZARDS=20; core.N_NOFLY_ZONES=2
        core.MIN_OPEN_PLATFORMS=14; core.MAX_OPEN_PLATFORMS=14; core.MAX_TOTAL_UAVS=14
    else:
        raise ValueError(scale)

def _core_result(scale,iseed,sseed):
    _configure(scale)
    base=core.build_instance(iseed)
    algs=[
        ('Greedy',lambda x:core.greedy_only(x,sseed)),
        ('ALNS-only',lambda x:core.alns_only(x,sseed,core.HEURISTIC_ITERS)),
        ('ALNS–SA',lambda x:core.alns_sa_only(x,sseed,core.HEURISTIC_ITERS)),
        ('TQ-ALNS',lambda x:core.tq_alns(x,sseed,core.HEURISTIC_ITERS,trace_file=None)),
    ]
    rows=[]
    for name,fn in algs:
        inst=copy.deepcopy(base); sol=fn(inst); core.compute_solution_obj(inst,sol)
        rows.append(dict(scale=scale,algorithm=name,objective=sol.obj,feasible=int(bool(sol.feasible)),
                         coverage=100.0*sol.coverage_ratio,open_platforms=len(core.open_set(sol)),runtime_s=sol.runtime_s))
    return rows

def main():
    raw=[]
    # Small-scale: Table-3 nominal scale, MIP/benchmark vs TQ-ALNS.
    for rid,(iseed,sseed) in enumerate(zip(INSTANCE_SEEDS,SEARCH_SEEDS),1):
        spec=exact.InstanceSpec(instance_id=f'R-S{rid:02d}',scale='small',n_points=16,n_candidate_platforms=7,
                                n_hazard_sources=3,n_no_fly_zones=2,n_uav_types=3,open_platforms_fixed=3,seed=7000+iseed)
        inst=exact.build_instance(spec)
        g=exact.solve_gurobi_mip(inst)
        a=exact.solve_alns_qsa(inst,sseed)
        for label,res in [('MIP / benchmark',g),('TQ-ALNS',a)]:
            raw.append(dict(Scale='Small-scale',Algorithm=label,Run=rid,Instance_seed=7000+iseed,Search_seed=sseed,
                            Objective=res.get('obj'),Feasible=int(bool(res.get('feasible'))),Coverage_pct=100.0*float(res.get('coverage') or 0),
                            Open_platforms=len(res.get('open_platforms') or []),Runtime_s=res.get('time_sec')))
    # Medium and large scales: four methods.
    for scale,label in [('medium','Medium-scale'),('large','Large-scale')]:
        for rid,(iseed,sseed) in enumerate(zip(INSTANCE_SEEDS,SEARCH_SEEDS),1):
            rows=_core_result(scale,iseed,sseed)
            for r in rows:
                raw.append(dict(Scale=label,Algorithm=r['algorithm'],Run=rid,Instance_seed=iseed,Search_seed=sseed,
                                Objective=r['objective'],Feasible=r['feasible'],Coverage_pct=r['coverage'],
                                Open_platforms=r['open_platforms'],Runtime_s=r['runtime_s']))

    # Per-run BKS and deviation.
    for scale in sorted(set(r['Scale'] for r in raw)):
        for run in range(1,11):
            rr=[r for r in raw if r['Scale']==scale and r['Run']==run and r['Feasible']==1 and r['Objective'] is not None]
            bks=min((r['Objective'] for r in rr),default=None)
            for r in [x for x in raw if x['Scale']==scale and x['Run']==run]:
                r['Run_BKS']=bks
                r['Dev_from_run_BKS_pct']=None if bks in (None,0) or not r['Feasible'] else 100.0*(r['Objective']-bks)/abs(bks)

    rawp=OUT/'table8_robustness_raw.csv'
    with rawp.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0])); w.writeheader(); w.writerows(raw)

    summary=[]
    order=[('Small-scale','MIP / benchmark'),('Small-scale','TQ-ALNS'),
           ('Medium-scale','Greedy'),('Medium-scale','ALNS-only'),('Medium-scale','ALNS–SA'),('Medium-scale','TQ-ALNS'),
           ('Large-scale','Greedy'),('Large-scale','ALNS-only'),('Large-scale','ALNS–SA'),('Large-scale','TQ-ALNS')]
    for scale,alg in order:
        rows=[r for r in raw if r['Scale']==scale and r['Algorithm']==alg]
        fr=[r for r in rows if r['Feasible']==1 and r['Objective'] is not None]
        objs=[r['Objective'] for r in fr]; s=_sd(objs); mo=_mean(objs)
        summary.append({
            'Instance size':scale,'Algorithm':alg,'Runs':len(rows),'Valid runs':len(fr),'Mean objective value':mo,
            'Mean dev. from run BKS (%)':_mean([r['Dev_from_run_BKS_pct'] for r in fr if r['Dev_from_run_BKS_pct'] is not None]),
            'Std. Dev.':s,'CV (%)':None if mo in (None,0) or s is None else 100.0*s/mo,
            'Mean coverage rate (%)':_mean([r['Coverage_pct'] for r in rows]),
            'Mean open platforms':_mean([r['Open_platforms'] for r in fr]),
            'Mean time (s)':_mean([r['Runtime_s'] for r in rows if r['Runtime_s'] is not None]),
            'Feasibility rate (%)':100.0*len(fr)/len(rows) if rows else 0.0,
        })
    sp=OUT/'table8_robustness_summary.csv'
    with sp.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print('Saved:',rawp); print('Saved:',sp)

if __name__=='__main__': main()
