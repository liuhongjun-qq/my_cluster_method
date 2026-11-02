#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare original HoSig and HoSig+Louvain batch reports.
Usage: python compare_hosig_fusion.py --hosig "report(2).json" --fusion "report(1).json" --output comparison_results
Only Python standard library is required.
"""
import argparse, csv, json, math, statistics
from pathlib import Path

METRICS = ('modularity', 'ari', 'nmi', 'ch', 'silhouette')

def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

def avg(values):
    values = [v for v in values if finite(v)]
    return statistics.mean(values) if values else None

def analyze(hosig, fusion):
    if hosig['total_events'] != fusion['total_events']:
        raise ValueError('The reports have different total_events')
    hs = {int(b['batch_id']):b for b in hosig['batches']}
    fs = {int(b['batch_id']):b for b in fusion['batches']}
    if hs.keys() != fs.keys():
        raise ValueError('Batch IDs do not match')
    rows=[]
    for bid in sorted(hs):
        h,f=hs[bid],fs[bid]; hm=h['metrics']; fm=f['evaluation']; hd=h['diagnostics']
        if (hd['n_events'] != f['events']or hm['truth_label_counts'] != fm['truth_label_counts']):
            raise ValueError(f'Batch {bid}: event counts or truth labels do not match')
        r={'batch_id':bid, 'events':f['events'],'truth_classes':fm['truth_class_count'],
           'hosig_clusters':hm['n_clusters'], 'fusion_communities':fm['n_communities'],
           'graph_edges':f['undirected_edges'], 'hosig_seconds':hd['timings_seconds']['total'],
           'fusion_seconds':f['seconds']}
        for metric in METRICS:
            x,y=hm.get(metric),fm.get(metric)
            r['hosig_'+metric] = x
            r['fusion_'+metric] = y
            r['delta_'+metric] = y-x if finite(x) and finite(y) else None
        r['ch_equals_one_suspected'] = (hm.get('ch')==1 or fm.get('ch')==1)
        rows.append(r)
    return rows

def summarize(rows):
    result={}
    for scope,selected in [('all',rows),('multiclass',[r for r in rows if r['truth_classes']>=2])]:
        sec={'batch_count':len(selected),'events':sum(r['events'] for r in selected)}
        for m in METRICS:
            h=[r['hosig_'+m] for r in selected]
            f=[r['fusion_'+m] for r in selected]
            valid=[r for r in selected if finite(r['hosig_'+m]) and finite(r['fusion_'+m])]
            stats={'hosig_mean_available':avg(h),'fusion_mean_available':avg(f),
                   'hosig_valid_n':sum(map(finite,h)),'fusion_valid_n':sum(map(finite,f)),
                   'paired_n':len(valid),'paired_hosig_mean':avg([r['hosig_'+m] for r in valid]),
                   'paired_fusion_mean':avg([r['fusion_'+m] for r in valid]),
                   'paired_delta_mean':avg([r['delta_'+m] for r in valid]),
                   'fusion_wins':sum(r['delta_'+m]>1e-10 for r in valid),
                   'hosig_wins':sum(r['delta_'+m]<-1e-10 for r in valid),
                   'ties':sum(abs(r['delta_'+m])<=1e-10 for r in valid)}
            sec[m]=stats
        for col in ['hosig_clusters','fusion_communities','hosig_seconds','fusion_seconds']:
            sec[col+'_mean']=avg([r[col] for r in selected])
        result[scope]=sec
    result['cautions']=[
      'All means are equally weighted per batch, NOT whole-dataset global metrics.',
      'Original HoSig CH/Silhouette are undefined when there is only one cluster; null is not zero.',
      'CH=1.0 appears repeatedly: inspect whether it is a fallback/placeholder before scholarly interpretation.',
      'Single-truth-class batches do not meaningfully evaluate type separation by ARI/NMI.',
      'Runtime scopes differ: original HoSig total includes signature extraction; fusion timing may cover other stages.',
      'IIRs are not compared: signature-based entity differs from fusion centroid-based entity.',
      'Same edge counts and truth counts do not prove adjacency weights or sample ordering are identical.'
    ]
    return result

def stringify(v):
    return '' if v is None else v

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hosig',required=True,help='Original HoSig report.json')
    p.add_argument('--fusion',required=True,help='Fusion report.json')
    p.add_argument('--output',default='comparison_results',help='Output directory')
    args=p.parse_args(); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    with open(args.hosig,encoding='utf-8') as fp: h=json.load(fp)
    with open(args.fusion,encoding='utf-8') as fp: f=json.load(fp)
    rows=analyze(h,f);summary=summarize(rows)
    with (out/'comparison_46_batches.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows([{k:stringify(v) for k,v in r.items()} for r in rows])
    with (out/'comparison_summary.json').open('w',encoding='utf-8') as fp:json.dump(summary,fp,ensure_ascii=False,indent=2)
    with (out/'comparison_summary.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        w=csv.writer(fp);w.writerow(['scope','metric','hosig_mean_available','hosig_valid_n','fusion_mean_available','fusion_valid_n','paired_n','paired_hosig_mean','paired_fusion_mean','paired_delta_mean','fusion_wins','hosig_wins','ties'])
        for scope in ['all','multiclass']:
            for m in METRICS:
                v=summary[scope][m];w.writerow([scope,m]+[stringify(v[k]) for k in ['hosig_mean_available','hosig_valid_n','fusion_mean_available','fusion_valid_n','paired_n','paired_hosig_mean','paired_fusion_mean','paired_delta_mean','fusion_wins','hosig_wins','ties']])
    print('Compared',len(rows),'batches,',sum(r['events'] for r in rows),'events')
    for scope in ['all','multiclass']:
        print('\n',scope,'batches:',summary[scope]['batch_count'])
        for m in METRICS:
            a=summary[scope][m];print(f"{m:12s} H={a['hosig_mean_available']!s:>19s} F={a['fusion_mean_available']!s:>19s} paired={a['paired_n']:2} wins={a['fusion_wins']}/{a['hosig_wins']}/{a['ties']}")
    print('\nSaved to:',out.resolve())
if __name__=='__main__':main()
