"""Score the predeclared baseline comparison on identical frozen validation rows."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
GATE = Path('/private/tmp/tcn-signal-date-holdout-20261127/daily')
REV = Path('/private/tmp/tcn-vs-five-minute-reversal-20261127')
TRAIN = Path('/private/tmp/tcn-clock-exclusive-20261127/gate/report.json')
PLAN = json.loads((HERE / 'evaluation_plan.json').read_text())
REFERENCE = {r['date']:r for r in json.loads((REV / 'rank_comparison.json').read_text())['daily']}
ARMS = ('reversal','linear','mlp','tcn')

def to_daily_ranks(labels, scores, axis):
    count = np.isfinite(labels).sum(axis=axis)
    mean = (count + 1) / 2
    def centered(arr):
        ranks = pd.DataFrame(arr).rank(axis=axis, method='average').to_numpy()
        return ranks - (mean if axis==0 else mean[:,None])
    target = centered(labels)
    norm_y = np.nansum(target**2,axis=axis)
    results={}
    for name, arr in scores.items():
        ranks = centered(arr)
        norm = np.sqrt(norm_y * np.nansum(ranks**2,axis=axis))
        coef = np.full(norm.shape, np.nan)
        np.divide(np.nansum(ranks*target,axis=axis), norm, out=coef, where=(count>=20)&(norm>0))
        results[name]=coef
    common = np.all(np.isfinite(list(results.values())),axis=0)
    return {name:float(x[common].mean()) for name,x in results.items()},int(common.sum())

def verify_keys(left,right):
    for k in ('code','t'):
        assert left[k].combine_chunks().equals(right[k].combine_chunks()), k

def top_one(scores, col, ids, y):
    order=np.argsort(col,kind='stable')
    periods=np.split(order,np.flatnonzero(np.diff(col[order]))+1)
    hold={name:np.full(ids[-1]+1,-10000,dtype=int) for name in scores}
    picked={name:[] for name in scores}
    for idx in periods:
        now=col[idx[0]]; k=max(1,len(idx)//100)
        for name,score in scores.items():
            allowed=idx[now-hold[name][ids[idx]]>=5]
            assert len(allowed)>=k
            # Canonically ordered tickers within each official date, used only to break equal scores.
            chosen=allowed[np.lexsort((ids[allowed],-score[allowed]))[:k]]
            hold[name][ids[chosen]]=now
            picked[name].extend(chosen.tolist())
    res={}
    for name,idx in picked.items():
        raw=y[np.array(idx)]
        valid=raw[np.isfinite(raw)]
        gross=np.expm1(valid)*10000
        res[name]=dict(selected=len(raw),valid=len(valid),gross_sum_bp=float(gross.sum()),
                       daily_mean_gross_bp=float(gross.mean()))
    return res

def analyze(day, file=None, demo=False):
    old=pq.read_table(GATE/(day+'.parquet'),columns=['code','t','p16','label','label_valid'])
    quote=pq.read_table(REV/'daily'/(day+'.parquet'),columns=['code','t','past_return','past_valid','y5_natural_valid'])
    verify_keys(old,quote)
    assert quote['past_valid'].to_numpy().all()
    score={'reversal':-quote['past_return'].to_numpy(), 'tcn':old['p16'].to_numpy()}
    if demo:
        score['mlp']=score['tcn']
        score['linear']=score['reversal']
    else:
        new=pq.read_table(file)
        verify_keys(old,new)
        if 'label_valid' in new.column_names:
            assert np.array_equal(new['label_valid'].to_numpy(),old['label_valid'].to_numpy())
        if 'label' in new.column_names:
            assert np.array_equal(new['label'].to_numpy(),old['label'].to_numpy(),equal_nan=True)
        for name in ('linear','mlp'):score[name]=new['p_'+name].to_numpy()
    for name,p in score.items(): assert np.isfinite(p).all(),(day,name)
    valid=old['label_valid'].to_numpy()
    y=old['label'].to_numpy()
    assert np.array_equal(valid,quote['y5_natural_valid'].to_numpy()) and np.array_equal(valid,np.isfinite(y))
    codes=old['code'].to_numpy()
    m=(old['t'].cast('int64').to_numpy()//60000+480)%1440
    assert np.all((codes[1:]>codes[:-1])|((codes[1:]==codes[:-1])&(m[1:]>m[:-1])))
    ids=np.r_[True,codes[1:]!=codes[:-1]].cumsum()-1
    times,j=np.unique(m,return_inverse=True)
    size=(int(ids[-1])+1,len(times))
    def dense(arr):
        a=np.full(size,np.nan);a[ids[valid],j[valid]]=arr[valid];return a
    label=dense(y)
    models={name:dense(score[name]) for name in ARMS}
    cs,ncs=to_daily_ranks(label,models,axis=0)
    ts,nts=to_daily_ranks(label,models,axis=1)
    ref=REFERENCE[day]
    for k,n in [('tcn','tcn_rank_ic'),('reversal','reversal_rank_ic')]:
        assert abs(cs[k]-ref[n])<1e-10, (day,k,cs[k],ref[n])
    assert ncs==ref['paired_minutes'] and ref['paired_rows']==int(valid.sum())
    tables=json.loads(TRAIN.read_text())['label_scale_statistics']
    scales=np.full(1440,np.nan,dtype=np.float32)
    for row in tables:scales[row['minute']]=row['std']
    assert np.isfinite(scales[times]).all()
    vscale=scales[m[valid]]
    huber={}
    for k,p in {'zero':np.zeros(len(y)), **{name:score[name] for name in ARMS if name!='reversal'}}.items():
        e=np.abs((y[valid]-p[valid])/vscale)
        huber[k]=float(np.where(e<=1,0.5*e*e,e-0.5).sum())
    return dict(day=day,candidates=len(y),valid=int(valid.sum()),cs_minute_count=ncs,ts_stock_count=nts,
                cross_section_ic=cs,within_stock_ic=ts,huber_sum=huber,
                top_one_percent=top_one(score,m,ids,y))

def aggregate(rows):
    total=sum(x['valid'] for x in rows)
    result={'dates':len(rows),'candidates':sum(x['candidates'] for x in rows),'valid':total,
            'cs_minute_count':sum(x['cs_minute_count'] for x in rows),
            'ts_stock_count':sum(x['ts_stock_count'] for x in rows)}
    for field in ('cross_section_ic','within_stock_ic'):
        panel={k:np.array([x[field][k] for x in rows]) for k in ARMS}
        result[field]={k:float(v.mean()) for k,v in panel.items()}
        rng=np.random.default_rng(20261127)
        n=len(rows);starts=rng.integers(n,size=(10000,(n+4)//5))
        draws=((starts[:,:,None]+np.arange(5))%n).reshape(10000,-1)[:,:n]
        stats={}
        for k in ('reversal','linear','mlp'):
            delta=panel['tcn']-panel[k]
            ci=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist()
            stats[k]=dict(mean_tcn_minus_comparator=float(delta.mean()),wins=int((delta>0).sum()),
                          ties=int((delta==0).sum()),paired_date_block95=ci)
        result[field+'_tcn_comparison']=stats
    result['huber']={k:sum(r['huber_sum'][k] for r in rows)/total for k in rows[0]['huber_sum']}
    top={}
    for k in ARMS:
        xs=[r['top_one_percent'][k] for r in rows]
        n=sum(x['valid'] for x in xs); gross=sum(x['gross_sum_bp'] for x in xs)/n
        top[k]=dict(n_selected=sum(x['selected'] for x in xs),valid=n,pooled_gross_bp=gross,
                    pooled_minus_10bp=gross-10,equal_date_gross_bp=float(np.mean([x['daily_mean_gross_bp'] for x in xs])))
    result['top_one_percent']=top
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-with-cached-only',action='store_true')
    parser.add_argument('--wait-for-seconds',type=int,default=3600)
    args=parser.parse_args()
    days=PLAN['evaluation_dates']
    if args.verify_with_cached_only:
        demo=analyze(days[0],demo=True)
        print('Reference checks passed for both earlier factors.',demo['day'],demo['cross_section_ic'])
        return
    dest=HERE/'scored_dates';dest.mkdir(exist_ok=True)
    rows=[]
    for n,day in enumerate(days,1):
        f=HERE/'predictions'/(day+'.parquet')
        limit=time.monotonic()+args.wait_for_seconds
        while not f.is_file() and time.monotonic()<limit:time.sleep(3)
        x=analyze(day,f)
        (dest/(day+'.json')).write_text(json.dumps(x,allow_nan=False)+'\n')
        rows.append(x)
        if n%10==0 or n==len(days):print(n,'dates scored',flush=True)
    summary=aggregate(rows)
    result={'plan':PLAN,'overall':summary,'monthly':{m:aggregate([r for r in rows if r['day'].startswith(m)]) for m in sorted({s[:7] for s in days})},'daily':rows}
    (HERE/'comparison.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
