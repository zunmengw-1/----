"""Recompute Gate16, two flat models, and zero under the project's official fixed validation scale."""
from pathlib import Path
from time import sleep, monotonic
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent
FROZEN = Path('/private/tmp/tcn-signal-date-holdout-20261127')
REV = Path('/private/tmp/tcn-vs-five-minute-reversal-20261127')
GATE = Path('/private/tmp/tcn-clock-exclusive-20261127')
OFFICIAL = json.loads((GATE/'three_audit.json').read_text())
PLAN = json.loads((ROOT/'evaluation_plan.json').read_text())
DATES = PLAN['evaluation_dates']
assert DATES == json.loads((FROZEN/'frozen_plan.json').read_text())['evaluation_dates']
assert hashlib.sha256((GATE/'gate/model.pt').read_bytes()).hexdigest() == PLAN['gate16_sha256']
assert hashlib.sha256((FROZEN/'frozen_plan.json').read_bytes()).hexdigest() == PLAN['panel_sha256']
SCALE = float(np.float32(json.loads((GATE/'gate/report.json').read_text())['config']['return_scale']))
assert SCALE == OFFICIAL['validation_scale']
SPOT = {DATES[0], DATES[len(DATES)//2], DATES[-1]}

def wait_file(path):
    end = monotonic() + 7200.
    while not path.is_file():
        if monotonic() > end:
            raise TimeoutError(str(path))
        sleep(2)
    return path

def same_keys(old, new):
    for col in ('code','t'):
        assert old[col].combine_chunks().equals(new[col].combine_chunks()), col

def global_loss(p, y):
    # Official evaluation buffers contain float64 physical returns; the scalar is the float32 checkpoint value.
    error = np.abs(np.asarray(y, dtype=np.float64)/SCALE -
                   np.asarray(p, dtype=np.float64)/SCALE)
    return float(np.where(error < 1., 0.5*error**2, error-0.5).sum(dtype=np.float64))

def audit_selection(old, y, scores, valid):
    codes = np.asarray(old['code'].to_numpy())
    # Physical timestamps in UTC ms differ by the same clock-minute intervals as Shanghai time.
    minutes = old['t'].cast('int64').to_numpy() // 60000
    order = np.argsort(minutes, kind='stable')
    groups = np.split(order, np.flatnonzero(np.diff(minutes[order])) + 1)
    selected = {k:[] for k in scores}
    cross = {k:[] for k in scores}
    last = {k:{} for k in scores}
    for group in groups:
        now = int(minutes[group[0]])
        eligible_y = group[valid[group]]
        assert len(eligible_y) >= 20
        ranked_y = rankdata(y[eligible_y], method='average')
        kcut = max(1,len(group)//100)
        for k, values in scores.items():
            ranked_scores = rankdata(values[eligible_y], method='average')
            assert ranked_scores.std() > 0 and ranked_y.std() > 0
            cross[k].append(float(np.corrcoef(ranked_y,ranked_scores)[0,1]))
            # Construct priority over all candidates first, then walk it until eligible slots fill.
            priority = group[np.lexsort((codes[group],-values[group]))]
            chosen = []
            for row in priority:
                code = codes[row]
                if now-last[k].get(code, -10**10) >= 5:
                    last[k][code] = now
                    chosen.append(int(row))
                    if len(chosen) == kcut:
                        break
            assert len(chosen)==kcut
            selected[k].extend(chosen)
    results = {}
    for k, idx in selected.items():
        obs = y[idx]
        present = np.isfinite(obs)
        results[k] = dict(cross_section_ic=float(np.mean(cross[k])), selected=len(idx),
                          valid=int(present.sum()),
                          selected_row_indices_sha256=hashlib.sha256(np.asarray(idx,dtype='<i8').tobytes()).hexdigest(),
                          daily_mean_gross_bp=float(np.expm1(obs[present].astype(np.float64)).mean()*1e4))
    return results

def date_result(day):
    new = pq.read_table(wait_file(ROOT/'predictions'/f'{day}.parquet'))
    old = pq.read_table(FROZEN/'daily'/f'{day}.parquet', columns=['t','code','label','label_valid','p16'])
    assert old.schema.metadata[b'model16_sha256'].decode() == PLAN['gate16_sha256']
    same_keys(old,new)
    y = old['label'].to_numpy()
    valid = old['label_valid'].to_numpy()
    assert np.array_equal(np.isfinite(y), valid)
    if 'label' in new.column_names:
        assert np.array_equal(y,new['label'].to_numpy(), equal_nan=True)
    if 'label_valid' in new.column_names:
        assert np.array_equal(valid,new['label_valid'].to_numpy())
    scores = {k:new['p_'+k].to_numpy() for k in ('mlp','linear')}
    scores['tcn'] = old['p16'].to_numpy()
    assert all(np.isfinite(x).all() for x in scores.values())
    sums = {k:global_loss(x[valid], y[valid]) for k,x in scores.items()}
    sums['zero'] = global_loss(0.,y[valid])
    n = int(valid.sum())
    record = dict(day=day,candidates=len(y),valid=n,sum_huber=sums,mean_huber={k:v/n for k,v in sums.items()})
    if day in SPOT:
        reversal = pq.read_table(REV/'daily'/f'{day}.parquet',columns=['t','code','past_return','past_valid'])
        same_keys(old,reversal)
        assert reversal['past_valid'].to_numpy().all()
        scores['reversal'] = -reversal['past_return'].to_numpy()
        record['spot_check'] = audit_selection(old,y,scores,valid)
    return record

def main():
    records=[]
    for idx,day in enumerate(DATES,1):
        records.append(date_result(day))
        if idx%10==0 or idx in (1,len(DATES)):
            print(f'{idx}/{len(DATES)} {day}',flush=True)
    for rec in records:
        if 'spot_check' not in rec:
            continue
        day = rec['day']
        scored = json.loads(wait_file(ROOT/'scored_dates'/f'{day}.json').read_text())
        for k,values in rec['spot_check'].items():
            expected = scored['top_one_percent'][k]
            assert (expected['selected'],expected['valid'])==(values['selected'],values['valid'])
            assert abs(values['cross_section_ic']-scored['cross_section_ic'][k]) < 1.e-10
            assert abs(values['daily_mean_gross_bp']-expected['daily_mean_gross_bp']) < 1.e-7
        rec['spot_check_agrees_with_main_scorer'] = True
    n = sum(x['valid'] for x in records)
    assert n==32834005 and sum(x['candidates'] for x in records)==33362162
    overall = {k:sum(x['sum_huber'][k] for x in records)/n for k in records[0]['sum_huber']}
    out = dict(scale=SCALE,delta=1.0,forecast_units='physical log return',dates=len(records),valid=n,
               candidates=sum(x['candidates'] for x in records),total_valid_weighted_huber=overall,
               same_code_time_order_all_dates=True,
               loss_arithmetic='Official fixed global return_scale, float64 rescaling of y and p and float64 accumulation, as in the official daily evaluator.',
               spot_check_protocol='3 prespecified dates (first, middle, last). Independent SciPy average-tied ranks minute by minute. Ordered full candidate universe then filled top 1% following a separate per-security five-minute lock by arm; future labels enter only for scoring.',
               daily=records)
    dest = ROOT/'global_huber_check.json'
    tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(dest)
    print(overall,flush=True)

if __name__=='__main__':
    main()
