"""Fit the inexpensive stage on fitting S1 only; select its gate on calibration."""
import argparse,json,time
from pathlib import Path
import numpy as np
import matching_v3 as v

def main():
    parser=argparse.ArgumentParser()
    for key in ('data','model','out'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--workers',type=int,default=6)
    a=parser.parse_args();data=Path(a.data);out=Path(a.out)
    if (out/'model.json').exists():raise FileExistsError('Use a fresh model directory')
    out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((data/'dataset.json').read_text());model=json.loads(Path(a.model).read_text())
    if model.get('feature_version')!=3 or meta['feature_count']!=153:raise ValueError('Expected version 3 features')
    x=np.load(data/'x.npy',mmap_mode='r');y=np.load(data/'y.npy');groups=np.load(data/'groups.npy')
    masks={}
    for kind in ('fit','calibration'):
        ids=set(meta['splits'][kind]);masks[kind]=np.array([eid in ids for eid in meta['queries']])[groups]
    fi,ci=masks['fit'],masks['calibration'];cheap=np.asarray(x[:,107:129]).copy();start=time.monotonic()
    fitted=v.v.lgb.train({'objective':'binary','metric':'binary_logloss','num_leaves':15,'learning_rate':.08,
        'min_data_in_leaf':100,'lambda_l2':3,'num_threads':a.workers,'verbosity':-1,'seed':81,
        'deterministic':True,'force_col_wise':True},v.v.lgb.Dataset(cheap[fi],label=y[fi]),num_boost_round=150)
    gate=fitted.predict(cheap[ci],num_threads=a.workers)
    full=v.model_booster(model['model_text']).predict(x[ci],num_threads=a.workers)
    trials=[];threshold=0.
    for t in (.00001,.00003,.0001,.0003,.001,.003,.01,.03,.05,.1,.2):
        keep=gate>=t;lost=int(((y[ci]==1)&~keep).sum());changed=int(((full>=model['threshold'])&~keep).sum())
        trials.append({'threshold':t,'full_model_fraction':float(keep.mean()),'true_pairs_lost':lost,'full_model_positive_pairs_lost':changed})
        if lost==0 and changed==0:threshold=t
    model.update(cascade_model_text=fitted.model_to_string(),cascade_threshold=threshold,cascade_features=list(range(107,129)))
    (out/'model.json').write_text(json.dumps(model),encoding='utf-8')
    report={'threshold':threshold,'calibration_trials':trials,'seconds':time.monotonic()-start}
    (out/'cascade_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
