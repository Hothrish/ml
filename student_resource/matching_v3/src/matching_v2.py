"""LightGBM matcher with native edit-distance features. MIT license."""
from pathlib import Path
import sys,os
deps=Path(__file__).resolve().parents[1]/'.matching_deps'
if deps.exists():sys.path.insert(0,str(deps))
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse,json,random,time
from functools import lru_cache
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein,JaroWinkler
import matching as m

BASE_FEATURES=m.features
def features(q,r,routes,fusion,views=None):
    out=BASE_FEATURES(q,r,routes,fusion,views)
    for a,b in ((q['name_sorted'],r['name_sorted']),
                (q['name_nospace'],r['name_nospace']),
                (q['name_cons'],r['name_cons']),
                (q.get('name_phon',''),r['name_phon']),
                (q['addr_norm'],r['addr_norm']),
                (' '.join(sorted(q['addr_norm'].split())),' '.join(sorted(r['addr_norm'].split())))):
        out.extend([Levenshtein.normalized_similarity(a,b),JaroWinkler.normalized_similarity(a,b),
                    fuzz.ratio(a,b)/100] if a and b else [0.,0.,0.])
    for a,b in ((q['name_sorted'],r['name_sorted']),(q['addr_norm'],r['addr_norm'])):
        out.extend([fuzz.token_set_ratio(a,b)/100,fuzz.WRatio(a,b)/100] if a and b else [0.,0.])
    return out

@lru_cache(maxsize=1)
def booster(model_text):return lgb.Booster(model_str=model_text)

def predict_scores(x,model):
    if model.get('type')!='lightgbm':raise ValueError('Use matching.py for the original logistic model')
    if not len(x):return np.empty(0,dtype=np.float32)
    x=np.asarray(x,dtype=np.float32)
    if model.get('feature_version')!=2 or x.shape[1]!=model['feature_count']:
        raise ValueError('Model and comparison features are incompatible')
    return booster(model['model_text']).predict(x,num_threads=1)

m.features=features
m.predict_scores=predict_scores

def build_country_cache(args):
    started=time.monotonic();out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    db=m.connect(args.index,readonly=True);meta=m.get_meta(db)
    if meta.get('status')!='ready':raise ValueError('Record index is not complete')
    codes={country:i+1 for i,(country,) in enumerate(db.execute('SELECT DISTINCT country FROM records'))}
    if len(codes)>65535:raise ValueError('Too many country values for this lookup')
    max_rid=db.execute('SELECT max(rid) FROM records').fetchone()[0]
    lookup=np.zeros(max_rid+1,dtype='<u2');count=0
    for rid,country in db.execute('SELECT rid,country FROM records ORDER BY rid'):
        lookup[rid]=codes[country];count+=1
        if count%1000000==0:m.log(f'Country lookup: {count:,}; {time.monotonic()-started:.1f}s')
    db.close()
    temp=Path(str(out)+'.bin.partial');lookup.tofile(temp);temp.replace(str(out)+'.bin')
    manifest={'base_inputs':meta['inputs'],'split':meta['split'],'max_rid':max_rid,'rows':count,'codes':codes}
    m.dump_json(str(out)+'.json',manifest)
    m.log(f'Country lookup ready: {count:,} records, {lookup.nbytes/1e6:.1f} MB')

def select_threshold(scores,groups,labels,truth):
    ids=sorted(truth);lookup={eid:i for i,eid in enumerate(ids)}
    group=np.array([lookup[eid] for eid in groups],dtype=np.int64);actual=np.array([len(truth[eid]) for eid in ids])
    best=(-1,1.)
    for t in np.r_[np.linspace(.005,.995,199),1.]:
        chosen=scores>=t
        predicted=np.bincount(group[chosen],minlength=len(ids))
        hit=np.bincount(group[chosen & (labels==1)],minlength=len(ids))
        score=np.where(actual==0,(predicted==0).astype(float),1.25*hit/np.maximum(.25*actual+predicted,1e-12)).mean()
        if score>=best[0]:best=(float(score),float(t))
    return best

def train(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'model.json').exists():raise FileExistsError('Use a fresh model directory')
    started=time.monotonic();truth=dict(m.iter_ground_truth(args.ground_truth))
    if args.splits:
        split=json.loads(Path(args.splits).read_text())
    else:
        ids=sorted(truth);random.Random(41).shuffle(ids);n=len(ids)
        if n<100:raise ValueError('Use at least 100 S1 queries')
        split={'fit':ids[:int(.7*n)],'threshold':ids[int(.7*n):int(.85*n)],'audit':ids[int(.85*n):]}
    fit=set(split['fit']);tune=set(split['threshold']);audit=set(split['audit'])
    assert not fit&tune and not fit&audit and not tune&audit
    x,groups,candidates,queries=m.load_features(args.index,args.queries,args.evidence,args.fields)
    if set(truth)!=fit|tune|audit or set(truth)!=set(queries):raise ValueError('Split/query coverage mismatch')
    y=np.asarray([cid in truth[eid] for eid,cid in zip(groups,candidates)],dtype=np.float32)
    fi=np.array([eid in fit for eid in groups]);ti=np.array([eid in tune for eid in groups]);ai=np.array([eid in audit for eid in groups])
    tune_truth={eid:truth[eid] for eid in tune};tune_groups=[eid for eid in groups if eid in tune]
    train_set=lgb.Dataset(x[fi],label=y[fi]);valid_set=lgb.Dataset(x[ti],label=y[ti],reference=train_set)
    trials=[];best=None
    for leaves in (15,31):
        params={'objective':'binary','metric':'binary_logloss','learning_rate':.05,
                'num_leaves':leaves,'min_data_in_leaf':50,'lambda_l2':2.,
                'feature_fraction':.9,'bagging_fraction':.9,'bagging_freq':1,
                'seed':41,'num_threads':args.workers,'verbosity':-1,'force_col_wise':True,
                'deterministic':True,'feature_pre_filter':False}
        model=lgb.train(params,train_set,num_boost_round=600,valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(100)])
        score,threshold=select_threshold(model.predict(x[ti],num_threads=args.workers),tune_groups,y[ti],tune_truth)
        trial={'leaves':leaves,'best_iteration':model.best_iteration,'calibration_macro_f0_5':score,'threshold':threshold}
        trials.append(trial);m.log(json.dumps(trial))
        if best is None or score>best[0]:best=(score,threshold,model)
    score,threshold,trained=best
    audit_scores=trained.predict(x[ai],num_threads=args.workers)
    audit_groups=[eid for eid in groups if eid in audit]
    audit_candidates=[cid for eid,cid in zip(groups,candidates) if eid in audit]
    pred=m.predictions(audit_scores,audit_groups,audit_candidates,threshold,audit)
    stats=json.loads((Path(args.evidence).parent/'blocking_stats.json').read_text())
    model={'version':1,'feature_version':2,'type':'lightgbm','model_text':trained.model_to_string(),
           'threshold':threshold,'config':stats['config'],'feature_count':x.shape[1],
           'retrieval':stats.get('retrieval','lsh'),'license':'MIT'}
    report={'model':'LightGBM with name/address comparison and native edit-distance features',
            'fit_s1':len(fit),'calibration_s1':len(tune),'audit_s1':len(audit),
            'fit_pairs':int(fi.sum()),'fit_positive_pairs':int(y[fi].sum()),
            'threshold':threshold,'calibration_macro_f0_5':score,'trials':trials,
            'audit':m.macro_score({eid:truth[eid] for eid in audit},pred),
            'audit_by_country':{},'seconds':time.monotonic()-started,
            'limitations':['This training-data audit is small and is not a test leaderboard score.',
                            'France has no labeled training examples.']}
    for country in sorted({queries[eid]['country_norm'] for eid in audit}):
        report['audit_by_country'][country]=m.macro_score({eid:truth[eid] for eid in audit if queries[eid]['country_norm']==country},pred)
    m.dump_json(out/'model.json',model);m.dump_json(out/'training_report.json',report)
    m.dump_json(out/'split_ids.json',split)
    with open(out/'audit_predictions.tsv','w',encoding='utf-8',newline='') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for eid in sorted(audit):f.write(eid+'\t'+','.join(pred[eid])+'\n')
    m.log(json.dumps(report,indent=2))

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    for key in ('index','queries','ground-truth','evidence','fields','out'):t.add_argument('--'+key,required=True)
    t.add_argument('--splits',help='Optional existing disjoint S1 split file; otherwise seed 41 creates a 70/15/15 split')
    t.add_argument('--workers',type=int,default=6)
    t=sub.add_parser('predict')
    for key in ('index','lsh-index','normalized-dir','model','out'):t.add_argument('--'+key,required=True)
    t.add_argument('--split',choices=('train','test'),required=True)
    t.add_argument('--queries');t.add_argument('--fields');t.add_argument('--limit',type=int,default=0)
    t.add_argument('--country-cache',help='Prefix of the optional exact country lookup .bin/.json files')
    t.add_argument('--workers',type=int,default=8);t.add_argument('--shard-size',type=int,default=1000)
    t=sub.add_parser('build-country-cache')
    t.add_argument('--index',required=True);t.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='train':train(args)
    elif args.command=='build-country-cache':build_country_cache(args)
    else:
        model=json.loads(Path(args.model).read_text())
        if model.get('type')!='lightgbm' or model.get('feature_version') not in (2,3) or model.get('retrieval','lsh')!='lsh':
            raise ValueError('This runner requires the supplied LightGBM version 2 LSH model')
        if args.country_cache:os.environ['MATCH_COUNTRY_CACHE']=str(Path(args.country_cache).resolve())
        else:os.environ.pop('MATCH_COUNTRY_CACHE',None)
        m.predict(args)

if __name__=='__main__':main()
