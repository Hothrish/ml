"""Prepare candidate features and train with disjoint S1 calibration/audit sets."""
import os,json,math,random,time,argparse,multiprocessing as mp
from pathlib import Path
from collections import Counter
from itertools import islice
import numpy as np
import matching_v3 as v

_truth=None
def init_worker(index,lsh,proto):v.init_predict(index,lsh,proto)
def one(item):
    i,q,truth=item
    candidates,_=v.v.m._BLOCKER.retrieve(q);views=v.query_views(q)
    x=np.asarray([v.features(q,r['matching_record'],r['routes'],r['fusion_score'],views) for r in candidates],dtype=np.float32)
    return i,x,[r['entity_id'] for r in candidates],[r['rid'] for r in candidates],np.asarray([r['entity_id'] in truth for r in candidates],dtype=np.uint8)

def prepare(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'dataset.json').exists():raise FileExistsError('Dataset already prepared')
    queries=list(v.b.iter_records([Path(a.queries)],1));truth=dict(v.b.iter_ground_truth(a.ground_truth))
    if {q['entity_id'] for q in queries}!=set(truth):raise ValueError('Query/truth mismatch')
    ids=sorted(truth);random.Random(81).shuffle(ids)
    fit=set(ids[:8000]);cal=set(ids[8000:10000]);audit=set(ids[10000:])
    if len(ids)!=12000:raise ValueError('This preparation uses 12000 S1 with an 8000/2000/2000 split')
    name=Counter();addr=Counter()
    for q in queries:
        if q['entity_id'] in fit:name.update(set(q['name_sorted'].split()));addr.update(set(q['addr_norm'].split()))
    idf={key:{t:math.log((len(fit)+1)/(n+1))+1 for t,n in counts.items()} for key,counts in (('name',name),('address',addr))}
    idf['default']=math.log(len(fit)+1)+1
    config=v.b.Config(lsh_bucket_limit=a.bucket,address_k=a.address)
    proto={'config':v.v.m.asdict(config),'feature_version':3,'idf':idf,
           'rerank_pool':a.pool,'pool_rarity':True,'text_rerank':True}
    os.environ.update(MATCH_COMPACT_LSH=str(Path(a.compact_index).resolve()),MATCH_COMPACT_RECORDS=str(Path(a.compact_records).resolve()),
        MATCH_COUNTRY_CACHE=str(Path(a.country_cache).resolve()),MATCH_TEXT_RERANK='1',MATCH_POOL_RARITY='1',MATCH_RERANK_POOL=str(a.pool))
    check=v.CompactLSHBlocker(a.index,a.lsh_index,config);check.close()
    chunks=out/'chunks';chunks.mkdir(exist_ok=True);started=time.monotonic();blocks=[];g=[];rids=[];labels=[];cids=[];total=0;chunk_id=0
    def save_chunk():
        nonlocal blocks,g,rids,labels,cids,chunk_id
        if not blocks:return
        np.save(chunks/f'x{chunk_id}.npy',np.concatenate(blocks));np.save(chunks/f'g{chunk_id}.npy',np.asarray(g,dtype=np.int32))
        np.save(chunks/f'r{chunk_id}.npy',np.asarray(rids,dtype=np.int32));np.save(chunks/f'y{chunk_id}.npy',np.concatenate(labels))
        (chunks/f'c{chunk_id}.json').write_text(json.dumps(cids),encoding='utf-8')
        blocks=[];g=[];rids=[];labels=[];cids=[];chunk_id+=1
    with mp.get_context('spawn').Pool(a.workers,init_worker,(a.index,a.lsh_index,proto)) as pool:
        items=((i,q,set(truth[q['entity_id']])) for i,q in enumerate(queries))
        for count,(i,x,ci,ri,y) in enumerate(pool.imap(one,items,chunksize=4),1):
            if len(x):blocks.append(x);g.extend([i]*len(x));rids.extend(ri);labels.append(y);cids.extend(ci);total+=len(x)
            if count%100==0:print(f'Prepared {count:,} S1; {total:,} pairs; {count/(time.monotonic()-started):.1f} S1/s',flush=True)
            if count%500==0:save_chunk()
    save_chunk()
    width=np.load(chunks/'x0.npy',mmap_mode='r').shape[1]
    arrays={key:np.lib.format.open_memmap(out/(key+'.npy'),mode='w+',dtype=dtype,shape=shape) for key,dtype,shape in
        [('x','float32',(total,width)),('groups','int32',(total,)),('rids','int32',(total,)),('y','uint8',(total,))]}
    start=0;candidates=[]
    for i in range(chunk_id):
        x=np.load(chunks/f'x{i}.npy',mmap_mode='r');end=start+len(x)
        for key,prefix in [('x','x'),('groups','g'),('rids','r'),('y','y')]:arrays[key][start:end]=np.load(chunks/f'{prefix}{i}.npy',mmap_mode='r')
        candidates.extend(json.loads((chunks/f'c{i}.json').read_text()));start=end
    for x in arrays.values():x.flush()
    meta={'prototype':proto,'queries':[q['entity_id'] for q in queries],'countries':[q['country_norm'] for q in queries],
          'truth':[truth[q['entity_id']] for q in queries],'splits':{'fit':sorted(fit),'calibration':sorted(cal),'audit':sorted(audit)},
          'rows':total,'feature_count':width,'seconds':time.monotonic()-started}
    (out/'candidates.json').write_text(json.dumps(candidates),encoding='utf-8')
    (out/'dataset.json').write_text(json.dumps(meta),encoding='utf-8')
    print('Dataset ready',total,width,flush=True)

def train(a):
    data=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'model.json').exists():raise FileExistsError('Model already exists')
    meta=json.loads((data/'dataset.json').read_text());x=np.load(data/'x.npy',mmap_mode='r');y=np.load(data/'y.npy')
    groups=np.load(data/'groups.npy');qids=meta['queries'];cids=json.loads((data/'candidates.json').read_text())
    sets={k:set(ids) for k,ids in meta['splits'].items()}
    masks={k:np.asarray([eid in ids for eid in qids])[groups] for k,ids in sets.items()}
    fi,ci,ai=(masks[k] for k in ('fit','calibration','audit'))
    truths={eid:ids for eid,ids in zip(qids,meta['truth'])}
    cal_truth={eid:truths[eid] for eid in meta['splits']['calibration']}
    cal_groups=[qids[i] for i in groups[ci]]
    train_set=v.v.lgb.Dataset(x[fi],label=y[fi]);valid_set=v.v.lgb.Dataset(x[ci],label=y[ci],reference=train_set)
    best=None;trials=[];started=time.monotonic()
    for leaves in (31,63):
        params={'objective':'binary','metric':'binary_logloss','learning_rate':.04,'num_leaves':leaves,
                'min_data_in_leaf':70,'lambda_l2':3.,'feature_fraction':.9,'bagging_fraction':.9,'bagging_freq':1,
                'seed':81,'num_threads':a.workers,'verbosity':-1,'force_col_wise':True,'deterministic':True,'feature_pre_filter':False}
        model=v.v.lgb.train(params,train_set,num_boost_round=700,valid_sets=[valid_set],
                 callbacks=[v.v.lgb.early_stopping(60,verbose=False),v.v.lgb.log_evaluation(100)])
        p=model.predict(x[ci],num_threads=a.workers)
        score,t=v.v.select_threshold(p,cal_groups,y[ci],cal_truth)
        trial={'leaves':leaves,'iterations':model.best_iteration,'calibration_f0_5':score,'threshold':t};trials.append(trial);print(trial,flush=True)
        if best is None or score>best[0]:best=score,t,model
    score,t,model=best
    result=dict(meta['prototype'],version=1,type='lightgbm',retrieval='lsh',feature_count=x.shape[1],threshold=t,model_text=model.model_to_string(),license='MIT')
    (out/'model.json').write_text(json.dumps(result),encoding='utf-8')
    p=model.predict(x[ai],num_threads=a.workers);audit_ids=meta['splits']['audit']
    pred=v.v.m.predictions(p,[qids[i] for i in groups[ai]],[cid for use,cid in zip(ai,cids) if use],t,set(audit_ids))
    audit_truth={eid:truths[eid] for eid in audit_ids}
    report={'fit_s1':8000,'calibration_s1':2000,'audit_s1':2000,'fit_pairs':int(fi.sum()),'fit_positives':int(y[fi].sum()),
            'threshold':t,'calibration_macro_f0_5':score,'trials':trials,'audit':v.v.m.macro_score(audit_truth,pred),'seconds':time.monotonic()-started}
    report['audit_singletons']={'count':sum(not ids for ids in audit_truth.values()),'correct':sum(not ids and not pred[eid] for eid,ids in audit_truth.items())}
    report['audit_by_country']={country:v.v.m.macro_score({eid:truths[eid] for eid,c in zip(qids,meta['countries']) if eid in audit_truth and c==country},pred) for country in sorted(set(meta['countries']))}
    (out/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'audit_predictions.json').write_text(json.dumps(pred),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for key in ('index','lsh-index','compact-index','compact-records','country-cache','queries','ground-truth','out'):a.add_argument('--'+key,required=True)
    a.add_argument('--pool',type=int,default=1000);a.add_argument('--bucket',type=int,default=2000);a.add_argument('--address',type=int,default=60);a.add_argument('--workers',type=int,default=8)
    a=sub.add_parser('train');a.add_argument('--data',required=True);a.add_argument('--out',required=True);a.add_argument('--workers',type=int,default=6)
    a=p.parse_args();prepare(a) if a.command=='prepare' else train(a)
if __name__=='__main__':main()
