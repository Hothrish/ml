"""MIT-licensed supervised baseline using the completed blocking indexes.

Train and threshold splits are by S1 entity. Test inference scores only retrieved
candidates and saves a checkpoint after every shard. No external entity data.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import re
import sqlite3
import shutil
import time
from itertools import islice
from functools import lru_cache
import numpy as np
from blocking import (Config, LSHBlocker, check_index, connect, dump_json, get_meta, set_meta,
                      fingerprint, iter_records, iter_ground_truth, source_parts, log)

VERSION = 1
COLS = ('entity_id','country','name_sorted','name_nospace','name_cons','house','locality')
EXTRA = ('addr_norm','addr_nums','addr_postcode','addr_region','addr_street','name_phon','name_legal')

class FastLSHBlocker(LSHBlocker):
    """Same candidate ranking, with a bounded cache for repeated band lookups."""
    def __init__(self,index,lsh_index,config):
        super().__init__(index,lsh_index,config)
        self._band_hits=lru_cache(maxsize=4096)(self._band_hits_uncached)
        available={row[1] for row in self.db.execute('PRAGMA table_info(records)')}
        requested=COLS[2:]+EXTRA
        if set(requested)<=available:
            self.matching_columns=requested
        self.country_lookup=None
        cache=os.environ.get('MATCH_COUNTRY_CACHE')
        if cache:
            manifest=json.loads(Path(cache+'.json').read_text())
            if manifest['base_inputs']!=self.meta['inputs'] or manifest['split']!=self.meta['split']:
                raise ValueError('Country lookup belongs to another record index')
            if Path(cache+'.bin').stat().st_size!=2*(manifest['max_rid']+1):
                raise ValueError('Incomplete country lookup')
            self.country_lookup=np.memmap(cache+'.bin',dtype='<u2',mode='r')
            self.country_codes=manifest['codes']

    def _band_hits_uncached(self,key):
        rows=self.lsh.execute('SELECT rid FROM bands WHERE key=? LIMIT ?',
                             (key,self.config.lsh_bucket_limit+1)).fetchall()
        return () if len(rows)>self.config.lsh_bucket_limit else tuple(row[0] for row in rows)

    def fuzzy(self,rec,view,source,k,global_rescue=False):
        from blocking import band_keys,VIEWS
        if not k:return []
        country=rec['country_norm'].casefold().strip()
        if global_rescue and not country:return []
        countries=[''] if global_rescue else ([country] if country else list(self.meta['countries']))
        votes=Counter()
        for c in countries:
            for key in band_keys(rec[VIEWS['name'] if global_rescue else VIEWS[view]],
                         'global_name' if global_rescue else view,source,c,self.config.lsh_bands):
                votes.update(self._band_hits(key))
        if global_rescue and votes:
            if self.country_lookup is not None:
                code=self.country_codes.get(country,-1)
                votes=Counter({rid:n for rid,n in votes.items() if self.country_lookup[rid]!=code})
            else:
                kept={};rids=list(votes)
                for offset in range(0,len(rids),500):
                    chunk=rids[offset:offset+500]
                    rows=self.db.execute('SELECT rid,country FROM records WHERE rid IN ('+
                                         ','.join('?' for _ in chunk)+')',chunk)
                    kept.update((rid,votes[rid]) for rid,c in rows if c!=country)
                votes=Counter(kept)
        ranked=sorted(votes,key=lambda rid:(-votes[rid],rid))[:k]
        return [(rid,float(votes[rid])) for rid in ranked]

    def close(self):
        if hasattr(self,'_band_hits'):self._band_hits.cache_clear()
        if getattr(self,'country_lookup',None) is not None:
            self.country_lookup._mmap.close()
            self.country_lookup=None
        super().close()

def fetch_records(db, ids, fields_db=None):
    result = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i+500]
        requested=COLS if fields_db else COLS+EXTRA
        for row in db.execute('SELECT '+','.join(requested)+' FROM records WHERE entity_id IN ('+
                              ','.join('?' for _ in chunk)+')', chunk):
            result[row[0]] = dict(zip(requested, row))
        if fields_db:
            for row in fields_db.execute('SELECT entity_id,'+','.join(EXTRA)+' FROM fields'+
                          ' WHERE entity_id IN ('+','.join('?' for _ in chunk)+')',chunk):
                result[row[0]].update(zip(EXTRA,row[1:]))
    if len(result) != len(set(ids)):
        raise ValueError('Candidate ID is missing from record index')
    if any('addr_norm' not in r for r in result.values()):raise ValueError('Candidate comparison fields missing')
    return result

def equal(a, b):
    return float(bool(a and b) and a == b)

def bigrams(value):
    return {value[i:i+2] for i in range(len(value)-1)} if len(value)>1 else set(value)

def jaccard(a,b):
    return len(a & b)/len(a | b) if a and b else 0.0

def ratio(a,b):
    return min(len(a),len(b))/max(len(a),len(b)) if a and b else 0.0

def query_views(q):
    return {'words':set(q['name_sorted'].split()), 'chars':bigrams(q['name_nospace']),
            'cons':bigrams(q['name_cons'].replace(' ','')),
            'local':bigrams(q['addr_locality']), 'nums':set(re.findall(r'\d+',q['name_sorted'])),
            'addr_chars':bigrams(q['addr_norm']), 'addr_words':set(q['addr_norm'].split()),
            'addr_nums':set(re.findall(r'\d+',q['addr_norm'])),
            'phon':bigrams(q.get('name_phon','')), 'street':set(q.get('addr_street','').split())}

def features(q, r, routes, fusion, views=None):
    v = views or query_views(q)
    words = set(r['name_sorted'].split())
    word = jaccard(v['words'], words)
    char = jaccard(v['chars'], bigrams(r['name_nospace']))
    cons = jaccard(v['cons'], bigrams(r['name_cons'].replace(' ','')))
    local = jaccard(v['local'],bigrams(r['locality']))
    house = equal(q['addr_house'],r['house'])
    nums = set(re.findall(r'\d+',r['name_sorted']))
    overlap = len(v['words'] & words)/min(len(v['words']),len(words)) if v['words'] and words else 0.0
    f = [equal(q['name_sorted'],r['name_sorted']), equal(q['name_nospace'],r['name_nospace']),
         equal(q['name_cons'],r['name_cons']), word,char,cons,overlap,
         ratio(q['name_nospace'],r['name_nospace']), house,
         float(bool(q['addr_house'] and r['house']) and q['addr_house']!=r['house']),
         float(bool(q['addr_house'])),float(bool(r['house'])),local,
         equal(q['addr_locality'],r['locality']),
         equal(q['country_norm'].casefold().strip(),r['country']),
         jaccard(v['nums'],nums), float(bool(v['nums'] and nums) and not (v['nums'] & nums)),
         float(not q['name_nospace']),float(not r['name_nospace']),
         float(not q['addr_norm']), float(r['entity_id'].startswith('S3-'))]
    scores = {}
    for route in ('name','address','skeleton','global_name'):
        info=routes.get(route)
        score=float(info.get('score') or 0)/8 if info else 0.0
        rank=1/math.sqrt(info['rank']) if info else 0.0
        f.extend([float(info is not None),score,rank])
        scores[route]=score
    exact=[k for k in routes if k.startswith('exact_')]
    f.extend([len(exact)/3, float(any(k.endswith('_house') for k in exact)),
              float(any(k.endswith('_locality') for k in exact)),fusion*10])
    addr=scores['address']
    f.extend([char*addr,word*addr,cons*addr,char*house,cons*house,char*local,cons*local,
              scores['name']*addr,scores['skeleton']*addr,house*local])
    # Hinge features allow nonlinear similarity cutoffs in a small logistic model.
    for value in (char,word,cons,local,addr):
        f.extend(max(0.0,value-cut) for cut in (.25,.5,.75,.9))
    awords=set(r['addr_norm'].split());anums=set(re.findall(r'\d+',r['addr_norm']))
    achar=jaccard(v['addr_chars'],bigrams(r['addr_norm']))
    aword=jaccard(v['addr_words'],awords)
    aoverlap=len(v['addr_words']&awords)/min(len(v['addr_words']),len(awords)) if v['addr_words'] and awords else 0.0
    anum=jaccard(v['addr_nums'],anums)
    phon=jaccard(v['phon'],bigrams(r['name_phon']))
    street=jaccard(v['street'],set(r['addr_street'].split()))
    f.extend([achar,aword,aoverlap,anum,phon,street,
              equal(q.get('addr_postcode',''),r['addr_postcode']),
              float(bool(q.get('addr_postcode') and r['addr_postcode']) and q['addr_postcode']!=r['addr_postcode']),
              equal(q.get('addr_region',''),r['addr_region']),
              equal(q.get('name_legal',''),r['name_legal']),float(not r['addr_norm']),
              ratio(q['addr_norm'],r['addr_norm']),char*achar,cons*achar,phon*achar,
              word*aword,house*achar,char*anum,cons*anum,phon*anum])
    for value in (achar,aword,anum,phon,street):
        f.extend(max(0.0,value-cut) for cut in (.25,.5,.75,.9))
    return f

def build_fields(args):
    out=Path(args.out)
    if out.exists():raise FileExistsError('Comparison-field file already exists; use a fresh path')
    selected=set()
    with open(args.candidates,encoding='utf-8') as f:
        next(f)
        for line in f:
            value=line.rstrip('\r\n').split('\t')[1]
            if value:selected.update(value.split(','))
    out.parent.mkdir(parents=True,exist_ok=True);temp=Path(str(out)+'.partial')
    if temp.exists():raise FileExistsError('Remove the incomplete comparison-field .partial file before retrying')
    db=sqlite3.connect(temp)
    db.execute('CREATE TABLE fields(entity_id TEXT PRIMARY KEY,'+','.join(x+' TEXT NOT NULL' for x in EXTRA)+')')
    found=0;scanned=0;batch=[];start=time.monotonic()
    try:
        for source in (2,3):
            # Parse only selected records; this avoids dictionaries for millions of unused rows.
            header=None
            for path in source_parts(args.normalized_dir,args.split,source):
                opener=gzip.open if str(path).endswith('.gz') else open
                with opener(path,'rt',encoding='utf-8') as f:
                    for line in f:
                        eid=line.split('\t',1)[0]
                        if eid=='entity_id':header=line.rstrip('\r\n').split('\t');continue
                        scanned+=1
                        if eid in selected:
                            rec=dict(zip(header,line.rstrip('\r\n').split('\t')))
                            batch.append((eid,*(rec.get(x,'') for x in EXTRA)));found+=1
                            if len(batch)>=5000:
                                db.executemany('INSERT INTO fields VALUES ('+','.join('?' for _ in range(1+len(EXTRA)))+')',batch)
                                db.commit();batch=[]
                        if scanned%1000000==0:log(f'Comparison fields: scanned {scanned:,}; kept {found:,}; {time.monotonic()-start:.1f}s')
        if batch:db.executemany('INSERT INTO fields VALUES ('+','.join('?' for _ in range(1+len(EXTRA)))+')',batch)
        db.commit()
        if found!=len(selected):raise ValueError('Not every candidate was found in normalized inputs')
    finally:db.close()
    temp.replace(out);log(f'PASS: saved comparison fields for {found:,} candidates')

def build_records(args):
    """Compact test index: exact name keys plus matcher fields, without FTS tables.

    Interrupted loading resumes from committed rows. LSH is built separately.
    """
    paths={s:source_parts(args.normalized_dir,args.split,s) for s in (2,3)}
    inputs={str(s):fingerprint(ps) for s,ps in paths.items()}
    path=Path(args.index);path.parent.mkdir(parents=True,exist_ok=True)
    existed=path.exists();db=connect(path);start=time.monotonic()
    if existed:
        meta=get_meta(db)
        if meta.get('inputs')!=inputs or meta.get('split')!=args.split or not meta.get('matching_fields'):
            db.close();raise ValueError('Existing index is incompatible; use another path')
        if meta['status']=='ready':db.close();check_index(path,args.normalized_dir,args.split);log('PASS: compact record index ready');return
    else:
        meta={'version':1,'status':'building','split':args.split,'partial_index':False,
              'inputs':inputs,'rows':{},'countries':{},'matching_fields':list(EXTRA),
              'retrieval':'MinHash plus indexed exact names; no FTS tables'}
        db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        db.execute('CREATE TABLE records(rid INTEGER PRIMARY KEY,entity_id TEXT UNIQUE NOT NULL,'+
                   'source INTEGER NOT NULL,country TEXT NOT NULL,script TEXT NOT NULL,'+
                   'name_sorted TEXT NOT NULL,name_nospace TEXT NOT NULL,name_cons TEXT NOT NULL,'+
                   'house TEXT NOT NULL,locality TEXT NOT NULL,'+','.join(x+' TEXT NOT NULL' for x in EXTRA)+')')
        set_meta(db,meta);db.commit()
    counts={s:db.execute('SELECT count(*) FROM records WHERE source=?',(s,)).fetchone()[0] for s in (2,3)}
    rid=sum(counts.values());batch=[]
    try:
        for source in (2,3):
            count=0
            for rec in iter_records(paths[source],source):
                count+=1
                if count<=counts[source]:continue
                rid+=1
                batch.append((rid,rec['entity_id'],source,rec['country_norm'].casefold().strip(),
                              rec['name_script'],rec['name_sorted'],rec['name_nospace'],rec['name_cons'],
                              rec['addr_house'],rec['addr_locality'],*(rec.get(x,'') for x in EXTRA)))
                if len(batch)>=5000:
                    db.executemany('INSERT INTO records VALUES ('+','.join('?' for _ in range(10+len(EXTRA)))+')',batch)
                    db.commit();batch=[]
                if count%100000==0:log(f'Compact S{source}: {count:,}; {time.monotonic()-start:.1f}s')
            if batch:db.executemany('INSERT INTO records VALUES ('+','.join('?' for _ in range(10+len(EXTRA)))+')',batch);batch=[]
            db.commit();meta['rows'][str(source)]=count
        for field in ('name_sorted','name_nospace','name_cons'):
            log(f'Indexing exact {field} lookups...')
            db.execute(f'CREATE INDEX IF NOT EXISTS i_{field} ON records(country,source,{field},house)')
            db.execute(f'CREATE INDEX IF NOT EXISTS i_{field}_locality ON records(country,source,{field},locality)')
        meta['countries']=dict(db.execute('SELECT country,count(*) FROM records GROUP BY country'))
        meta.update(status='ready',seconds=time.monotonic()-start);set_meta(db,meta);db.commit()
    finally:db.close()
    dump_json(str(path)+'.manifest.json',meta);log('PASS: compact record index ready')

def macro_score(truth, predicted):
    scores=[];tp=fp=fn=0
    for eid,actual in truth.items():
        actual=set(actual);guess=set(predicted.get(eid,()))
        hit=len(actual & guess)
        scores.append((1.0 if not guess else 0.0) if not actual else
                      1.25*hit/(.25*len(actual)+len(guess)))
        tp+=hit;fp+=len(guess)-hit;fn+=len(actual)-hit
    return {'macro_f0_5':sum(scores)/len(scores) if scores else 0.0,
            'query_count':len(scores),'true_positives':tp,'false_positives':fp,'false_negatives':fn,
            'micro_precision':tp/(tp+fp) if tp+fp else 0.0,
            'micro_recall':tp/(tp+fn) if tp+fn else 0.0}

def predictions(scores, groups, candidates, threshold, selected):
    result={eid:[] for eid in selected}
    for score,eid,cid in zip(scores,groups,candidates):
        if eid in result and score>=threshold:
            result[eid].append(cid)
    return result

def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-35,35)))

def fit_logistic(x,y,seed=41,epochs=60):
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<.05]=1
    x=np.ascontiguousarray((x-mean)/scale,dtype=np.float32)
    x=np.column_stack((np.ones(len(x),dtype=np.float32),x))
    weights=np.zeros(x.shape[1],dtype=np.float32)
    weights[0]=math.log(max(float(y.mean()),1e-6)/max(1-float(y.mean()),1e-6))
    m=np.zeros_like(weights);v=np.zeros_like(weights);step=0
    rng=np.random.default_rng(seed)
    for epoch in range(epochs):
        order=rng.permutation(len(y))
        for start in range(0,len(y),2048):
            idx=order[start:start+2048];batch=x[idx];target=y[idx]
            grad=batch.T@(sigmoid(batch@weights)-target)/len(idx)
            grad[1:]+=0.0002*weights[1:]
            step+=1;m=.9*m+.1*grad;v=.999*v+.001*grad*grad
            weights-=.02*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    return {'mean':mean.tolist(),'scale':scale.tolist(),'weights':weights.tolist()}

def predict_scores(x,model):
    if not len(x):return np.empty(0,dtype=np.float32)
    x=np.asarray(x,dtype=np.float32)
    return sigmoid(((x-np.asarray(model['mean'],dtype=np.float32))/
                     np.asarray(model['scale'],dtype=np.float32))@
                    np.asarray(model['weights'][1:],dtype=np.float32)+model['weights'][0])

def load_features(index,queries,evidence,fields):
    query={r['entity_id']:r for r in iter_records([Path(queries)],1)}
    views={eid:query_views(q) for eid,q in query.items()}
    blocks=[];groups=[];candidates=[];db=connect(index,readonly=True);field_db=connect(fields,readonly=True)
    started=time.monotonic()
    try:
        with gzip.open(evidence,'rt',encoding='utf-8') as stream:
            header=next(stream).rstrip('\n')
            if header!='source1_entity_id\tcandidate_entity_id\troutes_json\tfusion_score':
                raise ValueError('Unexpected evidence header')
            while lines:=list(islice(stream,10000)):
                x=[]
                rows=[line.rstrip('\r\n').split('\t') for line in lines]
                records=fetch_records(db,list({row[1] for row in rows}),field_db)
                for eid,cid,route,fusion in rows:
                    x.append(features(query[eid],records[cid],json.loads(route),float(fusion),views[eid]))
                    groups.append(eid);candidates.append(cid)
                blocks.append(np.asarray(x,dtype=np.float32))
                if len(groups)%100000==0:log(f'Features: {len(groups):,} pairs; {time.monotonic()-started:.1f}s')
    finally:db.close();field_db.close()
    if not blocks:raise ValueError('No candidate pairs to fit')
    return np.concatenate(blocks),groups,candidates,query

def train(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'model.json').exists():raise FileExistsError('Use a fresh model output directory')
    started=time.monotonic()
    stats=json.loads((Path(args.evidence).parent/'blocking_stats.json').read_text())
    if not stats.get('lsh_index'):raise ValueError('This model expects MinHash band evidence')
    truth=dict(iter_ground_truth(args.ground_truth))
    x,groups,candidates,queries=load_features(args.index,args.queries,args.evidence,args.fields)
    if set(truth)!=set(queries):raise ValueError('Queries and ground truth differ')
    ids=sorted(truth);random.Random(args.seed).shuffle(ids)
    n=len(ids)
    if n<100:raise ValueError('Use at least 100 S1 queries')
    fit=set(ids[:int(.7*n)]);tune=set(ids[int(.7*n):int(.85*n)]);audit=set(ids[int(.85*n):])
    y=np.asarray([cid in truth[eid] for eid,cid in zip(groups,candidates)],dtype=np.float32)
    rng=random.Random(args.seed)
    positives=[];negatives={eid:[] for eid in fit}
    for i,(eid,label) in enumerate(zip(groups,y)):
        if eid not in fit:continue
        if label:positives.append(i)
        else:negatives[eid].append(i)
    # Use all retrieved positives and a bounded random set of candidate negatives.
    chosen=positives[:]
    for idx in negatives.values():chosen.extend(rng.sample(idx,min(len(idx),30)))
    if len(set(y[chosen]))<2:raise ValueError('Fitting set needs both classes')
    model=fit_logistic(x[chosen],y[chosen],args.seed)
    scores=predict_scores(x,model)
    selected_threshold=None;best=-1
    tune_truth={eid:truth[eid] for eid in tune}
    # Threshold selection uses full candidate sets and the challenge macro metric.
    thresholds=np.unique(np.r_[np.linspace(.01,.99,99),1.0])
    for threshold in thresholds:
        metric=macro_score(tune_truth,predictions(scores,groups,candidates,float(threshold),tune))
        if metric['macro_f0_5']>=best:
            best=metric['macro_f0_5'];selected_threshold=float(threshold)
    model.update(version=VERSION,threshold=selected_threshold,config=stats['config'],
                 seed=args.seed,training_queries=len(fit),feature_count=x.shape[1],
                 license='MIT',description='Logistic matcher: names, address, numbers and LSH evidence')
    audit_truth={eid:truth[eid] for eid in audit}
    audit_pred=predictions(scores,groups,candidates,selected_threshold,audit)
    audit_candidates={eid:[] for eid in audit}
    for eid,cid in zip(groups,candidates):
        if eid in audit:audit_candidates[eid].append(cid)
    report={'model':'NumPy logistic regression with nonlinear comparison features',
            'fit_s1':len(fit),'threshold_s1':len(tune),'audit_s1':len(audit),
            'fit_pairs':len(chosen),'fit_positive_pairs':len(positives),
            'all_candidate_pairs':len(groups),'threshold':selected_threshold,
            'threshold_selection_macro_f0_5':best,'audit':macro_score(audit_truth,audit_pred),
            'audit_by_country':{},'audit_blocking_oracle':macro_score(audit_truth,
               {eid:list(set(truth[eid])&set(audit_candidates[eid])) for eid in audit}),
            'seconds':time.monotonic()-started,
            'limitations':['Audit is a small training-data estimate, not a test leaderboard score.',
              'France has no labeled training examples.',
              'Features use character/token overlap, phonetics and numbers; full edit-distance/TF-IDF features are not included.']}
    for country in sorted({queries[eid]['country_norm'] for eid in audit}):
        subset={eid:truth[eid] for eid in audit if queries[eid]['country_norm']==country}
        report['audit_by_country'][country]=macro_score(subset,audit_pred)
    dump_json(out/'model.json',model);dump_json(out/'training_report.json',report)
    dump_json(out/'split_ids.json',{'fit':sorted(fit),'threshold':sorted(tune),'audit':sorted(audit)})
    with open(out/'audit_predictions.tsv','w',encoding='utf-8',newline='') as handle:
        handle.write('source1_entity_id\tmatched_entity_ids\n')
        for eid in sorted(audit):handle.write(eid+'\t'+','.join(audit_pred[eid])+'\n')
    log(json.dumps(report,indent=2))

_BLOCKER=None
_MODEL=None
_FIELDS=None

def init_predict(index,lsh,model,fields=None):
    import signal
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    global _BLOCKER,_MODEL,_FIELDS
    _MODEL=model;_BLOCKER=FastLSHBlocker(index,lsh,Config(**model['config']))
    _FIELDS=connect(fields,readonly=True) if fields else None

def predict_one(rec):
    candidates,_=_BLOCKER.retrieve(rec)
    ids=[r['entity_id'] for r in candidates]
    records=({r['entity_id']:r['matching_record'] for r in candidates}
             if getattr(_BLOCKER,'matching_columns',None) else fetch_records(_BLOCKER.db,ids,_FIELDS))
    view=query_views(rec)
    x=[features(rec,records[r['entity_id']],r['routes'],r['fusion_score'],view) for r in candidates]
    scores=predict_scores(x,_MODEL)
    matches=[eid for eid,score in zip(ids,scores) if score>=_MODEL['threshold']]
    return rec['entity_id'],ids,matches

def predict(args):
    if args.workers<1 or args.shard_size<1:raise ValueError('workers/shard-size must be positive')
    model_path=Path(args.model);model=json.loads(model_path.read_text())
    if model['version']!=VERSION:raise ValueError('Incompatible model')
    meta=check_index(args.index,args.normalized_dir,args.split)
    blocker=FastLSHBlocker(args.index,args.lsh_index,Config(**model['config']));blocker.close()
    paths=[Path(args.queries)] if args.queries else source_parts(args.normalized_dir,args.split,1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if not (out/'checkpoint.json').exists() and any((out/name).exists() for name in ('matching_results.tsv','candidate_pairs.tsv')):
        raise FileExistsError('Output contains predictions from another run; choose a fresh output directory')
    identity={'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'inputs':fingerprint(paths),'base_inputs':meta['inputs'],
              'fields':fingerprint([Path(args.fields)]) if args.fields else None,
              'split':args.split,'limit':args.limit,'shard_size':args.shard_size}
    checkpoint=out/'checkpoint.json'
    if checkpoint.exists() and json.loads(checkpoint.read_text())!=identity:
        raise ValueError('Checkpoint belongs to different model, inputs or shard settings')
    dump_json(checkpoint,identity)
    shards=out/'shards';shards.mkdir(exist_ok=True)
    queries=iter_records(paths,1)
    if args.limit:queries=islice(queries,args.limit)
    total=pair_total=matched_total=0;started=time.monotonic();new_count=0
    with mp.get_context('spawn').Pool(args.workers,init_predict,(args.index,args.lsh_index,model,args.fields)) as pool:
        shard_num=0
        while chunk:=list(islice(queries,args.shard_size)):
            prefix=shards/f'{shard_num:06d}';manifest=Path(str(prefix)+'.json')
            candidate_path=Path(str(prefix)+'.candidates.tsv')
            match_path=Path(str(prefix)+'.matches.tsv')
            if manifest.exists():
                saved=json.loads(manifest.read_text())
                if saved['first']!=chunk[0]['entity_id'] or saved['last']!=chunk[-1]['entity_id'] or saved['rows']!=len(chunk):
                    raise ValueError('Resume shard/query mismatch')
                if not candidate_path.exists() or not match_path.exists():raise ValueError('Incomplete saved shard')
            else:
                tmpc=Path(str(candidate_path)+'.partial');tmpm=Path(str(match_path)+'.partial')
                pairs=matches=0
                with open(tmpc,'w',encoding='utf-8',newline='') as c,open(tmpm,'w',encoding='utf-8',newline='') as m:
                    for eid,ids,hits in pool.imap(predict_one,chunk,chunksize=4):
                        c.write(eid+'\t'+','.join(ids)+'\n');m.write(eid+'\t'+','.join(hits)+'\n')
                        pairs+=len(ids);matches+=len(hits);new_count+=1
                        if new_count%100==0:log(f'Matched {new_count:,} new S1; {new_count/(time.monotonic()-started):.1f} S1/s')
                tmpc.replace(candidate_path);tmpm.replace(match_path)
                saved={'first':chunk[0]['entity_id'],'last':chunk[-1]['entity_id'],
                       'rows':len(chunk),'pairs':pairs,'matches':matches}
                dump_json(manifest,saved)
            total+=saved['rows'];pair_total+=saved['pairs'];matched_total+=saved['matches'];shard_num+=1
    for suffix,name,column in (('candidates','candidate_pairs.tsv','candidate_entity_ids'),
                                ('matches','matching_results.tsv','matched_entity_ids')):
        dest=out/name;tmp=Path(str(dest)+'.partial')
        with open(tmp,'wb') as handle:
            handle.write(('source1_entity_id\t'+column+'\n').encode('utf-8'))
            for number in range(shard_num):
                with open(shards/f'{number:06d}.{suffix}.tsv','rb') as part:
                    shutil.copyfileobj(part,handle,length=8*1024*1024)
        tmp.replace(dest)
    report={'s1_records':total,'candidate_pairs':pair_total,'matches':matched_total,
            'new_s1_this_run':new_count,'seconds_this_run':time.monotonic()-started,
            'partial':bool(args.limit or args.queries),'split':args.split}
    dump_json(out/'prediction_report.json',report);log(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    for name in ('index','queries','ground-truth','evidence','fields','out'):t.add_argument('--'+name,required=True)
    t.add_argument('--seed',type=int,default=41)
    t=sub.add_parser('predict')
    for name in ('index','lsh-index','normalized-dir','model','out'):t.add_argument('--'+name,required=True)
    t.add_argument('--split',choices=('train','test'),required=True)
    t.add_argument('--queries');t.add_argument('--limit',type=int,default=0)
    t.add_argument('--fields',help='Optional comparison database; compact test indexes include these fields')
    t.add_argument('--workers',type=int,default=4);t.add_argument('--shard-size',type=int,default=1000)
    t=sub.add_parser('build-fields')
    for name in ('normalized-dir','split','candidates','out'):t.add_argument('--'+name,required=True)
    t=sub.add_parser('build-records')
    for name in ('normalized-dir','split','index'):t.add_argument('--'+name,required=True)
    args=p.parse_args()
    if args.command=='train':train(args)
    elif args.command=='predict':predict(args)
    elif args.command=='build-fields':build_fields(args)
    else:build_records(args)

if __name__=='__main__':main()
