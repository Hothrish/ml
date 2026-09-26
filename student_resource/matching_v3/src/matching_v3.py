"""Matcher with exact compact LSH search; experimental version 3."""
import os
import math,re,json
import heapq
from functools import lru_cache
from collections import Counter
from pathlib import Path
import matching_v2 as v
import blocking as b
from compact_lsh import Lookup
from compact_records import Lookup as RecordLookup
from rapidfuzz import fuzz,process
import numpy as np

class CompactLSHBlocker(v.m.FastLSHBlocker):
    def __init__(self,index,lsh,config=None):
        super().__init__(index,lsh,config)
        prefix=os.environ.get('MATCH_COMPACT_LSH')
        self.compact=Lookup(prefix,self.lsh_meta) if prefix else None
        if self.compact:self.db.execute('PRAGMA cache_size=-16384')
        text=os.environ.get('MATCH_COMPACT_RECORDS')
        self.text_lookup=RecordLookup(text,self.meta) if text else None
        if self.text_lookup:self.matching_columns=v.m.COLS[2:]+v.m.EXTRA
        self.rerank=os.environ.get('MATCH_TEXT_RERANK')=='1'
        self.rerank_pool=int(os.environ.get('MATCH_RERANK_POOL','20000'))
        self.pool_rarity=os.environ.get('MATCH_POOL_RARITY')=='1'
        if self.rerank and not self.text_lookup:raise ValueError('Text reranking needs compact records')
    def _band_hits_uncached(self,key):
        if self.compact is not None:return self.compact.get(key,self.config.lsh_bucket_limit)
        return super()._band_hits_uncached(key)
    def close(self):
        if getattr(self,'text_lookup',None) is not None:self.text_lookup.close()
        if getattr(self,'compact',None) is not None:self.compact.close()
        super().close()

    def retrieve(self,rec):
        self._text_scores={}
        try:return super().retrieve(rec)
        finally:self._text_scores.clear()

    def fuzzy(self,rec,view,source,k,global_rescue=False):
        budget=20000 if self.rerank and self.pool_rarity else max(k,self.rerank_pool)
        pool=super().fuzzy(rec,view,source,budget if self.rerank and k else k,global_rescue)
        if not self.rerank or len(pool)<=k:return pool
        if self.pool_rarity and len(pool)>self.rerank_pool:
            country=rec['country_norm'].casefold().strip()
            countries=[''] if global_rescue else ([country] if country else list(self.meta['countries']))
            rarity=Counter()
            for c in countries:
                for key in b.band_keys(rec[b.VIEWS['name'] if global_rescue else b.VIEWS[view]],
                                      'global_name' if global_rescue else view,source,c,self.config.lsh_bands):
                    hits=self._band_hits(key)
                    if hits:
                        weight=1/math.sqrt(len(hits))
                        for rid in hits:rarity[rid]+=weight
            keep={rid for rid,_ in pool[:self.rerank_pool]}
            keep.update(rid for rid,_ in heapq.nsmallest(self.rerank_pool,pool,key=lambda t:(-rarity[t[0]],-t[1],t[0])))
            pool=[item for item in pool if item[0] in keep]
        rows=[self.text_lookup.rerank(rid) for rid,_ in pool]
        def batch(q,field,scorer=fuzz.ratio):
            return process.cdist([q],[r[field] for r in rows],scorer=scorer,dtype=np.float64,workers=1)[0]
        name=batch(rec['name_sorted'],0);addr=batch(rec['addr_norm'],3)
        if view=='address' and not global_rescue:
            score=.75*np.maximum(addr,batch(rec['addr_norm'],3,fuzz.token_sort_ratio))+.25*name
        elif view=='skeleton' and not global_rescue:
            score=.75*batch(rec['name_cons'],2)+.25*addr
        else:
            score=.75*np.maximum(name,batch(rec['name_nospace'],1))+.25*addr
        ranked=np.lexsort(([rid for rid,_ in pool],[-votes for _,votes in pool],-score))[:k]
        return [pool[i] for i in ranked]

v.m.FastLSHBlocker=CompactLSHBlocker
b.LSHBlocker=CompactLSHBlocker

_idf={}
_base_init=v.m.init_predict
_base_views=v.m.query_views

def query_views(q):
    d=_base_views(q)
    d['number_sequence']=re.findall(r'\d+',q['addr_norm'])
    return d

def idf_similarity(a,c,kind):
    table=_idf.get(kind,{})
    default=_idf.get('default',1.)
    wa={t:table.get(t,default) for t in a};wc={t:table.get(t,default) for t in c}
    common=a&c
    dot=sum(wa[t]*wc[t] for t in common)
    denom=math.sqrt(sum(x*x for x in wa.values())*sum(x*x for x in wc.values()))
    overlap=sum(wa[t] for t in common)
    return [dot/denom if denom else 0.,overlap/min(sum(wa.values()),sum(wc.values())) if wa and wc else 0.]

def features(q,r,routes,fusion,views=None):
    views=views or query_views(q)
    f=v.features(q,r,routes,fusion,views)
    qw=views['words'];rw=set(r['name_sorted'].split());qa=views['addr_words'];ra=set(r['addr_norm'].split())
    f.extend(idf_similarity(qw,rw,'name'));f.extend(idf_similarity(qa,ra,'address'))
    qnums=views['number_sequence'];rnums=re.findall(r'\d+',r['addr_norm'])
    qs,rs=set(qnums),set(rnums)
    f.extend([min(len(q['name_nospace']),len(r['name_nospace']))/40,
              max(len(q['name_nospace']),len(r['name_nospace']))/40,
              len(qw)/10,len(rw)/10,len(q['addr_norm'])/150,len(r['addr_norm'])/150,
              len(qs-rs),len(rs-qs),len(qs&rs),float(bool(qnums) and qnums==rnums),
              float(bool(qnums) and sorted(qnums)==sorted(rnums)),abs(len(qnums)-len(rnums)),
              v.Levenshtein.distance(q['name_nospace'],r['name_nospace'])/20,
              fuzz.ratio(q['addr_house'],r['house'])/100 if q['addr_house'] and r['house'] else 0.])
    weights=_idf.get('name',{});default=_idf.get('default',1.)
    for a,c in ((qw,rw),(rw,qw)):
        if a and c:
            longest=max(a,key=lambda t:(len(t),t));rarest=max(a,key=lambda t:(weights.get(t,default),len(t),t))
            weighted=sum(weights.get(t,default)*max(fuzz.ratio(t,u) for u in c) for t in a)
            f.extend([max(fuzz.ratio(longest,u) for u in c)/100,
                      max(fuzz.ratio(rarest,u) for u in c)/100,
                      weighted/(100*sum(weights.get(t,default) for t in a))])
        else:f.extend([0.,0.,0.])
    return f

def configure_features(model):
    global _idf
    _idf=model.get('idf',{})
    v.m.features=features if model.get('feature_version')==3 else v.features
    v.m.query_views=query_views if model.get('feature_version')==3 else _base_views

def init_predict(index,lsh,model,fields=None):
    configure_features(model)
    _base_init(index,lsh,model,fields)

def predict_scores(x,model):
    if not len(x):return v.np.empty(0,dtype=v.np.float32)
    x=v.np.asarray(x,dtype=v.np.float32)
    if model.get('feature_version') not in (2,3) or x.shape[1]!=model['feature_count']:
        raise ValueError('Incompatible model features')
    return model_booster(model['model_text']).predict(x,num_threads=1)

@lru_cache(maxsize=2)
def model_booster(text):return v.lgb.Booster(model_str=text)

def cheap_features(q,records):
    """Native batched equivalents of the 22 edit-distance columns."""
    result=np.zeros((len(records),22),dtype=np.float32)
    pairs=[(q[k],[r[k] for r in records]) for k in ('name_sorted','name_nospace','name_cons','name_phon','addr_norm')]
    pairs.append((' '.join(sorted(q['addr_norm'].split())),[' '.join(sorted(r['addr_norm'].split())) for r in records]))
    for i,(a,values) in enumerate(pairs):
        if not a:continue
        mask=np.asarray([bool(s) for s in values])
        for j,scorer in enumerate((v.Levenshtein.normalized_similarity,v.JaroWinkler.normalized_similarity,fuzz.ratio)):
            values_out=process.cdist([a],values,scorer=scorer,dtype=np.float64,workers=1)[0]
            if j==2:values_out/=100.
            result[:,3*i+j]=np.where(mask,values_out,0.)
    for i,(a,values) in enumerate((pairs[0],pairs[4])):
        if not a:continue
        mask=np.asarray([bool(s) for s in values])
        for j,scorer in enumerate((fuzz.token_set_ratio,fuzz.WRatio)):
            values_out=process.cdist([a],values,scorer=scorer,dtype=np.float64,workers=1)[0]/100.
            result[:,18+2*i+j]=np.where(mask,values_out,0.)
    return result

v.m.init_predict=init_predict
v.m.predict_scores=predict_scores
_base_predict_one=v.m.predict_one

def predict_one(q):
    model=v.m._MODEL
    if model.get('cascade_model_text'):
        candidates,_=v.m._BLOCKER.retrieve(q)
        ids=[r['entity_id'] for r in candidates]
        if not candidates:return q['entity_id'],ids,[]
        records=[r['matching_record'] for r in candidates]
        gate=model_booster(model['cascade_model_text']).predict(cheap_features(q,records),num_threads=1)
        chosen=np.flatnonzero(gate>=model['cascade_threshold']);views=v.m.query_views(q)
        x=[v.m.features(q,records[i],candidates[i]['routes'],candidates[i]['fusion_score'],views) for i in chosen]
        scores=predict_scores(x,model)
        return q['entity_id'],ids,[ids[i] for i,p in zip(chosen,scores) if p>=model['threshold']]
    if not model.get('relative_threshold'):return _base_predict_one(q)
    candidates,_=v.m._BLOCKER.retrieve(q)
    ids=[r['entity_id'] for r in candidates];views=v.m.query_views(q)
    x=[v.m.features(q,r['matching_record'],r['routes'],r['fusion_score'],views) for r in candidates]
    scores=predict_scores(x,model)
    threshold=max(model['threshold'],model['relative_threshold']*float(max(scores,default=0)))
    return q['entity_id'],ids,[eid for eid,p in zip(ids,scores) if p>=threshold]

v.m.predict_one=predict_one

def main():
    import sys
    # Keep the original v2 command contract and add the compact prefix.
    for arg,env in (('--compact-index','MATCH_COMPACT_LSH'),('--compact-records','MATCH_COMPACT_RECORDS')):
        if arg in sys.argv:
            i=sys.argv.index(arg);os.environ[env]=str(Path(sys.argv[i+1]).resolve());del sys.argv[i:i+2]
    if '--text-rerank' in sys.argv:os.environ['MATCH_TEXT_RERANK']='1';sys.argv.remove('--text-rerank')
    if 'predict' in sys.argv and '--model' in sys.argv:
        model=json.loads(Path(sys.argv[sys.argv.index('--model')+1]).read_text(encoding='utf-8'))
        if model.get('feature_version')==3:
            os.environ['MATCH_TEXT_RERANK']='1' if model.get('text_rerank') else '0'
            os.environ['MATCH_POOL_RARITY']='1' if model.get('pool_rarity') else '0'
            os.environ['MATCH_RERANK_POOL']=str(model.get('rerank_pool',20000))
            if not os.environ.get('MATCH_COMPACT_RECORDS'):raise ValueError('Version 3 requires --compact-records')
    v.main()

if __name__=='__main__':main()
