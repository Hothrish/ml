import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import matching as m
from blocking import build_lsh, iter_records

class MatchingTests(unittest.TestCase):
    def test_macro_metric_and_singletons(self):
        truth={'a':['S2-1','S3-1'],'b':[],'c':['S2-2']}
        result=m.macro_score(truth,{'a':['S2-1'],'b':[],'c':[]})
        self.assertAlmostEqual(result['macro_f0_5'],(1.25/1.5+1)/3)
        self.assertEqual(m.macro_score({'a':[]},{'a':['S2-1']})['macro_f0_5'],0)

    def test_missing_values_are_not_exact_matches(self):
        self.assertEqual(m.equal('',''),0)
        self.assertEqual(m.jaccard(set(),set()),0)

    def test_fit_and_json_round_trip(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(1000,3)).astype(np.float32)
        y=(x[:,0]+x[:,1]>0).astype(np.float32)
        model=m.fit_logistic(x[:800],y[:800],epochs=25)
        scores=m.predict_scores(x[800:],json.loads(json.dumps(model)))
        self.assertGreater(np.mean((scores>=.5)==y[800:]),.95)

    def test_compact_index_prediction_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            fields=['entity_id','country_norm','name_core','name_sorted','name_nospace','name_cons',
                    'name_script','addr_norm','addr_core','addr_house','addr_locality',
                    'addr_nums','addr_postcode','addr_region','addr_street','name_phon','name_legal']
            def row(eid,name):
                return dict(zip(fields,[eid,'France',name,name,name.replace(' ',''),'abc','latin',
                         '12 rue paris','12 rue paris','12','paris','12','75001','paris','rue',name,'']))
            for source,names in [(1,['alpha bakery','beta cafe']),(2,['alpha bakery']),(3,['beta cafe'])]:
                with open(root/f'test_source{source}.tsv','w',encoding='utf-8') as h:
                    h.write('\t'.join(fields)+'\n')
                    for i,name in enumerate(names):h.write('\t'.join(row(f'S{source}-{i}',name)[k] for k in fields)+'\n')
            index=root/'test.sqlite';lsh=root/'lsh.sqlite'
            m.build_records(argparse.Namespace(normalized_dir=str(root),split='test',index=str(index)))
            m.build_records(argparse.Namespace(normalized_dir=str(root),split='test',index=str(index)))
            build_lsh(root,'test',index,lsh)
            q=next(iter_records([root/'test_source1.tsv']))
            db=m.connect(index,readonly=True);r=m.fetch_records(db,['S2-0'])['S2-0'];db.close()
            width=len(m.features(q,r,{},0))
            model={'version':1,'mean':[0]*width,'scale':[1]*width,'weights':[0]*(width+1),
                   'threshold':.5,'config':m.asdict(m.Config())}
            path=root/'model.json';path.write_text(json.dumps(model))
            args=argparse.Namespace(model=str(path),normalized_dir=str(root),split='test',
                index=str(index),lsh_index=str(lsh),out=str(root/'result'),queries=None,
                fields=None,limit=0,shard_size=1,workers=2)
            m.predict(args)
            original=(root/'result'/'matching_results.tsv').read_text()
            candidates=(root/'result'/'candidate_pairs.tsv').read_text()
            self.assertEqual(original.splitlines()[1:],candidates.splitlines()[1:])
            m.predict(args)
            self.assertEqual(original,(root/'result'/'matching_results.tsv').read_text())
            self.assertEqual(json.loads((root/'result'/'prediction_report.json').read_text())['new_s1_this_run'],0)
            model['threshold']=.6;path.write_text(json.dumps(model))
            with self.assertRaises(ValueError):m.predict(args)

if __name__=='__main__':unittest.main()
