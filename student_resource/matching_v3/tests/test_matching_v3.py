from pathlib import Path
import sys,json,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import matching_v3 as v
import numpy as np
from compact_lsh import Lookup

class MatchingV3Tests(unittest.TestCase):
    def test_compact_lookup_boundaries_and_large_buckets(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix=str(Path(folder)/'index');meta={'status':'ready'}
            np.asarray([-10,2,9],dtype='<i8').tofile(prefix+'.keys')
            np.asarray([0,2,3,6],dtype='<u8').tofile(prefix+'.offsets')
            np.asarray([4,6,8,2,3,7],dtype='<u4').tofile(prefix+'.rids')
            Path(prefix+'.json').write_text(json.dumps({'buckets':3,'rows':6,'lsh_manifest':meta}))
            lookup=Lookup(prefix,meta)
            self.assertEqual(lookup.get(-10,2),[4,6]);self.assertEqual(lookup.get(2,2),[8])
            self.assertEqual(lookup.get(9,2),());self.assertEqual(lookup.get(10,2),())
            lookup.close()
            with self.assertRaisesRegex(ValueError,'another LSH'):
                Lookup(prefix,{'status':'wrong'})

    def test_singleton_and_missing_true_pair_metric(self):
        score=v.v.m.macro_score({'a':[],'b':['S2-1','S3-1']},{'a':[],'b':['S2-1']})
        self.assertAlmostEqual(score['macro_f0_5'],(1+1.25/1.5)/2)
        self.assertEqual(v.v.m.macro_score({'a':[]},{'a':['S2-2']})['macro_f0_5'],0)

    def test_saved_model_and_feature_count(self):
        model=json.loads((Path(__file__).resolve().parents[1]/'models/model_v3.json').read_text())
        x=np.zeros((2,model['feature_count']),dtype=np.float32)
        expected=v.v.lgb.Booster(model_str=model['model_text']).predict(x,num_threads=1)
        np.testing.assert_allclose(v.predict_scores(x,model),expected)
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            v.predict_scores(x[:,:-1],model)

    def test_batched_features_match_scalar_including_empty_and_unicode(self):
        q={k:'' for k in ('name_sorted','name_nospace','name_cons','name_phon','addr_norm')}
        records=[dict(q),dict(name_sorted='cafe test',name_nospace='cafetest',name_cons='cftst',name_phon='cafe',addr_norm='12 rue principale'),dict(name_sorted='café',name_nospace='café',name_cons='cf',name_phon='',addr_norm='')]
        for query in [q,*records]:
            expected=[]
            for r in records:
                row=[]
                pairs=[(query[k],r[k]) for k in ('name_sorted','name_nospace','name_cons','name_phon','addr_norm')]
                pairs.append((' '.join(sorted(query['addr_norm'].split())),' '.join(sorted(r['addr_norm'].split()))))
                for a,b in pairs:
                    row.extend([v.v.Levenshtein.normalized_similarity(a,b),v.v.JaroWinkler.normalized_similarity(a,b),v.fuzz.ratio(a,b)/100] if a and b else [0.,0.,0.])
                for a,b in (pairs[0],pairs[4]):
                    row.extend([v.fuzz.token_set_ratio(a,b)/100,v.fuzz.WRatio(a,b)/100] if a and b else [0.,0.])
                expected.append(row)
            np.testing.assert_array_equal(v.cheap_features(query,records),np.asarray(expected,dtype=np.float32))

if __name__=='__main__':unittest.main()
