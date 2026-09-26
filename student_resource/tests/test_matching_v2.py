from pathlib import Path
import json
import sys
import unittest
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import matching_v2 as v
import numpy as np
from validate_matching_v2 import check_candidates

class MatchingV2Tests(unittest.TestCase):
    def test_threshold_accounts_for_singletons_and_missed_truth(self):
        score,t=v.select_threshold(np.array([.9,.8,.1]),['a','a','b'],np.array([1,0,0]),
                                   {'a':['x','missed'],'b':[]})
        self.assertGreater(t,.8)
        self.assertLessEqual(t,.9)
        self.assertAlmostEqual(score,(1.25/1.5+1)/2)

    def test_model_round_trip_and_feature_contract(self):
        model=json.loads((Path(__file__).resolve().parents[1]/'models/model_v2.json').read_text())
        x=np.zeros((2,model['feature_count']),dtype=np.float32)
        scores=v.predict_scores(x,model)
        fresh=v.lgb.Booster(model_str=model['model_text']).predict(x,num_threads=1)
        np.testing.assert_allclose(scores,fresh)
        self.assertTrue(np.all((scores>=0)&(scores<=1)))
        with self.assertRaisesRegex(ValueError,'incompatible'):
            v.predict_scores(x[:,:-1],model)

    def test_empty_candidate_list(self):
        self.assertEqual(len(v.predict_scores([],{'type':'lightgbm'})),0)
        self.assertEqual(v.m.macro_score({'singleton':[]},{'singleton':[]})['macro_f0_5'],1)

    def test_streaming_candidate_validation_rejects_unscored_match(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'matches.tsv';b=Path(folder)/'candidates.tsv'
            a.write_text('source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\t\n')
            b.write_text('source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\n')
            check_candidates(a,b)
            a.write_text('source1_entity_id\tmatched_entity_ids\nS1-1\tS2-2\nS1-2\t\n')
            with self.assertRaisesRegex(ValueError,'outside candidate'):
                check_candidates(a,b)

if __name__=='__main__':unittest.main()
