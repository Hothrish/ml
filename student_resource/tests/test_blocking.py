import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from blocking import (Blocker, Config, build_index, build_lsh, evaluate, generate, resume_index,
                      grams, iter_records, make_sample, source_parts)


def record(eid, name, address='', country='India', script='latin'):
    return dict(entity_id=eid, country_norm=country, name_core=name,
                name_sorted=' '.join(sorted(name.split())), name_nospace=name.replace(' ', ''),
                name_cons=''.join(c for c in name if c not in 'aeiou'), name_script=script,
                addr_norm=address, addr_core=address, addr_house=address.split()[0] if address else '',
                addr_locality='')


def write_records(path, records):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write('\t'.join(records[0]) + '\n')
        for rec in records:
            f.write('\t'.join(rec.values()) + '\n')


class BlockingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.s1 = [record('S1-1', 'shree venkateshwara traders', '12 mg road bengaluru'),
                   record('S1-2', 'ecole lumiere', '8 rue victor hugo paris', 'France'),
                   record('S1-3', '', '', ''),
                   record('S1-4', 'lonely café', '', 'Unseen Country')]
        self.s2 = [record('S2-1', 'sri venkateswara tradrs', '12 mg rd bangalore'),
                   record('S2-2', 'ecole lumiere', '8 rue victor hugo paris', 'France'),
                   record('S2-3', 'shree venkateshwara traders', '45 paris', 'France'),
                   record('S2-4', 'unrelated pharmacy', '999 other avenue', 'India')]
        self.s3 = [record('S3-1', 'venkateswara traders', 'bengaluru karnataka'),
                   record('S3-2', 'lumiere ecole', 'paris victor hugo 8', 'France')]
        for source, rows in ((1,self.s1),(2,self.s2),(3,self.s3)):
            write_records(self.root/f'train_source{source}.tsv',rows)
        self.index = self.root/'index.db'
        build_index(self.root, 'train', self.index)

    def tearDown(self):
        self.temp.cleanup()

    def test_typo_pair_and_cross_country_rescue(self):
        blocker = Blocker(self.index)
        try:
            found, _ = blocker.retrieve(self.s1[0])
            by_id = {r['entity_id']:r for r in found}
            self.assertIn('S2-1',by_id)
            self.assertIn('name',by_id['S2-1']['routes'])
            self.assertIn('S3-1',by_id)
            self.assertIn('global_name',by_id['S2-3']['routes'])
            self.assertEqual(len(found),len(by_id))
        finally:
            blocker.close()

    def test_france_empty_and_unknown_countries(self):
        blocker = Blocker(self.index)
        try:
            self.assertIn('S2-2',{r['entity_id'] for r in blocker.retrieve(self.s1[1])[0]})
            self.assertEqual(blocker.retrieve(self.s1[2])[0],[])
            # Unknown/missing country labels cannot cause a KeyError or unsupported bucket.
            blocker.retrieve(self.s1[3])
            unknown = dict(self.s1[0], country_norm='')
            self.assertIn('S2-1',{r['entity_id'] for r in blocker.retrieve(unknown)[0]})
        finally:
            blocker.close()

    def test_safe_fts_query_and_short_names(self):
        blocker = Blocker(self.index)
        try:
            blocker.retrieve(record('S1-x', '" OR * :() - café S.A.', country='F"rance'))
            self.assertTrue(grams('AB'))
            self.assertEqual(grams(''),[])
        finally:
            blocker.close()

    def test_budget_applies_after_union(self):
        blocker = Blocker(self.index,Config(max_candidates=1))
        try:
            results, stats = blocker.retrieve(self.s1[0])
            self.assertEqual(len(results),1)
            self.assertGreater(stats['before_cap'],stats['after_cap'])
        finally:
            blocker.close()

    def test_oversized_exact_bucket_refines_without_arbitrary_truncation(self):
        write_records(self.root/'train_source2.tsv',[
            record('S2-a','common traders','1 main road'),
            record('S2-b','common traders','2 other road'),
            record('S2-c','common traders','3 third road')])
        index = self.root/'oversized.db'
        build_index(self.root,'train',index)
        config = Config(name_k=0,address_k=0,skeleton_k=0,global_name_k=0,exact_bucket_limit=1)
        blocker = Blocker(index,config)
        try:
            found, stats = blocker.retrieve(record('S1-check','common traders','2 other road'))
            self.assertEqual([r['entity_id'] for r in found],['S2-b'])
            self.assertGreater(stats['oversized_exact_buckets'],0)
        finally:
            blocker.close()

    def test_end_to_end_evaluation_includes_missing_positives_and_singletons(self):
        query = self.root/'train_source1.tsv'
        out = self.root/'out'
        stats = generate(self.index,[query],out,split='train')
        gt = self.root/'gt.tsv'
        gt.write_text('source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-1,S2-missing\nS1-2\tS2-2,S3-2\nS1-3\t\nS1-4\t\n',encoding='utf-8')
        result = evaluate(self.index,[query],gt,out/'candidate_pairs.tsv',
                          out/'candidate_evidence.tsv.gz',out/'report.json')
        self.assertEqual(stats['s1_records'],4)
        self.assertAlmostEqual(result['pair_recall'],.8)
        self.assertEqual(result['singletons'],2)
        self.assertIn('S2-missing',result['missing_index_true_id_examples'])
        self.assertAlmostEqual(result['oracle_macro_f0_5_ceiling'],(2.5/2.75+3)/4)

    def test_wrong_split_and_existing_index_rejected(self):
        with self.assertRaises(ValueError):
            generate(self.index,[self.root/'train_source1.tsv'],self.root/'wrong',split='test')
        with self.assertRaises(FileExistsError):
            build_index(self.root,'train',self.index)

    def test_parallel_matches_serial_output(self):
        query = [self.root/'train_source1.tsv']
        generate(self.index,query,self.root/'serial',workers=1)
        generate(self.index,query,self.root/'parallel',workers=2)
        self.assertEqual((self.root/'serial'/'candidate_pairs.tsv').read_bytes(),
                         (self.root/'parallel'/'candidate_pairs.tsv').read_bytes())

    def test_audit_sample_excludes_tuning_queries(self):
        gt = self.root/'gt.tsv'
        gt.write_text('source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\tS2-2\nS1-3\t\nS1-4\t\n',encoding='utf-8')
        queries = [self.root/'train_source1.tsv']
        make_sample(gt,queries,self.root/'tune',2,17)
        make_sample(gt,queries,self.root/'audit',2,29,self.root/'tune'/'queries.tsv')
        tune = {r['entity_id'] for r in iter_records([self.root/'tune'/'queries.tsv'])}
        audit = {r['entity_id'] for r in iter_records([self.root/'audit'/'queries.tsv'])}
        self.assertFalse(tune & audit)
        self.assertEqual(len(tune | audit),4)

    def test_resume_after_stale_tail_removed(self):
        row = self.s2[0]
        for number, contents in ((0,'\t'.join(row)+'\n'+'\t'.join(row.values())+'\n'),
                                 (1,'\t'.join(row.values())+'\n')):
            with gzip.open(self.root/f'train_source2.part-{number:02d}.tsv.gz','wt',encoding='utf-8') as f:
                f.write(contents)
        index = self.root/'interrupted.db'
        with self.assertRaises(Exception):
            build_index(self.root,'train',index,batch_size=1)
        (self.root/'train_source2.part-01.tsv.gz').unlink()
        meta = resume_index(self.root,'train',index,batch_size=1)
        self.assertEqual(meta['rows'],{'2':1,'3':2})
        self.assertEqual(meta['status'],'ready')

    def test_lsh_routes_and_parallel_output(self):
        lsh=self.root/'lsh.db'
        meta=build_lsh(self.root,'train',self.index,lsh,bands=8,batch_size=2)
        self.assertEqual(meta['indexed_rows'],6)
        query=[self.root/'train_source1.tsv']
        generate(self.index,query,self.root/'lsh_serial',lsh_index=lsh,workers=1)
        generate(self.index,query,self.root/'lsh_parallel',lsh_index=lsh,workers=2)
        a=(self.root/'lsh_serial'/'candidate_pairs.tsv').read_bytes()
        b=(self.root/'lsh_parallel'/'candidate_pairs.tsv').read_bytes()
        self.assertEqual(a,b)
        lines=a.decode().splitlines()
        self.assertTrue(any('S2-1' in line for line in lines))
        self.assertTrue(any('S2-2' in line for line in lines))

    def test_deterministic_reservoir_includes_singletons(self):
        gt = self.root/'gt.tsv'
        gt.write_text('source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-2\tS2-2\nS1-3\t\nS1-4\t\n',encoding='utf-8')
        manifest = make_sample(gt,[self.root/'train_source1.tsv'],self.root/'sample',4,17)
        self.assertEqual(manifest['singletons'],2)
        self.assertEqual(len(list(iter_records([self.root/'sample'/'queries.tsv']))),4)

    def test_headerless_parts_and_literal_quotes(self):
        row = record('S1-quoted','"hello" traders')
        lines = ['\t'.join(row)+'\n','\t'.join(row.values())+'\n']
        for part,text in ((0,''.join(lines)),(1,'\t'.join(dict(row,entity_id='S1-next').values())+'\n')):
            with gzip.open(self.root/f'test_source1.part-{part:02d}.tsv.gz','wt',encoding='utf-8') as f:
                f.write(text)
        values = list(iter_records(source_parts(self.root,'test',1),1))
        self.assertEqual(len(values),2)
        self.assertEqual(values[0]['name_core'],'"hello" traders')

    def test_bad_schema_and_missing_part_rejected(self):
        bad = self.root/'bad.tsv'
        bad.write_text('entity_id\tname_core\nS1-a\tx\n',encoding='utf-8')
        with self.assertRaises(ValueError):
            list(iter_records([bad]))
        (self.root/'test_source2.part-01.tsv.gz').touch()
        with self.assertRaises(ValueError):
            source_parts(self.root,'test',2)


if __name__ == '__main__':
    unittest.main()
