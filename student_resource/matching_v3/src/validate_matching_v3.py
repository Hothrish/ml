"""Validate candidates one row at a time, then run the official match validator."""
import argparse
import importlib.util
from itertools import zip_longest

def check_candidates(matching,candidate):
    rows=pairs=0
    with open(matching,encoding='utf-8') as m,open(candidate,encoding='utf-8') as c:
        if m.readline().rstrip('\r\n')!='source1_entity_id\tmatched_entity_ids':
            raise ValueError('Unexpected matching header')
        if c.readline().rstrip('\r\n')!='source1_entity_id\tcandidate_entity_ids':
            raise ValueError('Unexpected candidate header')
        for line_number,(ml,cl) in enumerate(zip_longest(m,c),2):
            if ml is None or cl is None:raise ValueError('Candidate and matching row counts differ')
            mp=ml.rstrip('\r\n').split('\t');cp=cl.rstrip('\r\n').split('\t')
            if len(mp)!=2 or len(cp)!=2:raise ValueError(f'Malformed TSV row {line_number}')
            if mp[0]!=cp[0] or not mp[0].startswith('S1-'):
                raise ValueError(f'Candidate and matching S1 IDs differ at row {line_number}')
            mids=mp[1].split(',') if mp[1] else []
            cids=cp[1].split(',') if cp[1] else []
            if len(cids)!=len(set(cids)):raise ValueError(f'Duplicate candidate ID at row {line_number}')
            if any(not x.startswith(('S2-','S3-')) or x.strip()!=x for x in cids):
                raise ValueError(f'Invalid candidate ID at row {line_number}')
            if not set(mids)<=set(cids):raise ValueError(f'Match outside candidate set at row {line_number}')
            rows+=1;pairs+=len(cids)
            if rows%100000==0:print(f'Validated {rows:,} candidate rows',flush=True)
    print(f'PASS: {rows:,} candidate rows, {pairs:,} scored pairs; every match is a candidate',flush=True)

def main():
    p=argparse.ArgumentParser()
    for key in ('matching','candidate','test-dir','validator'):p.add_argument('--'+key,required=True)
    args=p.parse_args();check_candidates(args.matching,args.candidate)
    spec=importlib.util.spec_from_file_location('official_submission_validator',args.validator)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    errors,warnings=official.validate(args.matching,None,args.test_dir,check_ids=False)
    for warning in warnings:print('NOTE:',warning)
    if errors:raise ValueError('\n'.join(errors))
    print('PASS: official matching format and complete test S1 coverage')

if __name__=='__main__':main()
