"""Compact, exact representation of the existing sorted LSH postings."""
from pathlib import Path
import json,time,sqlite3,argparse
import numpy as np

def build(index,out):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if Path(str(out)+'.json').exists():raise FileExistsError('Use a fresh compact index prefix')
    db=sqlite3.connect(Path(index).resolve().as_uri()+'?mode=ro',uri=True)
    manifest=json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
    if manifest['status']!='ready':raise ValueError('LSH index is not complete')
    cur=db.execute('SELECT key,rid FROM bands ORDER BY key,rid')
    rows=unique=0;last=None;start=time.monotonic()
    paths=[Path(str(out)+suffix+'.partial') for suffix in ('.keys','.offsets','.rids')]
    with open(paths[0],'wb') as keys,open(paths[1],'wb') as offsets,open(paths[2],'wb') as rids:
        while batch:=cur.fetchmany(250000):
            values=np.asarray(batch,dtype='<i8');del batch
            changes=np.flatnonzero(np.r_[True,values[1:,0]!=values[:-1,0]])
            if last is not None and values[0,0]==last:changes=changes[1:]
            if values[:,1].min()<1 or values[:,1].max()>np.iinfo(np.uint32).max:
                raise ValueError('Record IDs do not fit uint32')
            values[changes,0].astype('<i8').tofile(keys)
            (changes.astype('<u8')+rows).tofile(offsets)
            values[:,1].astype('<u4').tofile(rids)
            last=int(values[-1,0]);rows+=len(values);unique+=len(changes)
            if rows%10000000==0:print(f'Compact postings: {rows:,}; {unique:,} buckets; {time.monotonic()-start:.1f}s',flush=True)
        np.asarray([rows],dtype='<u8').tofile(offsets)
    db.close()
    for p,suffix in zip(paths,('.keys','.offsets','.rids')):p.replace(str(out)+suffix)
    report={'version':1,'lsh_manifest':manifest,'rows':rows,'buckets':unique,
            'seconds':time.monotonic()-start,'bytes':8*unique+8*(unique+1)+4*rows}
    Path(str(out)+'.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='lsh_manifest'}),flush=True)

class Lookup:
    def __init__(self,prefix,lsh_meta):
        prefix=str(prefix);meta=json.loads(Path(prefix+'.json').read_text())
        if meta['lsh_manifest']!=lsh_meta:raise ValueError('Compact postings belong to another LSH index')
        for suffix,size in (('.keys',meta['buckets']*8),('.offsets',(meta['buckets']+1)*8),('.rids',meta['rows']*4)):
            if Path(prefix+suffix).stat().st_size!=size:raise ValueError('Incomplete compact index '+suffix)
        self.keys=np.memmap(prefix+'.keys',dtype='<i8',mode='r')
        self.offsets=np.memmap(prefix+'.offsets',dtype='<u8',mode='r')
        self.rids=np.memmap(prefix+'.rids',dtype='<u4',mode='r')
    def get(self,key,limit):
        i=int(np.searchsorted(self.keys,key))
        if i==len(self.keys) or self.keys[i]!=key:return ()
        a,b=int(self.offsets[i]),int(self.offsets[i+1])
        return () if b-a>limit else self.rids[a:b].tolist()
    def close(self):
        for array in (self.keys,self.offsets,self.rids):array._mmap.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();build(args.index,args.out)

if __name__=='__main__':main()
