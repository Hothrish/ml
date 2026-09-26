"""Memory-mapped normalized comparison records aligned to LSH record IDs."""
from pathlib import Path
import json,time,argparse,mmap
import numpy as np
import blocking as b
from functools import lru_cache

FIELDS=('entity_id','country','name_sorted','name_nospace','name_cons','house','locality',
        'addr_norm','addr_nums','addr_postcode','addr_region','addr_street','name_phon','name_legal','script')
RENAMES={'country':'country_norm','house':'addr_house','locality':'addr_locality','script':'name_script'}
def build(index,normalized_dir,split,out):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    if Path(str(out)+'.json').exists():raise FileExistsError('Use a fresh record prefix')
    meta=b.check_index(index,normalized_dir,split);count=offset=0;started=time.monotonic()
    data_path=Path(str(out)+'.data.partial');offset_path=Path(str(out)+'.offsets.partial')
    offsets=[0]
    with open(data_path,'wb') as data,open(offset_path,'wb') as idx:
        for source in (2,3):
            n=0
            for r in b.iter_records(b.source_parts(normalized_dir,split,source),source):
                values=[r.get(RENAMES.get(k,k),'') for k in FIELDS]
                values[1]=values[1].casefold().strip()
                line='\t'.join(values).encode('utf-8')
                data.write(line);offset+=len(line);offsets.append(offset);count+=1;n+=1
                if count%100000==0:
                    np.asarray(offsets,dtype='<u8').tofile(idx);offsets=[]
                if count%1000000==0:print(f'Compact records: {count:,}; {time.monotonic()-started:.1f}s',flush=True)
            if n!=meta['rows'][str(source)]:raise ValueError('Row order/count mismatch')
        np.asarray(offsets,dtype='<u8').tofile(idx)
    data_path.replace(str(out)+'.data');offset_path.replace(str(out)+'.offsets')
    manifest={'version':1,'rows':count,'fields':FIELDS,'base_inputs':meta['inputs'],'split':split,'data_bytes':offset,'seconds':time.monotonic()-started}
    Path(str(out)+'.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Compact records ready: {count:,}, {offset/1e9:.2f} GB',flush=True)

class Lookup:
    def __init__(self,prefix,meta):
        prefix=str(prefix);manifest=json.loads(Path(prefix+'.json').read_text())
        if manifest['base_inputs']!=meta['inputs'] or manifest['split']!=meta['split']:
            raise ValueError('Compact records belong to another base index')
        if Path(prefix+'.offsets').stat().st_size!=8*(manifest['rows']+1) or Path(prefix+'.data').stat().st_size!=manifest['data_bytes']:
            raise ValueError('Incomplete compact records')
        self._offsets=np.memmap(prefix+'.offsets',dtype='<u8',mode='r')
        self.offsets=np.asarray(self._offsets)
        self._file=open(prefix+'.data','rb')
        self.data=mmap.mmap(self._file.fileno(),0,access=mmap.ACCESS_READ)
        self.get=lru_cache(maxsize=4096)(self._get)
        self.rerank=lru_cache(maxsize=4096)(self._rerank)
    def _get(self,rid):
        if rid<1 or rid>=len(self.offsets):raise ValueError('Unknown record ID')
        a,c=int(self.offsets[rid-1]),int(self.offsets[rid])
        return dict(zip(FIELDS,self.data[a:c].decode('utf-8').split('\t')))
    def _rerank(self,rid):
        if rid<1 or rid>=len(self.offsets):raise ValueError('Unknown record ID')
        a,c=int(self.offsets[rid-1]),int(self.offsets[rid])
        values=self.data[a:c].decode('utf-8').split('\t',8)
        return values[2],values[3],values[4],values[7]
    def close(self):
        self.get.cache_clear();self.rerank.cache_clear();self._offsets._mmap.close();self.data.close();self._file.close()

def main():
    p=argparse.ArgumentParser()
    for key in ('index','normalized-dir','split','out'):p.add_argument('--'+key,required=True)
    a=p.parse_args();build(a.index,a.normalized_dir,a.split,a.out)
if __name__=='__main__':main()
