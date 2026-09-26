"""Disk-backed, multi-pass blocking for the supplied normalization schema.

Python 3.10+, standard library only; SQLite must include FTS5.
Run `python src/blocking.py --help`. No labels are used during retrieval.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import re
import sqlite3
import sys
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import islice

VERSION = 1
REQUIRED = {'entity_id', 'country_norm', 'name_core', 'name_sorted',
            'name_nospace', 'name_cons', 'name_script', 'addr_norm',
            'addr_core', 'addr_house', 'addr_locality'}
EXACT_FIELDS = ('name_sorted', 'name_nospace', 'name_cons')
VIEWS = {'name': 'name_core', 'address': 'addr_norm', 'skeleton': 'name_cons'}


@dataclass
class Config:
    # Each fuzzy budget applies separately to S2 and S3.
    name_k: int = 30
    address_k: int = 30
    skeleton_k: int = 20
    global_name_k: int = 5
    exact_bucket_limit: int = 100
    query_terms: int = 16       # 0 = all available grams; benchmark recall/cost
    max_candidates: int = 0   # 0 = no additional cap on the union
    lsh_bands: int = 8
    lsh_bucket_limit: int = 2000

    def validate(self):
        for k, v in asdict(self).items():
            if type(v) is not int or v < 0:
                raise ValueError(f'{k} must be a nonnegative integer')
        if not self.exact_bucket_limit:
            raise ValueError('exact_bucket_limit must be positive')
        if not 1 <= self.lsh_bands <= 16:
            raise ValueError('lsh_bands must be between 1 and 16')
        if not self.lsh_bucket_limit:
            raise ValueError('lsh_bucket_limit must be positive')
        return self


def log(message):
    print(message, flush=True)


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def text_open(path, mode='rt'):
    if str(path).endswith('.gz'):
        return gzip.open(path, mode, encoding='utf-8', newline='')
    return open(path, mode, encoding='utf-8', newline='')


def source_parts(directory, split, source):
    paths = sorted(Path(directory).glob(f'{split}_source{source}.part-*.tsv.gz'))
    if paths:
        numbers = [int(re.search(r'\.part-(\d+)\.', p.name)[1]) for p in paths]
        if numbers != list(range(len(numbers))):
            raise ValueError(f'Missing or unordered normalized parts: {paths}')
        return paths
    plain = Path(directory) / f'{split}_source{source}.tsv'
    if plain.exists():
        return [plain]
    raise FileNotFoundError(f'No normalized {split}_source{source} files in {directory}')


def iter_records(paths, source=None):
    """Read part-00's header and headerless later parts; also allow repeated headers.

The supplied normalizer writes unquoted TSV, including literal quote characters.
Do not use CSV quote interpretation on these files.
"""
    header = None
    for path in paths:
        with text_open(path) as handle:
            for line_number, line in enumerate(handle, 1):
                values = line.rstrip('\r\n').split('\t')
                if values[0] == 'entity_id':
                    if header is not None and values != header:
                        raise ValueError(f'Changed schema in {path}:{line_number}')
                    header = values
                    if len(set(header)) != len(header) or not REQUIRED <= set(header):
                        raise ValueError(f'Invalid normalized schema in {path}: {REQUIRED - set(header)}')
                    continue
                if header is None or len(values) != len(header):
                    raise ValueError(f'Missing header or malformed row in {path}:{line_number}')
                rec = dict(zip(header, values))
                if source and not rec['entity_id'].startswith(f'S{source}-'):
                    raise ValueError(f'Wrong source in {path}:{line_number}')
                yield rec
    if header is None:
        raise ValueError('No normalized header found')


def fingerprint(paths):
    return [{'name': p.name, 'bytes': p.stat().st_size,
             'mtime_ns': p.stat().st_mtime_ns} for p in paths]


def token(value):
    # ASCII alphabetic/digit encoding prevents FTS query syntax injection.
    return 'x' + value.casefold().encode('utf-8').hex()


def grams(value):
    """Unique character trigrams within words, with boundaries and short words.

Boundary padding supplies two grams for a two-letter token. This view tolerates
token reordering; exact nospace keys complement spacing differences.
"""
    result = set()
    for word in re.findall(r'\w+', value.casefold(), flags=re.UNICODE):
        padded = '^' + word + '$'
        result.update(token(padded[i:i + 3]) for i in range(len(padded) - 2))
    return sorted(result)


@lru_cache(maxsize=20000)
def minhash_signature(value):
    """32 seeded minimum hashes over within-token character trigrams."""
    terms = [term.encode('ascii') for term in grams(value)]
    if not terms:
        return ()
    return tuple(min(zlib.crc32(term, seed) for term in terms)
                 for seed in range(1, 33))


@lru_cache(maxsize=4096)
def _band_salt(view, band, source, country):
    data = f'{view}|{band}|{source}|{country}'.encode('utf-8')
    return int.from_bytes(__import__('hashlib').blake2b(data,digest_size=8).digest(),'big')


def _signed64(value):
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= (1 << 63) else value


def band_keys(value, view, source, country, bands):
    signature = minhash_signature(value)
    if not signature:
        return []
    return [_signed64(((signature[2*i] << 32) | signature[2*i+1]) ^
                      _band_salt(view,i,source,country)) for i in range(bands)]


def connect(path, readonly=False, cache_mb=128):
    if readonly:
        db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    else:
        db = sqlite3.connect(path)
    db.execute(f'PRAGMA cache_size={-1024 * cache_mb}')
    db.execute('PRAGMA temp_store=FILE')
    return db


def get_meta(db):
    return json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])


def set_meta(db, value):
    db.execute("INSERT OR REPLACE INTO metadata VALUES ('manifest', ?)", (json.dumps(value),))


def check_index(index, directory, split):
    db = connect(index, readonly=True)
    try:
        meta = get_meta(db)
        if meta['status'] != 'ready' or meta['version'] != VERSION:
            raise ValueError('Index is incomplete or incompatible; use a new index path')
        if meta['split'] != split or meta['partial_index']:
            raise ValueError('Index must be complete and belong to the requested split')
        for source in (2, 3):
            if fingerprint(source_parts(directory,split,source)) != meta['inputs'][str(source)]:
                raise ValueError('Normalized files changed since indexing; build a new index')
        return meta
    finally:
        db.close()


def build_index(directory, split, index, limit_per_source=0, batch_size=5000):
    index = Path(index)
    if index.exists():
        raise FileExistsError(f'{index} exists. Use a new path to avoid overwriting an index.')
    paths = {s: source_parts(directory, split, s) for s in (2, 3)}
    index.parent.mkdir(parents=True, exist_ok=True)
    db = connect(index)
    start = time.monotonic()
    manifest = {'version': VERSION, 'status': 'building', 'split': split,
                'partial_index': bool(limit_per_source), 'rows': {},
                'inputs': {str(s): fingerprint(ps) for s, ps in paths.items()},
                'countries': {}, 'sqlite_version': sqlite3.sqlite_version,
                'retrieval': 'FTS5 BM25 over unique character trigrams'}
    try:
        db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        set_meta(db, manifest)
        db.execute('''CREATE TABLE records(
            rid INTEGER PRIMARY KEY, entity_id TEXT UNIQUE NOT NULL,
            source INTEGER NOT NULL, country TEXT NOT NULL, script TEXT NOT NULL,
            name_sorted TEXT NOT NULL, name_nospace TEXT NOT NULL,
            name_cons TEXT NOT NULL, house TEXT NOT NULL, locality TEXT NOT NULL)''')
        for view in VIEWS:
            db.execute(f"CREATE VIRTUAL TABLE f_{view} USING fts5(terms, country, source, content='', detail=column)")
            db.execute(f"INSERT INTO f_{view}(f_{view},rank) VALUES ('rank','bm25(1.0,0.0,0.0)')")
            db.execute(f"CREATE VIRTUAL TABLE v_{view} USING fts5vocab(f_{view},'col')")
        db.commit()
        countries = Counter()
        rid = 0
        for source, files in paths.items():
            count = 0
            for rec in iter_records(files, source):
                if limit_per_source and count >= limit_per_source:
                    break
                rid += 1
                count += 1
                country = rec['country_norm'].casefold().strip()
                countries[country] += 1
                db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)',
                           (rid, rec['entity_id'], source, country, rec['name_script'],
                            *(rec[f] for f in EXACT_FIELDS), rec['addr_house'], rec['addr_locality']))
                for view, field in VIEWS.items():
                    db.execute(f'INSERT INTO f_{view}(rowid,terms,country,source) VALUES (?,?,?,?)',
                               (rid, ' '.join(grams(rec[field])), token(country), f's{source}'))
                if count % batch_size == 0:
                    db.commit()
                    if count % (batch_size * 10) == 0:
                        log(f'Indexed S{source}: {count:,}; elapsed {time.monotonic()-start:.1f}s')
            manifest['rows'][str(source)] = count
            db.commit()
            log(f'Indexed S{source}: {count:,} records')
        for field in EXACT_FIELDS:
            # Prefix supports the unrefined exact lookup. The trailing component
            # keeps refinement from scanning an entire very common-name bucket.
            db.execute(f'CREATE INDEX i_{field} ON records(country,source,{field},house)')
            db.execute(f'CREATE INDEX i_{field}_locality ON records(country,source,{field},locality)')
        for view in VIEWS:
            log(f'Optimizing {view} index...')
            db.execute(f"INSERT INTO f_{view}(f_{view}) VALUES ('optimize')")
        manifest.update(status='ready', countries=dict(countries),
                        seconds=round(time.monotonic() - start, 3))
        set_meta(db, manifest)
        db.commit()
    finally:
        db.close()
    dump_json(str(index) + '.manifest.json', manifest)
    return manifest


def resume_index(directory, split, index, batch_size=5000):
    """Continue a build from committed rows, after trailing stale parts were removed.

    Only accepts unchanged retained files in the same order. This specifically
    supports a normalizer rerun with fewer workers, which leaves old tail parts.
    Uncommitted rows from a failed batch are rolled back by SQLite.
    """
    db = connect(index)
    start = time.monotonic()
    try:
        manifest = get_meta(db)
        if manifest['status'] == 'ready':
            return check_index(index, directory, split)
        if manifest['status'] != 'building' or manifest['version'] != VERSION or manifest['split'] != split:
            raise ValueError('Index cannot be resumed: incompatible manifest')
        if manifest['partial_index']:
            raise ValueError('Partial smoke-test indexes cannot be resumed')
        paths = {s: source_parts(directory, split, s) for s in (2,3)}
        for source, files in paths.items():
            current = fingerprint(files)
            original = manifest['inputs'][str(source)]
            if len(current) > len(original) or original[:len(current)] != current:
                raise ValueError(f'S{source} normalized inputs changed within retained parts; rebuild a fresh index')
            manifest['inputs'][str(source)] = current
        counts = {source: db.execute('SELECT count(*) FROM records WHERE source=?',(source,)).fetchone()[0]
                  for source in (2,3)}
        rid = db.execute('SELECT coalesce(max(rid),0) FROM records').fetchone()[0]
        if rid != sum(counts.values()):
            raise ValueError('Noncontiguous index row IDs; rebuild a fresh index')
        if counts[3] and not counts[2]:
            raise ValueError('Unexpected source order in partial index')
        countries = Counter(dict(db.execute('SELECT country,count(*) FROM records GROUP BY country').fetchall()))
        for source, files in paths.items():
            count = 0
            for rec in iter_records(files, source):
                count += 1
                if count <= counts[source]:
                    # Check the checkpoint boundary before appending more rows.
                    if count == counts[source]:
                        old = db.execute('SELECT entity_id FROM records WHERE rid=?',(rid if source == 3 else count,)).fetchone()
                        if old and old[0] != rec['entity_id']:
                            raise ValueError('Normalized row order changed; rebuild a fresh index')
                    continue
                rid += 1
                country = rec['country_norm'].casefold().strip()
                countries[country] += 1
                db.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)',
                           (rid, rec['entity_id'], source, country, rec['name_script'],
                            *(rec[f] for f in EXACT_FIELDS), rec['addr_house'], rec['addr_locality']))
                for view, field in VIEWS.items():
                    db.execute(f'INSERT INTO f_{view}(rowid,terms,country,source) VALUES (?,?,?,?)',
                               (rid, ' '.join(grams(rec[field])), token(country), f's{source}'))
                if count % batch_size == 0:
                    db.commit()
                    if count % (batch_size * 10) == 0:
                        log(f'Indexed S{source}: {count:,}; resume elapsed {time.monotonic()-start:.1f}s')
            if count < counts[source]:
                raise ValueError(f'S{source} has fewer normalized records than this index already contains')
            manifest['rows'][str(source)] = count
            db.commit()
            log(f'Indexed S{source}: {count:,} records (resumed)')
        for field in EXACT_FIELDS:
            db.execute(f'CREATE INDEX IF NOT EXISTS i_{field} ON records(country,source,{field},house)')
            db.execute(f'CREATE INDEX IF NOT EXISTS i_{field}_locality ON records(country,source,{field},locality)')
        for view in VIEWS:
            log(f'Optimizing {view} index...')
            db.execute(f"INSERT INTO f_{view}(f_{view}) VALUES ('optimize')")
        manifest.update(status='ready',countries=dict(countries),
                        resume_seconds=round(time.monotonic()-start,3))
        set_meta(db,manifest)
        db.commit()
    finally:
        db.close()
    dump_json(str(index)+'.manifest.json',manifest)
    return manifest


def build_lsh(directory, split, base_index, lsh_index, bands=8, batch_size=5000):
    """Build an exact band lookup beside a completed record index.

    Reads the normalized sources once. Row order is tied to the base index and
    checked at source boundaries and every 100,000 rows. The band table is
    append-only during loading, then its lookup index is created in one pass.
    """
    if not 1 <= bands <= 16:
        raise ValueError('bands must be between 1 and 16')
    lsh_index = Path(lsh_index)
    if lsh_index.exists():
        raise FileExistsError(f'{lsh_index} exists; use a new LSH index path')
    base = check_index(base_index, directory, split)
    lsh_index.parent.mkdir(parents=True, exist_ok=True)
    source_db = connect(base_index, readonly=True)
    db = connect(lsh_index, cache_mb=64)
    began = time.monotonic()
    meta = {'version': VERSION, 'status': 'building', 'split':split,
            'bands':bands, 'base_rows':base['rows'], 'base_inputs':base['inputs'],
            'indexed_rows':0, 'algorithm':'seeded CRC32 MinHash, two hashes per band'}
    try:
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA synchronous=NORMAL')
        db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        set_meta(db,meta)
        db.execute('CREATE TABLE bands(key INTEGER NOT NULL,rid INTEGER NOT NULL)')
        db.commit()
        rid=0
        rows=[]
        for source in (2,3):
            count=0
            for rec in iter_records(source_parts(directory,split,source),source):
                rid+=1;count+=1
                if count==1 or count % 100000 == 0 or count==base['rows'][str(source)]:
                    old=source_db.execute('SELECT entity_id FROM records WHERE rid=?',(rid,)).fetchone()
                    if not old or old[0]!=rec['entity_id']:
                        raise ValueError(f'Normalized row order differs from base index at S{source} row {count}')
                country=rec['country_norm'].casefold().strip()
                for view,field in VIEWS.items():
                    value=rec[field]
                    rows.extend((key,rid) for key in band_keys(value,view,source,country,bands))
                # Global rescue indexes the same name with no country salt.
                rows.extend((key,rid) for key in band_keys(rec['name_core'],'global_name',source,'',bands))
                if count % batch_size == 0:
                    db.executemany('INSERT INTO bands VALUES (?,?)',rows)
                    rows.clear()
                    db.commit()
                    if count % 50000 == 0:
                        log(f'LSH S{source}: {count:,}; elapsed {time.monotonic()-began:.1f}s')
            if count != base['rows'][str(source)]:
                raise ValueError(f'S{source} normalized row count changed: {count} != {base["rows"][str(source)]}')
            if rows:
                db.executemany('INSERT INTO bands VALUES (?,?)',rows)
                rows.clear()
            db.commit()
            log(f'LSH S{source}: {count:,} records indexed')
        meta['indexed_rows']=rid
        log('Sorting LSH bands into the lookup index...')
        db.execute('CREATE INDEX band_lookup ON bands(key,rid)')
        meta.update(status='ready',seconds=round(time.monotonic()-began,3))
        set_meta(db,meta)
        db.commit()
    finally:
        db.close()
        source_db.close()
    dump_json(str(lsh_index)+'.manifest.json',meta)
    return meta


def check_lsh(lsh_index,base_index):
    db=connect(lsh_index,readonly=True)
    source_db=connect(base_index,readonly=True)
    try:
        meta=get_meta(db)
        base=get_meta(source_db)
        if (meta['status']!='ready' or meta['version']!=VERSION or
            meta['split']!=base['split'] or meta['base_inputs']!=base['inputs'] or
            meta['base_rows']!=base['rows']):
            raise ValueError('LSH index is incomplete or does not match the record index')
        return meta
    finally:
        db.close();source_db.close()


class Blocker:
    def __init__(self, index, config=None):
        self.config = (config or Config()).validate()
        self.db = connect(index, readonly=True)
        self.meta = get_meta(self.db)
        if self.meta['status'] != 'ready' or self.meta['version'] != VERSION:
            self.db.close()
            raise ValueError('Index is incomplete or belongs to a different code version')
        self.frequencies = lru_cache(maxsize=100000)(self._frequencies)

    def close(self):
        self.frequencies.cache_clear()
        self.db.close()


    def _frequencies(self, view, term):
        row = self.db.execute(f"SELECT doc FROM v_{view} WHERE term=? AND col='terms'", (term,)).fetchone()
        return row[0] if row else 0

    def fuzzy(self, rec, view, source, k, global_rescue=False):
        if not k:
            return []
        available = [(self.frequencies(view, t), t) for t in grams(rec[VIEWS[view]])]
        available = sorted((n, t) for n, t in available if n)
        if self.config.query_terms:
            available = available[:self.config.query_terms]
        if not available:
            return []
        expression = 'terms:(' + ' OR '.join(t for _, t in available) + ')'
        expression += f' AND source:s{source}'
        country = rec['country_norm'].casefold().strip()
        if country:
            expression += (' NOT ' if global_rescue else ' AND ') + 'country:' + token(country)
        # Known countries: rescue specifically targets other/unknown countries.
        # Missing query country: main searches already cover every country.
        elif global_rescue:
            return []
        rows = self.db.execute(f'SELECT rowid,rank FROM f_{view} WHERE f_{view} MATCH ? ORDER BY rank LIMIT ?',
                               (expression, k)).fetchall()
        return [(rid, -score) for rid, score in rows]

    def retrieve(self, rec):
        evidence = defaultdict(dict)
        diagnostics = Counter()
        country = rec['country_norm'].casefold().strip()

        def add(rid, route, rank, score=None):
            evidence[rid][route] = {'rank': rank, 'score': score}

        for source in (2, 3):
            for field in EXACT_FIELDS:
                value = rec[field]
                if not value:
                    continue
                # Missing country is handled by unrestricted fuzzy search.
                if not country:
                    continue
                rows = self.db.execute(f'SELECT rid FROM records WHERE country=? AND source=? AND {field}=? LIMIT ?',
                                       (country, source, value, self.config.exact_bucket_limit + 1)).fetchall()
                if len(rows) > self.config.exact_bucket_limit:
                    diagnostics['oversized_exact_buckets'] += 1
                    # Preserve selective address-refined exact hits instead of taking arbitrary IDs.
                    for col, query_field in (('house', 'addr_house'), ('locality', 'addr_locality')):
                        if not rec[query_field]:
                            continue
                        refined = self.db.execute(f'SELECT rid FROM records WHERE country=? AND source=? AND {field}=? AND {col}=? LIMIT ?',
                                                  (country, source, value, rec[query_field], self.config.exact_bucket_limit + 1)).fetchall()
                        if len(refined) <= self.config.exact_bucket_limit:
                            for (rid,) in refined:
                                add(rid, f'exact_{field}_{col}', 1)
                    continue
                for (rid,) in rows:
                    add(rid, 'exact_' + field, 1)
            for view, k in (('name', self.config.name_k), ('address', self.config.address_k),
                            ('skeleton', self.config.skeleton_k)):
                for rank, (rid, score) in enumerate(self.fuzzy(rec, view, source, k), 1):
                    add(rid, view, rank, score)
            for rank, (rid, score) in enumerate(self.fuzzy(rec, 'name', source, self.config.global_name_k, True), 1):
                add(rid, 'global_name', rank, score)

        diagnostics['before_cap'] = len(evidence)
        # Rank fusion uses ranks rather than incomparable BM25 magnitudes.
        def fusion(rid):
            return sum(1.0 / (60 + r['rank']) for r in evidence[rid].values())
        ranked = sorted(evidence, key=lambda rid: (-fusion(rid), rid))
        if self.config.max_candidates:
            ranked = ranked[:self.config.max_candidates]
        diagnostics['after_cap'] = len(ranked)
        details = {}
        matching_columns = getattr(self, 'matching_columns', ())
        extra_select = ',' + ','.join(matching_columns) if matching_columns else ''
        for offset in range(0,len(ranked),500):
            chunk = ranked[offset:offset+500]
            if getattr(self,'text_lookup',None) is not None:
                for rid in chunk:
                    r=self.text_lookup.get(rid)
                    details[rid]=(r['entity_id'],int(r['entity_id'][1]),r['country'],r['script'],*(r[k] for k in matching_columns))
            else:
                for row in self.db.execute('SELECT rid,entity_id,source,country,script' + extra_select + ' FROM records WHERE rid IN (' + ','.join('?' for _ in chunk) + ')',chunk):
                    details[row[0]] = row[1:]
        results = []
        for rid in ranked:
            eid, source, candidate_country, script = details[rid][:4]
            results.append({'entity_id': eid, 'source': source, 'country': candidate_country,
                            'script': script, 'routes': evidence[rid], 'fusion_score': fusion(rid),'rid':rid})
            if matching_columns:
                results[-1]['matching_record'] = dict(entity_id=eid,country=candidate_country,
                                                     **dict(zip(matching_columns,details[rid][4:])))
        return results, diagnostics


class LSHBlocker(Blocker):
    def __init__(self,index,lsh_index,config=None):
        super().__init__(index,config)
        self.lsh_meta=check_lsh(lsh_index,index)
        self.lsh=connect(lsh_index,readonly=True,cache_mb=64)
        if self.config.lsh_bands > self.lsh_meta['bands']:
            self.close()
            raise ValueError('Configuration requests more LSH bands than this index contains')

    def close(self):
        if hasattr(self,'lsh'):
            self.lsh.close()
        super().close()

    def fuzzy(self,rec,view,source,k,global_rescue=False):
        if not k:
            return []
        country=rec['country_norm'].casefold().strip()
        if global_rescue and not country:
            return []
        countries=list(self.meta['countries']) if not country else [country]
        view_key='global_name' if global_rescue else view
        source_field=VIEWS['name'] if global_rescue else VIEWS[view]
        votes=Counter()
        for candidate_country in ([''] if global_rescue else countries):
            for key in band_keys(rec[source_field],view_key,source,candidate_country,self.config.lsh_bands):
                ids=self.lsh.execute('SELECT rid FROM bands WHERE key=? LIMIT ?',
                                     (key,self.config.lsh_bucket_limit+1)).fetchall()
                if len(ids)>self.config.lsh_bucket_limit:
                    continue
                votes.update(row[0] for row in ids)
        if global_rescue and votes:
            kept={}
            for offset in range(0,len(votes),500):
                chunk=list(votes)[offset:offset+500]
                rows=self.db.execute('SELECT rid,country FROM records WHERE rid IN ('+
                                     ','.join('?' for _ in chunk)+')',chunk).fetchall()
                kept.update((rid,votes[rid]) for rid,c in rows if c!=country)
            votes=Counter(kept)
        ranked=sorted(votes,key=lambda rid:(-votes[rid],rid))[:k]
        return [(rid,float(votes[rid])) for rid in ranked]


def percentile(histogram, q):
    n = sum(histogram.values())
    if not n:
        return 0
    position = max(1, math.ceil(q * n))
    accumulated = 0
    for value, count in sorted(histogram.items()):
        accumulated += count
        if accumulated >= position:
            return value


_WORKER_BLOCKER = None


def _init_worker(index, config, lsh_index):
    global _WORKER_BLOCKER
    _WORKER_BLOCKER = LSHBlocker(index,lsh_index,config) if lsh_index else Blocker(index,config)


def _query_worker(rec):
    candidates, diagnostics = _WORKER_BLOCKER.retrieve(rec)
    return rec['entity_id'], rec['country_norm'], candidates, diagnostics


def query_stream(blocker, index, queries, workers, lsh_index=None):
    records = iter_records(queries, 1)
    if workers == 1:
        for rec in records:
            results, diagnostics = blocker.retrieve(rec)
            yield rec['entity_id'], rec['country_norm'], results, diagnostics
    else:
        # Bound records and result objects in flight; Windows uses spawn safely.
        with mp.get_context('spawn').Pool(workers, _init_worker, (str(index),blocker.config,
                                                                 str(lsh_index) if lsh_index else None)) as pool:
            while batch := list(islice(records, workers * 32)):
                yield from pool.imap(_query_worker, batch, chunksize=4)


def generate(index, queries, out_dir, config=None, split=None, emit_evidence=True, workers=1,
             lsh_index=None):
    if workers < 1:
        raise ValueError('workers must be positive')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / 'candidate_pairs.tsv'
    if result_path.exists() or (out_dir / 'blocking_stats.json').exists():
        raise FileExistsError(f'Use a fresh output directory: {out_dir}')
    blocker = LSHBlocker(index,lsh_index,config) if lsh_index else Blocker(index, config)
    if split and blocker.meta['split'] != split:
        blocker.close()
        raise ValueError('Cannot mix training and test indexes')
    start = time.monotonic()
    n = total = 0
    histogram, routes, diagnostics, countries = Counter(), Counter(), Counter(), Counter()
    tmp = result_path.with_suffix('.tsv.partial')
    evidence_path = out_dir / 'candidate_evidence.tsv.gz'
    evidence_tmp = out_dir / 'candidate_evidence.partial.tsv.gz'
    evidence_file = None
    # Unique query IDs are enforced using a temporary disk-backed SQLite table.
    blocker.db.execute('CREATE TEMP TABLE seen_s1(id TEXT PRIMARY KEY)')
    try:
        if emit_evidence:
            evidence_file = text_open(evidence_tmp, 'wt')
            evidence_file.write('source1_entity_id\tcandidate_entity_id\troutes_json\tfusion_score\n')
        with open(tmp, 'w', encoding='utf-8', newline='') as handle:
            handle.write('source1_entity_id\tcandidate_entity_ids\n')
            for eid, country, results, diag in query_stream(blocker,index,queries,workers,lsh_index):
                blocker.db.execute('INSERT INTO seen_s1 VALUES (?)', (eid,))
                handle.write(eid + '\t' + ','.join(r['entity_id'] for r in results) + '\n')
                if evidence_file:
                    for candidate in results:
                        evidence_file.write(eid + '\t' + candidate['entity_id'] + '\t' +
                                            json.dumps(candidate['routes'], separators=(',', ':')) + '\t' +
                                            str(candidate['fusion_score']) + '\n')
                n += 1
                total += len(results)
                histogram[len(results)] += 1
                countries[country] += 1
                diagnostics.update(diag)
                for r in results:
                    routes.update(r['routes'].keys())
                if n % 100 == 0:
                    log(f'Blocked {n:,} S1; {total/max(n,1):.1f} candidates/S1; {n/max(time.monotonic()-start,.001):.1f} S1/s')
    finally:
        if evidence_file:
            evidence_file.close()
        blocker.close()
    # Only publish completed output files; interrupted runs leave .partial files.
    tmp.replace(result_path)
    if emit_evidence:
        evidence_tmp.replace(evidence_path)
    stats = {'version': VERSION, 'config': asdict(config or Config()),
             's1_records': n, 'candidate_pairs': total, 'mean_candidates': total/max(n,1),
             'p50_candidates': percentile(histogram,.5), 'p95_candidates': percentile(histogram,.95),
             'p99_candidates': percentile(histogram,.99), 'max_candidates': max(histogram, default=0),
             'zero_candidate_records': histogram[0], 'countries': dict(countries),
             'candidate_route_counts': dict(routes), 'diagnostics': dict(diagnostics),
             'seconds': time.monotonic()-start,
             'queries_per_second': n / max(time.monotonic()-start,.001),
             'workers': workers, 'index_manifest': blocker.meta,
             'lsh_index':str(lsh_index) if lsh_index else None,
             'query_inputs': fingerprint(list(map(Path, queries)))}
    dump_json(out_dir / 'blocking_stats.json', stats)
    return stats


def iter_ground_truth(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        if handle.readline().rstrip('\r\n').split('\t') != ['source1_entity_id','matched_entity_ids']:
            raise ValueError('Unexpected ground-truth header')
        for line in handle:
            fields = line.rstrip('\r\n').split('\t')
            if len(fields) != 2 or not fields[0].startswith('S1-'):
                raise ValueError('Malformed ground-truth row')
            ids = fields[1].split(',') if fields[1] else []
            if len(ids) != len(set(ids)) or any(not x.startswith(('S2-', 'S3-')) for x in ids):
                raise ValueError('Malformed ground-truth matched IDs')
            yield fields[0], ids


def make_sample(gt, queries, out_dir, size=1000, seed=17, exclude_ids=None):
    if size < 1:
        raise ValueError('Sample size must be positive')
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError('Use an empty sample output directory')
    out_dir.mkdir(parents=True, exist_ok=True)
    excluded = set()
    if exclude_ids:
        with open(exclude_ids, encoding='utf-8') as handle:
            next(handle)
            excluded = {line.split('\t', 1)[0] for line in handle}
    rng, reservoir, population = random.Random(seed), [], 0
    for item in iter_ground_truth(gt):
        if item[0] in excluded:
            continue
        population += 1
        if len(reservoir) < size:
            reservoir.append(item)
        else:
            slot = rng.randrange(population)
            if slot < size:
                reservoir[slot] = item
    chosen = dict(reservoir)
    if len(chosen) != len(reservoir):
        raise ValueError('Duplicate S1 IDs in sampled ground truth')
    found = set()
    with open(out_dir / 'queries.tsv', 'w', encoding='utf-8', newline='') as handle:
        writer = None
        for rec in iter_records(queries, 1):
            if rec['entity_id'] not in chosen:
                continue
            if rec['entity_id'] in found:
                raise ValueError('Duplicate sampled S1 ID')
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(rec), delimiter='\t',
                                        lineterminator='\n', quoting=csv.QUOTE_NONE, quotechar=None)
                writer.writeheader()
            writer.writerow(rec)
            found.add(rec['entity_id'])
    if found != set(chosen):
        raise ValueError(f'{len(set(chosen)-found)} sampled S1 IDs missing from normalized data')
    with open(out_dir / 'ground_truth.tsv', 'w', encoding='utf-8', newline='') as handle:
        handle.write('source1_entity_id\tmatched_entity_ids\n')
        for eid, matches in sorted(chosen.items()):
            handle.write(eid + '\t' + ','.join(matches) + '\n')
    manifest = {'seed': seed, 'requested_size': size, 'sample_size': len(chosen),
                'population_s1': population, 'singletons': sum(not ids for ids in chosen.values()),
                'excluded_s1': len(excluded),
                'sampling': 'uniform reservoir over all S1 ground-truth rows, including singletons'}
    dump_json(out_dir / 'sample_manifest.json', manifest)
    return manifest


def evaluate(index, queries, ground_truth, candidate_file, evidence_file, report):
    """Evaluate a bounded held-out sample, against a full train index.

No missing indexed positive is removed from a denominator. Candidate IDs,
duplicate queries, and exact query/label coverage are checked before scoring.
"""
    records = {}
    for r in iter_records(queries, 1):
        if r['entity_id'] in records:
            raise ValueError('Duplicate evaluation query ID')
        records[r['entity_id']] = r
    truth = {}
    for eid, ids in iter_ground_truth(ground_truth):
        if eid in truth:
            raise ValueError('Duplicate ground-truth S1 ID')
        truth[eid] = ids
    if set(records) != set(truth):
        raise ValueError('Queries and ground truth must cover exactly the same sampled S1 IDs')
    db = connect(index, readonly=True)
    meta = get_meta(db)
    if meta['split'] != 'train' or meta['status'] != 'ready':
        db.close()
        raise ValueError('Evaluation requires a ready training index')
    predicted = {}
    with open(candidate_file, encoding='utf-8', newline='') as handle:
        if handle.readline().rstrip('\r\n') != 'source1_entity_id\tcandidate_entity_ids':
            raise ValueError('Invalid candidates header')
        for line in handle:
            eid, ids = line.rstrip('\r\n').split('\t')
            values = ids.split(',') if ids else []
            if eid in predicted or len(values) != len(set(values)):
                raise ValueError('Duplicate candidate or query ID')
            predicted[eid] = set(values)
    if set(predicted) != set(truth):
        raise ValueError('Candidates must contain exactly one row for every sampled S1')
    for values in predicted.values():
        for eid in values:
            if db.execute('SELECT 1 FROM records WHERE entity_id=?', (eid,)).fetchone() is None:
                raise ValueError(f'Candidate {eid} does not exist in this index')
    route_hits, unique_hits = Counter(), Counter()
    if evidence_file:
        remaining = {eid: set(ids) for eid, ids in predicted.items()}
        with text_open(evidence_file) as handle:
            handle.readline()
            for line in handle:
                eid, candidate, details, _ = line.rstrip('\r\n').split('\t')
                if eid not in remaining or candidate not in remaining[eid]:
                    raise ValueError('Evidence contains an extra or duplicate pair')
                remaining[eid].remove(candidate)
                if candidate in truth[eid]:
                    routes = list(json.loads(details))
                    route_hits.update(routes)
                    if len(routes) == 1:
                        unique_hits[routes[0]] += 1
        if any(remaining.values()):
            raise ValueError('Evidence is missing candidate pairs')
    hits = total = non_singletons = complete = 0
    macro_recall = oracle = 0.0
    singleton_candidates = singleton_count = 0
    slices = defaultdict(lambda: Counter(total=0, retrieved=0))
    histogram = Counter()
    missing_index = []
    missing_index_count = 0
    for eid, matched in truth.items():
        candidates = predicted[eid]
        histogram[len(candidates)] += 1
        recovered = len(candidates.intersection(matched))
        hits += recovered
        total += len(matched)
        if matched:
            non_singletons += 1
            recall = recovered / len(matched)
            macro_recall += recall
            complete += recovered == len(matched)
            # Oracle accepts exactly the retrieved true links (precision 1).
            oracle += 1.25 * recovered / (recovered + .25 * len(matched))
        else:
            singleton_count += 1
            singleton_candidates += bool(candidates)
            oracle += 1
        for mid in matched:
            row = db.execute('SELECT source,country,script FROM records WHERE entity_id=?', (mid,)).fetchone()
            labels = ['country:' + records[eid]['country_norm'], 'source:' + mid[:2]]
            if row:
                labels += ['script:' + row[2], 'country_agreement:' + str(row[1] == records[eid]['country_norm'].casefold().strip())]
            else:
                missing_index_count += 1
                if len(missing_index) < 20:
                    missing_index.append(mid)
                labels += ['script:missing_from_index']
            for label in labels:
                slices[label]['total'] += 1
                slices[label]['retrieved'] += mid in candidates
    db.close()
    n = len(truth)
    candidate_count = sum(len(v) for v in predicted.values())
    pool_size = sum(meta['rows'].values())
    result = {'evaluation_is_against_full_index': not meta['partial_index'],
              'query_count': n, 'true_links': total, 'retrieved_true_links': hits,
              'pair_recall': hits/total if total else None,
              'macro_recall_non_singletons': macro_recall/non_singletons if non_singletons else None,
              'all_links_retrieved_fraction_non_singletons': complete/non_singletons if non_singletons else None,
              'oracle_macro_f0_5_ceiling': oracle/n if n else None,
              'candidate_pairs': candidate_count, 'mean_candidates': candidate_count/max(n,1),
              'p95_candidates': percentile(histogram,.95), 'p99_candidates': percentile(histogram,.99),
              'max_candidates': max(histogram, default=0), 'zero_candidate_s1': histogram[0],
              'global_reduction_ratio': 1-candidate_count/(n*pool_size) if n and pool_size else None,
              'singletons': singleton_count, 'singletons_with_candidates': singleton_candidates,
              'missing_index_true_links': missing_index_count,
              'missing_index_true_id_examples': missing_index,
              'slices': {k: dict(v, recall=v['retrieved']/v['total']) for k,v in sorted(slices.items())},
              'retrieved_true_links_by_route': dict(route_hits),
              'true_links_only_retrieved_by_this_route': dict(unique_hits),
              'note': 'Oracle is a blocking ceiling, not classifier performance. Singleton candidates are not yet false matches.'}
    dump_json(report, result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Check Python and SQLite FTS5 support')
    b = sub.add_parser('build', help='Build reusable S2/S3 disk indexes')
    b.add_argument('--normalized-dir', required=True)
    b.add_argument('--split', choices=['train','test'], required=True)
    b.add_argument('--index', required=True)
    b.add_argument('--limit-per-source', type=int, default=0, help='SMOKE TEST ONLY: partial index, 0=all')
    c = sub.add_parser('check-index', help='Verify ready/full index and source-file fingerprints')
    c.add_argument('--index', required=True)
    c.add_argument('--normalized-dir', required=True)
    c.add_argument('--split', choices=['train','test'], required=True)
    r = sub.add_parser('resume', help='Resume an interrupted full index after removing stale tail parts')
    r.add_argument('--index', required=True)
    r.add_argument('--normalized-dir', required=True)
    r.add_argument('--split', choices=['train','test'], required=True)
    l = sub.add_parser('build-lsh', help='Build MinHash band index for fast candidate retrieval')
    l.add_argument('--normalized-dir',required=True)
    l.add_argument('--split',choices=['train','test'],required=True)
    l.add_argument('--index',required=True,help='Ready record index built with build/resume')
    l.add_argument('--lsh-index',required=True)
    l.add_argument('--bands',type=int,default=8)
    l.add_argument('--limit-per-source',type=int,default=0,
                   help='Reserved for tests; use a full record index in production')
    lc = sub.add_parser('check-lsh', help='Verify an existing MinHash index')
    lc.add_argument('--lsh-index', required=True)
    lc.add_argument('--index', required=True)
    s = sub.add_parser('sample', help='Select random training S1 queries including singletons')
    s.add_argument('--normalized-dir', required=True)
    s.add_argument('--ground-truth', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--size', type=int, default=1000)
    s.add_argument('--seed', type=int, default=17)
    s.add_argument('--exclude-ids', help='Prior sample queries.tsv; create a disjoint audit sample')
    g = sub.add_parser('generate', help='Retrieve and union candidate pairs; uses no labels')
    g.add_argument('--index', required=True)
    g.add_argument('--normalized-dir')
    g.add_argument('--queries', help='Normalized sample queries.tsv, instead of full S1 parts')
    g.add_argument('--split', choices=['train','test'], required=True)
    g.add_argument('--out', required=True)
    g.add_argument('--config')
    g.add_argument('--no-evidence', action='store_true')
    g.add_argument('--workers', type=int, default=1, help='Concurrent read-only retrieval processes; try 4 on an SSD')
    g.add_argument('--lsh-index', help='Use the fast MinHash index instead of FTS5 BM25')
    e = sub.add_parser('evaluate', help='Evaluate generated candidates on a bounded labeled sample')
    e.add_argument('--index', required=True)
    e.add_argument('--queries', required=True)
    e.add_argument('--ground-truth', required=True)
    e.add_argument('--candidates', required=True)
    e.add_argument('--evidence')
    e.add_argument('--report', required=True)
    return p


def main():
    args = parser().parse_args()
    if args.command == 'doctor':
        db = sqlite3.connect(':memory:')
        db.execute('CREATE VIRTUAL TABLE check_fts USING fts5(text)')
        db.close()
        log(f'PASS: Python {sys.version.split()[0]}, SQLite {sqlite3.sqlite_version}, FTS5 available')
    elif args.command == 'build':
        if args.limit_per_source < 0:
            raise ValueError('limit-per-source must be nonnegative')
        log(json.dumps(build_index(args.normalized_dir, args.split, args.index, args.limit_per_source), indent=2))
    elif args.command == 'check-index':
        meta = check_index(args.index, args.normalized_dir, args.split)
        log(f"PASS: {meta['split']} index, {sum(meta['rows'].values()):,} S2/S3 records")
    elif args.command == 'resume':
        meta = resume_index(args.normalized_dir,args.split,args.index)
        log(f"PASS: {meta['split']} index, {sum(meta['rows'].values()):,} S2/S3 records")
    elif args.command == 'build-lsh':
        if args.limit_per_source:
            raise ValueError('Build LSH against a completed record index; --limit-per-source is unused')
        meta=build_lsh(args.normalized_dir,args.split,args.index,args.lsh_index,args.bands)
        log(f"PASS: {meta['split']} LSH index, {meta['indexed_rows']:,} records")
    elif args.command == 'check-lsh':
        meta=check_lsh(args.lsh_index,args.index)
        log(f"PASS: {meta['split']} LSH index, {meta['indexed_rows']:,} records")
    elif args.command == 'sample':
        log(json.dumps(make_sample(args.ground_truth, source_parts(args.normalized_dir,'train',1),
                                   args.out, args.size, args.seed, args.exclude_ids), indent=2))
    elif args.command == 'generate':
        config = Config(**json.loads(Path(args.config).read_text(encoding='utf-8'))) if args.config else Config()
        config.validate()
        if bool(args.queries) == bool(args.normalized_dir):
            raise ValueError('Specify exactly one of --queries and --normalized-dir')
        if args.normalized_dir:
            check_index(args.index, args.normalized_dir, args.split)
        queries = [Path(args.queries)] if args.queries else source_parts(args.normalized_dir,args.split,1)
        stats = generate(args.index, queries, args.out, config, args.split, not args.no_evidence,
                         args.workers,args.lsh_index)
        log(json.dumps({k:stats[k] for k in ('s1_records','candidate_pairs','mean_candidates','p95_candidates','seconds')}, indent=2))
    else:
        result = evaluate(args.index, [Path(args.queries)], args.ground_truth,
                          args.candidates, args.evidence, args.report)
        log(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
