"""Append-only bounded closure; never overwrite previous packages or labels."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts import run_isolated_staging as staging
from strategic_graphrag.build_identity import source_fingerprint, sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/accuracy-closure-20261016'
BASE = ROOT / 'experiments/accuracy-recovery-20261016'
PORT = 8013


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def build():
    import pdfplumber
    from strategic_graphrag.pipeline.financial_table_extractor import extract_financial_table_triples
    package = OUT / 'package'
    package.mkdir(exist_ok=False)
    pdfs = list(staging.DEFAULT_PDFS) + [ROOT / f'experiments/acceptance-20261015/sources/TSMC-{y}-Annual-Report.pdf' for y in (2023, 2024)]
    identity = staging.identity_for(pdfs, 'accuracy-closure-five-20261016').to_dict()
    bid = identity['build_id']
    staging.phase_identity(package, identity, bid)
    registry = read(BASE / 'package/document_registry.json')
    registry['build_id'] = bid
    staging.json_write(package / 'document_registry.json', registry)
    summary = read(BASE / 'package/document_layer_summary.json')
    summary['build_id'] = bid
    for doc in summary['files']:
        src = BASE / 'package' / doc['snapshot']
        assert sha256_file(src) == doc['snapshot_sha256']
        data = read(src)
        data['build_id'] = bid
        staging.json_write(package / doc['snapshot'], data)
        doc['snapshot_sha256'] = sha256_file(package / doc['snapshot'])
    staging.json_write(package / 'document_layer_summary.json', summary)
    extraction = read(BASE / 'package/candidate_extraction.json')
    extraction['build_id'] = bid
    records = []
    for row in extraction['files']:
        filename = row['filename']
        assert row['document_sha256'] == identity['pdfs'][filename]['sha256']
        row['build_id'] = bid
        if filename.startswith('TSMC'):
            records.append(dict(filename=filename, mode='HASH_VERIFIED_CACHED_EXTRACTION'))
            continue
        started = time.perf_counter()
        additions = []
        pdf = next(p for p in pdfs if p.name == filename)
        with pdfplumber.open(pdf) as document:
            for number, page in enumerate(document.pages, 1):
                text = page.extract_text() or ''
                if 'consolidated balance sheets' not in text.casefold():
                    continue
                for fact in extract_financial_table_triples(page, text, row['year'], company_id='NVIDIA_CORPORATION', document_id=filename):
                    if fact['target'] not in {'INTANGIBLE_ASSETS', 'OTHER_ASSETS', 'RETAINED_EARNINGS'}:
                        continue
                    fact.update(source_page=number, source_filing=filename, document_sha256=row['document_sha256'])
                    additions.append(fact)
        row['accepted_triples'] = [f for f in row['accepted_triples'] if f['target'] not in {'INTANGIBLE_ASSETS', 'OTHER_ASSETS', 'RETAINED_EARNINGS'}] + additions
        records.append(dict(filename=filename, mode='FRESH_BALANCE_ROWS_OTHER_FAMILIES_CACHED', pages_examined=len(document.pages), additions=len(additions), seconds=time.perf_counter()-started, llm_calls=0))
        print(filename, len(additions), flush=True)
    staging.json_write(package / 'candidate_extraction.json', extraction)
    staging.phase_accepted(package, bid)
    start = time.perf_counter()
    staging.phase_indexes(package, bid)
    save('build-record.json', dict(build_id=bid, extraction=records, fresh_vector_seconds=time.perf_counter()-start, cost=None, token_count=None))
    save('package-hashes.json', {str(p.relative_to(package)): sha256_file(p) for p in package.rglob('*') if p.is_file()})
    save('implementation-freeze.json', dict(build_id=bid, source=source_fingerprint(), holdout_not_yet_created=True))
    print(bid, flush=True)


def import_graph():
    from deployment import request_contract_20261015 as campaign
    campaign.OUT = OUT
    campaign.PACKAGE = OUT / 'package'
    campaign.import_graph()


def serve():
    from deployment import request_contract_20261015 as campaign
    campaign.BASE = BASE
    campaign.OUT = OUT
    campaign.PACKAGE = OUT / 'package'
    campaign.serve(PORT)


def launch():
    with (OUT / 'server.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, '-m', 'deployment.accuracy_closure_20261016', 'serve'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    save('server-launch.json', dict(pid=process.pid, port=PORT))


if __name__ == '__main__':
    {'build': build, 'import': import_graph, 'serve': serve, 'launch': launch}[sys.argv[1]]()
