"""Seal immutable delivery artifacts, excluding private live stores and logs."""
import hashlib,json,socket,subprocess,os
from pathlib import Path
from deployment.acceptance_20261015 import ROOT,OUT,save
def main():
    initial=json.loads((OUT/'initial-inventory.json').read_text(encoding='utf-8'))
    original=Path(os.environ.get('GRAPHRAG_ORIGINAL_WORKSPACE', str(Path.home()/'OneDrive/Desktop/Nvidia-GraphRAG-Engine')))
    current=subprocess.check_output(['git','status','--porcelain'],cwd=original,text=True)
    before=next(x['status'] for x in initial['workspaces'] if x['path']==str(original))
    ports={}
    for port in [8000,8002,8003,5173,17691,17692,17693]:
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=1):ports[str(port)]='LISTENING'
        except OSError:ports[str(port)]='NOT_LISTENING'
    save('final-state.json',dict(original_git_status_unchanged=current==before,original_status=current,ports=ports,release_pointer_mutations=0,github_uploads=0))
    excluded={'server-round1.log','server-round2.log','delivery-manifest.json','verification-final.json','verification-complete.json'}
    paths=[p for p in OUT.iterdir() if p.is_file() and p.name not in excluded]
    paths.extend(p for folder in ['browser','five-package-round2','five-package','single-package','sources'] for p in (OUT/folder).rglob('*') if p.is_file())
    paths.extend(ROOT/n for n in json.loads((OUT/'final-source-fingerprint.json').read_text())['runtime_sources'])
    paths.extend(p for p in (ROOT/'deployment').glob('*.py') if p.name in ['acceptance_20261015.py','download_acceptance_corpus.py','record_graph_development_review.py','finish_source_reviews.py'] or 'five' in p.name)
    paths.extend([ROOT/'tests/test_acceptance_semantic_regressions.py',ROOT/'requirements.txt'])
    save('delivery-manifest.json',dict(schema='five-corpus-delivery/v1',files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))},
        exclusions=['private Neo4j state/credentials','mutable HTTP logs','runtime Chroma copies','manifest itself','verification outputs'],
        initial_seal_attempt='verification-final.json records the failed pre-manifest verification; preserved, not overwritten',acceptance='FAIL'))
if __name__=='__main__':main()
