"""Verify selected public bytes, regenerated scores and local relative links."""
import argparse,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def portable_summary(summary):
    # Preserve all values and byte digests. Only filesystem separator spelling is
    # platform-specific in the retained Windows summary; never omit comparisons.
    result=dict(summary)
    hashes=summary.get('raw_hashes',{})
    result['raw_hashes']={name.replace('\\','/'):digest for name,digest in hashes.items()}
    if len(result['raw_hashes'])!=len(hashes):raise ValueError('path canonicalization collision')
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--create-manifest',action='store_true');p.add_argument('--recomputed',type=Path);a=p.parse_args()
    root=a.root.resolve();manifest=root/'public_manifest.json'
    if a.create_manifest:
        paths=[x for x in root.rglob('*') if x.is_file() and x.suffix.lower() in {'.json','.jsonl','.md','.svg','.png'} and x!=manifest]
        record={str(x.relative_to(root)).replace('\\','/'):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(paths)}
        with manifest.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
    hashes=json.loads(manifest.read_text());bad=[name for name,digest in hashes.items() if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest]
    if bad:raise RuntimeError('public artifact hash mismatch: '+str(bad))
    if a.recomputed:
        actual=json.loads(a.recomputed.read_text());expected=json.loads((root/'summary-release.json').read_text())
        if portable_summary(actual)!=portable_summary(expected):raise RuntimeError('summary differs from retained raw recomputation')
    missing=[]
    for doc in [ROOT/'README.md',root/'README.md']:
        for link in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if link.startswith(('https:','http:','#')):continue
            target=(doc.parent/link.split('#',1)[0]).resolve()
            if not target.exists():missing.append(str(target))
    if missing:raise RuntimeError('missing relative links: '+str(missing))
    print(json.dumps(dict(hash_files=len(hashes),hash_mismatches=0,relative_link_missing=0,summary_equal=bool(a.recomputed))))
if __name__=='__main__':main()
