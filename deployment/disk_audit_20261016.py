"""Read-only, no-reparse disk audit. Never deletes, moves, or reads secrets."""
import collections,json,os,subprocess,tempfile,time
from pathlib import Path
from deployment.finalization_20261016 import ROOT,OUT,save

DESKTOP=Path(os.environ.get('GRAPHRAG_ORIGINAL_WORKSPACE', str(Path.home()/'OneDrive/Desktop/Nvidia-GraphRAG-Engine')))
REPARSE=0x400

def walk(root,skipped):
    pending=[root]
    while pending:
        path=pending.pop()
        try:
            if path.lstat().st_file_attributes & REPARSE:
                skipped.append(str(path));continue
            with os.scandir(path) as entries:
                for item in entries:
                    try:
                        stat=item.stat(follow_symlinks=False)
                        if stat.st_file_attributes & REPARSE:skipped.append(item.path)
                        elif item.is_dir(follow_symlinks=False):pending.append(Path(item.path))
                        elif item.is_file(follow_symlinks=False):yield Path(item.path),stat.st_size
                    except OSError as exc:skipped.append(dict(path=item.path,error=type(exc).__name__))
        except OSError as exc:skipped.append(dict(path=str(path),error=type(exc).__name__))

def audit():
    skipped=[];entries=[];started=time.time()
    disk=__import__('shutil').disk_usage(DESKTOP)
    for root in [DESKTOP,ROOT]:
        tracked=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
        buckets=collections.defaultdict(lambda:dict(files=0,bytes=0,tracked_files=0))
        for path,size in walk(root,skipped):
            rel=path.relative_to(root)
            # Three levels expose environment/runtime/package boundaries.
            key=Path(*rel.parts[:min(3,len(rel.parts)-1)]) if len(rel.parts)>1 else Path('.')
            bucket=buckets[str(key)];bucket['files']+=1;bucket['bytes']+=size
        for key,counts in sorted(buckets.items()):
            path=root/key
            prefix=Path(key).as_posix().rstrip('/')+'/'
            counts['tracked_files']=sum(1 for name in tracked if name.startswith(prefix))
            ignore=subprocess.run(['git','check-ignore',str(path)],cwd=root,capture_output=True,text=True)
            protected=any(word in key.lower() for word in ['.venv','configs','tmp','experiments','data','deployment','strategic_graphrag','tests'])
            entries.append(dict(path=str(path),logical_bytes=counts['bytes'],file_count=counts['files'],git_tracked_files=counts['tracked_files'],ignored=ignore.returncode==0,
                build='See candidate package/build_identity.json; association requires reference review',process='See process snapshot; dynamic references NOT_VERIFIED',
                references='NOT_VERIFIED',unique_copy='NOT_VERIFIED',recovery='NOT_VERIFIED; no deletion permitted',risk='HIGH' if protected else 'REVIEW_REQUIRED',
                recommendation='PRESERVE' if protected else 'MEASURED_ONLY_NOT_AUTHORIZED_FOR_DELETION'))
        print(str(root),'measured',flush=True)
    temp=Path(tempfile.gettempdir())
    # Only the explicitly project-named vector staging directories are inventoried.
    for path in temp.iterdir():
        if path.name.startswith('graphrag-vector-build') and path.is_dir():
            files=list(walk(path,skipped))
            entries.append(dict(path=str(path),file_count=len(files),logical_bytes=sum(n for _,n in files),git_tracked_files=0,ignored=None,
                build='NOT_VERIFIED',process='NOT_VERIFIED',references='NOT_VERIFIED',unique_copy='NOT_VERIFIED',recovery='Rebuild vector index only after manifest/dependency review',risk='REVIEW_REQUIRED',recommendation='A_CANDIDATE_NOT_APPROVED'))
    save('disk-audit/inventory.json',dict(measured_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),seconds=time.time()-started,entries=entries,skipped_reparse_or_inaccessible=skipped,
        disk_before=dict(total=disk.total,free=disk.free,used=disk.used),actual_deleted_files=0,actual_archived_files=0,logical_deleted_bytes=0,
        actual_released_bytes=0,limitations='Logical bytes, not physical allocation. No cleanup executed. Unique-copy, dynamic dependencies, restoration remain explicit review tasks.'))

if __name__=='__main__':audit()
