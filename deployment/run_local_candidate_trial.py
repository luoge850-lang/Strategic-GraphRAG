"""Exercise a real loopback HTTP service; retain records, never publish production."""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, payload):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--pdf-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8002)
    parser.add_argument('--reuse-runtime', action='store_true', help='validate existing writable runtime instead of overwriting')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--runtime-root', type=Path, default=ROOT)
    args = parser.parse_args()
    from scripts.run_isolated_staging import verify_package
    from deployment.live_acceptance import run, http, CASES, score
    import psutil
    if args.output.exists():
        parser.error('output must be new')
    before = verify_package(args.candidate)
    if before['status'] != 'PASS':
        raise ValueError('candidate integrity rejected')
    credentials = json.loads(args.credentials.read_text())
    if credentials['uri'].split('://', 1)[-1].split(':')[0] != '127.0.0.1':
        raise ValueError('only loopback credentials accepted')
    identity = json.loads((args.candidate / 'build_identity.json').read_text())
    manifest = json.loads((args.candidate / 'vector_index_manifest.json').read_text())
    runtime = args.runtime_root.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    data = runtime / 'data'
    vector = data / 'chroma_db'
    if vector.exists() and not args.reuse_runtime:
        raise ValueError('runtime vector path already exists; never overwrite')
    args.output.mkdir(parents=True)
    if not vector.exists():
        shutil.copytree(args.candidate / 'vector_index', vector)
    for name, record in identity['pdfs'].items():
        source = args.pdf_root / ('pdfs' if name == '2025-10-K.pdf' else 'pdfs_other') / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('PDF source hash mismatch')
        target = data / source.parent.name / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('existing PDF target differs')
        else:
            shutil.copyfile(source, target)
    env = {key: value for key, value in os.environ.items() if not key.endswith('API_KEY')}
    env.update(NEO4J_URI=credentials['uri'], NEO4J_USERNAME=credentials['username'],
               NEO4J_PASSWORD=credentials['password'], NEO4J_DATABASE=credentials['database'],
               GRAPHRAG_BUILD_ID=identity['build_id'], GRAPH_VECTOR_COLLECTION=manifest['collection'],
               GRAPH_EMBEDDING_BACKEND='chroma_onnx', API_AUTH_ENABLED='false',
               QUERY_CACHE_TTL_SECONDS='0', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = str(ROOT)
    started = time.perf_counter()
    check = subprocess.run([sys.executable, 'deployment/preflight.py', '--candidate-dir', str(args.candidate),
                            '--data-root', str(data), '--output', str(args.output.resolve() / 'preflight.json')], cwd=ROOT, env=env,
                           capture_output=True, timeout=90)
    if check.returncode:
        raise RuntimeError('real-store preflight rejected: see preflight.json')
    process = None
    peak = 0
    try:
        with (args.output / 'http-server.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'strategic_graphrag.api.server:app',
                                        '--host', '127.0.0.1', '--port', str(args.port)],
                                       cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
        base = f'http://127.0.0.1:{args.port}'
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('HTTP service exited')
            if http(base, '/health/live', timeout=1)['http_status'] == 200:
                break
            time.sleep(.5)
        else:
            raise RuntimeError('HTTP startup timed out')
        startup_ms = (time.perf_counter() - started) * 1000
        cold = run(base, args.output / 'cold-http')
        warm = run(base, args.output / 'warm-http')
        if args.dataset:
            import re
            development = []
            for line in args.dataset.read_text(encoding='utf-8').splitlines():
                family = json.loads(line)
                for item in [family, *family.get('candidate_variants', [])]:
                    if item.get('partition', family.get('partition')) != 'development':
                        continue
                    result = http(base, '/query', {'question': item['question'], 'max_paths': 10,
                        'retrieval_mode': 'graph', 'synthesize': False, 'use_cache': False,
                        'source_filing': item.get('candidate_filing') or family.get('candidate_filing'),
                        'cross_filing': False})
                    result.pop('_bytes')
                    response = result.get('response') or {}
                    reference = item.get('reference_answer', '')
                    kind = item.get('question_type')
                    numeric = None
                    # Score only explicit dollar references for single-value questions.
                    # Relationship/comparison prose is not assigned an automatic semantic verdict.
                    amount = re.search(r'\$([\d,]+(?:\.\d+)?)\s*(million|billion)', reference)
                    if amount and kind in {'single_year_fact', 'fact_year_in_later_disclosure', 'unit_conversion_or_calculation'}:
                        expected = float(amount.group(1).replace(',', ''))
                        unit = 'USD ' + amount.group(2) + 's'
                        numeric = score({'expected': expected, 'unit': unit}, response)
                    grades = item.get('relevance_grades', {})
                    citations = response.get('citations') or []
                    ranked = [f"{c.get('source_filing')}#{c.get('page')}" for c in citations]
                    supported = any(grades.get(key, 0) > 0 for key in ranked)
                    development.append({'family_id': family['family_id'], 'item_id': item['item_id'],
                        'question_type': kind, 'question': item['question'], 'reference_answer': reference,
                        'label_tier': item.get('label_tier', family.get('label_tier')),
                        'dataset_sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                        'build_id': identity['build_id'], 'relevance_grades': grades,
                        'known_support_citation_hit': supported if grades else None,
                        'numeric_score': numeric, 'semantic_answer_score': None, **result})
            save(args.output / 'development-answers.json', {'scope': 'previously used development families; not independent test',
                'unjudged_citations': 'not assumed irrelevant', 'records': development})
        rows = []
        for concurrency in (1, 2, 4):
            batch_start = time.perf_counter()
            def measure(item):
                repeat, case = item
                result = http(base, '/query', {'question': case['question'], 'max_paths': 10,
                    'retrieval_mode': 'graph', 'synthesize': False, 'use_cache': False,
                    'cross_filing': case.get('cross_filing', False), 'source_filing': case.get('source_filing')})
                result.pop('_bytes')
                return dict(concurrency=concurrency, repeat=repeat, case_id=case['id'],
                            score=score(case, result.get('response') or {}), **result)
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(measure, (repeat, case)) for repeat in range(2) for case in CASES]
                while any(not future.done() for future in futures):
                    try:
                        parent = psutil.Process(process.pid)
                        peak = max(peak, sum(item.memory_info().rss for item in [parent, *parent.children(recursive=True)]))
                    except psutil.Error:
                        pass
                    time.sleep(.02)
                measured = [future.result() for future in futures]
            elapsed = time.perf_counter() - batch_start
            rows.extend(measured)
            save(args.output / f'concurrency-{concurrency}.json', {'requests': measured,
                 'wall_seconds': elapsed, 'throughput_requests_per_second': len(measured) / elapsed})
        save(args.output / 'resources.json', {'build_id': identity['build_id'], 'platform': platform.platform(),
             'cpu': platform.processor(), 'logical_cpu_count': os.cpu_count(),
             'system_memory_bytes': psutil.virtual_memory().total, 'http_process_tree_sampled_peak_rss_bytes': peak,
             'memory_sampling_ms': 20, 'startup_including_preflight_ms': startup_ms,
             'fixed_smoke_queries': 40, 'development_queries': len(development) if args.dataset else 0,
             'concurrency': [1, 2, 4], 'repeat_per_case_per_concurrency': 2,
             'llm_generation_enabled': False, 'llm_generation_calls': 0,
             'embedding_inference': 'local ONNX; embedding cache pre-existing, not a model cold-load benchmark',
             'scope': 'loopback deterministic graph queries; peak excludes Neo4j and OS cache',
             'cold_summary': cold, 'warm_summary': warm})
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=20)
        save(args.output / 'immutable-after.json', verify_package(args.candidate))
    print(json.dumps({'status': 'RECORDED', 'output': str(args.output), 'build_id': identity['build_id']}))


if __name__ == '__main__':
    main()
