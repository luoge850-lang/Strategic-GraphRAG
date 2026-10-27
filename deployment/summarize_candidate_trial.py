"""Recompute trial quality, latency and failure counts from retained raw responses."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from live_acceptance import CASES, score


def summary(root):
    concurrency = []
    for level in (1, 2, 4):
        data = json.loads((root / f'concurrency-{level}.json').read_text(encoding='utf-8'))
        rows = data['requests']
        cases = {case['id']: case for case in CASES}
        good = sum(row['http_status'] == 200 and score(cases[row['case_id']], row.get('response') or {})['status'] == 'PASS' for row in rows)
        latencies = sorted(row['elapsed_ms'] for row in rows if row.get('elapsed_ms') is not None)
        successes = sum(row['http_status'] == 200 for row in rows)
        success_latencies = sorted(row['elapsed_ms'] for row in rows if row['http_status'] == 200)
        concurrency.append({'concurrency': level, 'scenario_passed': good, 'total': len(rows),
            'http_success': successes, 'p50_ms': statistics.median(latencies) if latencies else None,
            'http_status_counts': dict(Counter(str(row['http_status']) for row in rows)),
            'success_latency_p50_ms': statistics.median(success_latencies) if success_latencies else None,
            'success_latency_p95_ms': success_latencies[math.ceil(.95 * len(success_latencies)) - 1] if success_latencies else None,
            'successful_throughput_per_second': successes / data['wall_seconds'],
            'p95_ms_nearest_rank': latencies[math.ceil(.95 * len(latencies)) - 1] if latencies else None,
            'throughput_per_second': len(rows) / data['wall_seconds'],
            'sample_limit': 'ten requests on five fixed smoke questions, descriptive only',
            'passed_success_subset': {'numerator': good, 'denominator': successes}})
    development = None
    path = root / 'development-answers.json'
    if path.exists():
        records = json.loads(path.read_text(encoding='utf-8'))['records']
        for row in records:
            amount = re.search(r'\$([\d,]+(?:\.\d+)?)\s*(million|billion)', row['reference_answer'])
            row['numeric_score'] = None
            if amount and row['question_type'] in {'single_year_fact', 'fact_year_in_later_disclosure', 'unit_conversion_or_calculation'}:
                row['numeric_score'] = score({'expected': float(amount.group(1).replace(',', '')),
                    'unit': 'USD ' + amount.group(2) + 's'}, row.get('response') or {})
        numeric = [row for row in records if row.get('numeric_score') is not None]
        slices = {}
        for row in records:
            item = slices.setdefault(row['question_type'], {'requests': 0, 'numeric_judged': 0, 'numeric_passed': 0})
            item['requests'] += 1
            if row.get('numeric_score') is not None:
                item['numeric_judged'] += 1
                item['numeric_passed'] += row['numeric_score']['status'] == 'PASS'
        development = {'requests': len(records), 'families': len({r['family_id'] for r in records}),
            'http_success': sum(r['http_status'] == 200 for r in records),
            'numeric_passed': sum(r['numeric_score']['status'] == 'PASS' for r in numeric),
            'numeric_judged': len(numeric), 'label_tier': 'AI_PDF_SOURCE_DIAGNOSIS_NOT_HUMAN_REVIEW',
            'known_support_citation_hits': sum(r['known_support_citation_hit'] is True for r in records),
            'known_support_citation_judged': sum(r['known_support_citation_hit'] is not None for r in records),
            'status_counts': dict(Counter((r.get('response', {}).get('calculation') or {}).get('status', 'NO_CALCULATION') for r in records)),
            'type_slices': slices, 'numerical_failures': [r['item_id'] for r in numeric if r['numeric_score']['status'] != 'PASS'],
            'limits': 'numeric score covers eight explicit single-value references, not 26 semantic answers; citation hits are not citation correctness or Recall; development data used for repairs'}
    resources = json.loads((root / 'resources.json').read_text())
    return {'schema': 'candidate-trial-summary/v1', 'build_id': resources['build_id'],
            'concurrency': concurrency, 'development': development,
            'http_process_tree_peak_rss_bytes': resources['http_process_tree_sampled_peak_rss_bytes'],
            'startup_including_preflight_ms': resources['startup_including_preflight_ms'],
            'llm_generation_calls': resources['llm_generation_calls'],
            'raw_hashes': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.json')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--development-trial', type=Path)
    parser.add_argument('--chart', type=Path)
    args = parser.parse_args()
    result = summary(args.trial)
    if args.before:
        result['before_repair'] = summary(args.before)
    if args.development_trial:
        result['development'] = summary(args.development_trial)['development']
        result['development_trial_hashes'] = summary(args.development_trial)['raw_hashes']
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    if args.chart:
        before = result.get('before_repair', {}).get('development') or {}
        after = result.get('development') or {}
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="340" viewBox="0 0 1000 340">',
            '<rect width="1000" height="340" fill="#f5f3ec"/>',
            '<g font-family="Arial,sans-serif" fill="#202923">',
            '<text x="40" y="40" font-size="23">Evidence before confidence</text>',
            '<text x="40" y="70" font-size="13">AI/PDF development diagnosis, not an independent accuracy claim</text>']
        for index, (name, data) in enumerate([('Before repair', before), ('After repair', after)]):
            n, d = data.get('numeric_passed', 0), data.get('numeric_judged', 0)
            width = 320 * n / d if d else 0
            y = 115 + index * 85
            parts += [f'<text x="40" y="{y}" font-size="16">{name}: {n}/{d} explicit numerical references</text>',
                f'<rect x="40" y="{y+12}" width="320" height="24" fill="#dddccf"/>',
                f'<rect x="40" y="{y+12}" width="{width}" height="24" fill="#577b5f"/>']
        parts += ['<text x="500" y="115" font-size="16">Real HTTP query latency (warm database)</text>']
        for index, row in enumerate(result['concurrency']):
            y = 152 + index * 48
            parts += [f'<text x="500" y="{y}" font-size="14">Concurrency {row["concurrency"]}: {row["scenario_passed"]}/{row["total"]} passed</text>',
                f'<text x="500" y="{y+19}" font-size="13">p50 {row["p50_ms"]:.2f} ms / p95 {row["p95_ms_nearest_rank"]:.2f} ms</text>']
        parts += ['<text x="40" y="308" font-size="12">Five fixed smoke questions; ten requests per concurrency. No LLM generation. OS cache uncontrolled.</text>', '</g></svg>']
        with args.chart.open('x', encoding='utf-8') as stream:
            stream.write('\n'.join(parts))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
