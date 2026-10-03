"""Generate a small visual from retained scores; no invented trend or absent value."""
import argparse,json,html
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();s=json.loads(a.summary.read_text(encoding='utf-8'))
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="430" viewBox="0 0 1100 430">',
         '<rect width="1100" height="430" fill="#f5f3ec"/><g fill="#243027" font-family="Arial,sans-serif">',
         '<text x="35" y="38" font-size="24">A matching number is not a complete fact</text>',
         '<text x="35" y="65" font-size="13">build_c3b1d950e8acce78 · AI/PDF diagnosis · graph-only · generation off</text>']
    names={'natural_language':'Natural language','user_scope':'Fixed user-scope fixture: 2025','reference_scope_diagnostic':'Reference scope: diagnostic only'}
    for i,(mode,parts) in enumerate(s['modes'].items()):
        y=105+i*90;dev=parts['development_previously_used'];v=parts['frozen_adjacent_diagnostic_no_tuning']
        out.append(f'<text x="35" y="{y}" font-size="16">{html.escape(names[mode])}</text>')
        for j,(title,metric) in enumerate([('Number',dev['fact_fields']['number']),('Joint fact',dev['fact_fields']['joint'])]):
            x=35+j*255;width=160*metric['numerator']/metric['denominator'] if metric['denominator'] else 0
            out += [f'<rect x="{x}" y="{y+12}" width="160" height="18" fill="#dddfd4"/>',
                f'<rect x="{x}" y="{y+12}" width="{width}" height="18" fill="{["#688777","#ae765a"][j]}"/>',
                f'<text x="{x}" y="{y+49}" font-size="13">{title}: {metric["numerator"]}/{metric["denominator"]}</text>']
        number=v['fact_fields']['number'];joint=v['fact_fields']['joint']
        out += [f'<text x="570" y="{y+13}" font-size="14">Adjacent facts: number {number["numerator"]}/{number["denominator"]}; joint {joint["numerator"]}/{joint["denominator"]}</text>',
            f'<text x="570" y="{y+39}" font-size="13">26-form HTTP p50/p95: {dev["latency_p50_ms"]:.2f}/{dev["latency_p95_ms"]:.2f} ms</text>']
    out+=['<text x="35" y="389" font-size="12">Development: 26 forms / 20 families; fact labels: 8 forms / 7 families. Adjacent: 12 forms / 11 metric groups.</text>',
          '<text x="35" y="410" font-size="12">Different file scope can conflict with question text. This is not independent Gold accuracy or production capacity.</text></g></svg>']
    with a.output.open('x',encoding='utf-8') as f:f.write('\n'.join(out))
if __name__=='__main__':main()
