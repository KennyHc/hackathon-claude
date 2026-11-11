"""Inject agg.json (+ agg_final.json) into atlas.src.html -> index.html (open directly in a browser).

atlas.src.html holds the <head> content (title, meta, fonts, CSS) above the `<!--/HEAD-->` marker and the page below it.
"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
a = json.load(open(H + '/agg.json'))
keep = ['phases', 'monthly', 'window', 'families', 'exceptions', 'items_per_line', 'categories', 'live_entity', 'entities',
        'live_currency', 'live_formats_monthly', 'moncayo', 'lines_with_exc', 'exc_rows_total', 'max_items', 'max_items_segment',
        'rem_formats_live', 'pdf_types_live', 'open_items_types', 'n_customers', 'n_suppliers']
d = {k: a[k] for k in keep}
d['segments'] = [[s['segment'], s['family'], int(s['lines']), int(s['pairs'])] for s in a['segments']]
d['families'] = {f['family']: [int(f['lines']), int(f['pairs'])] for f in a['families']}
d['entities'] = [[e['entity'], e['country'], e['functional_currency'], e['role']] for e in a['entities']]
d['live_banks'] = 13; d['live_accounts'] = 49
d['final'] = json.load(open(H + '/agg_final.json'))   # python3 atlas/compute_final.py
src = open(H + '/atlas.src.html').read()
head, body = src.split('<!--/HEAD-->', 1)
body = body.replace('/*__DATA__*/null', json.dumps(d, separators=(',', ':')))
page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
        + head.strip() + '\n</head>\n<body style="margin:0">\n' + body.strip() + '\n</body>\n</html>\n')
open(H + '/index.html', 'w').write(page)
print(len(page), 'bytes')
