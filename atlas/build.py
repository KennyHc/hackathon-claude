"""Inject agg.json into atlas.src.html -> index.html (open directly in a browser)."""
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
src = open(H + '/atlas.src.html').read()
out = src.replace('/*__DATA__*/null', json.dumps(d, separators=(',', ':')))
page = '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body style="margin:0">\n' + out + '\n</body></html>\n'
open(H + '/index.html', 'w').write(page)
print(len(out), 'bytes')
