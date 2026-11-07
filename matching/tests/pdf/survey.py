"""Survey the free-text clause lines of PDFs (lines after the IBAN line, minus the synthetic footer). Dev only."""
import sys, re, collections
sys.path.insert(0, '.')
import pandas as pd, pypdf
ph = sys.argv[1] if len(sys.argv) > 1 else 'sample'
types = sys.argv[2].split(',') if len(sys.argv) > 2 else ['invoice', 'credit_note', 'proforma']
i = pd.read_csv(f'data/{ph}/pdfs/index.csv', dtype=str)
i = i[i.document_type.isin(types)]
cnt = collections.Counter(); ex = {}
for f, t in zip(i.file, i.document_type):
    try:
        txt = '\n'.join(p.extract_text() for p in pypdf.PdfReader(f'data/{ph}/pdfs/' + f).pages)
    except Exception as e:
        cnt[('ERR', str(e)[:40])] += 1; continue
    lines = txt.split('\n')
    k = max([j for j, l in enumerate(lines) if l.startswith('IBAN')] or [-1])
    tail = [l for l in lines[k + 1:] if 'sintético' not in l]
    pay = [l for l in lines if re.match(r'^(Zahlungs|Conditions|Payment|Condiz|Condi|Forma|Termos|Payment terms)', l)]
    for l in tail + pay:
        key = (t, re.sub(r'\d', '9', l)[:70])
        cnt[key] += 1; ex.setdefault(key, l)
for k, v in cnt.most_common(150):
    print(v, k[0], '|', ex.get(k, k)[:160])
