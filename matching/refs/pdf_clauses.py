"""PDF clause extractor for bank reconciliation.

Extracts free-text clauses and structured metadata from document PDFs
(POLICIES.md §5.7-5.9 and §3).
"""

from collections import Counter
from datetime import datetime
import multiprocessing as mp
import os
import re
import sys
import numpy as np
import pandas as pd
import pypdf

DISCLAIMER_RE = re.compile(r'Documento sint[eé]tico de prueba', re.IGNORECASE)

TERMS_RE = re.compile(
    r'^(Terms|Forma de pago|Condi[cç][oõ]es|Condizioni|Conditions|Zahlungsbedingungen|Warunki p[lł]atno[sś]ci|Voorwaarden|Betalingsvoorwaarden|Pagamento)\b',
    re.IGNORECASE
)

TOTAL_HEADER_RE = re.compile(
    r'^(TOTAL TTC|TOTALE|GESAMT|RAZEM|TOTAAL|TOTAL)\b',
    re.IGNORECASE
)

MONTH_MAP = {
    'january': '01', 'janvier': '01', 'januar': '01', 'gennaio': '01', 'janeiro': '01', 'enero': '01', 'styczeń': '01', 'stycznia': '01',
    'february': '02', 'fevrier': '02', 'février': '02', 'februar': '02', 'febbraio': '02', 'fevereiro': '02', 'febrero': '02', 'luty': '02', 'lutego': '02',
    'march': '03', 'mars': '03', 'märz': '03', 'maerz': '03', 'marzo': '03', 'março': '03', 'marco': '03', 'marzec': '03', 'marca': '03',
    'april': '04', 'avril': '04', 'aprile': '04', 'abril': '04', 'kwiecień': '04', 'kwietnia': '04',
    'may': '05', 'mai': '05', 'maggio': '05', 'maio': '05', 'mayo': '05', 'maj': '05', 'maja': '05',
    'june': '06', 'juin': '06', 'juni': '06', 'giugno': '06', 'junho': '06', 'junio': '06', 'czerwiec': '06', 'czerwca': '06',
    'july': '07', 'juillet': '07', 'juli': '07', 'luglio': '07', 'julho': '07', 'julio': '07', 'lipiec': '07', 'lipca': '07',
    'august': '08', 'août': '08', 'aout': '08', 'agosto': '08', 'sierpień': '08', 'sierpnia': '08',
    'september': '09', 'septembre': '09', 'settembre': '09', 'setembro': '09', 'septiembre': '09', 'wrzesień': '09', 'września': '09',
    'october': '10', 'octobre': '10', 'oktober': '10', 'ottobre': '10', 'outubro': '10', 'octubre': '10', 'październik': '10', 'października': '10',
    'november': '11', 'novembre': '11', 'listopad': '11', 'listopada': '11', 'noviembre': '11',
    'december': '12', 'décembre': '12', 'decembre': '12', 'dezember': '12', 'dicembre': '12', 'dezembro': '12', 'diciembre': '12', 'grudzień': '12', 'grudnia': '12'
}

OUTPUT_COLS = [
    'file', 'document_type', 'document_number', 'entity', 'party_id', 'document_date', 'src_phase',
    'clause_text', 'terms_text', 'n_instalments', 'retention_pct', 'retention_amount',
    'retention_release_date', 'amount_payable', 'advance_amount', 'advance_proforma',
    'replaces_document', 'cancels_document', 'ref_document', 'service_month',
    'cn_settlement', 'rappel', 'factoring', 'new_iban', 'customer_po', 'total'
]


def parse_amount(s):
    """Parse monetary amounts in both 1,234.56 and 1.234,56 formats."""
    if s is None:
        return np.nan
    s = str(s).strip()
    if not s:
        return np.nan
    m = re.search(r'([0-9\s.,\xa0]+)', s)
    if not m:
        return np.nan
    raw = m.group(1).strip()
    raw = re.sub(r'[\s\xa0]', '', raw)
    if not raw:
        return np.nan
    if '.' in raw and ',' in raw:
        if raw.rfind('.') > raw.rfind(','):
            raw = raw.replace(',', '')
        else:
            raw = raw.replace('.', '').replace(',', '.')
    elif ',' in raw:
        parts = raw.split(',')
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            raw = raw.replace(',', '.')
        else:
            raw = raw.replace(',', '')
    try:
        return float(raw)
    except Exception:
        return np.nan


def parse_date_to_iso(date_str):
    """Parse date string to ISO YYYY-MM-DD."""
    if not date_str:
        return None
    date_str = date_str.strip()
    for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%y'):
        try:
            return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
        except ValueError:
            pass
    return None


def extract_one_pdf(task):
    """Worker function to parse a single PDF file."""
    file_rel, doc_type, doc_num, entity, party_id, doc_date, src_phase, full_path = task
    res = {
        'file': file_rel,
        'document_type': doc_type,
        'document_number': doc_num,
        'entity': entity,
        'party_id': party_id,
        'document_date': doc_date,
        'src_phase': src_phase,
        'clause_text': None,
        'terms_text': None,
        'n_instalments': np.nan,
        'retention_pct': np.nan,
        'retention_amount': np.nan,
        'retention_release_date': None,
        'amount_payable': np.nan,
        'advance_amount': np.nan,
        'advance_proforma': None,
        'replaces_document': None,
        'cancels_document': None,
        'ref_document': None,
        'service_month': None,
        'cn_settlement': '',
        'rappel': False,
        'factoring': False,
        'new_iban': None,
        'customer_po': None,
        'total': np.nan
    }
    
    if not os.path.exists(full_path):
        return res
        
    try:
        reader = pypdf.PdfReader(full_path)
        text = '\n'.join(p.extract_text() for p in reader.pages)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
    except Exception:
        return res

    disc_idx = len(lines)
    for i, l in enumerate(lines):
        if DISCLAIMER_RE.search(l):
            disc_idx = i
            break

    # 1. Bank account notice
    if doc_type == 'bank_account_notice':
        start_idx = 0
        for i in range(disc_idx):
            if re.search(r'Change of bank account|Cambio de cuenta bancaria', lines[i], re.IGNORECASE):
                start_idx = i
                break
        clause_lines = lines[start_idx:disc_idx]
        res['clause_text'] = ' || '.join(clause_lines) if clause_lines else None
        
        full_body = ' '.join(lines[:disc_idx])
        m = re.search(r'(?:account|cuenta)\s+([A-Z]{2}\d{2}[A-Z0-9]{10,30})', full_body, re.IGNORECASE)
        if m:
            res['new_iban'] = m.group(1).strip()
        else:
            m2 = re.search(r'\b([A-Z]{2}\d{2}[A-Z0-9]{10,30})\b', full_body)
            if m2:
                res['new_iban'] = m2.group(1).strip()
        return res

    # 2. General documents: header search for customer_po and ref_document
    if 'credit_note' in doc_type:
        for i, l in enumerate(lines[:disc_idx]):
            m = re.match(r'^(?:Ref|R[eé]f|Rif)\.[:\s]*(.*)$', l, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                if not val and i + 1 < disc_idx:
                    val = lines[i+1].strip()
                if val:
                    val = val.split()[0]
                    res['ref_document'] = val
                break

    for i, l in enumerate(lines[:disc_idx]):
        m = re.match(r'^(?:Your PO|Su pedido|Vs\.\s*ordine|Ihre Bestellung|Votre commande|Seu pedido|Nr zamówienia|Uw bestelling|Nº pedido)[:\s]*(.*)$', l, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if not val and i + 1 < disc_idx:
                val = lines[i+1].strip()
            if val:
                val = val.split()[0]
                res['customer_po'] = val
            break

    # TOTAL block
    total_idx = None
    for i, l in enumerate(lines[:disc_idx]):
        if TOTAL_HEADER_RE.match(l):
            if re.match(r'^Total\s*(HT|hors)', l, re.IGNORECASE):
                continue
            total_idx = i
            rest = TOTAL_HEADER_RE.sub('', l).strip()
            if rest:
                res['total'] = parse_amount(rest)
            elif i + 1 < disc_idx:
                res['total'] = parse_amount(lines[i+1])
            break

    # Terms line
    terms_idx = None
    for i in range(disc_idx):
        if TERMS_RE.match(lines[i]):
            terms_idx = i
            res['terms_text'] = lines[i]
            break

    # Free-text clause lines
    if terms_idx is not None:
        start_clause = terms_idx + 1
    elif total_idx is not None:
        rest = TOTAL_HEADER_RE.sub('', lines[total_idx]).strip()
        if not rest and total_idx + 1 < disc_idx and not np.isnan(res['total']):
            start_clause = total_idx + 2
        else:
            start_clause = total_idx + 1
    else:
        start_clause = 0

    clause_lines = lines[start_clause:disc_idx]
    if clause_lines:
        res['clause_text'] = ' || '.join(clause_lines)

    joined_clauses = ' '.join(clause_lines) if clause_lines else ''
    full_text = ' '.join(lines)

    # Factoring
    if re.search(r'\b(assigned to|cedido a|c[eé]d[eé] [aà]|abtretung|ceduto a|cesja)\b', joined_clauses, re.IGNORECASE):
        res['factoring'] = True

    # Rappel
    if re.search(r'\b(rappel|volume rebate|rabais de volume|mengenrabatt|rabat)\b', joined_clauses, re.IGNORECASE):
        res['rappel'] = True

    # cn_settlement
    if re.search(r'(to be offset|deducted from your next payment|se compensar[aá]|compens[eé]|verrechnet|potr[aą]con)', joined_clauses, re.IGNORECASE):
        res['cn_settlement'] = 'offset'
    elif re.search(r'(will be paid to your bank account|refunded by bank transfer|reembolsado mediante transferencia|se abonar[aá] por transferencia|rembours[eé] par virement|[uü]berwiesen|przelewem)', joined_clauses, re.IGNORECASE):
        res['cn_settlement'] = 'paid_out'
    else:
        res['cn_settlement'] = ''

    # Replaces document
    m_rep = re.search(r'(?:cancels and replaces invoice|anula y sustituye a la factura|annule et remplace la facture|annulla e sostituisce la fattura|storniert und ersetzt die Rechnung)\s+([A-Z0-9/_-]+)', joined_clauses, re.IGNORECASE)
    if m_rep:
        res['replaces_document'] = m_rep.group(1).strip()

    # Cancels document
    m_can = re.search(r'(?:CANCELLING OF INVOICE\s*(?:N[º°o.]*)*|anulaci[oó]n de la factura\s*(?:n[º°o.]*)*|annullamento della fattura\s*(?:n[º°o.]*)*|annulation de la facture\s*(?:n[º°o.]*)*|stornierung der rechnung\s*(?:nr[.]*)*)\s*([A-Z0-9/_-]+)', joined_clauses, re.IGNORECASE)
    if m_can:
        res['cancels_document'] = m_can.group(1).strip()

    # Instalments
    inst_source = (res['terms_text'] or '') + ' ' + joined_clauses
    m_inst = re.search(r'(\d+)\s*(?:instalments?|vencimientos?|rate|échéances?|echeances?|raten|raty|parcelas?|termijnen)\b', inst_source, re.IGNORECASE)
    if m_inst:
        res['n_instalments'] = int(m_inst.group(1))

    # Retention
    m_ret = re.search(
        r'(?:Retention|Retenci[oó]n en garant[ií]a(?: del)?)\s*(\d+(?:\.\d+)?)\s*%\s*\(([^)]+)\)\s*(?:withheld until final acceptance|liberable a la recepci[oó]n definitiva)\s*\(([0-9/.-]+)\)\.\s*(?:Amount payable at due date|Importe a pagar al vencimiento):?\s*([0-9\s.,\xa0]+)',
        joined_clauses,
        re.IGNORECASE
    )
    if m_ret:
        res['retention_pct'] = float(m_ret.group(1))
        res['retention_amount'] = parse_amount(m_ret.group(2))
        res['retention_release_date'] = parse_date_to_iso(m_ret.group(3))
        res['amount_payable'] = parse_amount(m_ret.group(4))
    else:
        m_ret_part = re.search(
            r'(?:Retention|Retenci[oó]n en garant[ií]a(?: del)?)\s*(\d+(?:\.\d+)?)\s*%\s*\(([^)]+)\)\s*(?:withheld until final acceptance|liberable a la recepci[oó]n definitiva)\s*\(([0-9/.-]+)\)',
            joined_clauses,
            re.IGNORECASE
        )
        if m_ret_part:
            res['retention_pct'] = float(m_ret_part.group(1))
            res['retention_amount'] = parse_amount(m_ret_part.group(2))
            res['retention_release_date'] = parse_date_to_iso(m_ret_part.group(3))
            m_ap = re.search(r'(?:Amount payable at due date|Importe a pagar al vencimiento):?\s*([0-9\s.,\xa0]+)', joined_clauses, re.IGNORECASE)
            if m_ap:
                res['amount_payable'] = parse_amount(m_ap.group(1))

    # Advance to deduct
    m_adv = re.search(
        r'(?:Less advance received against proforma|A deducir anticipo recibido seg[uú]n proforma)\s+([A-Z0-9/_-]+)(?::|\s+de)\s*([0-9\s.,\xa0]+)\s*[A-Z]{3}',
        joined_clauses,
        re.IGNORECASE
    )
    if m_adv:
        res['advance_proforma'] = m_adv.group(1).strip()
        res['advance_amount'] = parse_amount(m_adv.group(2))

    # Service month
    m_sm = re.search(
        r'(?:Maintenance and service fee for the month of|Servicio de mantenimiento)\s+([A-Za-z\u00C0-\u017F]+)\s+(\d{4})',
        full_text,
        re.IGNORECASE
    )
    if m_sm:
        m_name = m_sm.group(1).lower()
        year = m_sm.group(2)
        if m_name in MONTH_MAP:
            res['service_month'] = f"{year}-{MONTH_MAP[m_name]}"

    return res


def extract(phase: str, extra_phases: list[str] = None) -> pd.DataFrame:
    """Extract PDF clauses and metadata for a given phase and optional extra phases."""
    index_path = f'data/{phase}/pdfs/index.csv'
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Primary index file not found: {index_path}")
        
    df_primary = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    tasks = [
        (r.file, r.document_type, r.document_number, r.entity, r.party_id, r.document_date, phase, f'data/{phase}/pdfs/{r.file}')
        for _, r in df_primary.iterrows()
    ]
    
    # Process extra_phases if given
    if extra_phases:
        oi_path = f'data/{phase}/open_items_at_start.csv.gz'
        oi_keys = set()
        if os.path.exists(oi_path):
            oi = pd.read_csv(oi_path, dtype=str, keep_default_na=False)
            oi_keys = set(zip(oi['entity'], oi['document_number']))
            
        primary_keys = set(zip(df_primary['entity'], df_primary['document_number']))
        
        for ep in extra_phases:
            ep_index = f'data/{ep}/pdfs/index.csv'
            if not os.path.exists(ep_index):
                continue
            df_ep = pd.read_csv(ep_index, dtype=str, keep_default_na=False)
            for _, r in df_ep.iterrows():
                key = (r.entity, r.document_number)
                if oi_keys and key not in oi_keys:
                    continue
                if key in primary_keys:
                    continue
                tasks.append(
                    (r.file, r.document_type, r.document_number, r.entity, r.party_id, r.document_date, ep, f'data/{ep}/pdfs/{r.file}')
                )
                
    print(f"Extracting clauses from {len(tasks)} PDFs (phase={phase}, extra={extra_phases}) with 8 workers...")
    t0 = datetime.now()
    with mp.Pool(8) as pool:
        rows = pool.map(extract_one_pdf, tasks, chunksize=100)
    t1 = datetime.now()
    elapsed = (t1 - t0).total_seconds()
    print(f"Extracted {len(rows)} PDFs in {elapsed:.2f}s ({len(rows)/max(0.01, elapsed):.1f} PDFs/s)")
    
    df = pd.DataFrame(rows)
    # Ensure correct columns and types
    df = df[OUTPUT_COLS]
    df['n_instalments'] = df['n_instalments'].astype('Int64')
    for col in ['retention_pct', 'retention_amount', 'amount_payable', 'advance_amount', 'total']:
        df[col] = df[col].astype('float64')
    for col in ['rappel', 'factoring']:
        df[col] = df[col].astype('bool')
        
    return df


def load(phase: str, extra_phases: list[str] = None, force: bool = False) -> pd.DataFrame:
    """Load cached pdf_clauses.parquet or extract if not present or force=True."""
    out_dir = f'matching/out/{phase}'
    out_file = os.path.join(out_dir, 'pdf_clauses.parquet')
    
    if os.path.exists(out_file) and not force:
        print(f"Loading cached {out_file}...")
        return pd.read_parquet(out_file)
        
    df = extract(phase, extra_phases=extra_phases)
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(out_file, index=False)
    print(f"Saved {len(df)} rows to {out_file}")
    return df


def run_census(sample_size=600):
    """Run a census on ~600 sample PDFs and print top clauses."""
    df_val = pd.read_csv('data/val/pdfs/index.csv', dtype=str, keep_default_na=False)
    sample = df_val.groupby(['entity', 'document_type'], group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(6, sample_size // max(1, len(df_val.groupby(['entity', 'document_type']))))), random_state=42)
    ).reset_index(drop=True)
    
    tasks = [
        (r.file, r.document_type, r.document_number, r.entity, r.party_id, r.document_date, 'val', f'data/val/pdfs/{r.file}')
        for _, r in sample.iterrows()
    ]
    with mp.Pool(8) as pool:
        rows = pool.map(extract_one_pdf, tasks)
        
    all_clause_lines = []
    for r in rows:
        if r['clause_text']:
            for line in r['clause_text'].split(' || '):
                all_clause_lines.append(line.strip())
                
    patterns = Counter()
    for l in all_clause_lines:
        m = re.sub(r'\d+', '#', l)
        patterns[m] += 1
        
    print(f"=== CENSUS: TOP 40 CLAUSE PHRASINGS (Sample N={len(sample)}) ===")
    for idx, (pat, count) in enumerate(patterns.most_common(40), 1):
        print(f"{idx:2d}. [{count:3d}] {pat}")
        
    return patterns.most_common(40)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract PDF clauses.")
    parser.add_argument('--phase', choices=['sample', 'val', 'train', 'train_lite', 'live', 'final'], default='sample', help="Target phase")
    parser.add_argument('--extra', type=str, default=None, help="Comma-separated list of extra phases (e.g. train,live)")
    parser.add_argument('--force', action='store_true', help="Force re-extraction even if parquet exists")
    parser.add_argument('--census', action='store_true', help="Run and print the clause census")
    args = parser.parse_args()
    
    if args.census:
        run_census()
        return

    extra_phases = [p.strip() for p in args.extra.split(',')] if args.extra else None
    df = load(phase=args.phase, extra_phases=extra_phases, force=args.force)
    
    print(f"\n=== EXTRACTION SUMMARY ({args.phase}) ===")
    print(f"Total documents: {len(df)}")
    for col in df.columns:
        if col in ['rappel', 'factoring']:
            print(f"  {col:24s}: {df[col].sum()} True ({df[col].mean()*100:.1f}%)")
        elif col == 'cn_settlement':
            non_empty = (df[col] != '').sum()
            print(f"  {col:24s}: {non_empty} non-empty ({dict(df[col].value_counts())})")
        else:
            n_valid = df[col].notna().sum()
            print(f"  {col:24s}: {n_valid} / {len(df)} ({n_valid/len(df)*100:.1f}%)")
            
    # Sanity checks
    ret_rows = df[df.retention_pct.notna() & df.total.notna()]
    if len(ret_rows) > 0:
        expected = ret_rows.retention_pct / 100.0 * ret_rows.total
        diff = (ret_rows.retention_amount - expected).abs()
        max_diff = diff.max()
        print(f"\nRetention sanity: {len(ret_rows)} rows checked, max abs diff = {max_diff:.4f}")
        assert max_diff < 0.05, f"Retention diff too high: {max_diff}"
        print("Retention sanity PASSED!")

    inst_rows = df[df.n_instalments.notna()]
    if len(inst_rows) > 0:
        min_n = inst_rows.n_instalments.min()
        max_n = inst_rows.n_instalments.max()
        print(f"Instalments sanity: min={min_n}, max={max_n}")
        assert 2 <= min_n and max_n <= 12, f"Instalments out of bounds: {min_n}..{max_n}"
        print("Instalments sanity PASSED!")


if __name__ == '__main__':
    main()
