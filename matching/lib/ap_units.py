"""AP matcher building blocks: name normalisation, payment-run rows -> AP items, bank line -> run unit."""
from __future__ import annotations
import itertools, re, unicodedata
import numpy as np
import pandas as pd

TOL = 0.011
_SUFFIX = re.compile(r"\b(S ?A ?U|S ?L ?U|S ?A|S ?L|S ?COOP|S ?P ?A|S ?R ?L|GMBH|LTD|LIMITED|LDA|EIRELI|CO|PTY|INC|LLC|CORP|AG|BV|E ?K|EFC|SAS|SARL|SRL|LTDA|SP ZOO|SP Z O O|KG|OY|AB|AS|PLC|CV|SC|SRO|DE CV|SA DE CV|S DE RL|PVT|CIA)\b")

def norm_name(s) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_core(s) -> str:
    """name without legal suffix tokens"""
    s = norm_name(s)
    for _ in range(3):
        s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

RUN_RX = re.compile(r"((?:PR|CF)\d{12})(?:-(\d{1,4}))?")

def extract_run(row_vals):
    for v in row_vals:
        if isinstance(v, str):
            m = RUN_RX.search(v)
            if m:
                return m.group(1), m.group(2)
    return None, None

# ----------------------------------------------------------------------------------------------------------------------
def prep_runs(runs: pd.DataFrame) -> pd.DataFrame:
    r = runs[["run_id", "run_type", "execution_date", "debtor_account_id", "end_to_end_id", "beneficiary", "beneficiary_iban", "currency",
              "transfer_amount", "document_number", "document_type", "document_amount", "entity"]].copy()
    r["run_raw"] = r.run_id
    r["run_id"] = r.run_id + "@" + r.debtor_account_id   # the same run id can be debited from two accounts
    r["document_number"] = r.document_number.astype(str)
    r["damt"] = r.document_amount.abs().round(2)
    r["kind"] = np.where(r.document_type.eq("CREDIT_NOTE"), "APC", "API")
    r["bnorm"] = r.beneficiary.map(norm_core)
    r["row_id"] = np.arange(len(r))
    return r

def resolve_rows(rows: pd.DataFrame, items: pd.DataFrame, chunk_runs: int = 30, ignore_currency: bool = False) -> pd.DataFrame:
    """Return rows with columns item_id, party_id, status in {exact, partial, ambiguous, none}, amt (to allocate)."""
    it = items[["item_id", "item_type", "party_id", "document_number", "currency", "total_amount", "open_amount", "issue_date"]].copy()
    it["document_number"] = it.document_number.astype(str)
    it = it.rename(columns={"item_type": "kind", "document_number": "document_number", "currency": "icur"})
    out = []
    run_ids = rows.run_id.unique()
    for i in range(0, len(run_ids), chunk_runs):
        rs = rows[rows.run_id.isin(run_ids[i:i + chunk_runs])]
        m = rs[["row_id", "run_id", "end_to_end_id", "document_number", "kind", "currency", "damt", "execution_date", "bnorm"]].merge(
            it, on=["document_number", "kind"], how="left")
        m = m[m.item_id.notna()]
        if not ignore_currency:
            m = m[m.currency == m.icur]
        m = m[m.issue_date <= m.execution_date + pd.Timedelta(days=3)]
        m["exact"] = ((m.damt - m.open_amount).abs() <= TOL) | ((m.damt - m.total_amount).abs() <= TOL)
        m["fits"] = m.damt <= m.total_amount + TOL
        out.append(m[["row_id", "run_id", "end_to_end_id", "item_id", "party_id", "exact", "fits", "total_amount", "open_amount"]])
    cand = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["row_id"])
    return cand

def decide_rows(rows: pd.DataFrame, cand: pd.DataFrame, alias: dict | None = None) -> pd.DataFrame:
    """Pick one item per run row using the unit supplier (votes of uniquely-exact rows)."""
    rows = rows.copy()
    rows["unit"] = rows.run_id + "|" + rows.end_to_end_id.astype(str)
    cand = cand.merge(rows[["row_id", "unit", "bnorm"]], on="row_id")
    ex = cand[cand.exact]
    nex = ex.groupby("row_id").item_id.transform("nunique")
    uniq = ex[nex == 1].drop_duplicates("row_id")
    votes = uniq.groupby(["unit", "party_id"]).size().rename("v").reset_index()
    votes = votes.sort_values(["unit", "v"], ascending=[True, False])
    top = votes.groupby("unit").head(2)
    first = top.groupby("unit").nth(0).set_index("unit")
    second = top.groupby("unit").nth(1).set_index("unit")
    unit_party = first.party_id.copy()
    amb = first.v.reindex(second.index) <= second.v
    unit_party = unit_party.drop(index=second.index[amb.values], errors="ignore")
    # alias fallback for units without a party
    ub = rows.drop_duplicates("unit").set_index("unit").bnorm
    if alias:
        miss = ub.index.difference(unit_party.index)
        add = ub.loc[miss].map(alias).dropna()
        unit_party = pd.concat([unit_party, add])
    rows = rows.set_index("row_id")
    cand["uparty"] = cand.unit.map(unit_party)
    # candidates restricted to the unit party (if known), else only uniquely exact ones
    known = cand[cand.uparty.notna() & (cand.party_id == cand.uparty)].copy()
    k_ex = known[known.exact].drop_duplicates(["row_id", "item_id"])
    cnt_ex = k_ex.groupby("row_id").item_id.transform("nunique")
    pick_ex = k_ex[cnt_ex == 1].drop_duplicates("row_id").assign(status="exact")
    done = set(pick_ex.row_id)
    k_in = known[~known.row_id.isin(done) & known.fits].drop_duplicates(["row_id", "item_id"])
    cnt_in = k_in.groupby("row_id").item_id.transform("nunique")
    pick_in = k_in[cnt_in == 1].drop_duplicates("row_id").assign(status="partial")
    done |= set(pick_in.row_id)
    # no unit party: unique exact over everything
    rest = cand[~cand.row_id.isin(done) & cand.uparty.isna()]
    rex = rest[rest.exact].drop_duplicates(["row_id", "item_id"])
    rc = rex.groupby("row_id").item_id.transform("nunique")
    pick_u = rex[rc == 1].drop_duplicates("row_id").assign(status="exact_noparty")
    picks = pd.concat([pick_ex, pick_in, pick_u])[["row_id", "item_id", "party_id", "status", "total_amount", "open_amount"]]
    return picks, unit_party

def build_alias(rows, cand) -> dict:
    """beneficiary (normalised) -> party_id from rows whose candidate is uniquely exact; used only if consistent."""
    c = cand.merge(rows[["row_id", "bnorm"]], on="row_id")
    ex = c[c.exact]
    n = ex.groupby("row_id").item_id.transform("nunique")
    u = ex[n == 1].drop_duplicates("row_id")
    g = u.groupby(["bnorm", "party_id"]).size().rename("n").reset_index()
    tot = g.groupby("bnorm").n.transform("sum")
    g = g[(g.n / tot >= 0.9) & (g.bnorm != "")]
    return dict(zip(g.bnorm, g.party_id))
