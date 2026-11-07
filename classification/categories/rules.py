"""classification.categories.rules: High-precision text/code rules for bank lines.

Rules fire on unambiguous indicators (bank transaction codes, proprietary subcodes,
exact accounting phrases in free text, own account sweeps, and reversal pairs)
to assign categories with near 100% precision and override model predictions.
"""
from __future__ import annotations

import re
from typing import Dict, Tuple
import numpy as np
import pandas as pd

# 21 standard category codes from data/masters/categories.csv
ALL_CATEGORIES = [
    "BANK_FEES", "INTEREST_INCOME", "INTEREST_EXPENSE", "PAYROLL",
    "SOCIAL_SECURITY", "TAX_PAYMENT", "TAX_REFUND", "LOAN_DRAWDOWN",
    "LOAN_REPAYMENT", "LEASING", "FX_DEAL", "CASH_POOL", "INTERCOMPANY",
    "INTERNAL_TRANSFER", "CUSTOMER_ADVANCE", "CUSTOMER_REFUND",
    "SUPPLIER_REFUND", "CARD_EXPENSES", "INSURANCE", "BANK_REVERSAL",
    "UNIDENTIFIED"
]

# Precompiled regex patterns
RE_CASH_POOL = re.compile(
    r"ZBA SWEEP|CASH CONCENTRATION|TRASPASO CASH POOLING|TRASPASO ENTRE CUENTAS ES00|TRASPASO PROPIO",
    re.IGNORECASE
)
RE_INTERNAL_TRANSFER = re.compile(r"TRASPASO (?:A|DE) CTA \d+", re.IGNORECASE)
RE_INTERCOMPANY = re.compile(
    r"INTRAGRUPO|\bIC VALTERMIA\b|FINANCIACION IC|TRANSF\.?\s*IC|PRESTAMO.*VALTERMIA|DEVOLUCION PRESTAMO.*VALTERMIA",
    re.IGNORECASE
)
RE_BANK_REVERSAL = re.compile(r"ANULACION OPERACION INDEBIDA", re.IGNORECASE)

RE_INTEREST_TEXT = re.compile(
    r"LIQUIDACION INTERESES|INTERESES A FAVOR|INTEREST EARNED|INTEREST CHARGED|INTEREST PAID",
    re.IGNORECASE
)
RE_FEE_TEXT = re.compile(
    r"COMISION|GASTOS|FRAIS VIREMENT|SERV\. EM\. TRANSF|COM\. TRANSF|LIQUIDACION CTA\.CTO|MANTENIMIENTO|CUSTODIA",
    re.IGNORECASE
)
RE_PAYROLL_TEXT = re.compile(r"\bPAYROLL\b|NOMINAS?\b|PAG NOMINAS|\bSALARIES\b", re.IGNORECASE)
RE_SOCIAL_SEC = re.compile(r"SOCIAL CHARGES|TGSS\. COTIZACION|CUOTAS DE LA SEGURIDAD SOCIAL", re.IGNORECASE)

RE_TAX_REFUND = re.compile(r"DEVOLUCION AEAT MOD \d+", re.IGNORECASE)
RE_TAX_PAY = re.compile(r"VAT RETURN|TAX AUTHORITY|TAX PAYMENT|DGFIP TVA|IMPUESTOS MOD \d+", re.IGNORECASE)

RE_LOAN_DD = re.compile(r"DISPOSICION POLIZA CREDITO", re.IGNORECASE)
RE_LOAN_REP = re.compile(r"AMORTIZACION DE PRESTAMO|PRES\.[A-Z]{2}\d{2}-\d+", re.IGNORECASE)

RE_LEASING = re.compile(r"CUOTA\s+(?:LEASING|LSG|RENTING|ARRENDAMIENTO)", re.IGNORECASE)
RE_INSURANCE = re.compile(r"INSURANCE PREMIUM|PRIMA SEGURO|SEGUROS MULTIRRIESGO", re.IGNORECASE)
RE_CARD = re.compile(r"ADEUDO MENSUAL DE TARJETA", re.IGNORECASE)
RE_FX_DEAL = re.compile(r"\bFX DEAL\b|VENTA DIVISA|COMPRA DIVISA", re.IGNORECASE)

RE_ADVANCE = re.compile(r"PROFORMA|PAGO PROFORMA|\bPF\d{2}-\d+", re.IGNORECASE)
RE_REFUND = re.compile(r"DEVOLUCION PAGO DUPLICADO|REFUND DUPLICATE|ABONO PAGO DOBLE", re.IGNORECASE)

FEE_BANK_CODES = {"17", "NCHG", "ACMT/MCOP/CHRG"}
INTEREST_BANK_CODES = {"NINT", "ACMT/MDOP/INTR"}
SALARY_BANK_CODES = {"NSAL", "PMNT/ICDT/SALA"}
LOAN_REP_BANK_CODES = {"05", "NLDP", "LDAS/CSLN/PPAY"}
FX_BANK_CODES = {"NFEX", "FORX/SPOT/OTHR"}


def apply_rules(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Applies rules to bank transactions dataframe.

    Returns:
        mask_fired: bool array indicating whether a rule fired for each row.
        rule_categories: str array containing the assigned category (or '' if none).
        rule_confidences: float array of confidence scores (0.99 for rule matches, 0.0 otherwise).
        rule_methods: str array describing the method name (e.g. 'rule:cash_pool').
    """
    n = len(df)
    rule_cat = np.full(n, "", dtype=object)
    rule_method = np.full(n, "", dtype=object)
    rule_conf = np.zeros(n, dtype=np.float64)

    # Prepare inputs
    amt = df["amount"].astype(float).values
    text = df["text"].fillna("").astype(str).values
    rem = df["remittance_info"].fillna("").astype(str).values
    bcode = df["bank_code"].fillna("").astype(str).values

    # Helper vectorizers
    def match_regex(series_arr: np.ndarray, pattern: re.Pattern) -> np.ndarray:
        return np.array([bool(pattern.search(s)) for s in series_arr], dtype=bool)

    text_match_cp = match_regex(text, RE_CASH_POOL)
    text_match_it = match_regex(text, RE_INTERNAL_TRANSFER)
    text_match_ic = match_regex(text, RE_INTERCOMPANY)
    text_match_br = match_regex(text, RE_BANK_REVERSAL)

    text_match_intr = match_regex(text, RE_INTEREST_TEXT)
    bcode_intr = np.isin(bcode, list(INTEREST_BANK_CODES))
    bcode_fee = np.isin(bcode, list(FEE_BANK_CODES))

    text_match_pay = match_regex(text, RE_PAYROLL_TEXT)
    bcode_pay = np.isin(bcode, list(SALARY_BANK_CODES))
    text_match_ss = match_regex(text, RE_SOCIAL_SEC)

    text_match_tr = match_regex(text, RE_TAX_REFUND)
    text_match_tp = match_regex(text, RE_TAX_PAY)

    text_match_ld = match_regex(text, RE_LOAN_DD)
    text_match_lr = match_regex(text, RE_LOAN_REP)
    bcode_lr = np.isin(bcode, list(LOAN_REP_BANK_CODES))

    text_match_ls = match_regex(text, RE_LEASING)
    text_match_ins = match_regex(text, RE_INSURANCE)
    text_match_card = match_regex(text, RE_CARD)
    text_match_fx = match_regex(text, RE_FX_DEAL)
    bcode_fx = np.isin(bcode, list(FX_BANK_CODES))

    text_match_adv = match_regex(text, RE_ADVANCE) | match_regex(rem, RE_ADVANCE)
    text_match_ref = match_regex(text, RE_REFUND) | match_regex(rem, RE_REFUND)

    # 1. CASH_POOL
    m_cp = (bcode == "CAMT/ACCB/ZABA") | text_match_cp
    rule_cat[m_cp] = "CASH_POOL"
    rule_method[m_cp] = "rule:cash_pool"

    # 2. INTERNAL_TRANSFER
    m_it = text_match_it & (rule_cat == "")
    rule_cat[m_it] = "INTERNAL_TRANSFER"
    rule_method[m_it] = "rule:internal_transfer"

    # 3. INTERCOMPANY
    m_ic = text_match_ic & (rule_cat == "")
    rule_cat[m_ic] = "INTERCOMPANY"
    rule_method[m_ic] = "rule:intercompany"

    # 4. BANK_REVERSAL (explicit reversal statement)
    m_br = text_match_br & (rule_cat == "")
    rule_cat[m_br] = "BANK_REVERSAL"
    rule_method[m_br] = "rule:bank_reversal"

    # 5. INTEREST_INCOME / INTEREST_EXPENSE
    is_interest = text_match_intr | bcode_intr
    m_ii = (amt > 0) & is_interest & (rule_cat == "")
    m_ie = (amt < 0) & is_interest & (rule_cat == "")
    rule_cat[m_ii] = "INTEREST_INCOME"
    rule_method[m_ii] = "rule:interest_income"
    rule_cat[m_ie] = "INTEREST_EXPENSE"
    rule_method[m_ie] = "rule:interest_expense"

    # 6. BANK_FEES
    m_fee = bcode_fee & (rule_cat == "")
    rule_cat[m_fee] = "BANK_FEES"
    rule_method[m_fee] = "rule:bank_fees"

    # 7. SOCIAL_SECURITY
    m_ss = (amt < 0) & text_match_ss & (rule_cat == "")
    rule_cat[m_ss] = "SOCIAL_SECURITY"
    rule_method[m_ss] = "rule:social_security"

    # 8. PAYROLL
    m_pay = (bcode_pay | ((amt < 0) & text_match_pay)) & (rule_cat == "")
    rule_cat[m_pay] = "PAYROLL"
    rule_method[m_pay] = "rule:payroll"

    # 9. TAXES
    m_tr = (amt > 0) & text_match_tr & (rule_cat == "")
    m_tp = (amt < 0) & text_match_tp & (rule_cat == "")
    rule_cat[m_tr] = "TAX_REFUND"
    rule_method[m_tr] = "rule:tax_refund"
    rule_cat[m_tp] = "TAX_PAYMENT"
    rule_method[m_tp] = "rule:tax_payment"

    # 10. LOANS
    m_ld = (amt > 0) & text_match_ld & (rule_cat == "")
    m_lr = (amt < 0) & (bcode_lr | text_match_lr) & (rule_cat == "")
    rule_cat[m_ld] = "LOAN_DRAWDOWN"
    rule_method[m_ld] = "rule:loan_drawdown"
    rule_cat[m_lr] = "LOAN_REPAYMENT"
    rule_method[m_lr] = "rule:loan_repayment"

    # 11. LEASING
    m_ls = (amt < 0) & text_match_ls & (rule_cat == "")
    rule_cat[m_ls] = "LEASING"
    rule_method[m_ls] = "rule:leasing"

    # 12. INSURANCE
    m_ins = (amt < 0) & text_match_ins & (rule_cat == "")
    rule_cat[m_ins] = "INSURANCE"
    rule_method[m_ins] = "rule:insurance"

    # 13. CARD_EXPENSES
    m_card = (amt < 0) & text_match_card & (rule_cat == "")
    rule_cat[m_card] = "CARD_EXPENSES"
    rule_method[m_card] = "rule:card_expenses"

    # 14. FX_DEAL
    m_fx = (bcode_fx | text_match_fx) & (rule_cat == "")
    rule_cat[m_fx] = "FX_DEAL"
    rule_method[m_fx] = "rule:fx_deal"

    # 15. CUSTOMER_ADVANCE
    m_adv = (amt > 0) & text_match_adv & (rule_cat == "")
    rule_cat[m_adv] = "CUSTOMER_ADVANCE"
    rule_method[m_adv] = "rule:customer_advance"

    # 16. REFUNDS
    m_cr = (amt < 0) & text_match_ref & (rule_cat == "")
    m_sr = (amt > 0) & text_match_ref & (rule_cat == "")
    rule_cat[m_cr] = "CUSTOMER_REFUND"
    rule_method[m_cr] = "rule:customer_refund"
    rule_cat[m_sr] = "SUPPLIER_REFUND"
    rule_method[m_sr] = "rule:supplier_refund"

    # 17. REVERSAL COUNTERPART PAIRING
    # For every ANULACION OPERACION INDEBIDA line, match the opposite amount in same account & nearby date
    anul_indices = np.where(m_br)[0]
    if len(anul_indices) > 0 and "account_id" in df.columns:
        accounts = df["account_id"].values
        dates = df["booking_date"].values if "booking_date" in df.columns else np.full(n, "")
        for idx in anul_indices:
            a_acc = accounts[idx]
            a_amt = amt[idx]
            a_date_m = dates[idx][:7] if len(dates[idx]) >= 7 else ""

            # Candidates: same account, opposite amount, unassigned category, not self
            opp_mask = (
                (accounts == a_acc) &
                (np.abs(amt + a_amt) < 1e-4) &
                (rule_cat == "") &
                (np.arange(n) != idx)
            )
            cands = np.where(opp_mask)[0]
            if len(cands) == 1:
                c_idx = cands[0]
                rule_cat[c_idx] = "BANK_REVERSAL"
                rule_method[c_idx] = "rule:bank_reversal_pair"
            elif len(cands) > 1 and a_date_m:
                c_same_month = [c for c in cands if dates[c][:7] == a_date_m]
                if len(c_same_month) == 1:
                    c_idx = c_same_month[0]
                    rule_cat[c_idx] = "BANK_REVERSAL"
                    rule_method[c_idx] = "rule:bank_reversal_pair"

    fired = rule_cat != ""
    rule_conf[fired] = 0.99
    return fired, rule_cat, rule_conf, rule_method
