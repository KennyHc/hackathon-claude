# K3 round 2 plan (14:30 → interim 15:45, final 16:45)

1. History-free payer (flag PARTY_ANYWHERE, default off until validated): entity-scoped index of master customers
   (data/masters ∪ masters_final via read_master(phase)) by their first two normalised name words → scan every word
   position of text / counterparty_name; a customer is a candidate when the text from that position shares ≥ min(len(name),
   12) chars with its name (truncation-safe). IBANs found anywhere in text (regex) → master iban. Candidates only accepted
   through STRONG amount fits (single open item / exact subset / discount / withholding), never weak ones.
2. Runtime harvest (flag PARTY_HARVEST): before the pass, keys_of(line) → party for every line claimed by batch/refs
   with a single party (added only when the key is unknown or collides); during the pass, own claims add their keys;
   optional second pass over no_party/no_fit lines.
3. unid.py: unid_evidence(lines, items, phase) → bank_txn_id, has_ref_like (digit run ≥ 4 matching the tail of an
   open item's document number in the same entity, or invoice-like token INV/FV/FA/NF/KWI/RE+digits), payer_known
   (resolver/anywhere/IBAN gives a customer of the line's entity), best_candidate_customer.
4. LOEO: --loeo PL01 drops that entity's train lines from alias learning (masters kept, as for ZA01). Report on val
   PL01 lines: party alloc F1, UNIDENTIFIED P/R of rule "unclaimed receipt → UNID" vs "… and no evidence → UNID",
   before/after 1–3.
5. Weak segments (ar_1toN_credit_note, ar_fx, ar_discount_late, ar_advance_deduction, ar_duplicate_payment): diagnose
   on val, fix in party.
6. WINDOW A/B: latest-due vs nearest-to-booking.
Alias cache key includes mtimes of train bank/golden + masters files + loeo.
