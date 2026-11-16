"""Evaluation helpers (pure Python). Labels enter only here, after inference, joined on (OBJECTID, year)."""
from collections import Counter
from .engine import binary


def join_rows(traces, labels, ml=None):
    """One row per reference field-year; every row is kept (abstentions, failures, missing predictions)."""
    by = {(int(t["OBJECTID"]), int(t["year"])): t for t in traces}
    mlby = {(int(m["OBJECTID"]), int(m["year"])): m for m in (ml or [])}
    rows = []
    for l in labels:
        k = (int(l["OBJECTID"]), int(l["year"]))
        t = by.get(k)
        m = mlby.get(k)
        rows.append(dict(OBJECTID=k[0], year=k[1], label=l["label"].lower(),
                         final=t["final"] if t else "NO OUTPUT (failure)", review=bool(t["review"]) if t else True,
                         reason_codes=";".join(t["reason_codes"]) if t else "no_rulebook_output",
                         rb=binary(t["final"]) if t else None,
                         ml=(m["pred"] if m and m.get("pred") in ("silage", "grain") else None)))
    return rows


def bin_metrics(y, p):
    """y, p in {'silage','grain'}; silage = positive. Only decided pairs."""
    tp = sum(1 for a, b in zip(y, p) if a == "silage" and b == "silage"); fn = sum(1 for a, b in zip(y, p) if a == "silage" and b == "grain")
    fp = sum(1 for a, b in zip(y, p) if a == "grain" and b == "silage"); tn = sum(1 for a, b in zip(y, p) if a == "grain" and b == "grain")
    sd = lambda a, b: a / b if b else None
    m = dict(n=len(y), tp=tp, fn=fn, fp=fp, tn=tn, precision_silage=sd(tp, tp + fp), recall_silage=sd(tp, tp + fn),
             precision_grain=sd(tn, tn + fn), recall_grain=sd(tn, tn + fp))
    m["balanced_accuracy"] = None if m["recall_silage"] is None or m["recall_grain"] is None else (m["recall_silage"] + m["recall_grain"]) / 2
    return m


def summarize(rows, key="rb"):
    n = len(rows)
    dec = [r for r in rows if r[key] in ("silage", "grain")]
    correct = sum(1 for r in dec if r[key] == r["label"])
    cov = lambda rs: (sum(1 for r in rs if r[key] in ("silage", "grain")), len(rs))
    return dict(n=n, decided=len(dec), coverage=len(dec) / n if n else None, correct=correct,
                correct_over_all=correct / n if n else None, decided_accuracy=correct / len(dec) if dec else None,
                binary=bin_metrics([r["label"] for r in dec], [r[key] for r in dec]),
                coverage_by_year={y: cov([r for r in rows if r["year"] == y]) for y in sorted({r["year"] for r in rows})},
                coverage_by_class={c: cov([r for r in rows if r["label"] == c]) for c in ("silage", "grain")},
                outcomes=dict(Counter(r["final"] for r in rows)) if key == "rb" else None)
