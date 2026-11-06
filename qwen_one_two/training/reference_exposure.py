"""Metadata-only matched exposure intervention; no new annotation authoring."""
from collections import Counter
from .inputs import check, canonical, digest


ANCHORS = {'train-reference-0000': 86, 'train-reference-0010': 86,
           'train-reference-0001': 85, 'train-reference-0011': 85}
CONTROLS = tuple(ANCHORS) + ('train-reference-0008', 'train-reference-0018')


def reallocate(windows, roles, desired):
    """Replace evenly spaced repeat occurrences, preserving firsts and slot roles.

    `desired` maps existing view identities to final exposure counts. Ordering
    is explicit and deterministic. Neither labels nor measured outcomes enter.
    """
    check(bool(windows) and all(len(w) == 8 for w in windows), 'eight-slot windows')
    original = [key for window in windows for key in window]
    before = Counter(original)
    check(set(before) <= set(roles) and set(desired) <= set(before), 'unknown view')
    check(all(type(n) is int and n >= before[k] for k, n in desired.items()), 'exposure decrease')
    result = list(original)
    for role in sorted({roles[k] for k in desired}):
        keys = [k for k in desired if roles[k] == role]
        remaining = {k: desired[k] - before[k] for k in keys}
        additions = []
        while any(remaining.values()):
            for key in keys:
                if remaining[key]:
                    additions.append(key)
                    remaining[key] -= 1
        seen = set(); candidates = []
        for index, key in enumerate(original):
            if key in seen and key not in desired and roles[key] == role:
                candidates.append(index)
            seen.add(key)
        check(len(candidates) >= len(additions), 'insufficient same-role repeat slots')
        # Integer midpoint quantiles avoid floating rounding and duplicate picks.
        for j, key in enumerate(additions):
            index = candidates[((2*j+1)*len(candidates))//(2*len(additions))]
            result[index] = key
    after = Counter(result)
    check(set(before) == set(after), 'distinct-view coverage changed')
    check(all(after[k] == n for k, n in desired.items()), 'exposure allocation mismatch')
    seen = set(); changes = []
    for index, (old, new) in enumerate(zip(original, result)):
        check(roles[old] == roles[new], 'slot role changed')
        if old != new:
            check(old in seen and old not in desired, 'first occurrence or anchor removed')
            changes.append(dict(window=index//8+1, slot=index%8, removed=old, added=new, role=roles[new]))
        seen.add(old)
    treatment = [result[i:i+8] for i in range(0, len(result), 8)]
    return treatment, dict(changes=changes, before=dict(before), after=dict(after),
                          control_sha256=digest(canonical(windows)),
                          treatment_sha256=digest(canonical(treatment)))
