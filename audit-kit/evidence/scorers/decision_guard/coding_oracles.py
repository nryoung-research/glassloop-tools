"""Authored reference functions for a development preservation bank.

These are specification oracles, not model-generated answers or test results.
They require no imports, files, network, or external state and do not mutate inputs.
"""


def locker_signal_totals(records):
    totals = {}
    for label, delta in records:
        totals[label] = totals.get(label, 0) + delta
    return [[label, totals[label]] for label in sorted(totals)]


def parcel_first_fit(loads, capacity):
    groups, current, used = [], [], 0
    for load in loads:
        if current and used + load > capacity:
            groups.append(current)
            current, used = [], 0
        current.append(load)
        used += load
    if current:
        groups.append(current)
    return groups


def ticket_stamp_runs(stamps):
    result = []
    for stamp in stamps:
        if result and result[-1][0] == stamp:
            result[-1][1] += 1
        else:
            result.append([stamp, 1])
    return result


def courtyard_clock_union(intervals):
    ordered = sorted([start, end] for start, end in intervals if start < end)
    result = []
    for start, end in ordered:
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def cabinet_rotate_segments(values, widths):
    result, offset = [], 0
    for width in widths:
        segment = values[offset:offset + width]
        result.extend(segment[1:] + segment[:1])
        offset += width
    return result


def ribbon_word_ledger(text):
    counts, first = {}, {}
    for index, word in enumerate(text.lower().split()):
        if word not in first:
            first[word] = index
        counts[word] = counts.get(word, 0) + 1
    order = sorted(counts, key=lambda word: (-counts[word], first[word]))
    return [[word, counts[word], first[word]] for word in order]


def mosaic_diagonal_sums(grid):
    if not grid or not grid[0]:
        return []
    sums = [0] * (len(grid) + len(grid[0]) - 1)
    for row_index, row in enumerate(grid):
        for column_index, value in enumerate(row):
            sums[row_index + column_index] += value
    return sums


def copper_centered_deltas(values):
    if not values:
        return []
    center = sum(values) // len(values)
    return [value - center for value in values]


def harbor_restock_events(stock, events):
    result = dict(stock)
    for label, delta in events:
        result[label] = max(0, result.get(label, 0) + delta)
    return result


def tapestry_serpentine(grid):
    result = []
    for index, row in enumerate(grid):
        result.extend(row if index % 2 == 0 else row[::-1])
    return result


def signal_suffix_balance(text):
    openings, closings = [], []
    for index, char in enumerate(text):
        if char == '(':
            openings.append(index)
        elif char == ')':
            if openings:
                openings.pop()
            else:
                closings.append(index)
    return {'unmatched_open': openings, 'unmatched_close': closings}


def orchard_gap_groups(values, gap):
    result = []
    for value in sorted(set(values)):
        if result and value - result[-1][-1] <= gap:
            result[-1].append(value)
        else:
            result.append([value])
    return result


def marble_shared_receipts(left, right):
    remaining = {}
    for label in right:
        remaining[label] = remaining.get(label, 0) + 1
    result = []
    for label in left:
        if remaining.get(label, 0):
            result.append(label)
            remaining[label] -= 1
    return result


def quartz_shift_calendar(days, shift):
    counts = [0] * 7
    for day in days:
        counts[(day + shift) % 7] += 1
    return counts


def loom_priority_merge(records, order):
    ranks = {label: index for index, label in enumerate(order)}
    result = sorted(records, key=lambda row: (0, ranks[row[0]]) if row[0] in ranks else (1, row[0]))
    return [list(row) for row in result]


def badge_pack_fields(fields):
    return ''.join(str(len(field)) + ':' + field for field in fields)
