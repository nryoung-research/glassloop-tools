"""Build a frozen, authored ordinary-coding MECHANISM TRAIN bank.

CPU standard library only. Does not import a model, consult benchmarks, run a
grader, execute submitted/generated code, or derive expected results from oracles.
All expected fixtures below are independently written constants. Refuses overwrite.
"""
import hashlib
import json
from pathlib import Path


AUTHORING_DATE = '2026-09-08'
HERE = Path(__file__).resolve().parent


def spec(function, signature, domain, behavior, cases):
    return {'function_name': function, 'signature': signature, 'domain': domain,
            'behavior': behavior,
            'cases': [{'case_id': '%02d' % (index + 1), 'args': args, 'expected': expected}
                      for index, (args, expected) in enumerate(cases)]}


SPECS = [
    spec('locker_signal_totals', 'locker_signal_totals(records)',
         'records is a list of [label, delta] pairs. label is a nonempty ASCII string; delta is an integer.',
         'Sum deltas separately for each label. Return [label, total] pairs sorted by label using Python string ordering. Include labels whose total is zero. Empty input returns [].',
         [([[]], []),
          ([[['b', -1], ['a', 2], ['a', -2]]], [['a', 0], ['b', -1]]),
          ([[['x', 3], ['x', 4], ['x', -2]]], [['x', 5]]),
          ([[['a', 1], ['A', 2], ['z', 0]]], [['A', 2], ['a', 1], ['z', 0]])]),
    spec('parcel_first_fit', 'parcel_first_fit(loads, capacity)',
         'capacity is a positive integer. loads is a list of integers between 0 and capacity inclusive.',
         'Partition loads in original order into nonempty consecutive groups. Append each next load to the current group if its sum would not exceed capacity; otherwise start a new group. This is sequential grouping, never placement into an earlier group. Zeros stay in the current group when one exists. Empty input returns [].',
         [([[], 3], []),
          ([[2, 1, 3, 0, 2], 3], [[2, 1], [3, 0], [2]]),
          ([[0, 0], 1], [[0, 0]]),
          ([[1, 1, 1], 1], [[1], [1], [1]]),
          ([[0, 2, 0, 2], 2], [[0, 2, 0], [2]])]),
    spec('ticket_stamp_runs', 'ticket_stamp_runs(stamps)',
         'stamps is a list of ASCII strings, including possibly empty strings.',
         'Encode consecutive equal strings as [string, run_length] pairs. Preserve run order; equal values separated by another value remain different runs. Empty input returns [].',
         [([[]], []),
          ([['a', 'a', 'b', 'a']], [['a', 2], ['b', 1], ['a', 1]]),
          ([['', '', 'x']], [['', 2], ['x', 1]]),
          ([['x']], [['x', 1]])]),
    spec('courtyard_clock_union', 'courtyard_clock_union(intervals)',
         'intervals is a list of integer [start, end] pairs with start <= end. Each pair denotes the half-open interval [start, end).',
         'Discard empty intervals. Sort remaining intervals and merge every overlapping or touching chain. Return the resulting [start, end] pairs in increasing start order. Duplicates and nesting do not create additional intervals. Empty coverage returns [].',
         [([[]], []),
          ([[[4, 7], [1, 3], [3, 4], [10, 10], [9, 11]]], [[1, 7], [9, 11]]),
          ([[[-4, -1], [-3, 0], [2, 2]]], [[-4, 0]]),
          ([[[1, 8], [2, 3], [1, 8]]], [[1, 8]]),
          ([[[2, 2], [0, 0]]], [])]),
    spec('cabinet_rotate_segments', 'cabinet_rotate_segments(values, widths)',
         'values is a list of integers. widths is a list of nonnegative integers whose sum equals len(values).',
         'Split values into consecutive segments with the stated widths, including zero-width segments. Rotate each nonempty segment left by exactly one position, then concatenate all segments. A singleton stays unchanged. Empty values returns [].',
         [([[], []], []),
          ([[1, 2, 3, 4, 5], [2, 0, 3]], [2, 1, 4, 5, 3]),
          ([[9, 8, 7], [1, 1, 1]], [9, 8, 7]),
          ([[], [0, 0]], []),
          ([[-1, 0, 1], [3]], [0, 1, -1])]),
    spec('ribbon_word_ledger', 'ribbon_word_ledger(text)',
         'text contains only ASCII letters, decimal digits, and literal spaces. A token is a maximal non-space run.',
         'Lowercase each token, count its occurrences, and record its first zero-based token index. Return [token, count, first_index] rows sorted by decreasing count, breaking ties by first_index. Digit-only tokens count normally. Empty or all-space text returns [].',
         [([''], []),
          ([' A a 10 b B a '], [['a', 3, 0], ['b', 2, 3], ['10', 1, 2]]),
          (['Z y z Y x'], [['z', 2, 0], ['y', 2, 1], ['x', 1, 4]]),
          (['   '], []),
          (['12 012 12'], [['12', 2, 0], ['012', 1, 1]])]),
    spec('mosaic_diagonal_sums', 'mosaic_diagonal_sums(grid)',
         'grid is a rectangular list of lists of integers. It may have zero rows or zero columns.',
         'For every cell at zero-based row r and column c, add its value to output index r+c. Return these anti-diagonal sums in increasing r+c order. With R>0 and C>0 the result has length R+C-1; with no cells return [].',
         [([[]], []),
          ([[[], []]], []),
          ([[[1, 2, 3], [4, 5, 6]]], [1, 6, 8, 6]),
          ([[[-2], [7], [0]]], [-2, 7, 0]),
          ([[[5]]], [5])]),
    spec('copper_centered_deltas', 'copper_centered_deltas(values)',
         'values is a list of integers.',
         'For nonempty values, compute center = floor(sum(values)/len(values)), using mathematical floor also for negative means. Return each original value minus center in original order. Empty input returns [].',
         [([[]], []),
          ([[1, 2, 5]], [-1, 0, 3]),
          ([[-3, 0, 1]], [-2, 1, 2]),
          ([[9]], [0]),
          ([[-2, -2]], [0, 0])]),
    spec('harbor_restock_events', 'harbor_restock_events(stock, events)',
         'stock maps nonempty ASCII strings to nonnegative integers. events is a list of [label, delta] pairs with integer delta.',
         'Copy stock, then process events in order. For each event, set that label to max(0, current_value + delta), treating a missing label as zero. Keep zero-valued labels, including newly seen labels. Return the resulting dictionary; dictionary iteration order is not graded.',
         [([{}, []], {}),
          ([{'a': 2}, [['a', -5], ['a', 1], ['b', 3], ['b', -1]]], {'a': 1, 'b': 2}),
          ([{}, [['new', -2]]], {'new': 0}),
          ([{'x': 3, 'z': 0}, []], {'x': 3, 'z': 0}),
          ([{'x': 0}, [['x', 4], ['x', -2], ['x', -5]]], {'x': 0})]),
    spec('tapestry_serpentine', 'tapestry_serpentine(grid)',
         'grid is a possibly ragged list of lists of integers; empty rows are permitted.',
         'Read even-indexed rows from left to right and odd-indexed rows from right to left, then concatenate them. Row indices are the original zero-based indices; empty rows still count for parity. Empty grid returns [].',
         [([[]], []),
          ([[[1, 2], [], [3], [4, 5, 6]]], [1, 2, 3, 6, 5, 4]),
          ([[[], [1, 2], [3, 4]]], [2, 1, 3, 4]),
          ([[[7], [8], []]], [7, 8])]),
    spec('signal_suffix_balance', 'signal_suffix_balance(text)',
         'text contains only ASCII letters, spaces, and parentheses.',
         'Match each closing parenthesis to the most recent still-unmatched earlier opening parenthesis, if any. Ignore letters and spaces for matching but count them in character indices. Return {"unmatched_open": [...], "unmatched_close": [...]} with unmatched zero-based character indices in ascending order.',
         [([''], {'unmatched_open': [], 'unmatched_close': []}),
          (['a)(b(c)'], {'unmatched_open': [2], 'unmatched_close': [1]}),
          ([')(('], {'unmatched_open': [1, 2], 'unmatched_close': [0]}),
          (['(a) ()'], {'unmatched_open': [], 'unmatched_close': []}),
          (['abc'], {'unmatched_open': [], 'unmatched_close': []})]),
    spec('orchard_gap_groups', 'orchard_gap_groups(values, gap)',
         'values is a list of integers and gap is a nonnegative integer.',
         'Remove duplicates and sort ascending. Partition that sorted sequence into groups, starting a new group exactly when the difference from the previous value is greater than gap. Return the list of groups. With gap zero, each distinct value is a singleton. Empty input returns [].',
         [([[], 2], []),
          ([[9, 1, 3, 2, 9, 7], 2], [[1, 2, 3], [7, 9]]),
          ([[1, 1, 2], 0], [[1], [2]]),
          ([[-5, -3, 0, 1], 2], [[-5, -3], [0, 1]]),
          ([[5], 99], [[5]])]),
    spec('marble_shared_receipts', 'marble_shared_receipts(left, right)',
         'left and right are lists of ASCII strings, including possibly empty strings.',
         'Scan left in order. Emit a value only when an unused equal occurrence remains in right, consuming one such occurrence. Return emitted values in left order. Thus multiplicity is the minimum count across both inputs. Neither input may be changed.',
         [([[], ['a']], []),
          ([['a', 'b', 'a', 'c'], ['a', 'a', 'd']], ['a', 'a']),
          ([['b', 'a', 'b'], ['b', 'a']], ['b', 'a']),
          ([['', '', 'x'], ['', 'x', 'x']], ['', 'x']),
          ([['a'], []], [])]),
    spec('quartz_shift_calendar', 'quartz_shift_calendar(days, shift)',
         'days is a list of integers from 0 through 6 inclusive; shift is any integer.',
         'Replace each day d with (d+shift) modulo 7. Return a seven-element list whose index j counts shifted days equal to j. Empty days yields seven zeros. Negative and large shifts wrap in the same way.',
         [([[], 0], [0, 0, 0, 0, 0, 0, 0]),
          ([[0, 6, 6], 1], [2, 1, 0, 0, 0, 0, 0]),
          ([[0, 1, 2], -1], [1, 1, 0, 0, 0, 0, 1]),
          ([[0, 1, 2, 3, 4, 5, 6], 100], [1, 1, 1, 1, 1, 1, 1]),
          ([[3, 3], 7], [0, 0, 0, 2, 0, 0, 0])]),
    spec('loom_priority_merge', 'loom_priority_merge(records, order)',
         'records is a list of [label, integer_value] pairs with ASCII string labels. order is a list of distinct ASCII string labels.',
         'Return fresh copies of the records, putting labels listed in order first in that exact priority order. Put all other labels afterward in ascending Python string order. For equal labels, preserve original record order. Values never affect sorting. Empty input returns [].',
         [([[], []], []),
          ([[['z', 1], ['b', 2], ['a', 3], ['z', 4], ['b', 5]], ['b', 'a']], [['b', 2], ['b', 5], ['a', 3], ['z', 1], ['z', 4]]),
          ([[['b', 1], ['A', 2], ['b', 0]], []], [['A', 2], ['b', 1], ['b', 0]]),
          ([[['x', 9], ['y', 8]], ['unused', 'y']], [['y', 8], ['x', 9]])]),
    spec('badge_pack_fields', 'badge_pack_fields(fields)',
         'fields is a list of printable ASCII strings (characters with codes 32 through 126), including possibly empty strings.',
         'Encode each field as its decimal character length, then a colon, then the unchanged field. Concatenate these encodings without any other separator. Use ordinary decimal without leading zeros, except zero itself. Empty fields encodes as ""; a single empty string encodes as "0:".',
         [([[]], ''),
          ([['']], '0:'),
          ([['a:b', '', '12']], '3:a:b0:2:12'),
          ([['a|b', 'x y']], '3:a|b3:x y'),
          ([['0123456789', ':']], '10:01234567891::')]),
]


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def build_payload():
    common = ('Implement exactly the requested function in Python. Return only a Python code block containing the function. '
              'Use no imports, file access, network access, input/output, or external state. Do not mutate input arguments. '
              'Inputs always satisfy the domain; invalid-input handling is not required. Integer means a Python integer, not a boolean.')
    items = []
    for index, item in enumerate(SPECS, 1):
        prompt = (common + '\n\nFunction: ' + item['signature'] + '\nDomain: ' + item['domain']
                  + '\nRequired behavior: ' + item['behavior'])
        result = dict(item, task_id='DGTRAIN/%02d' % index, role='train', prompt=prompt,
                      prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest())
        result['fixture_sha256'] = hashlib.sha256(canonical_bytes(item['cases'])).hexdigest()
        items.append(result)
    return {'kind': 'decision_guard_coding_bank_v1', 'version': 1, 'role': 'train',
            'purpose': 'Sixteen-task ordinary-coding mechanism preservation training bank; not a benchmark or generalization exam.',
            'authored_date': AUTHORING_DATE, 'provenance': 'New specifications and constant fixtures authored in this task without consulting benchmark task content.',
            'benchmark_disjointness': 'NOT_ESTABLISHED: independent authoring does not certify lexical or semantic disjointness.',
            'monitor_bank': 'NOT_BUILT', 'parent_traces': 'NOT_PRODUCED', 'parent_correctness': 'NOT_MEASURED',
            'candidate_results': 'NOT_PRODUCED',
            'execution_contract': {'call_style': 'function(*case.args)', 'comparison': 'Exact JSON-value equality, including container/value types; dictionary key order is ignored.',
                                   'input_mutation': 'Forbidden; check deep copies before and after each call.',
                                   'isolation': 'A future model-code grader must use an external OS-isolated, resource-limited, no-network execution environment. This builder never executes submissions.'},
            'source1_sha256': file_sha(__file__), 'oracle_source_sha256': file_sha(HERE / 'coding_oracles.py'),
            'n_tasks': len(items), 'n_cases': sum(len(item['cases']) for item in items), 'items': items}


def main():
    destination = HERE / 'coding_bank.json'
    payload = build_payload()
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write('\n')
    print(json.dumps({'path': str(destination), 'sha256': file_sha(destination), 'n_tasks': payload['n_tasks'],
                      'n_cases': payload['n_cases'], 'source1_sha256': payload['source1_sha256'],
                      'oracle_source_sha256': payload['oracle_source_sha256']}))


if __name__ == '__main__':
    main()
