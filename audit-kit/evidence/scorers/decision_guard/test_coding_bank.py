"""CPU-only tests of authored oracles, literal fixtures, and source bindings.

No model output, generated submission, benchmark, or external grader is executed.
"""
import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path
import unittest

import coding_oracles as oracle
from build_coding_bank import SPECS, build_payload, canonical_bytes


HERE = Path(__file__).resolve().parent


class CodingBankTests(unittest.TestCase):
    def test_literal_fixtures_and_no_argument_mutation(self):
        for item in SPECS:
            function = getattr(oracle, item['function_name'])
            for case in item['cases']:
                with self.subTest(function=item['function_name'], case=case['case_id']):
                    args = copy.deepcopy(case['args'])
                    actual = function(*args)
                    self.assertEqual(canonical_bytes(actual), canonical_bytes(case['expected']))
                    self.assertEqual(canonical_bytes(args), canonical_bytes(case['args']))

    def test_bank_identity_and_declared_scope(self):
        stored = json.loads((HERE / 'coding_bank.json').read_text(encoding='utf-8'))
        self.assertEqual(stored, build_payload())
        self.assertEqual(stored['n_tasks'], 16)
        self.assertGreaterEqual(stored['n_cases'], 64)
        self.assertEqual(stored['role'], 'train')
        self.assertEqual(stored['monitor_bank'], 'NOT_BUILT')
        self.assertEqual(stored['parent_traces'], 'NOT_PRODUCED')
        self.assertEqual(stored['parent_correctness'], 'NOT_MEASURED')
        self.assertEqual(stored['candidate_results'], 'NOT_PRODUCED')
        self.assertEqual(len({item['task_id'] for item in stored['items']}), 16)
        self.assertEqual(len({item['function_name'] for item in stored['items']}), 16)
        for item in stored['items']:
            self.assertEqual(item['role'], 'train')
            self.assertGreaterEqual(len(item['cases']), 4)
            self.assertEqual(hashlib.sha256(item['prompt'].encode()).hexdigest(), item['prompt_sha256'])
            self.assertEqual(hashlib.sha256(canonical_bytes(item['cases'])).hexdigest(), item['fixture_sha256'])

    def test_reference_functions_have_no_imports_or_external_io(self):
        tree = ast.parse((HERE / 'coding_oracles.py').read_text(encoding='utf-8'))
        self.assertEqual(len([node for node in tree.body if isinstance(node, ast.FunctionDef)]), 16)
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)))
        banned = {'open', 'exec', 'eval', 'compile', '__import__', 'input', 'print'}
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                             and node.func.id in banned for node in ast.walk(tree)))

    def test_signal_totals_permutation_and_split_invariance(self):
        records = [['a', 4], ['b', -2], ['a', -1], ['c', 0]]
        expected = [['a', 3], ['b', -2], ['c', 0]]
        for perm in itertools.permutations(records):
            self.assertEqual(oracle.locker_signal_totals(list(perm)), expected)
        self.assertEqual(oracle.locker_signal_totals([['a', 1], ['a', 3]]),
                         oracle.locker_signal_totals([['a', 4]]))

    def test_parcel_partition_conservation_capacity_and_maximality(self):
        for loads in itertools.product(range(3), repeat=5):
            groups = oracle.parcel_first_fit(list(loads), 2)
            self.assertEqual([value for group in groups for value in group], list(loads))
            self.assertTrue(all(group and sum(group) <= 2 for group in groups))
            self.assertTrue(all(sum(left) + right[0] > 2 for left, right in zip(groups, groups[1:])))

    def test_stamp_runs_roundtrip_and_no_adjacent_duplicate_groups(self):
        for stamps in itertools.product(('', 'a', 'b'), repeat=4):
            groups = oracle.ticket_stamp_runs(list(stamps))
            self.assertEqual([value for value, count in groups for _ in range(count)], list(stamps))
            self.assertTrue(all(count > 0 for _, count in groups))
            self.assertTrue(all(left[0] != right[0] for left, right in zip(groups, groups[1:])))

    def test_interval_union_coverage_order_and_idempotence(self):
        intervals = [[4, 7], [-3, -1], [-1, 2], [1, 5], [10, 10]]
        merged = oracle.courtyard_clock_union(intervals)
        before = {point for start, end in intervals for point in range(start, end)}
        after = {point for start, end in merged for point in range(start, end)}
        self.assertEqual(before, after)
        self.assertTrue(all(left[1] < right[0] for left, right in zip(merged, merged[1:])))
        self.assertEqual(oracle.courtyard_clock_union(merged), merged)
        self.assertEqual(oracle.courtyard_clock_union(intervals[::-1]), merged)

    def test_segment_rotations_restore_after_common_period(self):
        original = list(range(8))
        current = original
        for _ in range(6):
            current = oracle.cabinet_rotate_segments(current, [2, 3, 0, 3])
        self.assertEqual(current, original)
        once = oracle.cabinet_rotate_segments(original, [2, 3, 0, 3])
        self.assertEqual(sorted(once), original)

    def test_word_ledger_whitespace_case_and_count_conservation(self):
        expected = oracle.ribbon_word_ledger('A b a 10 B c')
        self.assertEqual(oracle.ribbon_word_ledger('  a   B A  10 b C  '), expected)
        self.assertEqual(sum(row[1] for row in expected), 6)
        self.assertEqual([row[2] for row in expected], [0, 1, 3, 5])

    def test_diagonal_sum_conservation_transpose_and_linearity(self):
        left = [[1, -2, 3], [4, 0, 6]]
        right = [[-1, 3, 0], [2, 2, -4]]
        transposed = [list(column) for column in zip(*left)]
        a = oracle.mosaic_diagonal_sums(left)
        b = oracle.mosaic_diagonal_sums(right)
        self.assertEqual(sum(a), sum(map(sum, left)))
        self.assertEqual(a, oracle.mosaic_diagonal_sums(transposed))
        added = [[x + y for x, y in zip(row1, row2)] for row1, row2 in zip(left, right)]
        self.assertEqual(oracle.mosaic_diagonal_sums(added), [x + y for x, y in zip(a, b)])

    def test_centering_translation_and_floor_remainder(self):
        for values in itertools.product(range(-2, 3), repeat=3):
            actual = oracle.copper_centered_deltas(list(values))
            self.assertGreaterEqual(sum(actual), 0)
            self.assertLess(sum(actual), len(values))
            self.assertEqual(actual, oracle.copper_centered_deltas([value + 19 for value in values]))
            self.assertEqual(actual[0] - actual[-1], values[0] - values[-1])

    def test_restock_composition_and_order_sensitivity(self):
        stock = {'a': 2}
        first, second = [['a', -5], ['b', 1]], [['a', 1], ['b', -9]]
        self.assertEqual(oracle.harbor_restock_events(stock, first + second),
                         oracle.harbor_restock_events(oracle.harbor_restock_events(stock, first), second))
        self.assertNotEqual(oracle.harbor_restock_events(stock, [['a', -5], ['a', 1]]),
                            oracle.harbor_restock_events(stock, [['a', 1], ['a', -5]]))

    def test_serpentine_involution_with_original_row_widths(self):
        original = [[1, 2], [], [3], [4, 5, 6], [7, 8]]
        flattened = oracle.tapestry_serpentine(original)
        reconstructed, offset = [], 0
        for row in original:
            reconstructed.append(flattened[offset:offset + len(row)])
            offset += len(row)
        self.assertEqual(oracle.tapestry_serpentine(reconstructed), [x for row in original for x in row])

    def test_unmatched_parentheses_deletion_leaves_balanced_text(self):
        for chars in itertools.product(('(', ')', 'a'), repeat=5):
            text = ''.join(chars)
            result = oracle.signal_suffix_balance(text)
            removed = set(result['unmatched_open'] + result['unmatched_close'])
            balance = 0
            for index, char in enumerate(text):
                if index not in removed:
                    balance += (char == '(') - (char == ')')
                    self.assertGreaterEqual(balance, 0)
            self.assertEqual(balance, 0)
            self.assertEqual(result['unmatched_open'], sorted(result['unmatched_open']))
            self.assertEqual(result['unmatched_close'], sorted(result['unmatched_close']))

    def test_gap_groups_permutation_duplicates_and_coarsening(self):
        values = [-4, -1, 0, 4, 9]
        for gap in range(7):
            groups = oracle.orchard_gap_groups(values, gap)
            self.assertEqual(groups, oracle.orchard_gap_groups(values[::-1] + values, gap))
            self.assertEqual([x for group in groups for x in group], values)
            self.assertTrue(all(b - a <= gap for group in groups for a, b in zip(group, group[1:])))
            self.assertTrue(all(right[0] - left[-1] > gap for left, right in zip(groups, groups[1:])))
            self.assertGreaterEqual(len(groups), len(oracle.orchard_gap_groups(values, gap + 1)))

    def test_shared_receipts_multiplicity_and_right_permutation(self):
        left, right = ['a', 'b', 'a', 'c', 'b'], ['c', 'a', 'b', 'b', 'b']
        output = oracle.marble_shared_receipts(left, right)
        self.assertEqual(output, ['a', 'b', 'c', 'b'])
        self.assertEqual(output, oracle.marble_shared_receipts(left, right[::-1]))
        for label in set(left + right):
            self.assertEqual(output.count(label), min(left.count(label), right.count(label)))

    def test_calendar_periodicity_mass_and_successive_shifts(self):
        days = [0, 0, 1, 4, 6]
        for shift in range(-8, 9):
            actual = oracle.quartz_shift_calendar(days, shift)
            self.assertEqual(sum(actual), len(days))
            self.assertEqual(actual, oracle.quartz_shift_calendar(days, shift + 21))
            shifted_days = [day for day, count in enumerate(actual) for _ in range(count)]
            self.assertEqual(oracle.quartz_shift_calendar(shifted_days, 3),
                             oracle.quartz_shift_calendar(days, shift + 3))

    def test_priority_stability_idempotence_and_fresh_record_lists(self):
        records = [['z', 8], ['b', 7], ['a', 6], ['b', 5], ['z', 4]]
        ordered = oracle.loom_priority_merge(records, ['b', 'a'])
        self.assertEqual(ordered, oracle.loom_priority_merge(ordered, ['b', 'a']))
        for label in ('z', 'b', 'a'):
            self.assertEqual([value for tag, value in records if tag == label],
                             [value for tag, value in ordered if tag == label])
        ordered[0][1] = -100
        self.assertEqual(records[1][1], 7)

    def test_length_packing_roundtrip_and_concatenation(self):
        def unpack(packed):
            result, cursor = [], 0
            while cursor < len(packed):
                colon = packed.index(':', cursor)
                length = int(packed[cursor:colon])
                cursor = colon + 1
                result.append(packed[cursor:cursor + length])
                cursor += length
            return result
        alphabet = ['', 'x', 'a:b', '|', '  ', '0123456789']
        for fields in itertools.product(alphabet, repeat=3):
            self.assertEqual(unpack(oracle.badge_pack_fields(list(fields))), list(fields))
            self.assertEqual(oracle.badge_pack_fields(list(fields[:1])) + oracle.badge_pack_fields(list(fields[1:])),
                             oracle.badge_pack_fields(list(fields)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
