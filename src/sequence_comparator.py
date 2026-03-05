import os
import json
import numpy as np
from difflib import SequenceMatcher


class SequenceComparator:
    def __init__(self):
        self.results = []

    def compare(self, original_seq, generated_seq, edge_id=None, verbose=False):
        similarity = SequenceMatcher(None, original_seq, generated_seq).ratio()
        edit_distance = self._levenshtein_distance(original_seq, generated_seq)

        matches = 0
        mismatches = 0
        min_len = min(len(original_seq), len(generated_seq))

        for i in range(min_len):
            if original_seq[i] == generated_seq[i]:
                matches += 1
            else:
                mismatches += 1

        result = {
            'edge_id': edge_id,
            'original_length': len(original_seq),
            'generated_length': len(generated_seq),
            'similarity': similarity,
            'edit_distance': edit_distance,
            'position_stats': {
                'matches': matches,
                'mismatches': mismatches,
                'match_rate': matches / min_len if min_len > 0 else 0,
                'length_diff': abs(len(original_seq) - len(generated_seq))
            }
        }

        self.results.append(result)

        if verbose:
            print(f"\n{'='*80}")
            print(f"Edge ID: {edge_id}")
            print(f"Original  ({len(original_seq):4d}): {original_seq[:100]}{'...' if len(original_seq) > 100 else ''}")
            print(f"Generated ({len(generated_seq):4d}): {generated_seq[:100]}{'...' if len(generated_seq) > 100 else ''}")
            print(f"Similarity: {similarity:.4f}")
            print(f"Edit Distance: {edit_distance}")
            print(f"Match Rate: {result['position_stats']['match_rate']:.4f} ({matches}/{min_len})")
            print(f"Length Diff: {result['position_stats']['length_diff']}")

        return result

    def _levenshtein_distance(self, s1, s2):
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def print_summary(self, comparison_results=None):
        if comparison_results is None:
            comparison_results = self.results

        if not comparison_results:
            print("No comparison results to display")
            return

        similarities = [r['similarity'] for r in comparison_results]
        edit_distances = [r['edit_distance'] for r in comparison_results]
        match_rates = [r['position_stats']['match_rate'] for r in comparison_results]
        length_diffs = [r['position_stats']['length_diff'] for r in comparison_results]

        print(f"\n{'='*80}")
        print(f"Sequence Comparison Summary (total {len(comparison_results)} pairs)")
        print(f"{'='*80}")
        print(f"Similarity    - mean: {np.mean(similarities):.4f}, median: {np.median(similarities):.4f}, range: [{np.min(similarities):.4f}, {np.max(similarities):.4f}]")
        print(f"Edit Distance - mean: {np.mean(edit_distances):.2f}, median: {np.median(edit_distances):.2f}, range: [{np.min(edit_distances):.0f}, {np.max(edit_distances):.0f}]")
        print(f"Match Rate    - mean: {np.mean(match_rates):.4f}, median: {np.median(match_rates):.4f}, range: [{np.min(match_rates):.4f}, {np.max(match_rates):.4f}]")
        print(f"Length Diff   - mean: {np.mean(length_diffs):.2f}, median: {np.median(length_diffs):.2f}, range: [{np.min(length_diffs):.0f}, {np.max(length_diffs):.0f}]")
        print(f"{'='*80}\n")

    def save_results(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if self.results:
            summary = {
                'total_comparisons': len(self.results),
                'avg_similarity': float(np.mean([r['similarity'] for r in self.results])),
                'avg_edit_distance': float(np.mean([r['edit_distance'] for r in self.results])),
                'avg_match_rate': float(np.mean([r['position_stats']['match_rate'] for r in self.results])),
            }
        else:
            summary = {}

        output = {
            'summary': summary,
            'details': self.results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Comparison results saved to: {filepath}")

    def clear_results(self):
        self.results = []
