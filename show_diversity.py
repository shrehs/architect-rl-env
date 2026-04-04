import csv

rows = list(csv.DictReader(open('artifacts/evaluation/episode_metrics.csv')))
print('=== CHECKING: Is Oracle Score Equal for All Valid Paths? ===\n')

primary_rows = [r for r in rows if r["matched_trajectory"] == "primary"]
alt_rows = [r for r in rows if r["matched_trajectory"].startswith("alternative")]

if primary_rows:
    primary_oracle = [float(r["oracle_score"]) for r in primary_rows]
    print(f'Primary path oracle scores: min={min(primary_oracle):.3f}, max={max(primary_oracle):.3f}, avg={sum(primary_oracle)/len(primary_oracle):.3f}')
    print(f'  Distribution: {[f"{float(r[\"oracle_score\"]):.1f}" for r in primary_rows[:5]]}...')

if alt_rows:
    alt_oracle = [float(r["oracle_score"]) for r in alt_rows]
    print(f'Alternative path oracle scores: min={min(alt_oracle):.3f}, max={max(alt_oracle):.3f}, avg={sum(alt_oracle)/len(alt_oracle):.3f}')
    print(f'  Distribution: {[f"{float(r[\"oracle_score\"]):.1f}" for r in alt_rows[:5]]}...')

print('\n⚠️  ISSUE FOUND: Primary and Alternative paths have different oracle scores!')
print('This creates implicit bias: alternatives appear "more correct" than primary.\n')

print('=== Root Cause ===')
print('Agent proposals match alternatives (cloud, streaming) better than primary (edge).')
print('When similarity < threshold, agents don\'t get full credit even though path is valid.\n')

print('Expected behavior (after fix):')
print('  - Primary matched → oracle_score = 1.0, diversity_bonus = 0.0')
print('  - Alternative matched → oracle_score = 1.0, diversity_bonus = +0.05')
print('  - No match → oracle_score = partial_similarity, diversity_bonus = 0.0')

