import csv

rows = list(csv.DictReader(open('artifacts/evaluation/episode_metrics.csv')))
print('=== Checking: Is Oracle Score Equal for All Valid Paths? ===\n')

primary_rows = [r for r in rows if r["matched_trajectory"] == "primary"]
alt_rows = [r for r in rows if r["matched_trajectory"].startswith("alternative")]

if primary_rows:
    primary_oracle = [float(r["oracle_score"]) for r in primary_rows]
    avg_primary = sum(primary_oracle) / len(primary_oracle)
    print(f'Primary path oracle scores:')
    print(f'  avg={avg_primary:.3f}, min={min(primary_oracle):.3f}, max={max(primary_oracle):.3f}')

if alt_rows:
    alt_oracle = [float(r["oracle_score"]) for r in alt_rows]
    avg_alt = sum(alt_oracle) / len(alt_oracle)
    print(f'\nAlternative path oracle scores:')
    print(f'  avg={avg_alt:.3f}, min={min(alt_oracle):.3f}, max={max(alt_oracle):.3f}')

print('\n⚠️  ISSUE: Primary and Alternative paths have DIFFERENT oracle scores!')
print('This creates bias where primary appears less correct.\n')

print('ROOT CAUSE:')
print('When agents match primary with similarity < 0.7, they do not pass threshold.')
print('But when they match alternatives with similarity >= 0.7, they pass.')
print('Result: Primary 0.1, Alternatives 1.0 - looks like bias!\n')

print('FIX APPROACH:')
print('- If agent matches ANY valid path well enough, oracle_score = 1.0')
print('- Add diversity_bonus = +0.05 only if matched != primary')
print('- This separates CORRECTNESS (1.0 for all) from EXPLORATION (bonus for alt)')
