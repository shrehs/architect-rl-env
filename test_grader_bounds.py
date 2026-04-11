#!/usr/bin/env python3
"""Test that grader returns scores strictly in (0, 1)"""
from env.tasks import default_task_grader

# Test 1: No trajectory
result1 = default_task_grader([], {'task_id': 'easy'})
print(f'Test 1 (empty trajectory): score={result1["score"]}, strictly in (0,1)? {0 < result1["score"] < 1}')

# Test 2: Oracle score of 0.0
result2 = default_task_grader([{'info': {'oracle_score': 0.0}}], {'task_id': 'medium'})
print(f'Test 2 (oracle=0.0): score={result2["score"]}, strictly in (0,1)? {0 < result2["score"] < 1}')

# Test 3: Oracle score of 1.0
result3 = default_task_grader([{'info': {'oracle_score': 1.0}}], {'task_id': 'hard'})
print(f'Test 3 (oracle=1.0): score={result3["score"]}, strictly in (0,1)? {0 < result3["score"] < 1}')

# Test 4: Oracle score of 0.75
result4 = default_task_grader([{'info': {'oracle_score': 0.75}}], {'task_id': 'easy'})
print(f'Test 4 (oracle=0.75): score={result4["score"]}, strictly in (0,1)? {0 < result4["score"] < 1}')

# Test 5: Exception handling
result5 = default_task_grader([{'info': None}], None)
print(f'Test 5 (exception): score={result5["score"]}, strictly in (0,1)? {0 < result5["score"] < 1}')

# Test 6: High score (should clamp to 0.99)
result6 = default_task_grader([{'info': {'oracle_score': 1.5}}], {'task_id': 'easy'})
print(f'Test 6 (oracle=1.5, overflow): score={result6["score"]}, clamped to 0.99? {result6["score"] == 0.99}')

print()
all_valid = all([
    0 < result1["score"] < 1,
    0 < result2["score"] < 1,
    0 < result3["score"] < 1,
    0 < result4["score"] < 1,
    0 < result5["score"] < 1,
    0 < result6["score"] < 1,
])

if all_valid:
    print('✅ SUCCESS: All grader scores are strictly within (0, 1)')
else:
    print('❌ FAILED: Some scores are out of range')
    exit(1)
