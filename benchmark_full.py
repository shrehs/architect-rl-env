#!/usr/bin/env python3
"""
Comprehensive benchmark: 3 tasks × 3 modes × 2 runs = 18 episodes
"""
import subprocess
import re
import json
import os
from collections import defaultdict
import statistics

# Configuration
tasks = ['easy', 'medium', 'hard']
modes = ['clean', 'noisy', 'adversarial']
runs_per_config = 2

# Regex patterns
start_re = re.compile(r'\[START\] task=(\w+)')
end_re = re.compile(r'\[END\] success=(\w+) steps=(\d+) rewards=(.*)')

# Data structure
results = defaultdict(lambda: defaultdict(lambda: {
    'success': [],
    'steps': [],
    'avg_episode_reward': [],
    'final_step_reward': [],
    'total_episode_reward': []
}))

print('🚀 Running Full Benchmark: 3 Tasks × 3 Modes × 2 Runs = 18 Episodes')
print('=' * 80)

run_count = 0
for task in tasks:
    for mode in modes:
        for run in range(runs_per_config):
            run_count += 1
            
            # Set environment variable and run
            env = os.environ.copy()
            env['EVAL_MODE'] = mode
            
            print(f'[{run_count:2d}/18] Task={task:8s} Mode={mode:12s} Run={run+1}', end=' ... ', flush=True)
            
            try:
                result = subprocess.run(
                    ['python', 'inference.py', '--task', task, '--num-episodes', '1'],
                    cwd='e:/Meta_R1',
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                # Parse output
                task_found = None
                success = None
                steps = None
                rewards = []
                
                for line in result.stdout.splitlines():
                    m_start = start_re.search(line)
                    if m_start:
                        task_found = m_start.group(1)
                    
                    m_end = end_re.search(line)
                    if m_end:
                        success = 1 if m_end.group(1).lower() == 'true' else 0
                        steps = int(m_end.group(2))
                        rewards = [float(x) for x in m_end.group(3).split(',') if x.strip()]
                
                if success is not None:
                    avg_ep_reward = sum(rewards) / len(rewards) if rewards else 0.0
                    final_reward = rewards[-1] if rewards else 0.0
                    total_reward = sum(rewards) if rewards else 0.0
                    
                    stats = results[task][mode]
                    stats['success'].append(success)
                    stats['steps'].append(steps)
                    stats['avg_episode_reward'].append(avg_ep_reward)
                    stats['final_step_reward'].append(final_reward)
                    stats['total_episode_reward'].append(total_reward)
                    
                    print('✓')
                else:
                    print('✗ (parse error)')
                    if result.stderr:
                        print(f"  stderr: {result.stderr[:100]}")
            except subprocess.TimeoutExpired:
                print('✗ (timeout)')
            except Exception as e:
                print(f'✗ ({str(e)[:40]})')

print()
print('=' * 80)
print('ANALYTICS REPORT')
print('=' * 80)

# Summary by task
print()
print('📊 PER-TASK SUMMARY (Aggregated across all modes)')
print('-' * 80)

for task in tasks:
    all_success = []
    all_steps = []
    all_rewards = []
    
    for mode in modes:
        all_success.extend(results[task][mode]['success'])
        all_steps.extend(results[task][mode]['steps'])
        all_rewards.extend(results[task][mode]['total_episode_reward'])
    
    if all_success:
        success_pct = sum(all_success)/len(all_success)*100
        avg_steps = sum(all_steps)/len(all_steps) if all_steps else 0
        avg_reward = sum(all_rewards)/len(all_rewards) if all_rewards else 0
        print(f'Task: {task.upper():10s} | Success Rate: {success_pct:6.1f}% | Avg Steps: {avg_steps:5.2f} | Avg Total Reward: {avg_reward:7.3f}')

# Detailed results
print()
print('📊 DETAILED RESULTS BY TASK & MODE')
print('-' * 80)

for task in tasks:
    print(f'\n🔹 TASK: {task.upper()}')
    print('  ' + '-' * 76)
    
    for mode in modes:
        stats = results[task][mode]
        s = stats['success']
        st = stats['steps']
        ar = stats['avg_episode_reward']
        fr = stats['final_step_reward']
        tr = stats['total_episode_reward']
        
        if not s:
            continue
        
        success_rate = sum(s) / len(s) * 100
        avg_steps = sum(st) / len(st)
        avg_avg_reward = sum(ar) / len(ar) if ar else 0
        avg_final_reward = sum(fr) / len(fr) if fr else 0
        avg_total_reward = sum(tr) / len(tr) if tr else 0
        
        print(f'  Mode: {mode.upper():12s}')
        print(f'    Success Rate:        {success_rate:6.1f}%')
        print(f'    Steps (avg):         {avg_steps:6.2f}')
        print(f'    Avg Step Reward:     {avg_avg_reward:7.3f}')
        print(f'    Final Step Reward:   {avg_final_reward:7.3f}')
        print(f'    Total Ep Reward:     {avg_total_reward:7.3f}')

# Comparative analytics
print()
print('=' * 80)
print('📈 COMPARATIVE ANALYTICS')
print('=' * 80)

print()
print('🔸 MODE COMPARISON (Aggregated across all tasks)')
print('-' * 80)

for mode in modes:
    all_success = []
    all_steps = []
    all_rewards = []
    
    for task in tasks:
        all_success.extend(results[task][mode]['success'])
        all_steps.extend(results[task][mode]['steps'])
        all_rewards.extend(results[task][mode]['total_episode_reward'])
    
    if all_success:
        success_rate = sum(all_success) / len(all_success) * 100
        avg_steps = sum(all_steps) / len(all_steps) if all_steps else 0
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        print(f'{mode.upper():12s} | Success: {success_rate:6.1f}% | Avg Steps: {avg_steps:5.2f} | Avg Total Reward: {avg_reward:7.3f}')

print()
print('🔸 DIFFICULTY IMPACT (Clean mode: baseline for learning capacity)')
print('-' * 80)

if 'clean' in modes:
    for task in tasks:
        stats = results[task]['clean']
        s = stats['success']
        if s:
            success_rate = sum(s) / len(s) * 100
            avg_total = sum(stats['total_episode_reward']) / len(stats['total_episode_reward'])
            print(f'{task.upper():8s}: Success {success_rate:6.1f}% | Avg Total Reward: {avg_total:7.3f}')

print()
print('🔸 ROBUSTNESS UNDER NOISE (Noisy vs Clean)')
print('-' * 80)

for task in tasks:
    clean_success = results[task]['clean']['success']
    noisy_success = results[task]['noisy']['success']
    
    if clean_success and noisy_success:
        clean_rate = sum(clean_success) / len(clean_success) * 100
        noisy_rate = sum(noisy_success) / len(noisy_success) * 100
        degradation = clean_rate - noisy_rate
        
        print(f'{task.upper():8s}: Clean {clean_rate:5.1f}% → Noisy {noisy_rate:5.1f}% (Δ {degradation:+6.1f}%)')

print()
print('🔸 ADVERSARIAL RESILIENCE (Adversarial vs Clean)')
print('-' * 80)

for task in tasks:
    clean_success = results[task]['clean']['success']
    adv_success = results[task]['adversarial']['success']
    
    if clean_success and adv_success:
        clean_rate = sum(clean_success) / len(clean_success) * 100
        adv_rate = sum(adv_success) / len(adv_success) * 100
        degradation = clean_rate - adv_rate
        
        print(f'{task.upper():8s}: Clean {clean_rate:5.1f}% → Adversarial {adv_rate:5.1f}% (Δ {degradation:+6.1f}%)')

print()
print('=' * 80)
print('✅ Benchmark Complete')
print('=' * 80)
