# Critical Fix: Oracle Gradient Restoration

## 🚨 The Problem: Evaluation Collapse

### Initial Issue (Discovered April 4, 2026)

The oracle was **binarizing all scores to 1.0**, meaning:
- ❌ Random agents appeared as smart as heuristic agents
- ❌ No learning signal for agents to improve
- ❌ Diversity exploration was hiding evaluation bugs
- ❌ Success rates showed 100% across the board (meaningless)

```python
# BROKEN CODE:
if best_score > 0:  # Even tiny partial matches!
    corrected_score = 1.0  # ← All get 1.0!
```

### Root Causes

1. **Binarized oracle score** - Converting continuous similarity (0.0-1.0) to binary (1.0 or 0.0)
2. **Universal fallback path** - "Hybrid Cloud" was always valid, matching random outputs
3. **Lenient success threshold** - Defined as `oracle_score >= 0.6`, which was always true
4. **Similarity scoring too generous** - Partial matches got credit as full matches

---

## ✅ The Solution: Continuous Gradient + Constraint-Based Paths

### Fix 1: Restored Continuous Oracle Scoring

```python
# FIXED CODE:
# Keep the actual similarity score, don't binarize!
return float(best_score)  # Range: 0.0 - 1.0

# Score interpretation:
# 0.0-0.3: Generic/random (no real match)
# 0.3-0.6: Partial match (some components right)
# 0.6-0.8: Good match (mostly constraint-aware)
# 0.8-1.0: Excellent (precise + reasoning-backed)
```

**Impact:** Agents now get proportional feedback for their quality.

### Fix 2: Tightened Similarity Computation

```python
# OLD (Lenient):
score = 0.0
if agent_model == oracle_model:
    score += 0.3
else:
    score += 0.1  # ← Still got partial credit for being wrong!

# NEW (Discriminative):
if agent_model == oracle_model:
    score += 0.33
else:
    score -= 0.1  # ← Penalize mismatches
```

**Impact:** Exact matches required, partial matches penalized.

### Fix 3: Added Generic Architecture Penalty

```python
# Penalize generic/vague architectures
arch = agent_structured.get("architecture", "").lower()
generic_terms = ["microservice", "api", "database", "standard", "modular"]
if any(term in arch for term in generic_terms):
    score -= 0.3  # ← Strong penalty for vagueness
```

**Impact:** Agents must propose specific architectural reasoning.

### Fix 4: Conditional Path Validity

```python
# OLD: Always include "Hybrid Cloud" as fallback
alternatives.append({
    "model": "hybrid",
    "deployment": "standard_cloud",
    "architecture": "service_oriented",
})  # ← Random agents could stumble into this!

# NEW: Only include if constraints actually suggest balance
if _has_any(data_size, ["small", "medium"]) or \
   (not urgent_latency and not budget_constrained and not huge_data):
    alternatives.append({...})
```

**Impact:** Only genuinely valid paths are available.

### Fix 5: Raised Success Threshold

```python
# OLD: success = oracle_score >= 0.6
# With binarized scoring, this was always true!

# NEW: success = oracle_score >= 0.8
# Now requires strong alignment
```

**Impact:** Only high-quality solutions count as successes.

---

## 📊 Results: Before vs After

### Evaluation Results (100 episodes, hard task)

| Agent | Mode | BEFORE | AFTER | Change |
|-------|------|--------|-------|--------|
| **random** | clean | oracle=1.0, success=100% | oracle=0.0, success=0% | ✅ Now correctly fails |
| **random** | noisy | oracle=1.0, success=100% | oracle=0.0, success=0% | ✅ Not lucky |
| **random** | adversarial | oracle=1.0, success=100% | oracle=0.0, success=0% | ✅ Properly penalized |
| **heuristic** | clean | oracle=1.0, success=100% | oracle=1.0, success=100% | ✅ Still excellent |
| **heuristic** | noisy | oracle=1.0, success=100% | oracle=0.69, success=58% | ✅ Shows degradation |
| **heuristic** | adversarial | oracle=1.0, success=100% | oracle=0.0, success=0% | ✅ Properly fails |
| **improved** | clean | oracle=1.0, success=100% | oracle=1.0, success=100% | ✅ Maintains quality |
| **improved** | noisy | oracle=1.0, success=100% | oracle=0.71, success=62% | ✅ Better resilience |
| **improved** | adversarial | oracle=1.0, success=100% | oracle=0.46, success=46% | ✅ Shows robustness |

### Key Insights

✅ **Learning signal restored** - Agents receive true feedback proportional to quality  
✅ **Random agents properly penalized** - No accidental successes  
✅ **Heuristic unrobust to adversarial** - Missing constraint awareness  
✅ **Improved shows better adversarial resilience** - 46% vs 0% for heuristic  
✅ **Success metric meaningful again** - 0% for random, 58-100% for real agents  

---

## 🎯 How It Works Together

### The Complete Flow

1. **Agent generates recommendation** (model, deployment, architecture)
   
2. **Similarity computed for each valid path:**
   ```
   - Model match: ±0.33
   - Deployment match: ±0.33  
   - Architecture match: ±0.34
   - Generic penalty: -0.3
   Range: [0.0, 1.0]
   ```

3. **Best matching path selected:**
   ```
   best_score = max(similarities)
   matched_path_idx = argmax(similarities)
   ```

4. **Oracle score = continuous similarity:**
   ```
   oracle_score = best_score  # NOT binarized!
   ```

5. **Success defined at high threshold:**
   ```
   success = oracle_score >= 0.8
   ```

6. **Diversity bonus applied orthogonally:**
   ```
   # Only for matched paths (matched_path_idx >= 0)
   freq_penalty = (1 - path_frequency) ** exploration_alpha
   contextual_bonus = 0.05 * freq_penalty
   # Completely separate from oracle_score
   ```

### Why This Matters

- **Correctness:** Agents must match actual valid architectures
- **Quality:** Oracle score reflects actual alignment quality  
- **Robustness:** Noise and adversarial inputs properly penalize weak agents
- **Learning:** Agents get continuous gradient to improve toward better solutions
- **Exploration:** Diversity bonuses reward novel solutions WITHOUT sacrificing correctness

---

## 🧪 Verification

### Check Continuous Scoring

```python
import pandas as pd

df = pd.read_csv('artifacts/evaluation_fixed200/episode_metrics.csv')

# Verify continuous range
assert (df['oracle_score'] >= 0.0).all() and (df['oracle_score'] <= 1.0).all()
print(f"Oracle score range: {df['oracle_score'].min():.3f} to {df['oracle_score'].max():.3f}")

# Should NOT be all 1.0!
assert df['oracle_score'].max() > 0.99
assert df['oracle_score'].min() < 0.1
print("✅ Continuous scoring restored!")

# Verify discrimination
random_avg = df[df['agent'] == 'random']['oracle_score'].mean()
heuristic_avg = df[df['agent'] == 'heuristic']['oracle_score'].mean()
improved_avg = df[df['agent'] == 'improved']['oracle_score'].mean()

print(f"\nAgent quality (oracle_score):")
print(f"  Random:    {random_avg:.3f}")
print(f"  Heuristic: {heuristic_avg:.3f}")
print(f"  Improved:  {improved_avg:.3f}")

# Should have clear separation
assert random_avg < 0.2
assert heuristic_avg > 0.5
assert improved_avg > 0.6
print("✅ Clear agent discrimination!")

# Verify success is hard to achieve
success_rate = (df['success'] == 1).sum() / len(df)
print(f"\nOverall success rate: {success_rate*100:.1f}%")
assert success_rate < 0.7  # Much harder to succeed now
print("✅ Success threshold is now meaningful!")
```

---

## 🔄 Impact on Exploration

The continuous oracle scoring + effective path validity means:

✅ **Diversity is now measured correctly** - Agents explore to improve, not just to explore  
✅ **Exploration bonuses are secondary** - Applied TO quality solutions, not INSTEAD of quality  
✅ **Learning is measurable** - Can see agents improving toward higher oracle scores  
✅ **Noise testing is meaningful** - Can distinguish agents by robustness  

---

## 📝 Summary

| Aspect | Before | After |
|--------|--------|-------|
| Oracle binarization | 1.0 allthe time | Continuous 0.0-1.0 |
| Random agent success | 100% | 0% |
| Heuristic robustness | Appears perfect | Visible degradation with noise |
| Learning signal | Blocked | Restored |
| Path validity | Universal fallback | Constraint-dependent |
| Success threshold | Trivial (≥0.6) | Meaningful (≥0.8) |
| Diversity bonus | Primary signal | Secondary signal |

The oracle gradient restoration ensures ArchitectEnv properly **measures both solution quality AND exploration diversity** without conflating them.
