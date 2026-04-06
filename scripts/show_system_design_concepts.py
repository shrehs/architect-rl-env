#!/usr/bin/env python3
from env.tasks import CONSTRAINT_CONCEPTS

print("\n" + "="*90)
print("SYSTEM DESIGN CONCEPTS COVERED")
print("="*90)

for concept_name, concept_info in sorted(CONSTRAINT_CONCEPTS.items()):
    print(f"\n{concept_name.upper():25s} -> {concept_info.get('concept', '?'):25s}")
    print(f"  {concept_info.get('description', '')}")
    if 'implications' in concept_info and concept_info['implications']:
        patterns = ", ".join(concept_info['implications'][:2])
        print(f"  Patterns: {patterns}")

print("\n" + "="*90)
print("Summary:")
print("  - 5 core constraints (must ask about)")
print("  - 7 advanced system design constraints")
print("  - Covers CAP theorem, load balancing, CDN, availability, message queues")
print("  - Evaluates constraint discovery, concept understanding, and reasoning quality")
print("="*90 + "\n")
