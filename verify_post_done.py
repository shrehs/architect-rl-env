import requests

base = "http://localhost:7860"

# reset
requests.post(f"{base}/reset")
print("✓ Reset called")

# run until done
for i in range(10):
    r = requests.post(f"{base}/step", json={"user_reply": "test"})
    data = r.json()
    print(f"  Step {i+1}: done={data.get('done', False)}")
    if data.get("done"):
        print(f"✓ Episode finished at step {i+1}")
        break

# one more step (should fail with 409)
print("\nAttempting step after done...")
r = requests.post(f"{base}/step", json={"user_reply": "extra"})
print(f"Status Code: {r.status_code}")
print(f"Response: {r.text}")

if r.status_code == 409:
    print("\n✅ VERIFIED: Post-done step returns 409 Conflict")
else:
    print(f"\n❌ FAILED: Expected 409, got {r.status_code}")
