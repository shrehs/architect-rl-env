#!/usr/bin/env python3
import urllib.request
import json

BASE_URL = "https://alive1111-architect-rl-env.hf.space"

def test_endpoint(name, endpoint, method="GET", body=None):
    """Test an endpoint and print results."""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print(f"Method: {method}")
    
    try:
        if method == "GET":
            r = urllib.request.urlopen(url, timeout=10)
        else:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), method=method)
            req.add_header("Content-Type", "application/json")
            r = urllib.request.urlopen(req, timeout=10)
        
        status = r.status
        response = json.loads(r.read())
        
        print(f"✅ Status: {status}")
        print(f"Response keys: {list(response.keys())}")
        if "mode" in response:
            print(f"Mode: {response['mode']}")
        if "step_count" in response:
            print(f"Step count: {response['step_count']}")
        if "reward" in response:
            print(f"Reward: {response['reward']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Run tests
print("Testing HF Space Deployment")
print("="*60)

test_endpoint("Root", "/")
test_endpoint("Health", "/health")
test_endpoint("Reset (easy)", "/reset?task_id=easy")
test_endpoint("Step", "/step", method="POST", body={"action_type": "ASK_LATENCY"})

print("\n" + "="*60)
print("HF Space tests complete!")
