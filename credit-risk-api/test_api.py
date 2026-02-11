"""
Quick API test script.
Tests the prediction endpoint with sample data.
"""
import requests
import json
import time


def test_health():
    """Test health endpoint."""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_prediction():
    """Test prediction endpoint."""
    print("\n🎯 Testing prediction endpoint...")

    # Sample request (Good customer - low risk)
    good_customer = {
        "loanamount": 100000,
        "totaldue": 120000,
        "termdays": 90,
        "latitude_gps": 6.5244,
        "longitude_gps": 3.3792,
        "bank_name_clients": "GTBank",
        "bank_branch_clients": "Lagos",
        "employment_status_clients": "Permanent",
        "level_of_education_clients": "HND/BSc",
        "hist_num_loans": 3,
        "hist_num_closed": 3,
        "hist_num_open": 0,
        "hist_avg_loan_amount": 80000,
        "hist_max_loan_amount": 100000,
        "hist_min_loan_amount": 50000,
        "hist_total_borrowed": 240000,
        "hist_avg_total_due": 288000,
        "hist_ontime_rate": 0.95,
        "hist_late_rate": 0.05,
        "hist_never_paid_rate": 0.0,
        "hist_avg_days_late": 1.5,
        "hist_closure_rate": 1.0,
        "days_since_last_loan": 120,
        "hist_avg_term_days": 90,
        "hist_max_term_days": 120,
        "has_open_loans": 0,
        "hist_interest_burden": 0.2
    }

    try:
        response = requests.post(
            "http://localhost:8000/api/v1/predict",
            json=good_customer,
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"\n   ✅ PREDICTION RESULT:")
            print(f"      Prediction: {result['prediction']}")
            print(f"      Default Probability: {result['default_probability']:.2%}")
            print(f"      Risk Category: {result['risk_category']}")
            print(f"      Confidence: {result['confidence']:.2%}")
            print(f"      Recommendation: {result['recommended_action']}")
            return True
        else:
            print(f"   ❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_bad_customer():
    """Test with bad customer (high risk)."""
    print("\n⚠️  Testing bad customer prediction...")

    # Sample request (Bad customer - high risk)
    bad_customer = {
        "loanamount": 500000,
        "totaldue": 700000,
        "termdays": 365,
        "latitude_gps": 6.5244,
        "longitude_gps": 3.3792,
        "bank_name_clients": "GTBank",
        "bank_branch_clients": "Lagos",
        "employment_status_clients": "Temporary",
        "level_of_education_clients": "Secondary",
        "hist_num_loans": 5,
        "hist_num_closed": 2,
        "hist_num_open": 3,
        "hist_avg_loan_amount": 300000,
        "hist_max_loan_amount": 500000,
        "hist_min_loan_amount": 100000,
        "hist_total_borrowed": 1500000,
        "hist_avg_total_due": 2100000,
        "hist_ontime_rate": 0.20,
        "hist_late_rate": 0.50,
        "hist_never_paid_rate": 0.30,
        "hist_avg_days_late": 45.0,
        "hist_closure_rate": 0.40,
        "days_since_last_loan": 30,
        "hist_avg_term_days": 180,
        "hist_max_term_days": 365,
        "has_open_loans": 1,
        "hist_interest_burden": 0.4
    }

    try:
        response = requests.post(
            "http://localhost:8000/api/v1/predict",
            json=bad_customer,
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"\n   ✅ PREDICTION RESULT:")
            print(f"      Prediction: {result['prediction']}")
            print(f"      Default Probability: {result['default_probability']:.2%}")
            print(f"      Risk Category: {result['risk_category']}")
            print(f"      Confidence: {result['confidence']:.2%}")
            print(f"      Recommendation: {result['recommended_action']}")
            return True
        else:
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_model_info():
    """Test model info endpoint."""
    print("\n📊 Testing model info endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/v1/model/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"   ✅ Model Information:")
            print(f"      Name: {info['model_name']}")
            print(f"      Version: {info['model_version']}")
            print(f"      Type: {info['model_type']}")
            print(f"      Features: {info['n_features']}")
            print(f"\n   Top 5 Features:")
            for i, feat in enumerate(info['top_features'][:5], 1):
                print(f"      {i}. {feat['feature']}: {feat['importance']:.4f}")
            return True
        else:
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("🧪 CREDIT RISK API TESTS")
    print("="*60)

    # Wait for API to start
    print("\n⏳ Waiting for API to start...")
    time.sleep(3)

    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Good Customer Prediction", test_prediction),
        ("Bad Customer Prediction", test_bad_customer),
        ("Model Info", test_model_info),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {name}")

    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\n   Total: {passed}/{total} tests passed")
    print("="*60)
