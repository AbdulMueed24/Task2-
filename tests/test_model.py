import joblib

def test_model_accuracy():
    model = joblib.load("src/model.pkl")
    assert hasattr(model, "score")
