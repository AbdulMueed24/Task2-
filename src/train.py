import joblib
from preprocess import load_data, preprocess_data
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = preprocess_data(load_data())
model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

joblib.dump(model, "src/model.pkl")
