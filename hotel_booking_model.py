import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class HotelBookingModel:
    #Initialize placeholders for the model
    def __init__(self):
        self.model = None
        self.ohe = None
        self.le = None
        self.is_trained = False

    #Fits the model with preprocessed features
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True

    #Predicting if Canceled or Not Canceled
    def predict(self, X_input):
        if not self.is_trained:
            raise Exception("Model not trained.")

        categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
        numerical_cols = [col for col in X_input.columns if col not in categorical_cols]

        for col in categorical_cols:
            X_input[col] = X_input[col].fillna("Missing").astype(str)

        X_cat = self.ohe.transform(X_input[categorical_cols])
        X_num = X_input[numerical_cols].to_numpy()
        X_processed = np.hstack((X_num, X_cat))

        y_pred_encoded = self.model.predict(X_processed)
        return self.le.inverse_transform(y_pred_encoded)

    #Shows the accuracy and performance of model(Optional)
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        print(f"Accuracy: {acc*100:.2f}%")
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", cr)
        return acc, cm, cr

    #Load PKL file saved from the python(ipynb) file
    def load_model(self, path='best_model.pkl'):
        print(f"Loading from {path}...")
        with open(path, 'rb') as file:
            self.model, self.ohe, self.le = pickle.load(file)
        self.is_trained = True
