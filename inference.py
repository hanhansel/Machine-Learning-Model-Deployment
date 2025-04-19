import pandas as pd
from hotel_booking_model import HotelBookingModel
#Optional, since we already have the hotel_booking_model.py but this code is used for later batch running predictions from new data files.
def load_input_data(csv_path):
    print(f"Loading from {csv_path}...")
    data = pd.read_csv(csv_path)
    return data

def run_inference(model_path, input_csv):
    model = HotelBookingModel()
    model.load_model(model_path)

    input_data = load_input_data(input_csv)
    predictions = model.predict(input_data)
    print(predictions)
    
    pd.DataFrame(predictions, columns=["prediction"]).to_csv("predictions.csv", index=False)
    return predictions

if __name__ == "__main__":
    run_inference("best_model.pkl", "new_data.csv")
