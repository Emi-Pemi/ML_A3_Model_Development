import joblib
import numpy as np
from huggingface_hub import hf_hub_download

class HousingPredictor:
    def __init__(self):
        try:
            print("🔄 Downloading model...")
            model_path = hf_hub_download(
                repo_id="Emi-Pemi/A3_Model_Development",
                filename="model.joblib",
                cache_dir=".cache"
            )
            self.model_data = joblib.load(model_path)
            self.weights = np.array(self.model_data['model_weights'])
            self.bias = self.model_data['model_bias']
            self.scaler_mean = np.array(self.model_data['scaler_mean'])
            self.scaler_scale = np.array(self.model_data['scaler_scale'])
            self.feature_names = self.model_data['feature_names']
            self.transformation = self.model_data.get('transformation', 'log')
            self.input_min = np.array(self.model_data['input_ranges']['min'])
            self.input_max = np.array(self.model_data['input_ranges']['max'])

            print("✅ Model loaded successfully.")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def _preprocess(self, X_raw):
        """Validate, clip, and scale the input"""
        X = np.clip(np.array(X_raw).reshape(1, -1), self.input_min, self.input_max)
        return (X - self.scaler_mean) / self.scaler_scale

    def predict(self, X_raw):
        """Run inference and return price in USD"""
        try:
            X_scaled = self._preprocess(X_raw)
            raw_pred = np.dot(X_scaled, self.weights) + self.bias

            if self.transformation == 'log':
                price = np.exp(raw_pred[0])
            elif self.transformation == 'sqrt':
                price = (raw_pred[0] ** 2) * 100000
            else:
                price = raw_pred[0] * 100000

            return float(np.clip(price, 50000, 500000))
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            raise

def main():
    predictor = HousingPredictor()

    print("\n📌 Please enter values for the following features:")
    print("Features: " + ", ".join(predictor.feature_names))
    print("➡️ Enter values separated by commas (no spaces):")
    print("Example: 3.0,25.0,5.0,1.0,1000.0,2.0,34.0,-118.0\n")

    user_input = input("Your input: ")
    try:
        values = [float(val) for val in user_input.strip().split(',')]
        if len(values) != len(predictor.feature_names):
            raise ValueError(f"Expected {len(predictor.feature_names)} values, got {len(values)}.")
        price = predictor.predict(values)
        print(f"\n🏡 Predicted House Price: ${price:,.2f}")

    except Exception as e:
        print(f"❌ Invalid input: {e}")

if __name__ == "__main__":
    main()
