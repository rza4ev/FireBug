    # predict_service.py

    import joblib
    import json
    import pandas as pd

    # 1. Modeli yüklə
    model = joblib.load("fire_ignition_xgb_model.joblib")

    # 2. Config-i yüklə
    with open("fire_model_config.json", "r") as f:
        config = json.load(f)

    FEATURE_COLS = config["feature_cols"]
    BEST_THRESHOLD = config["best_threshold"]

    def predict_risk(sensor_reading: dict):
        """
        sensor_reading: sensor JSON (dict formatında)
        return:
            prob_fire: yanğın ehtimalı (0–1)
            alert: threshold-a görə boolean (True → agentic AI işə düşməlidir)
        """
        df = pd.DataFrame([sensor_reading])[FEATURE_COLS]
        prob_fire = float(model.predict_proba(df)[0, 1])
        alert = prob_fire >= BEST_THRESHOLD
        return prob_fire, alert

    if __name__ == "__main__":
        # Test üçün bir nümunə
        sample = {
            "temp_c": 38,
            "humidity_pct": 18,
            "wind_speed_ms": 7.5,
            "wind_dir_deg": 110,
            "solar_rad_wm2": 800,
            "rain_last_24h_mm": 0.0,
            "vpd_kpa": 3.2,
            "co_ppm": 0.9,
            "co2_ppm": 650,
            "tvoc_ppb": 400,
            "h2_idx": 25,
            "ethanol_idx": 40,
            "hydrocarbon_idx": 35,
            "lat": 40.8,
            "lon": 47.6,
        }

        prob, alert = predict_risk(sample)
        print(f"P(ignite in next 5 min) = {prob*100:.4f}%")
        print("ALERT:", alert)
