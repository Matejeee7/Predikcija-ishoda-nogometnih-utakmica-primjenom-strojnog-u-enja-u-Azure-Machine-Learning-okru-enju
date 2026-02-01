
import json
import joblib
import numpy as np
import pandas as pd

FEATURE_NAMES = ['home_player_X1', 'home_player_X2', 'home_player_X3', 'home_player_X4', 'home_player_X5', 'home_player_X6', 'home_player_X7', 'home_player_X8', 'home_player_X9', 'home_player_X10', 'home_player_X11', 'away_player_X1', 'away_player_X2', 'away_player_X3', 'away_player_X4', 'away_player_X5', 'away_player_X6', 'away_player_X7', 'away_player_X8', 'away_player_X9', 'away_player_X10', 'away_player_X11', 'home_player_Y1', 'home_player_Y2', 'home_player_Y3', 'home_player_Y4', 'home_player_Y5', 'home_player_Y6', 'home_player_Y7', 'home_player_Y8', 'home_player_Y9', 'home_player_Y10', 'home_player_Y11', 'away_player_Y1', 'away_player_Y2', 'away_player_Y3', 'away_player_Y4', 'away_player_Y5', 'away_player_Y6', 'away_player_Y7', 'away_player_Y8', 'away_player_Y9', 'away_player_Y10', 'away_player_Y11', 'home_player_1', 'home_player_2', 'home_player_3', 'home_player_4', 'home_player_5', 'home_player_6', 'home_player_7', 'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11', 'away_player_1', 'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5', 'away_player_6', 'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10', 'away_player_11', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA', 'home_buildUpPlaySpeed', 'home_buildUpPlayDribbling', 'home_buildUpPlayPassing', 'home_chanceCreationPassing', 'home_chanceCreationCrossing', 'home_chanceCreationShooting', 'home_defencePressure', 'home_defenceAggression', 'home_defenceTeamWidth', 'away_buildUpPlaySpeed', 'away_buildUpPlayDribbling', 'away_buildUpPlayPassing', 'away_chanceCreationPassing', 'away_chanceCreationCrossing', 'away_chanceCreationShooting', 'away_defencePressure', 'away_defenceAggression', 'away_defenceTeamWidth']

def init():
    global model
    model = joblib.load("rf_model.joblib")

def run(raw_data):
    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

        # Očekujemo ili:
        # 1) {"features": {col: value, ...}}
        # 2) {"inputs": [[v1, v2, ...]]}
        if isinstance(data, dict) and "features" in data:
            row = [float(data["features"].get(c, 0.0)) for c in FEATURE_NAMES]
            X = np.array([row], dtype=float)
        elif isinstance(data, dict) and "inputs" in data:
            X = np.array(data["inputs"], dtype=float)
        else:
            return {"error": "Invalid input. Use {'features': {...}} or {'inputs': [[...]]}."}

        preds = model.predict(X).tolist()

        # ako postoji predict_proba
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X).tolist()

        return {"prediction": preds, "proba": proba}
    except Exception as e:
        return {"error": str(e)}
