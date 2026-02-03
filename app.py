import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_bundle():
    return joblib.load("ml/best_model.joblib")

bundle = load_bundle()

if isinstance(bundle, dict):
    model = bundle["model"]
    FEATURES = bundle["features"]
else:
    model = bundle
    FEATURES = model.feature_names_in_

st.title("Predikcija ishoda nogometne utakmice")

st.write("Unesite osnovne jačine ekipa (0–100):")

st.info(
    "Napomena: Model je treniran na značajkama izvedenima iz statistike posljednjih 5 utakmica (last5). "
    "Kako bi aplikacija bila jednostavna za korištenje, unos (0–100) se mapira u te značajke kao aproksimacija."
)

home_attack = st.slider("Napad domaće ekipe", 0, 100, 60)
away_attack = st.slider("Napad gostujuće ekipe", 0, 100, 60)
home_defense = st.slider("Obrana domaće ekipe", 0, 100, 60)
away_defense = st.slider("Obrana gostujuće ekipe", 0, 100, 60)

def attack_to_goals(x):
    return x / 20.0         

def defense_to_against(x):
    return (100 - x) / 25.0 

def points_from_attack(x):
    return x / 10.0        

if st.button("Predvidi ishod"):

    home_gf = attack_to_goals(home_attack)
    away_gf = attack_to_goals(away_attack)

    home_ga = defense_to_against(home_defense)
    away_ga = defense_to_against(away_defense)

    home_points = points_from_attack(home_attack)
    away_points = points_from_attack(away_attack)

    home_gd = home_gf - home_ga
    away_gd = away_gf - away_ga

    row = {
        "home_goals_for_last5": home_gf,
        "home_goals_against_last5": home_ga,
        "home_points_last5": home_points,
        "home_gd_last5": home_gd,
        "away_goals_for_last5": away_gf,
        "away_goals_against_last5": away_ga,
        "away_points_last5": away_points,
        "away_gd_last5": away_gd,
    }

    X = pd.DataFrame([row])

    X = X.reindex(columns=FEATURES)

    if X.isna().any().any():
        st.error("Nedostaju featurei za model!")
        st.write(X)
        st.stop()

    pred = model.predict(X)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]


    mapping = {
        0: "Pobjeda gostiju (A)",
        1: "Neriješeno (D)",
        2: "Pobjeda domaćih (H)"
    }

    st.success(f"Predikcija: {mapping[pred]}")

    if proba is not None:
        st.write("Vjerojatnosti po klasama:")
        st.write({
        "A (gosti)": float(proba[0]),
        "D (neriješeno)": float(proba[1]),
        "H (domaći)": float(proba[2]),
    })

    with st.expander("Debug (featurei poslani modelu)"):
        st.dataframe(X)
