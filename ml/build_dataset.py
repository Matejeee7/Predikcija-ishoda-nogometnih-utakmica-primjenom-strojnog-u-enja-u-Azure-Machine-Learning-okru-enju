import sqlite3
import pandas as pd
import numpy as np

DB_PATH = "database.sqlite"
OUT_CSV = "ml/matches_features.csv"

def result_class(home_goals, away_goals):
    if home_goals > away_goals:
        return 2  # HomeWin
    if home_goals == away_goals:
        return 1  # Draw
    return 0      # AwayWin

def main():
    conn = sqlite3.connect(DB_PATH)

    # Učitaj minimalno potrebne stupce
    df = pd.read_sql_query("""
        SELECT
            id, date, season,
            home_team_api_id, away_team_api_id,
            home_team_goal, away_team_goal
        FROM Match
        WHERE home_team_goal IS NOT NULL AND away_team_goal IS NOT NULL
              AND home_team_api_id IS NOT NULL AND away_team_api_id IS NOT NULL
              AND date IS NOT NULL
    """, conn)

    conn.close()

    # Parsiranje datuma i sortiranje kronološki
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Target
    df["y"] = [result_class(h, a) for h, a in zip(df["home_team_goal"], df["away_team_goal"])]

    # Pomoćne kolone: bodovi po utakmici
    df["home_points"] = np.where(df["home_team_goal"] > df["away_team_goal"], 3,
                          np.where(df["home_team_goal"] == df["away_team_goal"], 1, 0))
    df["away_points"] = np.where(df["away_team_goal"] > df["home_team_goal"], 3,
                          np.where(df["away_team_goal"] == df["home_team_goal"], 1, 0))

    # Funkcija koja za tim vraća “zadnjih N utakmica prije ovog reda”
    N = 5

    # Priprema praznih feature kolona
    feat_cols = [
        "home_goals_for_last5", "home_goals_against_last5", "home_points_last5", "home_gd_last5",
        "away_goals_for_last5", "away_goals_against_last5", "away_points_last5", "away_gd_last5",
    ]
    for c in feat_cols:
        df[c] = np.nan

    # Indeksi utakmica po timu (radi brzine)
    # Napravimo “long” formu svih nastupa timova
    home_part = df[["id","date","home_team_api_id","home_team_goal","away_team_goal","home_points"]].copy()
    home_part.columns = ["match_id","date","team_id","goals_for","goals_against","points"]
    away_part = df[["id","date","away_team_api_id","away_team_goal","home_team_goal","away_points"]].copy()
    away_part.columns = ["match_id","date","team_id","goals_for","goals_against","points"]

    long_df = pd.concat([home_part, away_part], ignore_index=True).sort_values("date")
    # Grupiraj po timu
    grouped = {tid: g.reset_index(drop=True) for tid, g in long_df.groupby("team_id")}

    # Map match_id -> pozicija u long_df po timu
    pos_map = {}
    for tid, g in grouped.items():
        for i, mid in enumerate(g["match_id"].values):
            pos_map[(tid, mid)] = i

    def lastN_stats(team_id, match_id):
        g = grouped.get(team_id)
        if g is None:
            return None
        i = pos_map.get((team_id, match_id))
        if i is None:
            return None
        start = max(0, i - N)
        hist = g.iloc[start:i]
        if len(hist) == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        gf = hist["goals_for"].mean()
        ga = hist["goals_against"].mean()
        pts = hist["points"].mean()
        gd = (hist["goals_for"] - hist["goals_against"]).mean()
        return (gf, ga, pts, gd)

    # Popuni featuree
    for idx, row in df.iterrows():
        mid = row["id"]
        ht = row["home_team_api_id"]
        at = row["away_team_api_id"]
        hs = lastN_stats(ht, mid)
        as_ = lastN_stats(at, mid)

        if hs:
            df.loc[idx, "home_goals_for_last5"] = hs[0]
            df.loc[idx, "home_goals_against_last5"] = hs[1]
            df.loc[idx, "home_points_last5"] = hs[2]
            df.loc[idx, "home_gd_last5"] = hs[3]
        if as_:
            df.loc[idx, "away_goals_for_last5"] = as_[0]
            df.loc[idx, "away_goals_against_last5"] = as_[1]
            df.loc[idx, "away_points_last5"] = as_[2]
            df.loc[idx, "away_gd_last5"] = as_[3]

    # Makni prve utakmice bez dovoljno povijesti (feature NaN)
    model_df = df.dropna(subset=feat_cols).copy()

    # Spremi
    model_df[feat_cols + ["y"]].to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV} | rows={len(model_df)}")

if __name__ == "__main__":
    main()
