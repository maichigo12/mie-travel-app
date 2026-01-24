# スコア型でホテルを評価する関数
import pandas as pd
def rank_hotels(hotels_df, user_pref):
    hotels = hotels_df.copy()
    scores = []

    for _, row in hotels.iterrows():
        score = 0

        score += user_pref["family"] * row.get("family", 0)
        score += user_pref["couple"] * row.get("couple", 0)
        score += user_pref["hot_spring"] * row.get("hot_spring", 0)
        score += user_pref["scenic"] * row.get("scenic", 0)
        score += user_pref["near_station"] * row.get("near_station", 0)
        score += user_pref["shopping"] * row.get("shopping", 0)

        scores.append(score)

    hotels["score"] = scores
    return hotels.sort_values("score", ascending=False)

