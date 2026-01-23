# スコア型でホテルを評価する関数
import pandas as pd
def rank_hotels(hotels_df, user_hotel_pref):
    results = []

    for _, row in hotels_df.iterrows():
        score = 0
        for key, value in user_hotel_pref.items():
            score += row[key] * value
            score -= distance_from_spots * 0.1 # 観光地に近いホテルが有利

        results.append({
            "name": row["name"],
            "area": row["area"],
            "type": row["type"],
            "score": score
        })

    return pd.DataFrame(results).sort_values("score", ascending=False)
