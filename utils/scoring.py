def calc_spot_scores(scores, spots_df):
    results = []
    for _, row in spots_df.iterrows():
        total = sum(scores[label] * row[label] for label in scores)
        results.append({"name": row["spot_name"], "score": total})
    return results
