import pandas as pd

def calc_spot_scores(label_scores, spots_df):
    df = spots_df.copy()

    # ラベル名とCSV列名が一致しているものだけ使う
    valid_labels = [label for label in label_scores.keys() if label in df.columns]

    if not valid_labels:
        return pd.DataFrame()  # 空ならエラー回避

    # スコア計算（モデルの確率 × 観光地のタグ）
    df["score"] = 0
    for label in valid_labels:
        df["score"] += df[label] * label_scores[label]

    # スコア順に並び替え
    df = df.sort_values("score", ascending=False)

    return df
