import streamlit as st
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.scoring import calc_spot_scores
from utils.hotel import rank_hotels
from utils.route import solve_tsp, make_google_map_url


# =====================
# 初期設定
# =====================
st.set_page_config(page_title="三重県1泊2日旅行プラン", layout="wide")

st.title("🧳 三重県 1泊2日 観光プラン提案アプリ")


# =====================
# モデル読み込み
# =====================
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "maichigo/mie-bert-travel"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME) 
    model.eval()  # 推論モード
    return tokenizer, model

tokenizer, model = load_model()

label_names = ["sea","mountain","nature","history",
               "play","shopping","food","family","rain"]


def predict_labels(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    scores = {label_names[i]: float(probs[i]) for i in range(len(label_names))}
    active = [k for k, v in scores.items() if v >= threshold]

    return scores, active



# =====================
# CSV 読み込み
# =====================
spots_df = pd.read_csv("data/mie_spot.csv")
hotels_df = pd.read_csv("data/mie_hotel.csv")

NAGOYA = {"name": "名古屋駅", "lat": 35.1709, "lon": 136.8815}


# =====================
# UI：文章入力
# =====================
st.header("① 行きたい旅行のイメージを入力")

text = st.text_area(
    "例：雨の日でも家族で楽しめる場所に行きたい",
    height=80
)

if not text:
    st.stop()


# =====================
# 観光地スコア計算
# =====================
scores, active_labels = predict_labels(text)

st.subheader("🔍 推定された旅行タイプ")
st.write(active_labels)

spot_ranking = calc_spot_scores(scores, spots_df)
if spot_ranking.empty:
    st.error("条件に合う観光地が見つかりませんでした")
    st.stop()

top_spots = spot_ranking.head(4)


# =====================
# 観光地表示
# =====================
st.header("② おすすめ観光地")

cols = st.columns(2)
for i, (_, row) in enumerate(top_spots.iterrows()):
    with cols[i % 2]:
        st.subheader(row["spot_name"])
        if "img_url" in row:
            st.image(row["img_url"], use_column_width=True)
        st.write(f"スコア：{row['score']:.2f}")


# =====================
# ホテル条件
# =====================
st.header("③ 宿泊条件")

col1, col2, col3 = st.columns(3)

with col1:
    family = st.checkbox("家族向け")
    hot_spring = st.checkbox("温泉")

with col2:
    couple = st.checkbox("カップル")
    scenic = st.checkbox("景色が良い")

with col3:
    near_station = st.checkbox("駅近")
    shopping = st.checkbox("買い物便利")

user_hotel_pref = {
    "family": int(family),
    "couple": int(couple),
    "hot_spring": int(hot_spring),
    "scenic": int(scenic),
    "near_station": int(near_station),
    "shopping": int(shopping)
}

ranked_hotels = rank_hotels(hotels_df, user_hotel_pref)
hotel = ranked_hotels.iloc[0]

st.success(f"🏨 おすすめ宿泊施設：{hotel['name']}（{hotel['area']}）")


# =====================
# Day1 / Day2 分割
# =====================
day1_df = top_spots.iloc[:2]
day2_df = top_spots.iloc[2:]

hotel_location = {
    "name": hotel["name"],
    "lat": hotel["lat"],
    "lon": hotel["lon"]
}

day1_locations = (
    [NAGOYA] +
    day1_df[["name","lat","lon"]].to_dict("records") +
    [hotel_location]
)

day2_locations = (
    [hotel_location] +
    day2_df[["name","lat","lon"]].to_dict("records") +
    [NAGOYA]
)

day1_route = solve_tsp(day1_locations)
day2_route = solve_tsp(day2_locations)


# =====================
# ルート表示
# =====================
st.header("④ 1泊2日モデルルート")

st.subheader("🗓 Day1")
st.write(" → ".join(day1_route))
st.markdown(f"[Googleマップで開く]({make_google_map_url(day1_route)})")

st.subheader("🗓 Day2")
st.write(" → ".join(day2_route))
st.markdown(f"[Googleマップで開く]({make_google_map_url(day2_route)})")
