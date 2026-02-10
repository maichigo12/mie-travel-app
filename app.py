import streamlit as st
import pandas as pd
import torch
import streamlit.components.v1 as components

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.scoring import calc_spot_scores
from utils.hotel import rank_hotels
from utils.route import solve_tsp, make_google_map_url
# from utils.route import solve_tsp


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

# =====================
# キーワードブースト関数（★ここに追加★）
# =====================
def adjust_scores_by_keywords(text, scores, label_names):
    """
    キーワードに基づいてスコアを調整する関数
    """
    keyword_boost = {
        'sea': ['海', 'ビーチ', '海岸', '波', 'マリン', '砂浜', '水族館', '海水浴', '真珠'],
        'mountain': ['山', '登山', 'ハイキング', '峠', '高原', '山頂', '渓谷', '山登り'],
        'nature': ['自然', '景色', '絶景', '風景', 'エコ', '森', '公園', '花', '紅葉', '星空', '川'],
        'history': ['歴史', '文化', '伝統', '寺', '神社', '城', '古い', '遺跡', '文化財', '伊勢', '武将', '忍者'],
        'play': ['遊ぶ', '体験', 'アクティビティ', 'レジャー', '楽しむ', 'テーマパーク', '動物園', '遊園地', '水族館'],
        'shopping': ['買い物', 'ショッピング', 'お土産', '店', 'モール', '商店街', '市場', 'アウトレット', '特産品'],
        'food': ['食べ', 'グルメ', '料理', 'レストラン', '美味', 'おいしい', 'カフェ', '食事', 'ランチ', '名物', '食べ歩き', '松阪牛', '伊勢海老'],
        'family': ['家族', '子供', 'ファミリー', '親子', '子ども', 'キッズ', '赤ちゃん', '3世代'],
        'rain': ['雨', '屋内', 'インドア', '雨天', '室内', '天候', '濡れない', '雨の日', '博物館', '美術館']
    }
    
    adjusted_scores = scores.copy()
    
    for label in label_names:
        if label in keyword_boost:
            for keyword in keyword_boost[label]:
                if keyword in text:
                    adjusted_scores[label] *= 1.8  # ブースト倍率
                    break
    
    return adjusted_scores



def predict_labels(text, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    scores = {label_names[i]: float(probs[i]) for i in range(len(label_names))}
    
    # ★ここでスコア調整を適用★
    scores = adjust_scores_by_keywords(text, scores, label_names)    
    
    active = [k for k, v in scores.items() if v >= threshold]

    return scores, active



# =====================
# CSV 読み込み
# =====================
spots_df = pd.read_csv("data/mie_spot.csv")
# ↓ ここを追加：カラム名の前後の空白を削除し、一律でクリーンにする
spots_df.columns = spots_df.columns.str.strip()

hotels_df = pd.read_csv("data/mie_hotel.csv")
# ↓ ついでにホテル側もやっておくと安全です
hotels_df.columns = hotels_df.columns.str.strip()

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
        # if "img_url" in row:
            # st.image(row["img_url"], use_column_width=True)
        st.write(row["description"])     
        st.write(f"スコア：{row['score']:.2f}")


# =====================
# ホテル条件
# =====================
st.header("③ 宿泊条件")

col1, col2, col3 = st.columns(3)

with col1:
    family = st.checkbox("家族向け")
    couple = st.checkbox("カップル")
    kids_room = st.checkbox("キッズルームあり")
    quiet = st.checkbox("静かな場所")


with col2:
    scenic = st.checkbox("景色が良い")
    beach_front = st.checkbox("海が目の前")
    hot_spring = st.checkbox("温泉")
    ocean_view_bath = st.checkbox("海の見えるお風呂")
    private_dining = st.checkbox("部屋食あり") 

with col3:
    near_station = st.checkbox("駅近")
    glamping = st.checkbox("グランピング")
    ise_shima_access = st.checkbox("伊勢志摩観光に便利")
    shopping = st.checkbox("買い物便利")

    
user_hotel_pref = {
    "family": int(family),
    "couple": int(couple),
    "hot_spring": int(hot_spring),
    "scenic": int(scenic),
    "near_station": int(near_station),
    "glamping": int(glamping),
    "shopping": int(shopping),
    "ocean_view_bath": int(ocean_view_bath),
    "quiet": int(quiet),
    "ise_shima_access": int(ise_shima_access),
    "beach_front": int(beach_front),
    "kids_room": int(kids_room),
    "private_dining": int(private_dining)
    

}

ranked_hotels = rank_hotels(hotels_df, user_hotel_pref)
hotel = ranked_hotels.iloc[0]

st.success(f"🏨 おすすめ宿泊施設：{hotel['name']}（{hotel['area']}）")
st.write(hotel["description"])


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

# 修正ポイント：CSVの "spot_name" を "name" にリネームしてから抽出する
day1_locations = (
    [NAGOYA] +
    day1_df[["spot_name", "lat", "lon"]].rename(columns={"spot_name": "name"}).to_dict("records") +
    [hotel_location]
)

day2_locations = (
    [hotel_location] +
    day2_df[["spot_name", "lat", "lon"]].rename(columns={"spot_name": "name"}).to_dict("records") +
    [NAGOYA]
)


# Day1：名古屋スタート → ホテルゴール
day1_route = solve_tsp(day1_locations, start_index=0, end_index=len(day1_locations)-1)

# Day2：ホテルスタート → 名古屋ゴール
day2_route = solve_tsp(day2_locations, start_index=0, end_index=len(day2_locations)-1)



# =====================
# ルート表示
# =====================
st.header("④ 1泊2日モデルルート")

# googlemapクリック表示
st.subheader("🗓 Day1")
st.write(" → ".join(day1_route))
st.markdown(f"[Googleマップで開く]({make_google_map_url(day1_route)})")

st.subheader("🗓 Day2")
st.write(" → ".join(day2_route))
st.markdown(f"[Googleマップで開く]({make_google_map_url(day2_route)})")

