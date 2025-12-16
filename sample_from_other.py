import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

# ----------------------
# 0. 설정
# ----------------------
PATH_TRAIN = "train.csv"
PATH_TEST_INDEX = "test_index.csv"
PATH_MATCH_INFO = "match_info.csv"       # 필요하면
PATH_SAMPLE_SUB = "sample_submission.csv"

K = 20   # 마지막 K 이벤트 사용 (20~32 사이 선택)

# ----------------------
# 1. 데이터 로드
# ----------------------
train = pd.read_csv(PATH_TRAIN)
test_index = pd.read_csv(PATH_TEST_INDEX)
match_info = pd.read_csv(PATH_MATCH_INFO)
sample_sub = pd.read_csv(PATH_SAMPLE_SUB)

test_events_list = []
for _, row in test_index.iterrows():
    df_ep = pd.read_csv(row["path"])
    test_events_list.append(df_ep)

test_events = pd.concat(test_events_list, ignore_index=True)

train["is_train"] = 1
test_events["is_train"] = 0

events = pd.concat([train, test_events], ignore_index=True)

# ----------------------
# 2. 기본 정렬 + episode 내 인덱스
# ----------------------
events = events.sort_values(["game_episode", "time_seconds", "action_id"]).reset_index(drop=True)

events["event_idx"] = events.groupby("game_episode").cumcount()
events["n_events"] = events.groupby("game_episode")["event_idx"].transform("max") + 1
events["ep_idx_norm"] = events["event_idx"] / (events["n_events"] - 1).clip(lower=1)

# ----------------------
# 3. 시간/공간 feature
# ----------------------
# Δt
events["prev_time"] = events.groupby("game_episode")["time_seconds"].shift(1)
events["dt"] = events["time_seconds"] - events["prev_time"]
events["dt"] = events["dt"].fillna(0.0)

# 이동량/거리
events["dx"] = events["end_x"] - events["start_x"]
events["dy"] = events["end_y"] - events["start_y"]
events["dist"] = np.sqrt(events["dx"]**2 + events["dy"]**2)

# 속도 (dt=0 보호)
events["speed"] = events["dist"] / events["dt"].replace(0, 1e-3)

# zone / lane (필요시 범위 조정)
events["x_zone"] = (events["start_x"] / (105/7)).astype(int).clip(0, 6)
events["lane"] = pd.cut(
    events["start_y"],
    bins=[0, 68/3, 2*68/3, 68],
    labels=[0, 1, 2],
    include_lowest=True
).astype(int)

# ----------------------
# 4. 라벨 및 episode-level 메타 (train 전용)
# ----------------------
train_events = events[events["is_train"] == 1].copy()

last_events = (
    train_events
    .groupby("game_episode", as_index=False)
    .tail(1)
    .copy()
)

labels = last_events[["game_episode", "end_x", "end_y"]].rename(
    columns={"end_x": "target_x", "end_y": "target_y"}
)

# episode-level 메타 (마지막 이벤트 기준)
ep_meta = last_events[["game_episode", "game_id", "team_id", "is_home", "period_id", "time_seconds"]].copy()
ep_meta = ep_meta.rename(columns={"team_id": "final_team_id"})

# game_clock (분 단위, 0~90+)
ep_meta["game_clock_min"] = np.where(
    ep_meta["period_id"] == 1,
    ep_meta["time_seconds"] / 60.0,
    45.0 + ep_meta["time_seconds"] / 60.0
)

# ----------------------
# 5. 공격 팀 플래그 (final_team vs 상대)
# ----------------------
# final_team_id를 전체 events에 붙임
events = events.merge(
    ep_meta[["game_episode", "final_team_id"]],
    on="game_episode",
    how="left"
)

events["is_final_team"] = (events["team_id"] == events["final_team_id"]).astype(int)

# ----------------------
# 6. 입력용 events에서 마지막 이벤트 타깃 정보 가리기
# ----------------------
# is_last 플래그
events["last_idx"] = events.groupby("game_episode")["event_idx"].transform("max")
events["is_last"] = (events["event_idx"] == events["last_idx"]).astype(int)

# labels는 이미 뽑아놨으니, 입력쪽에서만 end_x, end_y, dx, dy, dist, speed 지움
mask_last = events["is_last"] == 1
for col in ["end_x", "end_y", "dx", "dy", "dist", "speed"]:
    events.loc[mask_last, col] = np.nan

# ----------------------
# 7. 카테고리 인코딩 (type_name, result_name, team_id 등)
# ----------------------
events["type_name"] = events["type_name"].fillna("__NA_TYPE__")
events["result_name"] = events["result_name"].fillna("__NA_RES__")

le_type = LabelEncoder()
le_res = LabelEncoder()

events["type_id"] = le_type.fit_transform(events["type_name"])
events["res_id"] = le_res.fit_transform(events["result_name"])

# team_id는 그대로 써도 되지만, 문자열이면 숫자로 매핑
if events["team_id"].dtype == "object":
    le_team = LabelEncoder()
    events["team_id_enc"] = le_team.fit_transform(events["team_id"])
else:
    events["team_id_enc"] = events["team_id"].astype(int)

# ----------------------
# 8. 마지막 K 이벤트만 사용 (lastK)
# ----------------------
# rev_idx: 0이 마지막 이벤트
events["rev_idx"] = events.groupby("game_episode")["event_idx"].transform(
    lambda s: s.max() - s
)

lastK = events[events["rev_idx"] < K].copy()

# pos_in_K: 0~(K-1), 앞쪽 패딩 고려해서 뒤에 실제 이벤트가 모이게
def assign_pos_in_K(df):
    df = df.sort_values("event_idx")  # 오래된 → 최근
    L = len(df)
    df = df.copy()
    df["pos_in_K"] = np.arange(K - L, K)
    return df

lastK = lastK.groupby("game_episode", group_keys=False).apply(assign_pos_in_K)

# ----------------------
# 9. wide feature pivot
# ----------------------
# 사용할 이벤트 피처 선택
num_cols = [
    "start_x", "start_y",
    "end_x", "end_y",
    "dx", "dy", "dist", "speed",
    "dt",
    "ep_idx_norm",
    "x_zone", "lane",
    "is_final_team",
]

cat_cols = [
    "type_id",
    "res_id",
    "team_id_enc",
    "is_home",
    "period_id",
    "is_last",
]

feature_cols = num_cols + cat_cols

wide = lastK[["game_episode", "pos_in_K"] + feature_cols].copy()

# 숫자형 pivot
wide_num = wide.pivot_table(
    index="game_episode",
    columns="pos_in_K",
    values=num_cols,
    aggfunc="first"
)

# 범주형 pivot
wide_cat = wide.pivot_table(
    index="game_episode",
    columns="pos_in_K",
    values=cat_cols,
    aggfunc="first"
)

# 컬럼 이름 평탄화
wide_num.columns = [f"{c}_{int(pos)}" for (c, pos) in wide_num.columns]
wide_cat.columns = [f"{c}_{int(pos)}" for (c, pos) in wide_cat.columns]

X = pd.concat([wide_num, wide_cat], axis=1).reset_index()  # game_episode 포함

# episode-level 메타 붙이기
X = X.merge(ep_meta[["game_episode", "game_id", "game_clock_min", "final_team_id", "is_home", "period_id"]],
            on="game_episode", how="left")

# train 라벨 붙이기
X = X.merge(labels, on="game_episode", how="left")  # test는 NaN

# ----------------------
# 10. train/test 분리
# ----------------------
train_mask = X["game_episode"].isin(labels["game_episode"])
X_train = X[train_mask].copy()
X_test = X[~train_mask].copy()

y_train_x = X_train["target_x"].astype(float)
y_train_y = X_train["target_y"].astype(float)

# group용 game_id
groups = X_train["game_id"].values

# 모델 입력에서 빼야 할 컬럼들
drop_cols = [
    "game_episode",
    "game_id",
    "target_x",
    "target_y",
]

X_train_feat = X_train.drop(columns=drop_cols)
X_test_feat = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

# NaN 채우기 (LGBM은 NaN 다루긴 하지만, 깔끔하게)
X_train_feat = X_train_feat.fillna(0)
X_test_feat = X_test_feat.fillna(0)

# ----------------------
# 11. LightGBM 학습 (GroupKFold)
# ----------------------
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}

gkf = GroupKFold(n_splits=5)

models_x = []
models_y = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train_feat, y_train_x, groups)):
    print(f"Fold {fold}")

    X_tr, X_val = X_train_feat.iloc[tr_idx], X_train_feat.iloc[val_idx]
    y_tr_x, y_val_x = y_train_x.iloc[tr_idx], y_train_x.iloc[val_idx]
    y_tr_y, y_val_y = y_train_y.iloc[tr_idx], y_train_y.iloc[val_idx]

    dtrain_x = lgb.Dataset(X_tr, label=y_tr_x)
    dvalid_x = lgb.Dataset(X_val, label=y_val_x, reference=dtrain_x)

    model_x = lgb.train(
        params,
        dtrain_x,
        num_boost_round=3000,
        valid_sets=[dtrain_x, dvalid_x],
        valid_names=["train", "valid"],
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    models_x.append(model_x)

    dtrain_y = lgb.Dataset(X_tr, label=y_tr_y)
    dvalid_y = lgb.Dataset(X_val, label=y_val_y, reference=dtrain_y)

    model_y = lgb.train(
        params,
        dtrain_y,
        num_boost_round=3000,
        valid_sets=[dtrain_y, dvalid_y],
        valid_names=["train", "valid"],
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    models_y.append(model_y)

# ----------------------
# 12. test 예측 + 앙상블
# ----------------------
pred_x_folds = []
pred_y_folds = []

for model_x, model_y in zip(models_x, models_y):
    pred_x_folds.append(model_x.predict(X_test_feat, num_iteration=model_x.best_iteration))
    pred_y_folds.append(model_y.predict(X_test_feat, num_iteration=model_y.best_iteration))

pred_x = np.mean(pred_x_folds, axis=0)
pred_y = np.mean(pred_y_folds, axis=0)

# 필드 범위로 클립
pred_x = np.clip(pred_x, 0, 105)
pred_y = np.clip(pred_y, 0, 68)

# ----------------------
# 13. submission 생성
# ----------------------
sub = sample_sub.copy()

# X_test에는 game_episode가 있으니, test_index와 align
pred_df = X_test[["game_episode"]].copy()
pred_df["end_x"] = pred_x
pred_df["end_y"] = pred_y

sub = sub.drop(columns=["end_x", "end_y"], errors="ignore")
sub = sub.merge(pred_df, on="game_episode", how="left")

sub.to_csv("submission_lgbm_lastK.csv", index=False)
print("Saved submission_lgbm_lastK.csv")
