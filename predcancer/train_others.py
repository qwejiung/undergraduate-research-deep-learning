from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from predcancer.settings import from_test_probs
from predcancer.utils import dic_match, get_sens_spec_f1


def Train_XGB(train_df, valid_df, test_df, full, setting):
    # ------------------ 1. Feature & Label 준비 --------------------
    exp_name = setting["exp_name"]
    feature_cols = [
        c for c in full.columns if c not in ("subject_id", "c_time", "label")
    ]

    X_train = train_df[feature_cols].copy()
    X_valid = valid_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    y_train = train_df["label"].values
    y_valid = valid_df["label"].values
    y_test = test_df["label"].values

    # ------------------ 2. 모델 정의 --------------------
    my_key_to_xgb_key = {
        "seed": "random_state",
    }
    clf = XGBClassifier(
        **{
            my_key_to_xgb_key.get(k, k): setting[k]
            for k in [
                "learning_rate",
                "max_depth",
                "min_child_weight",
                "subsample",
                "colsample_bytree",
                "reg_lambda",
                "gamma",
                "scale_pos_weight",
                "seed",
            ]
        }
    )

    # ------------------ 3. 학습 (조기 종료 옵션) --------------------
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    # ------------------ 4. 검증 성능 --------------------
    y_pred_prob_valid = clf.predict_proba(X_valid)[:, 1]
    valid_auc = roc_auc_score(y_valid, y_pred_prob_valid)
    valid_ap = average_precision_score(y_valid, y_pred_prob_valid)

    valid_sensitivity, valid_specificity, valid_f1, val_thres, val_thres_f1 = (
        get_sens_spec_f1(y_valid, y_pred_prob_valid)
    )

    print(f"XGBoost valid AUROC={valid_auc:.3f} | PR-AUC={valid_ap:.3f}")

    # ------------------ 5. 테스트 성능 --------------------
    y_pred_prob_test = clf.predict_proba(X_test)[:, 1]

    results = {
        "val_auroc": valid_auc,
        "val_ap": valid_ap,
        "val_sensitivity": valid_sensitivity,
        "val_specificity": valid_specificity,
        "val_f1": valid_f1,
        **from_test_probs(exp_name, y_pred_prob_test, y_test, val_thres, val_thres_f1),
    }
    return results


def Train_LR(train_df, valid_df, test_df, full, setting):
    exp_name = setting["exp_name"]

    # ------------------ 1. Feature & Label 준비 --------------------
    feature_cols = [
        c for c in full.columns if c not in ("subject_id", "c_time", "label")
    ]

    X_train = train_df[feature_cols].copy()
    X_valid = valid_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    y_train = train_df["label"].values
    y_valid = valid_df["label"].values
    y_test = test_df["label"].values

    # ------------------ 2. Scaling --------------------
    scaler_lr = StandardScaler()
    X_train_scaled = scaler_lr.fit_transform(X_train)
    X_valid_scaled = scaler_lr.transform(X_valid)
    X_test_scaled = scaler_lr.transform(X_test)

    # ------------------ 3. 모델 학습 --------------------
    clf = LogisticRegression(
        max_iter=1000,
        class_weight=setting["class_weight"],  # 1:3 비율 보정
        random_state=setting["seed"],
        C=setting["c"],
    )
    clf.fit(X_train_scaled, y_train)

    # ------------------ 4. 모델 평가 --------------------
    y_pred_prob_valid = clf.predict_proba(X_valid_scaled)[:, 1]
    valid_auc = roc_auc_score(y_valid, y_pred_prob_valid)
    valid_ap = average_precision_score(y_valid, y_pred_prob_valid)

    valid_sensitivity, valid_specificity, valid_f1, val_thres, val_thres_f1 = (
        get_sens_spec_f1(y_valid, y_pred_prob_valid)
    )

    print(f"Logistic Regression valid AUROC={valid_auc:.3f} | PR-AUC={valid_ap:.3f}")

    y_pred_prob_test = clf.predict_proba(X_test_scaled)[:, 1]

    results = {
        "val_auroc": valid_auc,
        "val_ap": valid_ap,
        "val_sensitivity": valid_sensitivity,
        "val_specificity": valid_specificity,
        "val_f1": valid_f1,
        **from_test_probs(exp_name, y_pred_prob_test, y_test, val_thres, val_thres_f1),
    }

    return results
