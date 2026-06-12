import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def train_model(X_train, y_train):

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("\n" + "=" * 50)
    print("LIGHTGBM MODEL EVALUATION")
    print("=" * 50)

    print(classification_report(y_test, y_pred))

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.3f}")

    return y_pred


def show_feature_importance(model, feature_names):

    importance = model.feature_importances_

    ranked_features = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop Feature Importance")
    print("-" * 50)

    for feature, score in ranked_features:
        print(f"{feature}: {score}")