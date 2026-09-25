from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel

import numpy as np
import pandas as pd
import xgboost as xgb


def update_benchmark_metrics(y_test, y_pred, feat_sel, model, fold, df):
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Acc: {acc}, Prec:{prec}, Recall:{recall}, AUC:{auc}, F1:{f1}")

    df = df._append(
        {
            "model": model,
            "feat_selection": feat_sel,
            "fold": fold,
            "accuracy": acc,
            "precision": prec,
            "recall": recall,
            "AUC": auc,
            "F1": f1,
        },
        ignore_index=True,
    )

    return df

def select_feat_correlation(X_train, y_train, n_feats):
    
    X_norm = MinMaxScaler().fit_transform(X_train)
    chi_selector = SelectKBest(f_classif, k=n_feats)
    chi_selector.fit(X_norm, y_train)
    
    chi_support = chi_selector.get_support()
    chi_features = X_train.loc[:, chi_support].columns.tolist()
    
    return chi_features



def gridsearch_LogReg(X_train, y_train, seed):

    param_grid = {
        "C": [0.01, 0.1, 1, 10], 
        "penalty": ['l2', 'l1'],
    }
    
    rf = LogisticRegression()
    best_clf = RandomizedSearchCV(
        rf, param_grid, n_iter=100, cv=4, random_state=seed, verbose=0,
    )
    best_clf = best_clf.fit(X_train, y_train)
    
    print("Best estimator found by grid search:")
    print(best_clf.best_estimator_)
    
    return best_clf


def gridsearch_RF(X_train, y_train, seed):

    param_grid = {
        "n_estimators": [int(x) for x in np.linspace(100, 900, 5)],
        "max_depth": [int(x) for x in np.linspace(10, 50, 5)],
    }
    
    rf = RandomForestClassifier()
    best_clf = RandomizedSearchCV(
        rf, param_grid, n_iter=100, cv=4, random_state=seed, verbose=0,
    )
    best_clf = best_clf.fit(X_train, y_train)
    
    print("Best estimator found by grid search:")
    print(best_clf.best_estimator_)
    
    return best_clf


def gridsearch_XGB(X_train, y_train, seed):
    
    param_grid = {
        "n_estimators": [int(x) for x in np.linspace(100, 900, 5)],
        "max_depth": [int(x) for x in np.linspace(5, 25, 5)],
    }
    
    xgb_clf = xgb.XGBClassifier(tree_method="hist")
    best_clf = RandomizedSearchCV(
        xgb_clf, param_grid, n_iter=100, cv=4, random_state=seed, verbose=0
    )
    best_clf = best_clf.fit(X_train, y_train)

    print("Best estimator found by grid search:")
    print(best_clf.best_estimator_)

    return best_clf


def select_feat_RFE(X_train, y_train, model, n_feats, seed):

    if model=="LG":
        best_clf = gridsearch_LogReg(X_train, y_train, seed)
        clf = LogisticRegression(        
            C=best_clf.best_estimator_.C,
            penalty=best_clf.best_estimator_.penalty,
        )
    elif model=="RF":
        best_clf = gridsearch_RF(X_train, y_train, seed)
        clf = RandomForestClassifier(
            n_estimators=best_clf.best_estimator_.n_estimators,
            max_depth=best_clf.best_estimator_.max_depth,
        )
    elif model=="XGB":
        best_clf = gridsearch_XGB(X_train, y_train, seed)
        clf = xgb.XGBClassifier(
            n_estimators=best_clf.best_estimator_.n_estimators,
            max_depth=best_clf.best_estimator_.max_depth,
        )
        
    rfe_selector = RFE(
        estimator=clf,
        n_features_to_select=n_feats,
        step=20,
    )
    rfe_selector.fit(X_train, y_train)
    rfe_support = rfe_selector.get_support()
    rfe_features = X_train.loc[:, rfe_support].columns.tolist()

    return rfe_features


def select_feat_from_model(X_train, y_train, model, n_feats, seed):

    if model=="LG":
        best_clf = gridsearch_LogReg(X_train, y_train, seed)
        clf = LogisticRegression(        
            C=best_clf.best_estimator_.C,
            penalty=best_clf.best_estimator_.penalty,
        )
    elif model=="RF":
        best_clf = gridsearch_RF(X_train, y_train, seed)
        clf = RandomForestClassifier(
            n_estimators=best_clf.best_estimator_.n_estimators,
            max_depth=best_clf.best_estimator_.max_depth,
        )
    elif model=="XGB":
        best_clf = gridsearch_XGB(X_train, y_train, seed)
        clf = xgb.XGBClassifier(
            n_estimators=best_clf.best_estimator_.n_estimators,
            max_depth=best_clf.best_estimator_.max_depth,
        )
        
    embed_selector = SelectFromModel(
        estimator=clf,
        threshold=None,
        max_features=n_feats,
    )
    embed_selector.fit(X_train, y_train)
    embed_support = embed_selector.get_support()
    embed_features = X_train.loc[:, embed_support].columns.tolist()

    return embed_features

    
def prepare_train_test(X, y, train_idx, test_idx, feat_sel, n_feats, seed):
    
    col_names = X.columns

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = np.array(y)[train_idx], np.array(y)[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_names)
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_names)

    feats = col_names

    if len(feat_sel.split("_")) > 1:
        feat_sel, model = feat_sel.split("_")

    if feat_sel=="correlation":
        feats = select_feat_correlation(X_train, y_train, n_feats)
    elif feat_sel=="RFE":
        feats = select_feat_RFE(X_train, y_train, model, n_feats, seed)
    elif feat_sel=="EMBED":
        feats = select_feat_from_model(X_train, y_train, model, n_feats, seed)
        
    X_train = X_train[feats]
    X_test = X_test[feats]

    return X_train, X_test, y_train, y_test


def fit_logreg(X_train, X_test, y_train, y_test, df, feat_sel, fold, seed):

    best_clf = gridsearch_LogReg(X_train, y_train, seed)
    
    clf = LogisticRegression(
        C=best_clf.best_estimator_.C,
        penalty=best_clf.best_estimator_.penalty,
        random_state=seed
    ).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    df = update_benchmark_metrics(y_test, y_pred, feat_sel, "LG", fold, df)
    
    return df


def optimize_fit_rf(X_train, X_test, y_train, y_test, df, feat_sel, fold, seed):
    
    best_clf = gridsearch_RF(X_train, y_train, seed)

    clf = RandomForestClassifier(
        n_estimators=best_clf.best_estimator_.n_estimators,
        max_depth=best_clf.best_estimator_.max_depth,
        min_samples_split=2,
        random_state=seed,
    ).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    df = update_benchmark_metrics(y_test, y_pred, feat_sel, "RF", fold, df)
    
    return df


def optimize_fit_xgb(X_train, X_test, y_train, y_test, df, feat_sel, fold, seed):
    
    le = LabelEncoder().fit(y_train)
    y_train_xgb = le.transform(y_train)
    y_test_xgb = le.transform(y_test)

    best_clf = gridsearch_XGB(X_train, y_train_xgb, seed)
    
    clf = xgb.XGBClassifier(
        tree_method="hist",
        n_estimators=best_clf.best_estimator_.n_estimators,
        max_depth=best_clf.best_estimator_.max_depth,
    ).fit(X_train, y_train_xgb)
    y_pred = clf.predict(X_test)
    df = update_benchmark_metrics(y_test, y_pred, feat_sel, "XGB", fold, df)

    return df

def optimize_fit_cal_xgb(X_train, X_test, y_train, y_test, df, feat_sel, fold, seed):

    le = LabelEncoder().fit(y_train)
    y_train_xgb = le.transform(y_train)
    y_test_xgb = le.transform(y_test)

    best_clf = gridsearch_XGB(X_train, y_train_xgb, seed)
    
    clf = xgb.XGBClassifier(
        tree_method="hist",
        n_estimators=best_clf.best_estimator_.n_estimators,
        max_depth=best_clf.best_estimator_.max_depth,
    )
    clf = CalibratedClassifierCV(clf, cv=5, method="isotonic").fit(X_train, y_train_xgb)
    y_pred = clf.predict(X_test)
    df = update_benchmark_metrics(y_test, y_pred, feat_sel, "Cal. XGB", fold, df)

    return df