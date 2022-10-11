import catboost
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import make_column_transformer
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def check_mb(reference, received, threshold = 0.5):
    reference = reference.copy()
    received = received.copy()

    ddt = pd.DataFrame(index = [
        "DriftDetectorClassifier"
        ],
        columns = reference.columns)

    cat_features = reference.select_dtypes(exclude = 'number').columns.tolist()
    num_features = reference.select_dtypes(include = 'number').columns.tolist()

    if len(cat_features) > 0: 
        for cat_col in cat_features:     
            cat_transformer = make_column_transformer(
                (OneHotEncoder(sparse = False), [cat_col]),
                remainder = 'drop')
            
            reference_trans = cat_transformer.fit_transform(reference)
            received_trans = cat_transformer.transform(received)

            reference_cat = pd.DataFrame(reference_trans, 
                                            columns = cat_transformer.get_feature_names_out())
            received_cat = pd.DataFrame(received_trans, 
                                            columns = cat_transformer.get_feature_names_out())

            reference_cat['target'] = 0
            received_cat['target'] = 1

            df_merge = pd.concat([reference_cat, received_cat], axis = 0)
            X, y = df_merge.drop(['target'], axis = 1), df_merge['target']

            skf = StratifiedKFold(n_splits = 5)
                
            cv_score = np.empty(5) 

            for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                model = catboost.CatBoostClassifier(silent = True)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                cv_score[idx] = roc_auc_score(y_test, pred)

            drift_measurement = np.mean(cv_score)

            if drift_measurement > threshold:
                is_drift = True
            else:
                is_drift = False

            ddt.loc["DriftDetectorClassifier", cat_col] = is_drift

        for num_cols in num_features:
            reference['target'] = 0
            received['target'] = 1

            df_merge = pd.concat([reference, received], axis = 0)
            X, y = df_merge[num_cols], df_merge['target']

            skf = StratifiedKFold(n_splits = 10)
                
            cv_score = np.empty(10) 

            for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                model = catboost.CatBoostClassifier(silent = True)
                model.fit(X_train.values.reshape(-1, 1), y_train)
                pred = model.predict(X_test.values.reshape(-1, 1))
                cv_score[idx] = roc_auc_score(y_test, pred)

            drift_measurement = np.mean(cv_score)

            if drift_measurement > threshold:
                is_drift = True
            else:
                is_drift = False

            ddt.loc["DriftDetectorClassifier", num_cols] = is_drift

    else:

        for num_cols in num_features:
            reference['target'] = 0
            received['target'] = 1

            df_merge = pd.concat([reference, received], axis = 0)
            X, y = df_merge[num_cols], df_merge['target']

            skf = StratifiedKFold(n_splits = 10)
                
            cv_score = np.empty(2) 

            for idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                model = catboost.CatBoostClassifier(silent = True)
                model.fit(X_train.values.reshape(-1, 1), y_train)
                pred = model.predict(X_test.values.reshape(-1, 1))
                cv_score[idx] = roc_auc_score(y_test, pred)

            drift_measurement = np.mean(cv_score)

            if drift_measurement > threshold:
                is_drift = True
            else:
                is_drift = False

            ddt.loc["DriftDetectorClassifier", num_cols] = is_drift
    
    print("Numerical and Categorical Features Drift Table: Drift detected = True")

    return ddt