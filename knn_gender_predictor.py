# knn_gender_predictor.py
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

class PerFeatureTransformer:
    """
    params mapping example:
      {"GPA": "standard", "Major": "ordinal", "Program": "onehot"}
    Supported types: "standard" (numeric), "ordinal" (small categories),
                     "onehot" (categorical)
    """
    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        self.ord_encoders: Dict[str, OrdinalEncoder] = {}
        self.ohe_encoders: Dict[str, OneHotEncoder] = {}
        self.order = []  # track order of final feature columns

    def fit(self, df: pd.DataFrame, params: Dict[str, str]) -> None:
        self.order = []
        for feature, ftype in params.items():
            if ftype == "standard":
                scaler = StandardScaler()
                scaler.fit(df[[feature]])
                self.scalers[feature] = scaler
                self.order.append(feature)
            elif ftype == "ordinal":
                enc = OrdinalEncoder()
                enc.fit(df[[feature]])
                self.ord_encoders[feature] = enc
                self.order.append(feature)
            elif ftype == "onehot":
                ohe = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')
                ohe.fit(df[[feature]])
                self.ohe_encoders[feature] = ohe
                # store names for ordering
                cols = [f"{feature}__{c}" for c in ohe.categories_[0]]
                self.order.extend(cols)
            else:
                raise ValueError(f"Unknown feature type: {ftype}")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols_out = []
        for feature in list(set(list(self.scalers.keys()) + list(self.ord_encoders.keys()) + list(self.ohe_encoders.keys()))):
            pass  # we will generate using params order below
        # Reconstruct according to original fit order
        X_parts = []
        for feature_or_col in self.order:
            # detect if it's a onehot created column (contains '__')
            if "__" in feature_or_col:
                feat, cat = feature_or_col.split("__", 1)
                ohe = self.ohe_encoders[feat]
                # transform and find index of cat in categories_
                arr = ohe.transform(df[[feat]])
                # determine column index
                idx = list(ohe.categories_[0]).index(cat)
                X_parts.append(arr[:, idx:idx+1])
            else:
                # feature could be in scalers or ord_encoders
                if feature_or_col in self.scalers:
                    arr = self.scalers[feature_or_col].transform(df[[feature_or_col]])
                    X_parts.append(arr)
                elif feature_or_col in self.ord_encoders:
                    arr = self.ord_encoders[feature_or_col].transform(df[[feature_or_col]])
                    X_parts.append(arr)
                else:
                    raise RuntimeError(f"Unexpected feature: {feature_or_col}")
        X = np.hstack(X_parts) if len(X_parts) > 0 else np.empty((len(df), 0))
        return X

    def fit_transform(self, df: pd.DataFrame, params: Dict[str, str]) -> np.ndarray:
        self.fit(df, params)
        return self.transform(df)

class KNNGenderPredictor:
    def __init__(self, student_df: pd.DataFrame, username: str = "user"):
        self.df = student_df.copy()
        # Ensure consistent column names
        self.df.columns = [c.strip().title() for c in self.df.columns]
        if 'Gpa' in self.df.columns:
            self.df.rename(columns={'Gpa':'GPA'}, inplace=True)
        self.username = username
        self.transformer = PerFeatureTransformer()
        self.params = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None

    def train_val_test_split(self, test_size: float = 0.2, val_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # first split off test
        train_val, test = train_test_split(self.df, test_size=test_size, random_state=seed, stratify=self.df['Gender'])
        # then split train_val into train + val
        val_relative = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=val_relative, random_state=seed, stratify=train_val['Gender'])
        print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

    def get_feature_matrix_and_labels(self, df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        # prepare params: numeric -> standard; categorical with >2 classes -> onehot; else ordinal
        params = {}
        for f in features:
            if np.issubdtype(df[f].dtype, np.number):
                params[f] = "standard"
            else:
                # if small cardinality (<=4), ordinal else onehot; user can override by passing different params
                if df[f].nunique() <= 4:
                    params[f] = "ordinal"
                else:
                    params[f] = "onehot"
        # Fit transformer on training data only (we assume fit called earlier)
        self.params = params
        # It's required to fit transformer on training data only; caller should ensure that
        return self.transformer.transform(df), df['Gender'].values

    def fit_transform_on_train(self, train_df: pd.DataFrame, features: List[str]):
        # Fit transformer on train only
        params = {}
        for f in features:
            if np.issubdtype(train_df[f].dtype, np.number):
                params[f] = "standard"
            else:
                # treat categorical with small cardinality as ordinal (Major/Program likely have small categories)
                params[f] = "onehot" if train_df[f].nunique() > 4 else "ordinal" if train_df[f].nunique() <= 4 else "onehot"
        self.params = params
        X_train = self.transformer.fit_transform(train_df, params)
        y_train = train_df['Gender'].values
        return X_train, y_train

    def transform_df(self, df: pd.DataFrame):
        return self.transformer.transform(df)

    def get_knn_accuracy_vs_k(self, k_values: List[int], distance: str = "euclidean") -> List[float]:
        # use self.X_train/self.y_train and self.X_val/self.y_val (must be prepared)
        if self.X_train is None or self.X_val is None:
            raise RuntimeError("Train/val/test not prepared. Call prepare_data() first.")
        scores = []
        for k in k_values:
            if distance == "euclidean":
                clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            elif distance == "manhattan":
                clf = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
            elif distance == "cosine":
                clf = KNeighborsClassifier(n_neighbors=k, metric='cosine')
            else:
                raise ValueError("Unsupported distance")
            clf.fit(self.X_train, self.y_train)
            preds = clf.predict(self.X_val)
            scores.append(accuracy_score(self.y_val, preds))
        return scores

    def plot_knn_accuracy_vs_k(self, k_values: List[int], distance: str = "euclidean", save_path: str = None) -> None:
        scores = self.get_knn_accuracy_vs_k(k_values, distance)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(k_values, scores, marker='o')
        ax.set_xlabel("k")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title(f"KNN accuracy vs k ({distance})")
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    def get_knn_f1_heatmap(self, k_values: List[int], distances: List[str]) -> pd.DataFrame:
        rows = []
        for d in distances:
            row = []
            for k in k_values:
                if d == "euclidean":
                    clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
                elif d == "manhattan":
                    clf = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
                elif d == "cosine":
                    clf = KNeighborsClassifier(n_neighbors=k, metric='cosine')
                else:
                    raise ValueError("Unsupported distance")
                clf.fit(self.X_train, self.y_train)
                preds = clf.predict(self.X_val)
                f1 = f1_score(self.y_val, preds, average='macro')
                row.append(f1)
            rows.append(row)
        df = pd.DataFrame(rows, index=distances, columns=k_values)
        return df

    def plot_knn_f1_heatmap(self, f1_scores_df: pd.DataFrame, save_path: str = None) -> None:
        fig, ax = plt.subplots(figsize=(12,4))
        sns.heatmap(f1_scores_df, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title("Validation F1 (macro) for k (cols) x distance (rows)")
        if save_path:
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    def get_knn_f1_single_feature_table(self, k_values: List[int], features: List[str], distance: str = "euclidean") -> pd.DataFrame:
        # evaluate single-feature models on the TEST set
        results = {}
        for feat in features:
            # Create a temporary transformer fitted only to the single feature
            tmp_transformer = PerFeatureTransformer()
            params = {feat: "standard" if np.issubdtype(self.X_train.dtype, np.number) else "ordinal"}
            # Instead, more robust approach:
            # Fit on train DataFrame taking appropriate param based on dtype
            train_df = self.train_df  # stored earlier by prepare_data
            feat_params = {}
            if np.issubdtype(train_df[feat].dtype, np.number):
                feat_params[feat] = "standard"
            else:
                feat_params[feat] = "ordinal"
            tmp_transformer.fit(train_df, feat_params)
            X_train_feat = tmp_transformer.transform(train_df)
            val_df = self.val_df
            X_test_feat = tmp_transformer.transform(self.test_df)
            y_test = self.test_df['Gender'].values
            # Evaluate for each k
            f1s = []
            for k in k_values:
                if distance == "euclidean":
                    clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
                elif distance == "manhattan":
                    clf = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
                elif distance == "cosine":
                    clf = KNeighborsClassifier(n_neighbors=k, metric='cosine')
                else:
                    raise ValueError("Unsupported distance")
                clf.fit(X_train_feat, train_df['Gender'].values)
                preds = clf.predict(X_test_feat)
                f1s.append(f1_score(y_test, preds, average='macro'))
            results[feat] = f1s
        res_df = pd.DataFrame(results, index=k_values)
        return res_df

    # -------------------------
    # Helper to prepare data
    # -------------------------
    def prepare_data(self, features: List[str], test_size: float = 0.2, val_size: float = 0.2, seed: int = 42):
        train, val, test = self.train_val_test_split(test_size=test_size, val_size=val_size, seed=seed)
        self.train_df, self.val_df, self.test_df = train, val, test
        # Fit transformer on train only
        X_train, y_train = self.fit_transform_on_train(train, features)
        X_val = self.transform_df(val)
        X_test = self.transform_df(test)
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, val['Gender'].values, test['Gender'].values
        print("Prepared feature matrices.")

if __name__ == "__main__":
    # quick demo (requires student_dataset.csv)
    df = pd.read_csv("student_dataset.csv")
    predictor = KNNGenderPredictor(df)
    features = ["GPA", "Major", "Program"]
    predictor.prepare_data(features)
    k_vals = list(range(1,22,2))
    accs = predictor.get_knn_accuracy_vs_k(k_vals, distance="euclidean")
    predictor.plot_knn_accuracy_vs_k(k_vals, distance="euclidean")
    f1_df = predictor.get_knn_f1_heatmap(k_vals, ["euclidean", "manhattan", "cosine"])
    predictor.plot_knn_f1_heatmap(f1_df)
