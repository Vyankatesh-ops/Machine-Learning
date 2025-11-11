# poly_regression.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# ---------------------------------------------------------------
# Q3.0: Run Polynomial Regression (with or without regularization)
# ---------------------------------------------------------------
def run_poly_regression(X_train, y_train, X_val, y_val, X_test, y_test,
                        degree=1, regularizer=None, reg_strength=0.0, random_state: int = 42):
    """
    Fit polynomial regression model with optional regularization.
    regularizer: None, 'l1', 'l2'
    reg_strength: alpha (for Lasso/Ridge)
    Returns dict with train/val/test MSEs and fitted model & coefficients & feature_names
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xtr = poly.fit_transform(X_train)
    Xv = poly.transform(X_val)
    Xt = poly.transform(X_test)

    if regularizer is None:
        model = LinearRegression()
    elif regularizer == 'l1':
        model = Lasso(alpha=reg_strength, max_iter=10000, random_state=random_state)
    elif regularizer == 'l2':
        model = Ridge(alpha=reg_strength, random_state=random_state)
    else:
        raise ValueError("regularizer must be None, 'l1', or 'l2'")

    model.fit(Xtr, y_train)
    ytr_pred = model.predict(Xtr)
    yv_pred = model.predict(Xv)
    yt_pred = model.predict(Xt)

    res = {
        'model': model,
        'poly': poly,
        'train_mse': mean_squared_error(y_train, ytr_pred),
        'val_mse': mean_squared_error(y_val, yv_pred),
        'test_mse': mean_squared_error(y_test, yt_pred),
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'feature_names': poly.get_feature_names_out()
    }
    return res


# ---------------------------------------------------------------
# Evaluate Across Polynomial Degrees
# ---------------------------------------------------------------
def evaluate_across_degrees(X_train, y_train, X_val, y_val, X_test, y_test,
                            degrees=range(1, 7), regularizer=None, reg_strength=0.0):
    records = []
    for d in degrees:
        r = run_poly_regression(X_train, y_train, X_val, y_val, X_test, y_test,
                                degree=d, regularizer=regularizer, reg_strength=reg_strength)
        records.append({
            'degree': d,
            'train_mse': r['train_mse'],
            'val_mse': r['val_mse'],
            'test_mse': r['test_mse'],
            'res': r
        })
    return records


# ---------------------------------------------------------------
# Plot MSE vs Degree
# ---------------------------------------------------------------
def plot_degree_vs_mse(records, title_prefix="", save_path=None):
    degrees = [r['degree'] for r in records]
    train = [r['train_mse'] for r in records]
    val = [r['val_mse'] for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, train, marker='o', label='Train MSE')
    ax.plot(degrees, val, marker='o', label='Validation MSE')
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('MSE')
    ax.set_title(f'{title_prefix} Degree vs MSE')
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------
# Tune Regularization Strength (α)
# ---------------------------------------------------------------
def tune_regularization_for_degree(X_train, y_train, X_val, y_val, degree,
                                   regularizer='l2', alphas=None):
    if alphas is None:
        alphas = np.logspace(-4, 2, 30)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xtr = poly.fit_transform(X_train)
    Xv = poly.transform(X_val)

    best_alpha = None
    best_mse = np.inf
    results = []

    for a in alphas:
        if regularizer == 'l2':
            model = Ridge(alpha=a)
        elif regularizer == 'l1':
            model = Lasso(alpha=a, max_iter=10000)
        else:
            raise ValueError("regularizer must be 'l1' or 'l2'")

        model.fit(Xtr, y_train)
        yv_pred = model.predict(Xv)
        mse = mean_squared_error(y_val, yv_pred)
        results.append((a, mse))
        if mse < best_mse:
            best_mse = mse
            best_alpha = a

    results_df = pd.DataFrame(results, columns=['alpha', 'val_mse']).set_index('alpha')
    return best_alpha, best_mse, results_df


# ---------------------------------------------------------------
# Plot α vs Validation MSE (log scale)
# ---------------------------------------------------------------
def plot_alpha_vs_val_mse(results_df, title="", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(results_df.index, results_df['val_mse'], marker='o')
    ax.set_xlabel('Regularization Strength (alpha)')
    ax.set_ylabel('Validation MSE')
    ax.set_title(f'{title} Regularization Strength vs Validation MSE')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------
# Show Non-zero Coefficients (for L1/L2 regularization)
# ---------------------------------------------------------------
def coefficients_nonzero(model, feature_names):
    """
    Returns a DataFrame of non-zero coefficients and their magnitudes
    for the given model (useful for L1/L2 regularization analysis).
    """
    coefs = model.coef_.ravel() if hasattr(model.coef_, "ravel") else model.coef_
    nonzero_mask = np.abs(coefs) > 1e-6
    nonzero_features = np.array(feature_names)[nonzero_mask]
    nonzero_values = coefs[nonzero_mask]

    coef_df = pd.DataFrame({
        "Feature": nonzero_features,
        "Coefficient": nonzero_values
    }).sort_values(by="Coefficient", ascending=False)

    print("\n🔹 Non-zero Coefficients:")
    print(coef_df.to_string(index=False))
    return coef_df
