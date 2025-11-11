# main.py
import os
import numpy as np
import pandas as pd
from student_analysis import StudentDatasetAnalysis
from knn_gender_predictor import KNNGenderPredictor
from poly_regression import evaluate_across_degrees, plot_degree_vs_mse, tune_regularization_for_degree, coefficients_nonzero, run_poly_regression

PLOTS_DIR = "all_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def run_q1():
    print("=== Q1 Analysis ===")
    sa = StudentDatasetAnalysis("student_dataset.csv", plots_dir=PLOTS_DIR)
    # visualizations
    sa.plot_gender_distribution()
    sa.plot_major_distribution()
    sa.plot_program_distribution()
    sa.plot_gpa_distribution()
    sa.plot_program_by_major()
    sa.plot_gpa_by_major()
    sa.plot_gpa_by_program()
    sa.plot_gpa_by_program_and_major()
    sa.plot_sampled_dataset()
    sa.plot_entire_dataset_summary()
    # stats and program-major
    mean, std = sa.gpa_mean_std()
    counts = sa.count_students_per_program_major_pair()
    sa.visualize_students_per_program_major_pair(counts)
    # sampling Q1.2
    rand_avg, rand_std = sa.get_gpa_mean_std_random(n=500, repeats=50, random_state=42)
    strat_avg, strat_std = sa.get_gpa_mean_std_stratified(n=500, repeats=50, random_state=42)
    print("Random sampling:", rand_avg, rand_std)
    print("Stratified sampling:", strat_avg, strat_std)
    # Q1.3
    exact = sa.get_gender_balanced_counts(n=300, repeats=5)
    stratA = sa.sample_gender_uniform_random(n=300, repeats=5, random_state=42)
    print("Exact equal counts (repeats):", exact)
    print("Strategy A counts (repeats):", stratA)
    sa.plot_avg_max_gender_diff_vs_sample_size([300,600,900,1200,1500], repeats=10)
    # Q1.4 GPA uniform
    sampled_gpa_uniform = sa.sample_gpa_uniform(n=100, bins=10, random_state=42)
    sa.plot_gpa_histogram_comparison(sampled_gpa_uniform, bins=10)
    # Q1.5 program-major balanced
    sampled_pm = sa.sample_program_major_balanced(n=60, random_state=42)
    sa.show_program_major_counts_and_heatmap(sampled_pm)
    print("Q1 done. Plots saved to:", PLOTS_DIR)

def run_q2():
    print("\n=== Q2 KNN Gender Prediction ===")
    df = pd.read_csv("student_dataset.csv")
    predictor = KNNGenderPredictor(df, username="student_user")
    features = ["GPA", "Major", "Program"]
    predictor.prepare_data(features, test_size=0.2, val_size=0.2, seed=42)
    k_values = list(range(1,22,2))
    # Accuracy vs k for Euclidean
    predictor.plot_knn_accuracy_vs_k(k_values, distance="euclidean", save_path=os.path.join(PLOTS_DIR, "knn_accuracy_euclidean.png"))
    # get best k by validation accuracy
    accs = predictor.get_knn_accuracy_vs_k(k_values, distance="euclidean")
    best_k = k_values[int(np.argmax(accs))]
    print("Best k (validation, euclidean) =", best_k)
    # F1 heatmap for distances
    f1_df = predictor.get_knn_f1_heatmap(k_values, ["euclidean", "manhattan", "cosine"])
    predictor.plot_knn_f1_heatmap(f1_df, save_path=os.path.join(PLOTS_DIR, "knn_f1_heatmap.png"))
    print("F1 heatmap saved.")
    # Single feature table (on test set) - evaluate for each single feature
    try:
        single_feat_table = predictor.get_knn_f1_single_feature_table(k_values, features, distance="euclidean")
        single_feat_table.to_csv(os.path.join(PLOTS_DIR, "knn_single_feature_f1_table.csv"))
        print("Single feature F1 table saved.")
    except Exception as e:
        print("Warning: single-feature table generation encountered an issue:", e)

def run_q3():
    print("\n=== Q3 Polynomial Regression ===")
    # prepare features & target for GPA prediction
    df = pd.read_csv("student_dataset.csv")
    # Build simple feature matrix:
    # Numeric: GPA is target. For predictors use encoded major/program and maybe gender.
    X = pd.DataFrame()
    # One-hot encode categorical predictors
    X = pd.get_dummies(df[['Major','Program','Gender']], drop_first=True)
    y = df['GPA'].values
    # split
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    # Evaluate degrees 1..6 with no regularization, L1, L2 (we'll choose reg strengths by scanning)
    degrees = range(1,7)
    recs_none = evaluate_across_degrees(X_train.values, y_train, X_val.values, y_val, X_test.values, y_test, degrees=degrees, regularizer=None)
    plot_degree_vs_mse(recs_none, title_prefix="No Regularization", save_path=os.path.join(PLOTS_DIR, "degree_vs_mse_none.png"))
    # Tune for L2 (Ridge) for each degree and report best alpha
    best_by_degree_l2 = {}
    for d in degrees:
        best_alpha, best_mse, _ = tune_regularization_for_degree(X_train.values, y_train, X_val.values, y_val, degree=d, regularizer='l2')
        best_by_degree_l2[d] = (best_alpha, best_mse)
    print("Best Ridge alphas by degree:", best_by_degree_l2)
    # pick best degree according to val mse from recs_none or from L2 best_mse comparisons
    # For demonstration, pick degree with smallest validation MSE from recs_none
    best_degree = min(recs_none, key=lambda r: r['val_mse'])['degree']
    print("Best degree (no regularization by val MSE) =", best_degree)
    # For that degree, tune both L1 and L2
    best_alpha_l2, best_mse_l2, df_l2 = tune_regularization_for_degree(X_train.values, y_train, X_val.values, y_val, degree=best_degree, regularizer='l2')
    best_alpha_l1, best_mse_l1, df_l1 = tune_regularization_for_degree(X_train.values, y_train, X_val.values, y_val, degree=best_degree, regularizer='l1')
    print(f"Best L2 alpha for degree {best_degree}: {best_alpha_l2} (val mse {best_mse_l2})")
    print(f"Best L1 alpha for degree {best_degree}: {best_alpha_l1} (val mse {best_mse_l1})")
    # Fit final models for the best configurations and report test MSEs
    res_none = run_poly_regression(X_train.values, y_train, X_val.values, y_val, X_test.values, y_test, degree=best_degree, regularizer=None)
    res_l2 = run_poly_regression(X_train.values, y_train, X_val.values, y_val, X_test.values, y_test, degree=best_degree, regularizer='l2', reg_strength=best_alpha_l2)
    res_l1 = run_poly_regression(X_train.values, y_train, X_val.values, y_val, X_test.values, y_test, degree=best_degree, regularizer='l1', reg_strength=best_alpha_l1)
    print("Test MSEs -- None:", res_none['test_mse'], " L2:", res_l2['test_mse'], " L1:", res_l1['test_mse'])
    # Non-zero coefficients
    nz_l1 = coefficients_nonzero(res_l1['model'], res_l1['feature_names'])
    nz_l2 = coefficients_nonzero(res_l2['model'], res_l2['feature_names'])
    print("Non-zero coefficients (L1):", nz_l1[:10])
    print("Top coefficients (L2):", nz_l2[:10])
    print("Q3 done. Plots & tables saved under:", PLOTS_DIR)

if __name__ == "__main__":
    run_q1()
    run_q2()
    run_q3()
