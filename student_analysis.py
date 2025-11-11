# student_analysis.py
import os
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

class StudentDatasetAnalysis:
    def __init__(self, csv_path: str, plots_dir: str = "plots"):
        self.df = pd.read_csv(csv_path)
        # Normalise column names to consistent format used below
        self.df.columns = [c.strip().title() for c in self.df.columns]
        # Ensure expected columns exist
        expected = {"Gender", "Major", "Program", "Gpa"}
        if not expected.issubset(set(self.df.columns)):
            raise ValueError(f"CSV must contain columns: {expected}. Found: {self.df.columns.tolist()}")
        # Rename Gpa -> GPA for convenience
        if "Gpa" in self.df.columns:
            self.df.rename(columns={"Gpa": "GPA"}, inplace=True)
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)
        print(f"Loaded dataset with shape: {self.df.shape}")

    # -------------------------
    # Q1.1 Visualizations
    # -------------------------
    def _save_or_show(self, fig, name: str, save: bool = True):
        path = os.path.join(self.plots_dir, name)
        fig.tight_layout()
        if save:
            fig.savefig(path, dpi=150)
            plt.close(fig)
        else:
            plt.show()

    def plot_gender_distribution(self, save: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(6,4))
        order = self.df['Gender'].value_counts().index
        sns.countplot(x='Gender', data=self.df, order=order, ax=ax)
        ax.set_title("Gender Distribution")
        ax.set_ylabel("Count")
        self._save_or_show(fig, "gender_distribution.png", save)

    def plot_major_distribution(self, save: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(6,4))
        order = self.df['Major'].value_counts().index
        sns.countplot(x='Major', data=self.df, order=order, ax=ax)
        ax.set_title("Major Distribution")
        self._save_or_show(fig, "major_distribution.png", save)

    def plot_program_distribution(self, save: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(6,4))
        order = self.df['Program'].value_counts().index
        sns.countplot(x='Program', data=self.df, order=order, ax=ax)
        ax.set_title("Program Distribution")
        self._save_or_show(fig, "program_distribution.png", save)

    def plot_gpa_distribution(self, bins: int = 20, save: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(7,4))
        sns.histplot(self.df['GPA'], bins=bins, kde=True, ax=ax)
        ax.set_title("GPA Distribution")
        ax.set_xlabel("GPA")
        self._save_or_show(fig, "gpa_distribution.png", save)

    def plot_program_by_major(self, save: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.countplot(x='Major', hue='Program', data=self.df, ax=ax)
        ax.set_title("Program conditioned on Major")
        self._save_or_show(fig, "program_by_major.png", save)

    def plot_gpa_by_major(self, save: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(7,5))
        sns.boxplot(x='Major', y='GPA', data=self.df, ax=ax)
        ax.set_title("GPA conditioned on Major")
        self._save_or_show(fig, "gpa_by_major.png", save)

    def plot_gpa_by_program(self, save: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(7,5))
        sns.boxplot(x='Program', y='GPA', data=self.df, ax=ax)
        ax.set_title("GPA conditioned on Program")
        self._save_or_show(fig, "gpa_by_program.png", save)

    def plot_gpa_by_program_and_major(self, save: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(x='Program', y='GPA', hue='Major', data=self.df, ax=ax)
        ax.set_title("GPA conditioned on Program and Major")
        self._save_or_show(fig, "gpa_by_program_and_major.png", save)

    def plot_sampled_dataset(self, save: bool = True) -> None:
        sample_df = self.df.sample(100, random_state=42).reset_index(drop=True)
        # scatter with jitter on categorical axes
        fig, ax = plt.subplots(figsize=(10,6))
        # Convert categorical to numeric jitter positions for plotting
        majors = sample_df['Major'].astype('category').cat.codes
        programs = sample_df['Program'].astype('category').cat.codes
        # scatter: x by major index, hue by gender, size by program index
        sns.scatterplot(x=sample_df['Major'], y=sample_df['GPA'], hue=sample_df['Gender'],
                        style=sample_df['Program'], s=80, ax=ax)
        ax.set_title("Random sample of 100 students: Gender, Major, Program vs GPA")
        self._save_or_show(fig, "sampled_100.png", save)

    def plot_entire_dataset_summary(self, sample_size: int = 500, save: bool = True) -> None:
        # Pairplot on a sample (pairplot on full 10k is heavy)
        sample = self.df.sample(min(sample_size, len(self.df)), random_state=42)
        # Convert categorical to numeric for pairplot: encode major/program/gender temporarily
        temp = sample.copy()
        temp['Major_code'] = temp['Major'].astype('category').cat.codes
        temp['Program_code'] = temp['Program'].astype('category').cat.codes
        temp['Gender_code'] = temp['Gender'].astype('category').cat.codes
        cols = ['GPA', 'Major_code', 'Program_code', 'Gender_code']
        pairplot = sns.pairplot(temp[cols], diag_kind='kde')
        pairplot.fig.suptitle("Pairplot summary (sample)", y=1.02)
        outpath = os.path.join(self.plots_dir, "pairplot_summary.png")
        pairplot.savefig(outpath, dpi=150)
        plt.close()

    # -------------------------
    # Q1.1 (b) GPA stats
    # -------------------------
    def gpa_mean_std(self) -> Tuple[float, float]:
        mean = float(self.df['GPA'].mean())
        std = float(self.df['GPA'].std())
        print(f"GPA Mean: {mean:.4f}, GPA Std: {std:.4f}")
        return mean, std

    # -------------------------
    # Q1.1 (c) program-major pair counts & heatmap
    # -------------------------
    def count_students_per_program_major_pair(self) -> pd.DataFrame:
        counts = self.df.groupby(['Program', 'Major']).size().reset_index(name='Count')
        print(counts)
        return counts

    def visualize_students_per_program_major_pair(self, counts_df: pd.DataFrame, save: bool = True) -> None:
        pivot = counts_df.pivot(index='Program', columns='Major', values='Count')
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
        ax.set_title("Counts per (Program, Major)")
        self._save_or_show(fig, "program_major_heatmap.png", save)

    # -------------------------
    # Q1.2 Simple vs Stratified Sampling
    # -------------------------
    def get_gpa_mean_std_random(self, n: int = 500, repeats: int = 50, random_state: int = None) -> Tuple[float, float]:
        rng = np.random.RandomState(random_state)
        means = []
        for i in range(repeats):
            sample = self.df.sample(n, replace=False, random_state=rng.randint(0, 2**31-1))
            means.append(sample['GPA'].mean())
        avg_mean = float(np.mean(means))
        std_of_means = float(np.std(means, ddof=0))
        print(f"[Random] avg mean GPA: {avg_mean:.6f}, std of sample means: {std_of_means:.6f}")
        return avg_mean, std_of_means

    def get_gpa_mean_std_stratified(self, n: int = 500, repeats: int = 50, random_state: int = None) -> Tuple[float, float]:
        rng = np.random.RandomState(random_state)
        proportions = self.df['Major'].value_counts(normalize=True).to_dict()
        majors = list(proportions.keys())
        means = []
        for i in range(repeats):
            pieces = []
            for maj in majors:
                count = int(round(n * proportions[maj]))
                # ensure we don't sample more than available
                count = min(count, len(self.df[self.df['Major'] == maj]))
                pieces.append(self.df[self.df['Major'] == maj].sample(count, replace=False, random_state=rng.randint(0, 2**31-1)))
            sample = pd.concat(pieces).reset_index(drop=True)
            # if due to rounding sample size differs, adjust by sampling extra randomly
            if len(sample) < n:
                extra = self.df.drop(sample.index, errors='ignore').sample(n - len(sample), random_state=rng.randint(0, 2**31-1))
                sample = pd.concat([sample, extra])
            means.append(sample['GPA'].mean())
        avg_mean = float(np.mean(means))
        std_of_means = float(np.std(means, ddof=0))
        print(f"[Stratified] avg mean GPA: {avg_mean:.6f}, std of sample means: {std_of_means:.6f}")
        return avg_mean, std_of_means

    # -------------------------
    # Q1.3 Gender-Balanced Cohort
    # -------------------------
    def get_gender_balanced_counts(self, n: int = 300, repeats: int = 5) -> List[Dict[str,int]]:
        # exact same count across genders: if 3 genders -> each gets n/3 (floor), remainder distributed
        genders = sorted(self.df['Gender'].unique())
        k = len(genders)
        base = n // k
        results = []
        for _ in range(repeats):
            counts = {g: base for g in genders}
            rem = n - base * k
            # distribute remainder arbitrarily to first rem genders
            for i in range(rem):
                counts[genders[i]] += 1
            results.append(counts)
        print("Gender balanced counts (exact split):", results)
        return results

    def sample_gender_uniform_random(self, n: int = 300, repeats: int = 5, random_state: int = None) -> List[Dict[str,int]]:
        # Strategy A: pick a gender uniformly at random then pick a student from that gender
        rng = np.random.RandomState(random_state)
        genders = list(self.df['Gender'].unique())
        results = []
        for _ in range(repeats):
            sampled = []
            for i in range(n):
                gender_choice = rng.choice(genders)
                candidate = self.df[self.df['Gender'] == gender_choice].sample(1, random_state=rng.randint(0, 2**31-1))
                sampled.append(candidate.iloc[0])
            sampled_df = pd.DataFrame(sampled)
            counts = sampled_df['Gender'].value_counts().to_dict()
            # ensure all genders present in dict
            counts_full = {g: counts.get(g, 0) for g in genders}
            results.append(counts_full)
        print("Strategy A sampled gender counts:", results)
        return results

    def plot_avg_max_gender_diff_vs_sample_size(self, sample_sizes: List[int], repeats: int = 10, save: bool = True) -> None:
        genders = list(self.df['Gender'].unique())
        avg_max_rel_diffs = []
        rng = np.random.RandomState(42)
        for n in sample_sizes:
            max_rel_diffs = []
            for _ in range(repeats):
                # strategy A: sample n using "pick uniform gender then pick student from that gender"
                sampled = []
                for i in range(n):
                    gender_choice = rng.choice(genders)
                    candidate = self.df[self.df['Gender'] == gender_choice].sample(1, random_state=rng.randint(0, 2**31-1))
                    sampled.append(candidate.iloc[0])
                sampled_df = pd.DataFrame(sampled)
                counts = sampled_df['Gender'].value_counts().reindex(genders, fill_value=0)
                diff = counts.max() - counts.min()
                max_rel_diffs.append(diff / n)
            avg = float(np.mean(max_rel_diffs))
            avg_max_rel_diffs.append(avg)
        # plot histogram-like bar chart of avg_max_rel_diffs vs sample size
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(sample_sizes, avg_max_rel_diffs, marker='o')
        ax.set_xlabel("Sample size")
        ax.set_ylabel("Average max relative difference")
        ax.set_title("Avg max relative gender difference vs sample size (Strategy A)")
        self._save_or_show(fig, "avg_max_rel_diff_vs_sample_size.png", save)

    # -------------------------
    # Q1.4 GPA-Uniform Cohort
    # -------------------------
    def sample_gpa_uniform(self, n: int = 100, bins: int = 10, random_state: int = None) -> pd.DataFrame:
        rng = np.random.RandomState(random_state)
        # compute bins from entire dataset
        hist_vals, bin_edges = np.histogram(self.df['GPA'], bins=bins)
        # target 1 per bin = n / bins approximately
        per_bin = n // bins
        remainder = n - per_bin * bins
        sampled_rows = []
        for i in range(bins):
            left, right = bin_edges[i], bin_edges[i+1]
            candidates = self.df[(self.df['GPA'] >= left) & (self.df['GPA'] <= right)]
            desired = per_bin + (1 if i < remainder else 0)
            if len(candidates) == 0:
                continue
            # if not enough candidates, sample with replacement; else without
            replace = desired > len(candidates)
            chosen = candidates.sample(desired, replace=replace, random_state=rng.randint(0, 2**31-1))
            sampled_rows.append(chosen)
        sampled_df = pd.concat(sampled_rows).reset_index(drop=True)
        # If rounding causes mismatch in total, adjust by random sampling
        if len(sampled_df) < n:
            extra = self.df.sample(n - len(sampled_df), random_state=rng.randint(0, 2**31-1))
            sampled_df = pd.concat([sampled_df, extra]).reset_index(drop=True)
        elif len(sampled_df) > n:
            sampled_df = sampled_df.sample(n, random_state=rng.randint(0, 2**31-1)).reset_index(drop=True)
        return sampled_df

    def plot_gpa_histogram_comparison(self, sampled_df: pd.DataFrame, bins: int = 10, save: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(self.df['GPA'], bins=bins, label='Original', stat='density', kde=False, alpha=0.5, ax=ax)
        sns.histplot(sampled_df['GPA'], bins=bins, label='Sampled (GPA-uniform)', stat='density', kde=False, alpha=0.5, ax=ax)
        ax.legend()
        ax.set_title("GPA histogram: original vs GPA-uniform sampled")
        self._save_or_show(fig, "gpa_histogram_comparison.png", save)

    # -------------------------
    # Q1.5 Program-Major Balanced Cohort
    # -------------------------
    def sample_program_major_balanced(self, n: int = 60, random_state: int = None) -> pd.DataFrame:
        rng = np.random.RandomState(random_state)
        combos = self.df.groupby(['Program', 'Major']).size().reset_index(name='Count')
        valid_combos = list(zip(combos['Program'], combos['Major']))
        m = len(valid_combos)
        base = n // m
        remainder = n - base * m
        sampled_list = []
        # shuffle combos to distribute remainder
        order = list(range(m))
        rng.shuffle(order)
        for idx, combo_idx in enumerate(order):
            prog, maj = valid_combos[combo_idx]
            desired = base + (1 if idx < remainder else 0)
            candidates = self.df[(self.df['Program'] == prog) & (self.df['Major'] == maj)]
            replace = desired > len(candidates)
            chosen = candidates.sample(desired, replace=replace, random_state=rng.randint(0, 2**31-1))
            sampled_list.append(chosen)
        sampled_df = pd.concat(sampled_list).reset_index(drop=True)
        return sampled_df

    def show_program_major_counts_and_heatmap(self, sampled_df: pd.DataFrame, save: bool = True) -> None:
        counts = sampled_df.groupby(['Program','Major']).size().reset_index(name='Count')
        print("Sampled program-major counts:")
        print(counts)
        pivot = counts.pivot(index='Program', columns='Major', values='Count').fillna(0).astype(int)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.heatmap(pivot, annot=True, fmt='d', cmap='mako', ax=ax)
        ax.set_title("Sampled program-major counts")
        self._save_or_show(fig, "sampled_program_major_heatmap.png", save)

if __name__ == "__main__":
    # quick demo when run directly
    sa = StudentDatasetAnalysis("student_dataset.csv")
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
    sa.gpa_mean_std()
    counts = sa.count_students_per_program_major_pair()
    sa.visualize_students_per_program_major_pair(counts)
    sa.get_gpa_mean_std_random()
    sa.get_gpa_mean_std_stratified()
    sa.get_gender_balanced_counts()
    sa.sample_gender_uniform_random()
    sa.plot_avg_max_gender_diff_vs_sample_size([300,600,900,1200,1500])
    sampled = sa.sample_gpa_uniform()
    sa.plot_gpa_histogram_comparison(sampled)
    sampled_pm = sa.sample_program_major_balanced()
    sa.show_program_major_counts_and_heatmap(sampled_pm)
