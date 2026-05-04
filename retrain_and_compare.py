# -*- coding: utf-8 -*-
"""
retrain_and_compare.py

Retrains the model using discriminatory samples
and compares SPD + accuracy before and after retraining.

Run after FairAgent has generated:
  - test_dataset_model_ready.csv
  - bank-full.csv (original dataset)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────
# STEP 1: Load and encode original dataset
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("  FAIRAGENT - Retrain and Compare")
print("=" * 60)

# Load original bank dataset
COLS = ['age','job','marital','education','default',
        'balance','housing','loan','contact','day',
        'month','duration','campaign','pdays',
        'previous','poutcome','y']

print("\n[1/6] Loading original dataset...")
df_raw = pd.read_csv('bank-small-biased.csv', sep=';', names=COLS)
print(f"  Original dataset: {df_raw.shape[0]:,} rows")

# Encode all categorical columns
encoders = {}
df_enc = df_raw.copy()
for col in df_enc.columns:
    if df_enc[col].dtype == object or str(df_enc[col].dtype) == 'str':
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

print(f"  Encoded successfully")
print(f"  Marital mapping: {dict(zip(encoders['marital'].classes_, encoders['marital'].transform(encoders['marital'].classes_)))}")
print(f"  y mapping:       {dict(zip(encoders['y'].classes_, encoders['y'].transform(encoders['y'].classes_)))}")

# ─────────────────────────────────────────────────────────────
# STEP 2: Train ORIGINAL model
# ─────────────────────────────────────────────────────────────

print("\n[2/6] Training ORIGINAL model...")

X = df_enc.drop(columns=['y'])
y = df_enc['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_original = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
model_original.fit(X_train, y_train)

y_pred_original = model_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)
print(f"  Original accuracy: {accuracy_original:.4f} ({accuracy_original*100:.2f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 3: Calculate SPD BEFORE retraining
# ─────────────────────────────────────────────────────────────

print("\n[3/6] Calculating SPD BEFORE retraining...")

def calculate_spd(model, X_data, df_with_marital, marital_col='marital'):
    """
    Calculate Statistical Parity Difference for marital status
    SPD = P(Y=1|privileged) - P(Y=1|unprivileged)
    """
    y_pred = model.predict(X_data)

    # Get marital values aligned with X_data index
    marital_vals = df_with_marital.loc[X_data.index, marital_col]

    # married=1 is privileged (most frequent)
    priv_mask   = (marital_vals == 1).values   # married
    unpriv_mask = (marital_vals == 2).values   # single

    priv_pred   = y_pred[priv_mask]
    unpriv_pred = y_pred[unpriv_mask]

    if len(priv_pred) == 0 or len(unpriv_pred) == 0:
        return 0.0

    spd = float(priv_pred.mean() - unpriv_pred.mean())
    return round(spd, 6)

def calculate_spd_full(model, X_full, marital_series):
    """Calculate SPD on full dataset"""
    y_pred = model.predict(X_full)
    priv_mask   = (marital_series == 1).values  # married
    unpriv_mask = (marital_series == 2).values  # single
    priv_pred   = y_pred[priv_mask]
    unpriv_pred = y_pred[unpriv_mask]
    if len(priv_pred) == 0 or len(unpriv_pred) == 0:
        return 0.0
    spd = float(priv_pred.mean() - unpriv_pred.mean())
    return round(spd, 6)

# SPD on test set before retraining
spd_before_test = calculate_spd(
    model_original, X_test, df_enc
)

# SPD on full dataset before retraining
spd_before_full = calculate_spd_full(
    model_original, X, df_enc['marital']
)

print(f"  SPD (test set) BEFORE:  {spd_before_test:+.6f}")
print(f"  SPD (full set) BEFORE:  {spd_before_full:+.6f}")
print(f"  Interpretation: {'Married customers get more positive outcomes' if spd_before_full > 0 else 'Single/divorced customers get more positive outcomes'}")

# ─────────────────────────────────────────────────────────────
# STEP 4: Load discriminatory samples and retrain
# ─────────────────────────────────────────────────────────────

print("\n[4/6] Loading discriminatory samples...")

try:
    disc_df = pd.read_csv(
        'fairagent_results/test_dataset_model_ready.csv'
    )
    print(f"  Discriminatory samples: {disc_df.shape[0]:,} rows")
    print(f"  Columns: {list(disc_df.columns)}")

    # Separate features and labels
    X_disc = disc_df.drop(columns=['y'])
    y_disc = disc_df['y']

    # Make sure columns match
    missing_cols = set(X_train.columns) - set(X_disc.columns)
    extra_cols   = set(X_disc.columns) - set(X_train.columns)

    if missing_cols:
        print(f"  Adding missing columns: {missing_cols}")
        for col in missing_cols:
            X_disc[col] = 0

    if extra_cols:
        print(f"  Removing extra columns: {extra_cols}")
        X_disc = X_disc.drop(columns=list(extra_cols))

    # Align column order
    X_disc = X_disc[X_train.columns]

    print(f"  Discriminatory X shape: {X_disc.shape}")
    print(f"  Discriminatory y distribution: {y_disc.value_counts().to_dict()}")

except FileNotFoundError:
    print("  ERROR: test_dataset_model_ready.csv not found!")
    print("  Make sure FairAgent has run first.")
    exit(1)

# ─────────────────────────────────────────────────────────────
# Combine original training data + discriminatory samples
# ─────────────────────────────────────────────────────────────

print("\n[5/6] Retraining model with discriminatory samples...")

# Combine original training data with discriminatory samples
X_train_new = pd.concat([X_train, X_disc], ignore_index=True)
y_train_new = pd.concat([y_train, y_disc], ignore_index=True)

print(f"  Original training size:  {len(X_train):,}")
print(f"  Discriminatory samples:  {len(X_disc):,}")
print(f"  New training size:       {len(X_train_new):,}")

# Retrain model
model_retrained = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
model_retrained.fit(X_train_new, y_train_new)

# Accuracy after retraining (on same test set)
y_pred_retrained = model_retrained.predict(X_test)
accuracy_retrained = accuracy_score(y_test, y_pred_retrained)
print(f"\n  Retrained accuracy: {accuracy_retrained:.4f} ({accuracy_retrained*100:.2f}%)")

# ─────────────────────────────────────────────────────────────
# STEP 5: Calculate SPD AFTER retraining
# ─────────────────────────────────────────────────────────────

# SPD on test set after retraining
spd_after_test = calculate_spd(
    model_retrained, X_test, df_enc
)

# SPD on full dataset after retraining
spd_after_full = calculate_spd_full(
    model_retrained, X, df_enc['marital']
)

print(f"  SPD (test set) AFTER:   {spd_after_test:+.6f}")
print(f"  SPD (full set) AFTER:   {spd_after_full:+.6f}")

# ─────────────────────────────────────────────────────────────
# STEP 6: Print comparison report
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  COMPARISON REPORT")
print("=" * 60)

acc_change = accuracy_retrained - accuracy_original
spd_change_test = spd_after_test - spd_before_test
spd_change_full = spd_after_full - spd_before_full

print(f"""
  ACCURACY:
  ─────────────────────────────────────────
  Before retraining : {accuracy_original:.4f} ({accuracy_original*100:.2f}%)
  After retraining  : {accuracy_retrained:.4f} ({accuracy_retrained*100:.2f}%)
  Change            : {acc_change:+.4f} ({acc_change*100:+.2f}%)
  {"[OK] Accuracy maintained" if abs(acc_change) < 0.01 else "[WARN] Accuracy changed significantly"}

  SPD (marital) - TEST SET:
  ─────────────────────────────────────────
  Before retraining : {spd_before_test:+.6f}
  After retraining  : {spd_after_test:+.6f}
  Change            : {spd_change_test:+.6f}
  {"[OK] Bias reduced" if abs(spd_after_test) < abs(spd_before_test) else "[WARN] Bias increased or unchanged"}

  SPD (marital) - FULL DATASET:
  ─────────────────────────────────────────
  Before retraining : {spd_before_full:+.6f}
  After retraining  : {spd_after_full:+.6f}
  Change            : {spd_change_full:+.6f}
  {"[OK] Bias reduced" if abs(spd_after_full) < abs(spd_before_full) else "[WARN] Bias increased or unchanged"}

  DISCRIMINATORY SAMPLES ADDED:
  ─────────────────────────────────────────
  Training set size before : {len(X_train):,}
  Discriminatory samples   : {len(X_disc):,}
  Training set size after  : {len(X_train_new):,}
  Increase                 : {len(X_disc)/len(X_train)*100:.1f}%
""")

# ─────────────────────────────────────────────────────────────
# STEP 6: Detailed classification report
# ─────────────────────────────────────────────────────────────

print("  CLASSIFICATION REPORT - BEFORE:")
print("  " + "-" * 40)
report_before = classification_report(
    y_test, y_pred_original,
    target_names=['No (0)', 'Yes (1)']
)
for line in report_before.split('\n'):
    print("  " + line)

print("  CLASSIFICATION REPORT - AFTER:")
print("  " + "-" * 40)
report_after = classification_report(
    y_test, y_pred_retrained,
    target_names=['No (0)', 'Yes (1)']
)
for line in report_after.split('\n'):
    print("  " + line)

# ─────────────────────────────────────────────────────────────
# STEP 7: Visualizations
# ─────────────────────────────────────────────────────────────

print("\n[6/6] Generating comparison charts...")

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    plt.style.use('ggplot')

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# ── Chart 1: Accuracy Before vs After ────────────────────────
ax1 = axes[0]
bars = ax1.bar(
    ['Before\nRetraining', 'After\nRetraining'],
    [accuracy_original * 100, accuracy_retrained * 100],
    color=['#3498DB', '#2ECC71'],
    width=0.4,
    alpha=0.85,
    edgecolor='black',
    linewidth=0.8
)
for bar, val in zip(bars, [accuracy_original, accuracy_retrained]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.2,
        f'{val*100:.2f}%',
        ha='center', va='bottom',
        fontsize=12, fontweight='bold'
    )
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title(
    'Model Accuracy\nBefore vs After Retraining',
    fontsize=13, fontweight='bold'
)
ax1.set_ylim(
    min(accuracy_original, accuracy_retrained) * 100 - 2,
    max(accuracy_original, accuracy_retrained) * 100 + 3
)
# Add change annotation
ax1.annotate(
    f'Change: {acc_change*100:+.2f}%',
    xy=(0.5, 0.05),
    xycoords='axes fraction',
    ha='center',
    fontsize=11,
    color='green' if acc_change >= 0 else 'red',
    fontweight='bold'
)

# ── Chart 2: SPD Before vs After ─────────────────────────────
ax2 = axes[1]
spd_values = [spd_before_test, spd_after_test]
colors_spd = [
    '#E74C3C' if abs(v) > 0.01 else '#2ECC71'
    for v in spd_values
]
bars2 = ax2.bar(
    ['Before\nRetraining', 'After\nRetraining'],
    spd_values,
    color=colors_spd,
    width=0.4,
    alpha=0.85,
    edgecolor='black',
    linewidth=0.8
)
for bar, val in zip(bars2, spd_values):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        val + (0.0005 if val >= 0 else -0.001),
        f'{val:+.4f}',
        ha='center',
        va='bottom' if val >= 0 else 'top',
        fontsize=12, fontweight='bold'
    )
ax2.axhline(y=0, color='green', linestyle='-',
            linewidth=2, label='Perfect fairness (0)')
ax2.axhline(y=0.1, color='orange', linestyle='--',
            linewidth=1.5, label='Bias threshold (+0.1)')
ax2.axhline(y=-0.1, color='orange', linestyle='--',
            linewidth=1.5)
ax2.set_ylabel('SPD Value (closer to 0 = fairer)', fontsize=12)
ax2.set_title(
    'SPD (Marital)\nBefore vs After Retraining',
    fontsize=13, fontweight='bold'
)
ax2.legend(fontsize=9)
ax2.annotate(
    f'Change: {spd_change_test:+.6f}',
    xy=(0.5, 0.05),
    xycoords='axes fraction',
    ha='center',
    fontsize=11,
    color='green' if abs(spd_after_test) < abs(spd_before_test)
          else 'red',
    fontweight='bold'
)

# ── Chart 3: Combined summary ─────────────────────────────────
ax3 = axes[2]
metrics_labels  = [
    'Accuracy\nBefore (%)',
    'Accuracy\nAfter (%)',
    'SPD Before\n(x100)',
    'SPD After\n(x100)'
]
metrics_values  = [
    accuracy_original * 100,
    accuracy_retrained * 100,
    spd_before_test * 100,
    spd_after_test * 100
]
colors3 = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']
bars3   = ax3.bar(
    metrics_labels, metrics_values,
    color=colors3, alpha=0.85,
    edgecolor='black', linewidth=0.8
)
for bar, val in zip(bars3, metrics_values):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f'{val:.2f}',
        ha='center', va='bottom',
        fontsize=10, fontweight='bold'
    )
ax3.set_ylabel('Value', fontsize=12)
ax3.set_title(
    'Combined Comparison\n(All Metrics)',
    fontsize=13, fontweight='bold'
)

fig.suptitle(
    'FairAgent: Model Retraining Impact Analysis\n'
    'Fairness vs Accuracy Tradeoff',
    fontsize=15, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig(
    'fairagent_results/retraining_comparison.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("  [OK] fairagent_results/retraining_comparison.png")

# ─────────────────────────────────────────────────────────────
# Save text report
# ─────────────────────────────────────────────────────────────

report_path = 'fairagent_results/retraining_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("  FAIRAGENT - RETRAINING COMPARISON REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write("SETUP\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Original training size   : {len(X_train):,}\n")
    f.write(f"  Discriminatory samples   : {len(X_disc):,}\n")
    f.write(f"  New training size        : {len(X_train_new):,}\n")
    f.write(f"  Test size                : {len(X_test):,}\n\n")

    f.write("ACCURACY\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Before : {accuracy_original:.4f} ({accuracy_original*100:.2f}%)\n")
    f.write(f"  After  : {accuracy_retrained:.4f} ({accuracy_retrained*100:.2f}%)\n")
    f.write(f"  Change : {acc_change:+.4f} ({acc_change*100:+.2f}%)\n\n")

    f.write("SPD - MARITAL (TEST SET)\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Before : {spd_before_test:+.6f}\n")
    f.write(f"  After  : {spd_after_test:+.6f}\n")
    f.write(f"  Change : {spd_change_test:+.6f}\n")
    f.write(
        f"  Result : {'Bias REDUCED' if abs(spd_after_test) < abs(spd_before_test) else 'Bias INCREASED or UNCHANGED'}\n\n"
    )

    f.write("SPD - MARITAL (FULL DATASET)\n")
    f.write("-" * 40 + "\n")
    f.write(f"  Before : {spd_before_full:+.6f}\n")
    f.write(f"  After  : {spd_after_full:+.6f}\n")
    f.write(f"  Change : {spd_change_full:+.6f}\n")
    f.write(
        f"  Result : {'Bias REDUCED' if abs(spd_after_full) < abs(spd_before_full) else 'Bias INCREASED or UNCHANGED'}\n\n"
    )

    f.write("INTERPRETATION\n")
    f.write("-" * 40 + "\n")
    if abs(spd_after_test) < abs(spd_before_test):
        f.write(
            "  Retraining with discriminatory samples\n"
            "  REDUCED marital bias (SPD closer to 0)\n"
        )
    else:
        f.write(
            "  Retraining did not reduce marital bias.\n"
            "  Consider fairness constraints during training.\n"
        )

    if abs(acc_change) < 0.01:
        f.write(
            "  Accuracy was maintained within 1%.\n"
            "  Low fairness-accuracy tradeoff observed.\n"
        )
    else:
        f.write(
            f"  Accuracy changed by {acc_change*100:+.2f}%.\n"
            "  Fairness-accuracy tradeoff observed.\n"
        )

print(f"  [OK] {report_path}")

print("\n" + "=" * 60)
print("  [OK] Retraining analysis complete!")
print("  Outputs saved to: fairagent_results/")
print("    - retraining_comparison.png")
print("    - retraining_report.txt")
print("=" * 60)