"""
Reading Patterns in Native vs. Heritage Speakers
Programming in Python Final Project

Student: Efsun Dicle DUTAR
Instructor: Ali ASGARI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -----------------------------
# Data Simulation
# -----------------------------

np.random.seed(42)

n = 100

data = pd.DataFrame({
    "Participant_ID": range(1, n + 1),
    "Group": np.random.choice(["Native", "Heritage"], n)
})

data["Fixation_Duration"] = np.where(
    data["Group"] == "Heritage",
    np.random.normal(260, 40, n),
    np.random.normal(210, 30, n)
)

data["Reading_Time"] = np.where(
    data["Group"] == "Heritage",
    np.random.normal(1550, 200, n),
    np.random.normal(1350, 180, n)
)

# -----------------------------
# Data Cleaning
# -----------------------------

print("Missing Values")
print(data.isnull().sum())

print("\nDescriptive Statistics")
print(data.describe())

print("\nGroup Means")
print(data.groupby("Group")[["Fixation_Duration",
                             "Reading_Time"]].mean())

# Save dataset

data.to_csv(
    "simulated_meco_reading_data.csv",
    index=False
)

# -----------------------------
# Visualization 1
# -----------------------------

sns.boxplot(
    x="Group",
    y="Fixation_Duration",
    data=data
)

plt.title("Fixation Duration by Group")
plt.xlabel("Group")
plt.ylabel("Milliseconds")
plt.show()

# -----------------------------
# Visualization 2
# -----------------------------

sns.histplot(
    data=data,
    x="Reading_Time",
    hue="Group",
    kde=True
)

plt.title("Reading Time Distribution")
plt.xlabel("Milliseconds")
plt.show()

# -----------------------------
# Visualization 3
# -----------------------------

sns.boxplot(
    x="Group",
    y="Reading_Time",
    data=data
)

plt.title("Reading Time by Group")
plt.xlabel("Group")
plt.ylabel("Milliseconds")
plt.show()

# -----------------------------
# Independent Samples t-test
# -----------------------------

native = data[
    data["Group"] == "Native"
]["Reading_Time"]

heritage = data[
    data["Group"] == "Heritage"
]["Reading_Time"]

t_stat, p_value = stats.ttest_ind(
    native,
    heritage
)

print("\nIndependent Samples t-test")
print("T statistic:", round(t_stat, 3))
print("P value:", p_value)

if p_value < 0.05:
    print("Result: Significant difference between groups.")
else:
    print("Result: No significant difference between groups.")
print("\nSummary")

print(f"Number of participants: {len(data)}")

print(f"Native speakers: {len(native)}")

print(f"Heritage speakers: {len(heritage)}")

print(f"T statistic: {t_stat:.3f}")

print(f"P value: {p_value:.6f}")
