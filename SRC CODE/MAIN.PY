import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")

# ---------------------------------------
# LOAD DATA
# ---------------------------------------

original_path = "../DATA/archive/investments_VC.csv"
df = pd.read_csv(original_path, encoding="latin1")

# CLEAN COLUMN NAMES
df.columns = df.columns.str.strip()

# ---------------------------------------
# SELECT REQUIRED COLUMNS
# ---------------------------------------

columns_needed = [
    "category_list",
    "status",
    "country_code",
    "funding_rounds",
    "founded_year"
]

df = df[columns_needed]

# ---------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------

df = df.dropna(subset=["status", "funding_rounds", "founded_year"])

# ---------------------------------------
# CREATE SUCCESS VARIABLE
# ---------------------------------------

df["success"] = df["status"].apply(
    lambda x: 1 if x in ["acquired", "ipo"] else 0
)

# ---------------------------------------
# CREATE STARTUP AGE
# ---------------------------------------

current_year = 2026
df["startup_age"] = (current_year - df["founded_year"]).astype(int)

# ---------------------------------------
# CLEAN INDUSTRY COLUMN
# ---------------------------------------

df["category_list"] = (
    df["category_list"]
    .fillna("Unknown")
    .str.strip()
    .replace("", "Unknown")
)

df["primary_industry"] = (
    df["category_list"]
    .str.split("|")
    .str[0]   # FIXED: using first category (more logical)
    .fillna("Unknown")
)

# ---------------------------------------
# REMOVE EXTREME VALUES
# ---------------------------------------

df = df[df["startup_age"] < 60]

# ---------------------------------------
# SAVE CLEANED DATASET (IMPORTANT)
# ---------------------------------------

output_dir = os.path.dirname(original_path)
cleaned_file_path = os.path.join(output_dir, "cleaned_startups.csv")

df.to_csv(cleaned_file_path, index=False)

print(f"\n✅ Cleaned dataset saved at: {cleaned_file_path}")

# ---------------------------------------
# QUICK CHECK
# ---------------------------------------

print("\nTop Industries:")
print(df["primary_industry"].value_counts().head(10))

# ---------------------------------------
# EDA VISUALIZATIONS
# ---------------------------------------

os.makedirs("OUTPUT", exist_ok=True)

# 1. TOP INDUSTRIES
plt.figure(figsize=(10,6))
df["primary_industry"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Startup Industries")
plt.xlabel("Industry")
plt.ylabel("Number of Startups")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("OUTPUT/top_industries.png")
plt.close()

# 2. TOP COUNTRIES
plt.figure(figsize=(10,6))
df["country_code"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Countries by Number of Startups")
plt.xlabel("Country")
plt.ylabel("Number of Startups")
plt.tight_layout()
plt.savefig("OUTPUT/top_countries.png")
plt.close()

# 3. SUCCESS RATE BY INDUSTRY
industry_success = df.groupby("primary_industry")["success"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
industry_success.plot(kind="bar")
plt.title("Startup Success Rate by Industry")
plt.xlabel("Industry")
plt.ylabel("Success Rate")
plt.tight_layout()
plt.savefig("OUTPUT/success_by_industry.png")
plt.close()

# 4. SUCCESS RATE BY COUNTRY
country_success = df.groupby("country_code")["success"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
country_success.plot(kind="bar")
plt.title("Startup Success Rate by Country")
plt.xlabel("Country")
plt.ylabel("Success Rate")
plt.tight_layout()
plt.savefig("OUTPUT/success_by_country.png")
plt.close()

# 5. STARTUP FOUNDING TREND
plt.figure(figsize=(10,6))
df["founded_year"].value_counts().sort_index().plot()
plt.title("Startup Founding Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Startups")
plt.tight_layout()
plt.savefig("OUTPUT/startup_trends.png")
plt.close()

# 6. AGE VS SUCCESS
plt.figure(figsize=(10,6))
sns.boxplot(x="success", y="startup_age", data=df)
plt.title("Startup Age vs Success")
plt.xlabel("Success")
plt.ylabel("Startup Age")
plt.tight_layout()
plt.savefig("OUTPUT/age_vs_success.png")
plt.close()

# 7. FUNDING ROUNDS DISTRIBUTION
plt.figure(figsize=(10,6))
sns.histplot(df["funding_rounds"], bins=30)
plt.title("Distribution of Startup Funding Rounds")
plt.xlabel("Funding Rounds")
plt.ylabel("Number of Startups")
plt.tight_layout()
plt.savefig("OUTPUT/funding_round_distribution.png")
plt.close()

# 8. FUNDING VS SUCCESS
plt.figure(figsize=(10,6))
sns.boxplot(x="success", y="funding_rounds", data=df)
plt.title("Funding Rounds vs Startup Success")
plt.xlabel("Success")
plt.ylabel("Funding Rounds")
plt.tight_layout()
plt.savefig("OUTPUT/funding_vs_success.png")
plt.close()

# 9. STARTUP AGE DISTRIBUTION
plt.figure(figsize=(10,6))
sns.histplot(df["startup_age"], bins=30)
plt.title("Distribution of Startup Age")
plt.xlabel("Startup Age")
plt.ylabel("Number of Startups")
plt.tight_layout()
plt.savefig("OUTPUT/startup_age_distribution.png")
plt.close()

# 10. SUCCESSFUL STARTUPS BY COUNTRY
success_country = df[df["success"] == 1]["country_code"].value_counts().head(10)

plt.figure(figsize=(10,6))
success_country.plot(kind="bar")
plt.title("Top Countries by Successful Startups")
plt.xlabel("Country")
plt.ylabel("Number of Successful Startups")
plt.tight_layout()
plt.savefig("OUTPUT/successful_startups_country.png")
plt.close()

# ---------------------------------------
# MACHINE LEARNING
# ---------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

features = ["startup_age", "funding_rounds", "country_code", "primary_industry"]

X = pd.get_dummies(df[features], drop_first=True)
y = df["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# LOGISTIC REGRESSION
log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

# RANDOM FOREST
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# FEATURE IMPORTANCE
importances = pd.Series(rf_model.feature_importances_, index=X.columns)

print("\nTOP FACTORS INFLENCING STARTUP SUCCESS:")
print(importances.sort_values(ascending=False).head(10))