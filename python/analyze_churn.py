
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "telecom_customers.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDuplicate customer IDs:", df["customer_id"].duplicated().sum())


df = df.drop_duplicates(subset=["customer_id"]).copy()
df["monthly_charges"] = df["monthly_charges"].fillna(df["monthly_charges"].median())
df["tech_support"] = df["tech_support"].fillna("Unknown")
df["payment_method"] = df["payment_method"].fillna("Unknown")
df["churn_flag"] = df["churn_flag"].fillna(0).astype(int)


df["tenure_band"] = pd.cut(
    df["tenure_months"], bins=[0, 6, 12, 24, 48, 72],
    labels=["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49-72 months"],
    include_lowest=True
)
df["charge_band"] = pd.cut(
    df["monthly_charges"], bins=[0, 40, 70, 100, np.inf],
    labels=["Low", "Medium", "High", "Very High"], include_lowest=True
)
df["estimated_annual_revenue"] = df["monthly_charges"] * 12
df["is_month_to_month"] = np.where(df["contract_type"] == "Month-to-month", 1, 0)


total_customers = len(df)
churned = int(df["churn_flag"].sum())
churn_rate = df["churn_flag"].mean()
avg_monthly_charges = df["monthly_charges"].mean()
print("\n=== Executive KPIs ===")
print(f"Customers: {total_customers:,}")
print(f"Churned customers: {churned:,}")
print(f"Churn rate: {churn_rate:.1%}")
print(f"Average monthly charges: {avg_monthly_charges:.2f}")


segment_cols = ["contract_type", "tenure_band", "tech_support", "internet_service",
                "payment_method", "charge_band", "late_payment_flag"]
for col in segment_cols:
    summary = (df.groupby(col, observed=False)
                 .agg(customers=("customer_id", "count"),
                      churn_rate=("churn_flag", "mean"),
                      avg_monthly_charges=("monthly_charges", "mean"))
                 .reset_index()
                 .sort_values("churn_rate", ascending=False))
    summary.to_csv(OUT / f"churn_by_{col}.csv", index=False)
    print(f"\nChurn by {col}:\n", summary.to_string(index=False))


numeric = df[["tenure_months", "monthly_charges", "complaints_last_6m",
              "late_payment_flag", "churn_flag"]]
numeric.corr(numeric_only=True).to_csv(OUT / "numeric_correlation.csv")


def save_bar(data, x, y, title, filename, ylabel):
    plt.figure(figsize=(10, 5))
    plt.bar(data[x].astype(str), data[y])
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=160)
    plt.close()

contract_summary = df.groupby("contract_type", as_index=False).agg(
    customers=("customer_id", "count"), churn_rate=("churn_flag", "mean"))
save_bar(contract_summary, "contract_type", "churn_rate",
         "Churn Rate by Contract Type", "churn_by_contract.png", "Churn rate")

tenure_summary = df.groupby("tenure_band", observed=False, as_index=False).agg(
    customers=("customer_id", "count"), churn_rate=("churn_flag", "mean"))
save_bar(tenure_summary, "tenure_band", "churn_rate",
         "Churn Rate by Customer Tenure", "churn_by_tenure.png", "Churn rate")

support_summary = df.groupby("tech_support", as_index=False).agg(
    customers=("customer_id", "count"), churn_rate=("churn_flag", "mean"))
save_bar(support_summary, "tech_support", "churn_rate",
         "Churn Rate by Tech Support", "churn_by_support.png", "Churn rate")

plt.figure(figsize=(9, 5))
plt.scatter(df["tenure_months"], df["monthly_charges"],
            c=df["churn_flag"], alpha=0.45)
plt.title("Tenure vs Monthly Charges (colour indicates churn flag)")
plt.xlabel("Tenure (months)")
plt.ylabel("Monthly charges")
plt.tight_layout()
plt.savefig(OUT / "tenure_vs_charges.png", dpi=160)
plt.close()


month_to_month = df[df["contract_type"] == "Month-to-month"]
m2m_rate = month_to_month["churn_flag"].mean() if len(month_to_month) else 0
target_relative_reduction = 0.10
potential_saved = int(round(month_to_month["churn_flag"].sum() * target_relative_reduction))
scenario = pd.DataFrame([{
    "month_to_month_customers": len(month_to_month),
    "month_to_month_churn_rate": m2m_rate,
    "assumed_relative_churn_reduction": target_relative_reduction,
    "illustrative_customers_retained": potential_saved,
    "note": "Scenario only; validate with an experiment before claiming impact."
}])
scenario.to_csv(OUT / "retention_scenario.csv", index=False)
print("\nIllustrative retention scenario:\n", scenario.to_string(index=False))

df.to_csv(OUT / "telecom_customers_clean.csv", index=False)
print(f"\nCompleted. Outputs saved to: {OUT}")
