# Install dependencies before running:
# pip install pandas matplotlib seaborn scikit-learn numpy

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    # 1. Load dataset with error handling
    csv_path = "smart_inventory_dataset.csv"
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print("--- Loaded Dataset ---")
    print(df.head(), "\n")

    # 2. Check and handle missing values
    print("--- Missing Values by Column ---")
    print(df.isnull().sum(), "\n")

    # Fill numeric NaNs with column medians
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 3. Flag items that need reorder
    df['Needs_Reorder'] = df['Stock_Quantity'] < df['Reorder_Level']

    # 4. Basic stats
    print("--- Statistical Summary ---")
    print(df.describe(), "\n")

    # 5. Visualizations
    product = df['Item_Name'].iloc[0] if 'Item_Name' in df.columns else 'Item'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    sns.lineplot(x='Sales_Last_30_Days', y='Stock_Quantity', data=df, ax=axes[0, 0])
    axes[0, 0].set_title("Sales vs Stock Quantity")
    axes[0, 0].set_xlabel("Sales Last 30 Days")
    axes[0, 0].set_ylabel("Stock Quantity")
    axes[0, 0].grid(True)

    axes[0, 1].hist(df['Stock_Quantity'], bins=15, color='skyblue', edgecolor='black')
    axes[0, 1].set_title("Stock Quantity Distribution")
    axes[0, 1].set_xlabel("Stock Quantity")
    axes[0, 1].set_ylabel("Frequency")

    sns.boxplot(y='Sales_Last_30_Days', data=df, ax=axes[1, 0])
    axes[1, 0].set_title("Sales Variability")
    axes[1, 0].set_ylabel("Sales Last 30 Days")

    sns.scatterplot(x='Unit_Price', y='Stock_Quantity', hue='Category', data=df, ax=axes[1, 1])
    axes[1, 1].set_title("Unit Price vs Stock Quantity")
    axes[1, 1].set_xlabel("Unit Price")
    axes[1, 1].set_ylabel("Stock Quantity")

    plt.show()

    # 6. Items needing reorder
    reorder_items = df[df['Needs_Reorder']]
    print("--- Items Needing Reorder ---")
    print(reorder_items[['Item_ID', 'Item_Name', 'Stock_Quantity', 'Reorder_Level']], "\n")

    # 7. Supplier inventory value
    supplier_value = (
        df.groupby('Supplier')
          .apply(lambda x: (x['Unit_Price'] * x['Stock_Quantity']).sum())
          .reset_index(name='Total_Inventory_Value')
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Supplier', y='Total_Inventory_Value', data=supplier_value)
    plt.title("Inventory Value by Supplier")
    plt.tight_layout()
    plt.show()

    # 8. Urgent restock items
    urgent = reorder_items.sort_values('Restock_Lead_Time_Days')
    print("--- Urgent Restock (Top 10) ---")
    print(
        urgent[
            ['Item_ID','Item_Name','Stock_Quantity','Reorder_Level','Restock_Lead_Time_Days']
        ].head(10),
        "\n"
    )

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Category', data=reorder_items)
    plt.title("Reorder Needs by Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ---------- AI/ML PIPELINE ----------
    # AI/ML is used below for sales prediction and reorder score calculation

    # 9. Feature engineering (AI/ML input features)
    features = ['Stock_Quantity', 'Reorder_Level', 'Unit_Price', 'Restock_Lead_Time_Days']
    target   = 'Sales_Last_30_Days'
    X        = df[features].values
    y        = df[target].values

    # 9A. Product Clustering (Fast/Medium/Slow Movers)
    if df['Sales_Last_30_Days'].nunique() > 3:
        cluster_features = df[['Sales_Last_30_Days']]
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Sales_Cluster'] = kmeans.fit_predict(cluster_features)

        cluster_means = df.groupby('Sales_Cluster')['Sales_Last_30_Days'].mean().sort_values()
        cluster_labels = {cluster: label for cluster, label in zip(cluster_means.index, ['Slow', 'Medium', 'Fast'])}
        df['Sales_Cluster_Label'] = df['Sales_Cluster'].map(cluster_labels)

        print("--- Product Clusters (Fast/Medium/Slow Movers) ---")
        print(df[['Item_ID', 'Item_Name', 'Sales_Last_30_Days', 'Sales_Cluster_Label']].head(), "\n")

        plt.figure(figsize=(8, 5))
        sns.countplot(x='Sales_Cluster_Label', data=df, order=['Fast', 'Medium', 'Slow'])
        plt.title("Product Movement Category Distribution")
        plt.xlabel("Movement Category")
        plt.ylabel("Number of Products")
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough variance in sales data to perform clustering.\n")

    # 10. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 11. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # 12. Model training & evaluation (ML model: Linear Regression)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Linear Regression MSE: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}\n")

    # 13. Predict across full dataset & compute reorder score (AI prediction)
    X_full = scaler.transform(df[features])
    df['Predicted_Sales_Next_30_Days'] = model.predict(X_full)

    df['Reorder_Score'] = (
        df['Predicted_Sales_Next_30_Days'] - df['Stock_Quantity']
    ) / df['Restock_Lead_Time_Days']

    df['Reorder_Score'].replace([np.inf, -np.inf], 0, inplace=True)
    df['Reorder_Score'].fillna(0, inplace=True)

    # 14. Top AI-based reorder recommendations
    recs = df[df['Reorder_Score'] > 0].sort_values(
        by='Reorder_Score', ascending=False
    )
    print("Top 10 AI-Based Reorder Recommendations ---")
    print(
        recs[
            ['Item_ID','Item_Name','Stock_Quantity',
             'Predicted_Sales_Next_30_Days','Restock_Lead_Time_Days','Reorder_Score']
        ].head(10),
        "\n"
    )

    # 15.Current vs Expected Sales Barplot
    plt.figure(figsize=(12, 6))
    subset = df[['Item_Name', 'Sales_Last_30_Days', 'Predicted_Sales_Next_30_Days']].sort_values(by='Sales_Last_30_Days', ascending=False).head(15)
    subset_melted = pd.melt(subset, id_vars='Item_Name', value_vars=['Sales_Last_30_Days', 'Predicted_Sales_Next_30_Days'],
                            var_name='Metric', value_name='Sales')
    sns.barplot(x='Item_Name', y='Sales', hue='Metric', data=subset_melted)
    plt.title("Current vs Predicted Sales for Top 15 Items")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 16. Save full results
    output_path = "inventory_predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"Analysis complete. Results saved to {output_path}\n")

if __name__ == "__main__":
    main()
