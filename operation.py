import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("smart_inventory_dataset.csv")

# Basic info and stats
print("--- Dataset Overview ---")
print(df.head())
print(df.describe())

# 1. Check for items below reorder level
df['Needs_Reorder'] = df['Stock_Quantity'] < df['Reorder_Level']

# 2. Plot category-wise stock distribution
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Stock_Quantity', data=df, estimator=sum)
plt.title("Total Stock Quantity by Category")
plt.ylabel("Total Quantity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Items that need to be reordered
reorder_items = df[df['Needs_Reorder'] == True]
print("\n--- Items that need Reorder ---")
print(reorder_items[['Item_ID', 'Item_Name', 'Stock_Quantity', 'Reorder_Level']])

# 4. Sales vs Stock Comparison
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Sales_Last_30_Days', y='Stock_Quantity', hue='Category', data=df)
plt.title("Sales vs Stock Quantity")
plt.xlabel("Sales in Last 30 Days")
plt.ylabel("Current Stock")
plt.tight_layout()
plt.show()

# 5. Supplier-wise inventory value
supplier_value = df.groupby('Supplier').apply(lambda x: (x['Unit_Price'] * x['Stock_Quantity']).sum()).reset_index(name='Total_Inventory_Value')
plt.figure(figsize=(8, 5))
sns.barplot(x='Supplier', y='Total_Inventory_Value', data=supplier_value)
plt.title("Inventory Value by Supplier")
plt.tight_layout()
plt.show()

# 6. Restock urgency: reorder items with short lead time
urgent_restock = reorder_items.sort_values(by='Restock_Lead_Time_Days')
print("\n--- Urgent Restock Items ---")
print(urgent_restock[['Item_ID', 'Item_Name', 'Stock_Quantity', 'Reorder_Level', 'Restock_Lead_Time_Days']].head(10))

# 7. Category-wise reorder needs
plt.figure(figsize=(10, 6))
sns.countplot(x='Category', data=reorder_items)
plt.title("Reorder Needs by Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nAnalysis Complete.")
