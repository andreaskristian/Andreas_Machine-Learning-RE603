import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('dataset_karyawan_missing.csv')

# Handling Missing Values
df['Departemen'] = df['Departemen'].fillna(df['Departemen'].mode()[0])
df['Gaji'] = df['Gaji'].fillna(df['Gaji'].median())

# Plot 1: Distribusi Gaji Karyawan
plt.figure(figsize=(8, 5))
sns.histplot(df['Gaji'], kde=True, color='royalblue', bins=15)
plt.title('Distribusi Gaji Karyawan', fontsize=14)
plt.xlabel('Gaji', fontsize=12)
plt.ylabel('Frekuensi', fontsize=12)
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.show()

# Plot 2: Distribusi Gaji per Departemen
plt.figure(figsize=(8, 5))
sns.boxplot(x='Departemen', y='Gaji', data=df, palette='viridis')
plt.title('Distribusi Gaji per Departemen', fontsize=14)
plt.xlabel('Departemen', fontsize=12)
plt.ylabel('Gaji', fontsize=12)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()

# Plot 3: Distribusi Gaji berdasarkan Status Karyawan
plt.figure(figsize=(8, 5))
sns.boxplot(x='Status_Karyawan', y='Gaji', data=df, palette='Set2')
plt.title('Distribusi Gaji berdasarkan Status Karyawan', fontsize=14)
plt.xlabel('Status Karyawan', fontsize=12)
plt.ylabel('Gaji', fontsize=12)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()

# Plot 4: Heatmap Korelasi Fitur Numerik
plt.figure(figsize=(8, 6))
# Kita ubah kategori ke angka sementara khusus untuk melihat korelasinya di heatmap
df_heatmap = df.copy()
df_heatmap['Departemen_Enc'] = pd.factorize(df_heatmap['Departemen'])[0]
df_heatmap['Status_Enc'] = pd.factorize(df_heatmap['Status_Karyawan'])[0]

corr = df_heatmap[['Usia', 'Lama_Kerja', 'Departemen_Enc', 'Status_Enc', 'Gaji']].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f",
            linewidths=0.5, annot_kws={"size": 12})
plt.title('Heatmap Korelasi Variabel', fontsize=14)
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Feature Engineering (One-Hot Encoding)
df_encoded = pd.get_dummies(df.drop(['ID', 'Nama'], axis=1), drop_first=True)

X = df_encoded.drop('Gaji', axis=1)
y = df_encoded['Gaji']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Model
model_lr = LinearRegression()
model_rf = RandomForestRegressor(random_state=42)

model_lr.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# Prediksi dan Evaluasi
pred_lr = model_lr.predict(X_test)
pred_rf = model_rf.predict(X_test)

print(f"Linear Regression - R2: {r2_score(y_test, pred_lr):.4f}, MAE: {mean_absolute_error(y_test, pred_lr):.0f}")
print(f"Random Forest     - R2: {r2_score(y_test, pred_rf):.4f}, MAE: {mean_absolute_error(y_test, pred_rf):.0f}")

# Plot 5: Perbandingan Prediksi vs Aktual
plt.figure(figsize=(10, 5))
plt.scatter(y_test, pred_lr, alpha=0.6, label='Linear Regression', color='red', marker='x')
plt.scatter(y_test, pred_rf, alpha=0.6, label='Random Forest', color='green', marker='o')

# Garis Ideal
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', label='Prediksi Sempurna')

plt.title('Gaji Aktual vs Prediksi Model', fontsize=14)
plt.xlabel('Gaji Aktual')
plt.ylabel('Gaji Prediksi')
plt.legend()
plt.tight_layout()
plt.show()

