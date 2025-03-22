import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import normaltest

from transformations import apply_transformations
from visualization import plot_correlation_heatmap

# Veri Setini Yükle
df = pd.read_csv("data/penguins.csv")

# Eksik Değerleri Doldur
df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
for col in ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']:
    df[col] = df[col].fillna(df[col].mean())
    
# Kategorik Değişkenleri Sayısala Çevir
df_encoded = pd.get_dummies(df, drop_first=True)

# Korelasyon Isı Haritası Oluştur
plot_correlation_heatmap(df_encoded)

# Basit Doğrusal Regresyon (Simple Linear Regression)
X_simple = df[['bill_length_mm']]
y_simple = df['body_mass_g']
simple_model = LinearRegression().fit(X_simple, y_simple)
print(f"Basit R²: {simple_model.score(X_simple, y_simple):.4f}")

# Çoklu Doğrusal Regresyon (Multiple Linear Regression)
X = df_encoded.drop('body_mass_g', axis=1)
y = df_encoded['body_mass_g']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

multi_model = LinearRegression().fit(X_train, y_train)
y_pred = multi_model.predict(X_test)
print(f"Çoklu R²: {r2_score(y_test, y_pred):.4f}")
print(f"Hata (MSE): {mean_squared_error(y_test, y_pred):.2f}")

# Normalizasyon + D’Agostino K² Testi
scaler = StandardScaler()
normalized = scaler.fit_transform(df[['body_mass_g']])
stat, p_value = normaltest(normalized)
p_val = p_value[0]
print(f"Normalize p-değeri: {p_val:.4f} → {'Başarılı' if p_val > 0.05 else 'Başarısız'}")

# Dönüşümleri Uygula ve Karşılaştır (Box-Cox, Log, Kareköklü)
apply_transformations(df[['body_mass_g']])