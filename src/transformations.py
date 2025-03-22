import pandas as pd
import numpy as np
from scipy.stats import normaltest
from sklearn.preprocessing import PowerTransformer
import os

def apply_transformations(data):
    """
    Dönüşümler: Box-Cox, Logaritmik ve Kareköklü
    Normality test: D'Agostino K²
    Sonuçlar: outputs/transformation_results.csv
    """

    results = []

    # Orijinal veri
    stat_orig, p_orig = normaltest(data)
    results.append({
        'Dönüşüm': 'Orijinal',
        'p-değeri': round(p_orig[0], 4),
        'Sonuç': 'Başarılı' if p_orig[0] > 0.05 else 'Başarısız'
    })

    # Box-Cox (negatif ya da sıfır değer varsa 1e-6 eklenir)
    boxcox_data = PowerTransformer(method='box-cox').fit_transform(data + 1e-6)
    _, p_boxcox = normaltest(boxcox_data)
    results.append({
        'Dönüşüm': 'Box-Cox',
        'p-değeri': round(p_boxcox[0], 4),
        'Sonuç': 'Başarılı' if p_boxcox[0] > 0.05 else 'Başarısız'
    })

    # Log dönüşümü
    log_data = np.log1p(data)
    _, p_log = normaltest(log_data)
    results.append({
        'Dönüşüm': 'Log',
        'p-değeri': round(p_log[0], 4),
        'Sonuç': 'Başarılı' if p_log[0] > 0.05 else 'Başarısız'
    })

    # Kareköklü dönüşüm
    sqrt_data = np.sqrt(data)
    _, p_sqrt = normaltest(sqrt_data)
    results.append({
        'Dönüşüm': 'Karekök',
        'p-değeri': round(p_sqrt[0], 4),
        'Sonuç': 'Başarılı' if p_sqrt[0] > 0.05 else 'Başarısız'
    })

    # Sonuçları kaydet
    df_results = pd.DataFrame(results)
    os.makedirs("outputs", exist_ok=True)
    df_results.to_csv("outputs/transformation_results.csv", index=False)
    print("Dönüşüm sonuçları outputs/transformation_results.csv dosyasına kaydedildi.")