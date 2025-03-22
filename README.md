# Penguin ML Insights

Bu proje, Makine Öğrenmesi dersi kapsamında işlenen temel konuların uygulamalı olarak gösterilmesi amacıyla hazırlanmıştır. Palmer Penguins veri seti kullanılarak farklı veri işleme teknikleri, regresyon modelleri ve dağılım analizleri gerçekleştirilmiştir.

---

## Kapsanan Konular

- Eksik verilerin tespiti ve giderilmesi  
- Kategorik verilerin sayısal forma dönüştürülmesi  
- Korelasyon analizi ve görselleştirilmesi  
- Simple Linear Regression  
- Multiple Linear Regression  
- Modelin test edilmesi (train/test split, R² ve MSE hesaplamaları)  
- Normalizasyon (StandardScaler)  
- D’Agostino K² testi ile normal dağılım kontrolü  
- Box-Cox, Log ve Kareköklü dönüşümlerin karşılaştırılması

---

## Klasör Yapısı
penguin-ml-insights/
│ 
├── data/ # Veri dosyası 
│      └── penguins.csv │ 
├── src/ # Kaynak kodlar 
│   ├── ml_assignment.py # Ana çalışma dosyası 
│   ├── transformations.py # Dönüşüm fonksiyonlarıve normality testleri 
│   └── visualization.py # Korelasyon grafiği üretimi 
│ 
├── outputs/ # Üretilen görsel ve tablolar 
│   ├── correlation_heatmap.png 
│   └── transformation_results.csv 
│ 
└── README.md # Proje açıklama dosyası

---

## Çalıştırma Talimatları

1. `penguins.csv` dosyasını `data/` klasörüne yerleştirin.  
2. Gerekli kütüphanelerin kurulu olduğundan emin olun.  
3. Aşağıdaki komutla proje başlatılabilir:

```bash
python src/ml_assignment.py
```

## Gereksinimler

Aşağıdaki Python kütüphaneleri gereklidir:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`

### Kurulum

Gerekli kütüphaneleri yüklemek için aşağıdaki komutu çalıştırın:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
