import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_correlation_heatmap(df):
    """
    Sayısal sütunlar üzerinden korelasyon matrisi oluşturur ve çıktıyı PNG olarak kaydeder.
    """

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Korelasyon Matrisi")

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print("Korelasyon matrisi outputs/correlation_heatmap.png olarak kaydedildi.")