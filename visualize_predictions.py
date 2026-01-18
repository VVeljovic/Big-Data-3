import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_predictions(json_folder="output/predictions_json"):
    """Učitava sve JSON predikcije u DataFrame"""
    files = glob.glob(os.path.join(json_folder, "*.json"))

    if not files:
        print("Nema JSON fajlova.")
        return None

    df_list = []
    for f in files:
        if os.path.getsize(f) == 0:
            continue
        try:
            tmp = pd.read_json(f, lines=True)
            if not tmp.empty:
                df_list.append(tmp)
        except (ValueError, KeyError):
            continue

    if not df_list:
        print("Nema validnih podataka za prikaz.")
        return None

    df = pd.concat(df_list, ignore_index=True)
    df["window_start"] = pd.to_datetime(df["window_start"])
    df["high_severity_prob"] = df["probability"].apply(lambda x: x["values"][1])
    df = df.sort_values("window_start")

    return df


def plot_probability_histogram(df):
    """Graf 1: Histogram verovatnoća visoke ozbiljnosti"""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(df["high_severity_prob"], bins=20, color="steelblue", edgecolor="black")
    ax.axvline(x=0.5, color="red", linestyle="--", label="Prag odluke (0.5)")
    ax.set_xlabel("Verovatnoća visoke ozbiljnosti")
    ax.set_ylabel("Broj prozora")
    ax.legend()

    plt.tight_layout()
    plt.savefig("output/probability_histogram.png", dpi=150)
    plt.show()


def plot_weather_analysis(df):
    """Graf 2: Analiza po vremenskim uslovima"""
    weather_stats = df.groupby("Weather_Condition").agg({
        "prediction": "mean",
        "high_severity_prob": "mean",
        "accident_count": "sum"
    }).round(3)

    weather_stats.columns = ["Avg_Prediction", "Avg_Probability", "Total_Accidents"]
    weather_stats = weather_stats.sort_values("Avg_Probability", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.barh(weather_stats.index, weather_stats["Avg_Probability"], color="coral")
    ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Prosečna verovatnoća visoke ozbiljnosti")
    ax.set_title("Top 15 vremenskih uslova po riziku ozbiljnih nesreća")

    plt.tight_layout()
    plt.savefig("output/weather_risk_analysis.png", dpi=150)
    plt.show()


def print_summary(df):
    """Ispisuje sumarnu statistiku"""
    print("\n" + "="*50)
    print("SUMARNA STATISTIKA PREDIKCIJA")
    print("="*50)
    print(f"Ukupno prozora: {len(df)}")
    print(f"Ukupno nesreća: {df['accident_count'].sum()}")
    print(f"\nPredikcije:")
    print(f"  - Niska ozbiljnost: {(df['prediction'] == 0).sum()} ({(df['prediction'] == 0).mean()*100:.1f}%)")
    print(f"  - Visoka ozbiljnost: {(df['prediction'] == 1).sum()} ({(df['prediction'] == 1).mean()*100:.1f}%)")
    print(f"\nProsečna verovatnoća visoke ozbiljnosti: {df['high_severity_prob'].mean():.3f}")
    print(f"Min: {df['high_severity_prob'].min():.3f}, Max: {df['high_severity_prob'].max():.3f}")
    print("="*50)


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    df = load_predictions()

    if df is not None and not df.empty:
        print_summary(df)

        print("\n[1/7] Histogram verovatnoća...")
        plot_probability_histogram(df)

        print("[2/7] Analiza vremenskih uslova...")
        plot_weather_analysis(df)

