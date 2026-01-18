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
    ax.set_title("Distribucija verovatnoća visoke ozbiljnosti")
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


def plot_accidents_over_time(df):
    """Graf 3: Nesreće kroz vreme"""
    df_copy = df.copy()
    df_copy["window_start"] = pd.to_datetime(df_copy["window_start"], utc=True)
    df_copy = df_copy.set_index("window_start")
    df_daily = df_copy.resample("D").agg({
        "accident_count": "sum",
        "high_severity_prob": "mean",
        "prediction": "mean"
    }).dropna()

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.bar(df_daily.index, df_daily["accident_count"], alpha=0.6, color="steelblue", label="Broj nesreća")
    ax1.set_ylabel("Broj nesreća", color="steelblue")
    ax1.set_xlabel("Datum")

    ax2 = ax1.twinx()
    ax2.plot(df_daily.index, df_daily["high_severity_prob"], color="red", linewidth=2, label="Prosečan rizik")
    ax2.set_ylabel("Prosečna verovatnoća visoke ozbiljnosti", color="red")
    ax2.set_ylim(0, 1)

    plt.title("Broj nesreća i rizik kroz vreme")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.tight_layout()
    plt.savefig("output/accidents_over_time.png", dpi=150)
    plt.show()


def plot_feature_correlation(df):
    """Graf 4: Korelacija feature-a sa predikcijom"""
    numeric_cols = ["Visibility_mi", "Wind_Speed_mph", "Temperature_F",
                    "Humidity_percent", "Pressure_in", "accident_count",
                    "high_severity_prob"]

    available_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[available_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(len(available_cols)))
    ax.set_yticks(range(len(available_cols)))
    ax.set_xticklabels(available_cols, rotation=45, ha="right")
    ax.set_yticklabels(available_cols)

    for i in range(len(available_cols)):
        for j in range(len(available_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, label="Korelacija")
    ax.set_title("Korelaciona matrica")
    plt.tight_layout()
    plt.savefig("output/correlation_matrix.png", dpi=150)
    plt.show()


def plot_day_night_comparison(df):
    """Graf 5: Poređenje Dan vs Noć"""
    if "Sunrise_Sunset" not in df.columns:
        return

    day_night = df.groupby("Sunrise_Sunset").agg({
        "prediction": "mean",
        "high_severity_prob": "mean",
        "accident_count": "sum"
    }).round(3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Graf 1: Prosečna verovatnoća
    axes[0].bar(day_night.index, day_night["high_severity_prob"], color=["gold", "darkblue"])
    axes[0].set_ylabel("Prosečna verovatnoća visoke ozbiljnosti")
    axes[0].set_title("Rizik po dobu dana")
    axes[0].set_ylim(0, 1)

    # Graf 2: Ukupan broj nesreća
    axes[1].bar(day_night.index, day_night["accident_count"], color=["gold", "darkblue"])
    axes[1].set_ylabel("Ukupan broj nesreća")
    axes[1].set_title("Broj nesreća po dobu dana")

    plt.tight_layout()
    plt.savefig("output/day_night_analysis.png", dpi=150)
    plt.show()


def plot_temperature_vs_severity(df):
    """Graf 6: Temperatura vs Ozbiljnost nesreća"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = df["prediction"].map({0: "green", 1: "red"})
    scatter = ax.scatter(df["Temperature_F"], df["high_severity_prob"],
                         c=colors, alpha=0.5, s=df["accident_count"]*5)

    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.5, label="Prag odluke")
    ax.set_xlabel("Temperatura (°F)")
    ax.set_ylabel("Verovatnoća visoke ozbiljnosti")
    ax.set_title("Temperatura vs Verovatnoća ozbiljnih nesreća\n(veličina = broj nesreća)")

    # Legenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Niska ozbiljnost'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Visoka ozbiljnost')
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig("output/temperature_vs_severity.png", dpi=150)
    plt.show()


def plot_visibility_analysis(df):
    """Graf 7: Analiza vidljivosti"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Graf 1: Box plot vidljivosti po predikciji
    low_sev = df[df["prediction"] == 0]["Visibility_mi"]
    high_sev = df[df["prediction"] == 1]["Visibility_mi"]

    axes[0].boxplot([low_sev, high_sev], labels=["Niska ozbiljnost", "Visoka ozbiljnost"])
    axes[0].set_ylabel("Vidljivost (milje)")
    axes[0].set_title("Vidljivost po ozbiljnosti nesreća")

    # Graf 2: Vidljivost kroz vreme
    df_copy = df.copy()
    df_copy["window_start"] = pd.to_datetime(df_copy["window_start"], utc=True)
    df_daily = df_copy.set_index("window_start").resample("D").agg({
        "Visibility_mi": "mean",
        "high_severity_prob": "mean"
    }).dropna()

    ax2 = axes[1]
    ax2.plot(df_daily.index, df_daily["Visibility_mi"], color="blue", label="Vidljivost")
    ax2.set_ylabel("Prosečna vidljivost (milje)", color="blue")
    ax2.set_xlabel("Datum")

    ax3 = ax2.twinx()
    ax3.plot(df_daily.index, df_daily["high_severity_prob"], color="red", alpha=0.7, label="Rizik")
    ax3.set_ylabel("Prosečan rizik", color="red")
    ax3.set_ylim(0, 1)

    axes[1].set_title("Vidljivost i rizik kroz vreme")

    plt.tight_layout()
    plt.savefig("output/visibility_analysis.png", dpi=150)
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

        print("[3/7] Nesreće kroz vreme...")
        plot_accidents_over_time(df)

        print("[4/7] Korelaciona matrica...")
        plot_feature_correlation(df)

        print("[5/7] Dan vs Noć analiza...")
        plot_day_night_comparison(df)

        print("[6/7] Temperatura vs Ozbiljnost...")
        plot_temperature_vs_severity(df)

        print("[7/7] Analiza vidljivosti...")
        plot_visibility_analysis(df)

        print("\nSvi grafici sačuvani u output/ folderu!")
