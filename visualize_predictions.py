import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_predictions(json_folder="output/predictions_json"):
    files = glob.glob(os.path.join(json_folder, "*.json"))

    if not files:
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
        return None

    df = pd.concat(df_list, ignore_index=True)
    df["window_start"] = pd.to_datetime(df["window_start"])
    df["high_severity_prob"] = df["probability"].apply(lambda x: x["values"][1])
    df = df.sort_values("window_start")

    return df


def plot_probability_histogram(df):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(df["high_severity_prob"], bins=20, color="steelblue", edgecolor="black")
    ax.axvline(x=0.5, color="red", linestyle="--", label="Prag odluke (0.5)")
    ax.set_title("Histogram verovatnoće visoke ozbiljnosti nesreća")
    ax.set_xlabel("Verovatnoća visoke ozbiljnosti")
    ax.set_ylabel("Broj prozora")
    ax.legend()

    plt.tight_layout()
    plt.savefig("output/probability_histogram.png", dpi=150)
    plt.show()


def plot_weather_analysis(df):
    weather_stats = df.groupby("Weather_Condition").agg({
        "prediction": "mean",
        "high_severity_prob": "mean",
        "accident_count": "sum"
    }).round(3)

    weather_stats.columns = ["Avg_Prediction", "Avg_Probability", "Total_Accidents"]
    weather_stats = weather_stats[weather_stats.index != "Clear"]
    weather_stats = weather_stats.sort_values("Avg_Probability", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.barh(weather_stats.index, weather_stats["Avg_Probability"], color="coral")
    ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Prosečna verovatnoća visoke ozbiljnosti")
    ax.set_title("Vremenski uslovi po ozbiljnosti nesreća")

    plt.tight_layout()
    plt.savefig("output/weather_risk_analysis.png", dpi=150)
    plt.show()


def plot_hourly_weekday_heatmap(df):
   
    df_copy = df.copy()
    df_copy["hour"] = df_copy["window_start"].dt.hour
    df_copy["weekday"] = df_copy["window_start"].dt.dayofweek

    pivot = df_copy.pivot_table(
        values="high_severity_prob",
        index="hour",
        columns="weekday",
        aggfunc="mean"
    )

    day_names = ["Pon", "Uto", "Sre", "Čet", "Pet", "Sub", "Ned"]
    pivot.columns = [day_names[i] for i in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{h:02d}:00" for h in pivot.index])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Prosečna verovatnoća visoke ozbiljnosti")

    ax.set_xlabel("Dan u nedelji")
    ax.set_ylabel("Sat")
    ax.set_title("Rizik ozbiljnih nesreća po satu i danu u nedelji")

    plt.tight_layout()
    plt.savefig("output/hourly_weekday_heatmap.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    df = load_predictions()

    if df is not None and not df.empty:
        plot_probability_histogram(df)
        plot_weather_analysis(df)
        plot_hourly_weekday_heatmap(df)

