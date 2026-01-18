import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import folium

def visualize_predictions(json_folder="output/predictions_json"):
    files = glob.glob(f"{json_folder}/*.json")
    if not files:
        print("Nema JSON fajlova za vizualizaciju.")
        return
    
    df_list = [pd.read_json(f, lines=True) for f in files]
    df = pd.concat(df_list)

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Start_Lng, df.Start_Lat))

    gdf.plot(column='prediction', cmap='coolwarm', legend=True, figsize=(12,8))
    plt.title("Predikcija nezgoda")
    plt.show()

    m = folium.Map(location=[df.Start_Lat.mean(), df.Start_Lng.mean()], zoom_start=6)
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row.Start_Lat, row.Start_Lng],
            radius=3,
            color='red' if row.prediction==1 else 'blue',
            fill=True
        ).add_to(m)
    m.save("output/map.html")
    print("Interaktivna karta sačuvana u output/map.html")

if __name__ == "__main__":
    visualize_predictions()
