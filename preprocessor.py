import pandas as pd

df = pd.read_csv('us_accidents.csv')

df.dropna()

df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

rename_dict = {
    'Distance(mi)': 'Distance_mi',
    'Temperature(F)': 'Temperature_F',
    'Wind_Chill(F)': 'Wind_Chill_F',
    'Humidity(%)': 'Humidity_percent',
    'Pressure(in)': 'Pressure_in',
    'Visibility(mi)': 'Visibility_mi',
    'Wind_Speed(mph)': 'Wind_Speed_mph',
    'Precipitation(in)': 'Precipitation_in'
}

df = df.rename(columns=rename_dict)

df.to_csv('us_accidents_cleaned.csv', index=False)