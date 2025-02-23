import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 50,000 synthetic meter IDs
num_samples = 50000
meter_ids = [f"MTR-{str(i).zfill(5)}" for i in range(1, num_samples + 1)]

# Generate synthetic substations and feeders
substations = [f"Substation-{np.random.randint(1, 100)}" for _ in range(num_samples)]
feeders = [f"Feeder-{np.random.randint(1, 200)}" for _ in range(num_samples)]

# Generate random GPS coordinates within South Africa (approx range)
latitudes = np.random.uniform(-35.0, -22.0, num_samples)  # South Africa lat range
longitudes = np.random.uniform(16.0, 33.0, num_samples)  # South Africa lon range

# Assign random city names (example cities in South Africa)
cities = np.random.choice(
    ["Johannesburg", "Cape Town", "Durban", "Pretoria", "Port Elizabeth", "Bloemfontein"], 
    num_samples
)

# Assign random regions
regions = np.random.choice(
    ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Free State"], 
    num_samples
)

# ✅ Generate a Date-Time column (random dates within a year)
start_date = pd.to_datetime("2023-01-01 00:00:00")  # Start date
end_date = pd.to_datetime("2023-12-31 23:00:00")  # End date
date_times = pd.date_range(start=start_date, end=end_date, periods=num_samples)

# Create DataFrame
location_data = pd.DataFrame({
    "Meter ID": meter_ids,
    "Substation": substations,
    "Feeder": feeders,
    "Latitude": latitudes,
    "Longitude": longitudes,
    "City": cities,
    "Region": regions,
    "Date Time Hour Beginning": date_times
})

# Save dataset
file_path = "data/location_data.csv"
location_data.to_csv(file_path, index=False)

print(f"Location dataset saved successfully: {file_path}")
