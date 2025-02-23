import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define time range (about 2 months of hourly readings for multiple meters)
date_range = pd.date_range(start="2024-01-01", periods=24 * 60, freq="H")  # 60 days

# Define locations (Provinces & Cities)
locations = [
    ("Gauteng", "Johannesburg", -26.2041, 28.0473),
    ("Gauteng", "Pretoria", -25.7479, 28.2293),
    ("Western Cape", "Cape Town", -33.9249, 18.4241),
    ("KwaZulu-Natal", "Durban", -29.8587, 31.0218),
    ("Eastern Cape", "Port Elizabeth", -33.9608, 25.6022),
    ("Mpumalanga", "Nelspruit", -25.4745, 30.9703),
    ("Free State", "Bloemfontein", -29.0852, 26.1596),
    ("Limpopo", "Polokwane", -23.9000, 29.4500),
    ("North West", "Rustenburg", -25.6676, 27.2421),
    ("Northern Cape", "Kimberley", -28.7282, 24.7491)
]

# Define customer types
customer_types = ["Residential", "Commercial", "Industrial"]

# Number of smart meters to generate data for
num_meters = 1000  # More meters over a longer period to reach 50,000 rows

# Initialize data storage
data = []

for meter_id in range(1, num_meters + 1):
    province, city, lat, lon = locations[np.random.randint(0, len(locations))]
    customer_type = np.random.choice(customer_types)

    for timestamp in date_range:
        if len(data) >= 50_000:  # Stop when we reach 50,000 rows
            break

        # Base energy consumption
        base_consumption = np.random.uniform(2, 10) if customer_type == "Residential" else \
                           np.random.uniform(10, 50) if customer_type == "Commercial" else \
                           np.random.uniform(50, 200)

        # Simulating peak hours (morning 6-9 AM, evening 5-9 PM)
        hour = timestamp.hour
        if hour in [6, 7, 8, 17, 18, 19, 20]:
            base_consumption *= np.random.uniform(1.2, 1.5)

        # Simulate Load Shedding Impact (20% chance of reduction)
        load_shedding = np.random.choice([0, 1], p=[0.8, 0.2])
        if load_shedding:
            base_consumption *= np.random.uniform(0.3, 0.7)

        # Simulate Solar Power (only for Residential & Commercial)
        solar_generation = np.random.uniform(0, base_consumption * 0.5) if customer_type != "Industrial" else 0

        # Grid Stability Metrics
        voltage = np.random.uniform(215, 245)  # Normal voltage in SA varies
        frequency = np.random.uniform(49.3, 50.7)  # Slight frequency variation
        power_factor = np.random.uniform(0.85, 1.0)  # Efficiency of energy use

        # Simulate Fraud Anomalies (5% chance per meter)
        fraud = np.random.choice([0, 1], p=[0.95, 0.05])

        # Store Data
        data.append([
            meter_id, timestamp, province, city, lat, lon, customer_type,
            round(base_consumption, 2), round(solar_generation, 2), round(voltage, 1), 
            round(frequency, 2), round(power_factor, 2), fraud, load_shedding
        ])

        # Stop when we reach 50,000 rows
        if len(data) >= 50_000:
            break

# Convert to DataFrame
columns = [
    "Meter ID", "Timestamp", "Province", "City", "Latitude", "Longitude",
    "Customer Type", "Energy Consumption (kWh)", "Solar Generation (kWh)",
    "Voltage (kV)", "Frequency (Hz)", "Power Factor", "Fraud", "Load Shedding"
]

df_smart_grid = pd.DataFrame(data, columns=columns)

# Save dataset to CSV
df_smart_grid.to_csv("data/smart_meter_grid_south_africa.csv", index=False)

print("✅ Smart Meter Grid Data (50,000 rows) Generated and Saved as 'smart_meter_grid_south_africa_50k.csv'")
print(df_smart_grid.head())  # Display first few rows
