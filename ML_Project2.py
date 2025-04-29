import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import random

# Load dataset
df = pd.read_csv("isl_wise_train_detail_03082015_v1.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Remove single quotes from string values
df = df.map(lambda x: x.strip().replace("'", "") if isinstance(x, str) else x)

# Convert time columns to datetime format
df["arrival_time"] = pd.to_datetime(df["arrival_time"], format="%H:%M:%S", errors="coerce")
df["departure_time"] = pd.to_datetime(df["departure_time"], format="%H:%M:%S", errors="coerce")

# Calculate travel time in minutes
df["travel_time_mins"] = (df["departure_time"] - df["arrival_time"]).dt.total_seconds() / 60
df["travel_time_mins"] = df["travel_time_mins"].apply(lambda x: x if x >= 0 else np.nan)

# Extract hour information for clustering
df["arrival_hour"] = df["arrival_time"].dt.hour
df["departure_hour"] = df["departure_time"].dt.hour

# Select relevant features
features = ["distance", "travel_time_mins", "arrival_hour", "departure_hour"]
df_clean = df.dropna(subset=features).copy()

# Create a pipeline with StandardScaler and KMeans
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=4, random_state=42, n_init=10))
])

# Fit pipeline
df_clean["cluster"] = pipeline.fit_predict(df_clean[features])

# Save the trained pipeline
joblib.dump(pipeline, "kmeans_pipeline.pkl")

# Load trained pipeline
ML_Model = joblib.load("kmeans_pipeline.pkl")

# Streamlit UI
st.title("Train Route Finder")
st.write("Find the best train routes based on travel time and clustering")

# User input for source and destination
col1, col2 = st.columns(2)
source = col1.selectbox("Select your starting station:", df["source_station_name"].unique())
destination = col2.selectbox("Select your destination station:", df["destination_station_name"].unique())

if "booked_tickets" not in st.session_state:
    st.session_state.booked_tickets = []

def find_indirect_routes_via_ga(source, destination, max_transfers=3):
    stations = df["source_station_name"].unique().tolist()
    if source not in stations or destination not in stations:
        return []

    def route_exists(r):
        for i in range(len(r) - 1):
            if df[(df["source_station_name"] == r[i]) & (df["destination_station_name"] == r[i+1])].empty:
                return False
        return True

    def fitness(route):
        total_time = 0
        for i in range(len(route) - 1):
            leg = df[(df["source_station_name"] == route[i]) & (df["destination_station_name"] == route[i+1])]
            if leg.empty:
                return float("inf")
            best_leg = leg.sort_values(by="travel_time_mins").iloc[0]
            total_time += best_leg["travel_time_mins"]
        return total_time

    def create_population(size=30):
        population = []
        for _ in range(size):
            middle_count = random.randint(0, max_transfers)
            middle = random.sample([s for s in stations if s not in [source, destination]], k=middle_count)
            route = [source] + middle + [destination]
            population.append(route)
        return population

    def crossover(parent1, parent2):
        middle1 = parent1[1:-1]
        middle2 = parent2[1:-1]
        new_middle = random.sample(list(set(middle1 + middle2)), k=min(len(middle1), len(middle2)))
        return [source] + new_middle + [destination]

    def mutate(route):
        middle = route[1:-1]
        if len(middle) > 1:
            i, j = random.sample(range(len(middle)), 2)
            middle[i], middle[j] = middle[j], middle[i]
        return [route[0]] + middle + [route[-1]]

    population = create_population()
    for _ in range(20):
        population = [route for route in population if route_exists(route)]
        if not population:
            break
        population.sort(key=fitness)
        population = population[:20]
        children = [mutate(crossover(random.choice(population), random.choice(population))) for _ in range(20)]
        population += children

    population = [route for route in population if route_exists(route)]
    if not population:
        return []
    best_route = min(population, key=fitness)
    return best_route if fitness(best_route) != float("inf") else []

if st.button("Find Best Routes"):
    direct_routes = df[(df["source_station_name"] == source) & (df["destination_station_name"] == destination)]

    if not direct_routes.empty:
        possible_routes_clean = direct_routes.dropna(subset=features).copy()
        possible_routes_clean["cluster"] = ML_Model.predict(possible_routes_clean[features])
        most_common_cluster = possible_routes_clean["cluster"].mode()[0]
        best_routes = possible_routes_clean[possible_routes_clean["cluster"] == most_common_cluster]
        best_routes = best_routes.sort_values(by="travel_time_mins").head(3)
    else:
        best_route = find_indirect_routes_via_ga(source, destination, max_transfers=3)
        best_routes = []
        if best_route:
            total_time = 0
            st.write(f"Top Indirect Route: {' → '.join(best_route)}")
            for i in range(len(best_route) - 1):
                leg = df[(df["source_station_name"] == best_route[i]) & (df["destination_station_name"] == best_route[i+1])]
                if leg.empty:
                    st.warning(f"No available train from {best_route[i]} to {best_route[i+1]}. Skipping this leg.")
                    continue
                best_leg = leg.sort_values(by="travel_time_mins").iloc[0]
                st.write(f"Train: {best_leg['train_no.']} - {best_leg['train_name']}")
                st.write(f"Departure: {best_leg['departure_time']} from {best_route[i]}")
                st.write(f"Arrival: {best_leg['arrival_time']} at {best_route[i+1]}")
                st.write(f"Travel Time: {best_leg['travel_time_mins']} mins")
                total_time += best_leg["travel_time_mins"]
            st.write(f"Total Travel Time: {total_time:.2f} mins")

    if best_routes:
        st.write(f"Top 3 Best Routes from {source} to {destination}:")
        for idx, route in best_routes.iterrows():
            st.write(f"Train: {route['train_no.']} - {route['train_name']}")
            st.write(f"Departure Time: {route['departure_time']}")
            st.write(f"Arrival Time: {route['arrival_time']}")
            st.write(f"Travel Time: {route['travel_time_mins']} mins")
            if st.button(f"Book Ticket for {route['train_no.']} ({idx})"):
                st.session_state.booked_tickets.append(route.to_dict())
                st.write("Ticket booked successfully!")
            st.write("---")
    elif not direct_routes.empty:
        st.write("No optimized routes found with clustering. Showing available direct options.")
        st.write(direct_routes)
    else:
        st.write("No routes found, please try a different selection.")

st.write("### Booked Tickets")
for ticket in st.session_state.booked_tickets:
    st.write(ticket)
