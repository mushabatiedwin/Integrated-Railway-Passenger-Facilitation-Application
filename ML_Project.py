import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

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

# Fit pipeline and predict clusters
df_clean["cluster"] = pipeline.fit_predict(df_clean[features])

# Compute silhouette score
scaled_features = StandardScaler().fit_transform(df_clean[features])
score = silhouette_score(scaled_features, df_clean["cluster"])
print(f"Silhouette Score: {score:.3f}")

# Save the trained pipeline
joblib.dump(pipeline, "kmeans_pipeline.pkl")

# Load trained pipeline
ML_Model = joblib.load("kmeans_pipeline.pkl")

# Streamlit UI
st.title("Train Route Finder")
st.write("Find the best train routes based on travel time and clustering")

# ===================== Route Finder =====================
# User input for source and destination
col1, col2 = st.columns(2)
source = col1.selectbox("Select your starting station:", df["source_station_name"].unique())
destination = col2.selectbox("Select your destination station:", df["destination_station_name"].unique())

if "booked_tickets" not in st.session_state:
    st.session_state.booked_tickets = []

if st.button("Find Best Routes"):
    # Find direct routes
    direct_routes = df[(df["source_station_name"] == source) & (df["destination_station_name"] == destination)]

    if not direct_routes.empty:
        possible_routes_clean = direct_routes.dropna(subset=features).copy()
        possible_routes_clean["cluster"] = ML_Model.predict(possible_routes_clean[features])
        most_common_cluster = possible_routes_clean["cluster"].mode()[0]
        best_routes = possible_routes_clean[possible_routes_clean["cluster"] == most_common_cluster]
        best_routes = best_routes.sort_values(by="travel_time_mins").head(3)
    else:
        # Find indirect routes
        transfer_stations = df[df["source_station_name"] == source]["destination_station_name"].unique()
        indirect_routes = []
        for transfer in transfer_stations:
            first_leg = df[(df["source_station_name"] == source) & (df["destination_station_name"] == transfer)]
            second_leg = df[(df["source_station_name"] == transfer) & (df["destination_station_name"] == destination)]
            if not first_leg.empty and not second_leg.empty:
                for _, f in first_leg.iterrows():
                    for _, s in second_leg.iterrows():
                        total_travel_time = f["travel_time_mins"] + s["travel_time_mins"]
                        indirect_routes.append({
                            "train_1": f.get("train_no.", "Unknown"), "train_name_1": f.get("train_name", "Unknown"),
                            "departure_time_1": f.get("departure_time", "Unknown"),
                            "arrival_time_1": f.get("arrival_time", "Unknown"),
                            "train_2": s.get("train_no.", "Unknown"), "train_name_2": s.get("train_name", "Unknown"),
                            "departure_time_2": s.get("departure_time", "Unknown"),
                            "arrival_time_2": s.get("arrival_time", "Unknown"),
                            "total_travel_time": total_travel_time
                        })

        if indirect_routes:
            best_routes = sorted(indirect_routes, key=lambda x: x["total_travel_time"])[:3]
        else:
            best_routes = []

    if best_routes:
        st.write(f"Top 3 Best Routes from {source} to {destination}:")
        for idx, route in enumerate(best_routes):
            if "train_1" in route:
                st.write(f"Train 1: {route['train_1']} - {route['train_name_1']}")
                st.write(f"Departure Time: {route['departure_time_1']}")
                st.write(f"Arrival Time: {route['arrival_time_1']}")
                st.write(f"Transfer at: {transfer}")
                st.write(f"Train 2: {route['train_2']} - {route['train_name_2']}")
                st.write(f"Departure Time: {route['departure_time_2']}")
                st.write(f"Arrival Time: {route['arrival_time_2']}")
            else:
                st.write(f"Train: {route.get('train_no.', 'Unknown')} - {route.get('train_name', 'Unknown')}")
                st.write(f"Departure Time: {route.get('departure_time', 'Unknown')}")
                st.write(f"Arrival Time: {route.get('arrival_time', 'Unknown')}")
                st.write(f"Travel Time: {route.get('travel_time_mins', 'Unknown')} mins")

            if st.button(f"Book Ticket for {route.get('train_1', route.get('train_no.', 'Unknown'))} ({idx})"):
                st.session_state.booked_tickets.append(route)
                st.write("Ticket booked successfully!")
            st.write("---")
    else:
        st.write("No routes found, please try a different selection.")

# ===================== Booked Tickets =====================
st.write("### Booked Tickets")
for ticket in st.session_state.booked_tickets:
    st.write(ticket)

# ===================== Data Visualization =====================
st.write("### Data Visualization")

# Silhouette Score
st.subheader("Silhouette Score")
st.write(f"The silhouette score for the clustering is: **{score:.3f}**")

# Heatmap
st.subheader("Feature Correlation Heatmap")
fig_heatmap, ax_heatmap = plt.subplots()
corr = df_clean[features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_heatmap)
st.pyplot(fig_heatmap)

# Boxplot
st.subheader("Distribution of Features by Cluster")
selected_feature = st.selectbox("Select feature for boxplot:", features)
fig_boxplot, ax_boxplot = plt.subplots()
sns.boxplot(data=df_clean, x="cluster", y=selected_feature, ax=ax_boxplot, palette="Set2")
ax_boxplot.set_title(f"{selected_feature.replace('_', ' ').title()} Distribution by Cluster")
st.pyplot(fig_boxplot)

# Streamlit run ML_Project.py
