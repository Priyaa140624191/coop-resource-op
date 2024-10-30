import pandas as pd
import numpy as np
import plotly.graph_objects as go

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    return df


# def clean_data(df):
#     """
#     Clean the data by replacing null values in the 'Address 2' column with an empty string.
#
#     Parameters:
#         df (pd.DataFrame): The input DataFrame.
#
#     Returns:
#         pd.DataFrame: Cleaned DataFrame.
#     """
#     # Replace null values in the 'Address 2' column with an empty string
#     if "Address 2" in df.columns:
#         df["Address 2"].fillna("", inplace=True)
#
#     return df


def basic_analysis(df):
    """
    Perform basic analysis on the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary containing summary statistics.
    """
    summary_stats = df.describe(include='all').to_dict()
    return summary_stats

def filter_data(df, store_type=None, min_size=None):
    """
    Filter the data based on store type and minimum size.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        store_type (str): Store type to filter by.
        min_size (float): Minimum size to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if store_type:
        df = df[df['Store Type'] == store_type]
    if min_size:
        df = df[df['Approx Store Size (SQ/F)'] >= min_size]
    return df


def store_type_distribution(df):
    """
    Calculate the distribution of different Store Types.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: Distribution of Store Types.
    """
    return df['Store Type'].value_counts()


def solution_type_distribution(df):
    """
    Calculate the distribution of different Solution Types.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: Distribution of Solution Types.
    """
    return df['Solution Type'].value_counts()


def store_solution_crosstab(df):
    """
    Create a crosstab of Store Type and Solution Type.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Crosstab of Store Type vs. Solution Type.
    """
    return pd.crosstab(df['Store Type'], df['Solution Type'])


import streamlit as st
import folium
from streamlit_folium import folium_static

import streamlit as st
import folium
from streamlit_folium import folium_static


def plot_store_locations(df):
    """
    Plot the store locations on a map using latitude and longitude.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None: Displays the map in Streamlit.
    """
    # Create a map centered at the average latitude and longitude
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    store_map = folium.Map(location=map_center, zoom_start=6)

    # Add small red circle markers to the map
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,  # Size of the circle
            color='red',  # Border color
            fill=True,
            fill_color='red',  # Fill color
            fill_opacity=0.6
        ).add_to(store_map)

    # Display the map in Streamlit
    folium_static(store_map)


from sklearn.cluster import DBSCAN
import numpy as np


def perform_clustering(df, eps_miles):
    """
    Perform DBSCAN clustering on the store locations based on distance.

    Parameters:
        df (pd.DataFrame): The input DataFrame with 'Latitude' and 'Longitude' columns.
        eps_miles (float): The maximum distance (in miles) for two points to be considered in the same neighborhood.

    Returns:
        np.ndarray: Cluster labels for each point.
    """
    # Convert miles to kilometers (1 mile = 1.60934 kilometers)
    eps_km = eps_miles * 1.60934

    # DBSCAN requires distance in radians for geographic data
    earth_radius_km = 6371.0
    eps_rad = eps_km / earth_radius_km

    # Convert latitude and longitude to radians
    coords = np.radians(df[['latitude', 'longitude']])

    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps_rad, min_samples=2, metric='haversine').fit(coords)
    labels = db.labels_

    return labels


def plot_clusters(df, labels):
    """
    Plot clusters on a map using folium.

    Parameters:
        df (pd.DataFrame): The input DataFrame with 'Latitude' and 'Longitude' columns.
        labels (np.ndarray): Cluster labels for each point.

    Returns:
        None: Displays the map in Streamlit.
    """
    # Create a map centered at the average latitude and longitude
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    store_map = folium.Map(location=map_center, zoom_start=6)

    # Generate color map for clusters
    unique_labels = set(labels)
    colors = [
        f"#{hex(np.random.randint(0, 256))[2:]}{hex(np.random.randint(0, 256))[2:]}{hex(np.random.randint(0, 256))[2:]}"
        for _ in range(len(unique_labels))]

    # Plot each point with a color based on its cluster label
    for idx, row in df.iterrows():
        label = labels[idx]
        color = colors[label] if label != -1 else 'black'  # Black for noise points
        store_ref = row.get('Karcher reference', 'Unknown')
        postcode = row.get('Postcode', 'Unknown')
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"Cluster {label}, Store Ref: {store_ref}, Postcode: {postcode}" if label != -1 else f"Noise, Store Ref: {store_ref}, Postcode: {postcode}"
        ).add_to(store_map)

    # Display the map in Streamlit
    folium_static(store_map)


def calculate_cleaning_requirements(df, travel_time_per_store=0.5, work_hours_per_day=8):
    """
    Calculate the total cost, total time, and number of people needed to clean all stores.

    Parameters:
        df (pd.DataFrame): The input DataFrame with store information.
        travel_time_per_store (float): Estimated travel time between stores in hours. Default is 0.5 hours.
        work_hours_per_day (int): Number of work hours per day. Default is 8 hours.

    Returns:
        dict: A dictionary with the total cleaning time, total cost, number of people required, and total mobile time.
    """
    # Step 1: Convert 'Cost Per Hour (Labour Only)' to numeric
    df['Cost Per Hour (Labour Only)'] = df['Cost Per Hour (Labour Only)'].replace({'£': ''}, regex=True).astype(float)

    # Step 2: Calculate cleaning hours per store
    df['Cleaning Hours Per Store'] = df['Approx Store Size (SQ/F)'] / df['Productivity_SQ_F_PerHour']
    total_cleaning_hours = df['Cleaning Hours Per Store'].sum()

    # Step 3: Calculate total cleaning cost for the fixed approach
    people_needed_fixed = total_cleaning_hours / work_hours_per_day
    fixed_cost = people_needed_fixed * df['Cost Per Hour (Labour Only)'].mean() * work_hours_per_day

    # Step 4: Calculate mobile approach costs (including travel time)
    df['Total Time Per Store (Including Travel)'] = df['Cleaning Hours Per Store'] + travel_time_per_store
    total_mobile_hours = df['Total Time Per Store (Including Travel)'].sum()
    people_needed_mobile = total_mobile_hours / work_hours_per_day
    mobile_cost = people_needed_mobile * df['Cost Per Hour (Labour Only)'].mean() * work_hours_per_day

    # Prepare the results
    results = {
        "Total Cleaning Hours (Fixed)": total_cleaning_hours,
        "People Needed (Fixed)": people_needed_fixed,
        "Total Fixed Cost": fixed_cost,
        "Total Cleaning Hours (Mobile)": total_mobile_hours,
        "People Needed (Mobile)": people_needed_mobile,
        "Total Mobile Cost": mobile_cost
    }

    return results


def calculate_idle_time_by_store_type(df, work_hours_per_day=8):
    """
    Calculate total cleaning hours and idle time for each store type,
    based on the given productivity and store size values.

    Parameters:
        df (pd.DataFrame): The input DataFrame with store information.
        work_hours_per_day (int): Number of work hours per day. Default is 8 hours.

    Returns:
        pd.DataFrame: A DataFrame with total cleaning hours, idle time, and idle time percentage for each store type.
    """
    # Step 1: Calculate cleaning time for each store
    df['Cleaning Hours Per Store'] = df['Approx Store Size (SQ/F)'] / df['Productivity_SQ_F_PerHour']

    # Step 2: Group by 'Store Type' to get total cleaning hours for each store type
    cleaning_hours_by_store_type = df.groupby('Store Type')['Cleaning Hours Per Store'].sum().reset_index()
    cleaning_hours_by_store_type.columns = ['Store Type', 'Total Cleaning Hours']

    # Step 3: Calculate the number of stores for each store type
    store_counts = df.groupby('Store Type').size().reset_index(name='Number of Stores')

    # Step 4: Merge the store counts with the total cleaning hours
    cleaning_hours_by_store_type = cleaning_hours_by_store_type.merge(store_counts, on='Store Type')

    # Step 5: Calculate total available hours (assuming 8 hours per day per store)
    cleaning_hours_by_store_type['Total Available Hours'] = cleaning_hours_by_store_type[
                                                                'Number of Stores'] * work_hours_per_day

    # Step 6: Calculate idle time (Total Available Hours - Total Cleaning Hours)
    cleaning_hours_by_store_type['Idle Time'] = cleaning_hours_by_store_type['Total Available Hours'] - \
                                                cleaning_hours_by_store_type['Total Cleaning Hours']

    # Step 7: Calculate idle time percentage
    cleaning_hours_by_store_type['Idle Time Percentage'] = (cleaning_hours_by_store_type['Idle Time'] /
                                                            cleaning_hours_by_store_type['Total Available Hours']) * 100

    return cleaning_hours_by_store_type


def compare_fixed_vs_mobile(df, hourly_rate=13.77, travel_time_minutes=30, work_hours_per_day=8):
    """
    Compare fixed allocation and mobile team approaches for cleaning stores.
    Calculate cost savings and idle time reduction when switching from fixed allocation to mobile teams.

    Parameters:
        df (pd.DataFrame): The input DataFrame with store information.
        hourly_rate (float): Hourly rate for labor.
        travel_time_minutes (int): Fixed travel time in minutes between stores.
        work_hours_per_day (int): Number of work hours per day. Default is 8 hours.

    Returns:
        dict: A dictionary containing the cost and idle time comparison for fixed and mobile approaches.
    """
    # Step 1: Calculate fixed allocation cost and idle time
    num_stores = len(df)
    total_fixed_cost = num_stores * hourly_rate * work_hours_per_day
    total_available_hours = num_stores * work_hours_per_day
    total_cleaning_hours = df['Approx Store Size (SQ/F)'].sum() / df['Productivity_SQ_F_PerHour'].mean()
    fixed_idle_time = total_available_hours - total_cleaning_hours

    # Step 2: Calculate mobile team cost and idle time
    total_available_minutes = work_hours_per_day * 60  # 8 hours * 60 minutes
    df['Cleaning Time (Minutes)'] = (df['Approx Store Size (SQ/F)'] / df['Productivity_SQ_F_PerHour']) * 60
    df['Total Time Per Store (Minutes)'] = df['Cleaning Time (Minutes)'] + travel_time_minutes

    stores_covered = 0
    accumulated_time = 0
    for time in df['Total Time Per Store (Minutes)']:
        if accumulated_time + time > total_available_minutes:
            break
        accumulated_time += time
        stores_covered += 1

    total_cleaners_needed = (num_stores + stores_covered - 1) // stores_covered
    total_mobile_cost = total_cleaners_needed * hourly_rate * work_hours_per_day
    mobile_idle_time = total_cleaners_needed * work_hours_per_day - total_cleaning_hours

    # Step 3: Calculate savings
    cost_savings = total_fixed_cost - total_mobile_cost
    idle_time_reduction = fixed_idle_time - mobile_idle_time

    # Step 4: Prepare the results
    results = {
        "Fixed Cost": total_fixed_cost,
        "Mobile Cost": total_mobile_cost,
        "Cost Savings": cost_savings,
        "Fixed Idle Time": fixed_idle_time,
        "Mobile Idle Time": mobile_idle_time,
        "Idle Time Reduction": idle_time_reduction
    }

    return results


def nearest_neighbor_tsp(df, start_index=0, num_stores=100):
    """
    Approximate TSP solution using the Nearest Neighbor algorithm for a given number of stores.
    """
    # Limit to specified number of stores (default to 100)
    subset_df = df.iloc[:num_stores].reset_index(drop=True)

    # Initialize
    unvisited = set(range(len(subset_df)))
    path = [start_index]
    unvisited.remove(start_index)
    current_index = start_index

    while unvisited:
        # Find the nearest neighbor
        nearest_index = min(unvisited, key=lambda i: np.hypot(
            subset_df.iloc[i]['latitude'] - subset_df.iloc[current_index]['latitude'],
            subset_df.iloc[i]['longitude'] - subset_df.iloc[current_index]['longitude']
        ))
        # Update path
        path.append(nearest_index)
        unvisited.remove(nearest_index)
        current_index = nearest_index

    return path


def visualize_optimal_path_folium(df, optimal_path):
    # Create a map centered at the average latitude and longitude
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    store_map = folium.Map(location=map_center, zoom_start=9)

    # Plot the stores as markers
    for index in optimal_path:
        row = df.iloc[index]
        store_ref = row.get('Karcher reference', 'Unknown')
        postcode = row.get('Postcode', 'Unknown')
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Store {index}, Store Ref: {store_ref}, Postcode: {postcode}",
            icon=folium.Icon(color='green', icon='fa-shopping-basket', prefix='fa')
        ).add_to(store_map)

    # Draw the optimal path
    path_coordinates = [(df.iloc[i]['latitude'], df.iloc[i]['longitude']) for i in optimal_path]
    folium.PolyLine(
        locations=path_coordinates,
        color='red',
        weight=5,
        opacity=0.7
    ).add_to(store_map)

    return store_map

#osrm

import requests

import time

def get_route_via_osrm(lat1, lon1, lat2, lon2, retries=3):
    """
    Get the driving route from point A (lat1, lon1) to point B (lat2, lon2) using OSRM, with retry logic.
    """
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    for attempt in range(retries):
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200 and 'routes' in data and data['routes']:
            route_geometry = data['routes'][0]['geometry']['coordinates']
            return route_geometry
        else:
            print(f"Error fetching route from {lat1},{lon1} to {lat2},{lon2}, attempt {attempt + 1}")
            time.sleep(1)  # Delay before retrying

    return None  # Return None if all retries fail


import folium

def visualize_route_on_map(df, optimal_path):
    # Create a map centered at the average latitude and longitude
    map_center = [df['latitude'].mean(), df['longitude'].mean()]
    store_map = folium.Map(location=map_center, zoom_start=6)

    # Plot the stores as markers
    for index in optimal_path:
        row = df.iloc[index]
        store_ref = row.get('Karcher reference', 'Unknown')
        postcode = row.get('Postcode', 'Unknown')
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Store {index}, Store Ref: {store_ref}, Postcode: {postcode}",
            icon=folium.Icon(color='green', icon='fa-shopping-basket', prefix='fa')
        ).add_to(store_map)

    # Draw the road routes between stores
    for i in range(len(optimal_path) - 1):
        start_index = optimal_path[i]
        end_index = optimal_path[i + 1]
        start_lat, start_lon = df.iloc[start_index]['latitude'], df.iloc[start_index]['longitude']
        end_lat, end_lon = df.iloc[end_index]['latitude'], df.iloc[end_index]['longitude']

        # Get the route via OSRM
        route = get_route_via_osrm(start_lat, start_lon, end_lat, end_lon)

        if route:
            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in route],  # Flip to (lat, lon)
                color='red',
                weight=5,
                opacity=0.7
            ).add_to(store_map)

    return store_map
