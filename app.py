import streamlit as st
import pandas as pd
import resource_op as ro
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static

# Improved visualization for cost comparison
def visualize_cost_savings_waterfall(comparison_results):
    # Create waterfall chart for cost comparison
    fig = go.Figure(go.Waterfall(
        name="Cost Comparison",
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Fixed Cost", "Savings", "Mobile Cost"],
        y=[comparison_results['Fixed Cost'],
           -comparison_results['Cost Savings'],
           comparison_results['Mobile Cost']],
        text=["Fixed Cost", "Savings", "Mobile Cost"],
        decreasing={"marker": {"color": "green"}},
        increasing={"marker": {"color": "red"}},
        totals={"marker": {"color": "blue"}}
    ))

    fig.update_layout(
        title="Waterfall Chart of Cost Savings",
        xaxis_title="Cost Type",
        yaxis_title="Cost (in currency)",
        height=500,
        width=800
    )

    return fig

def main():
    st.title("Store Data Analysis App")
    st.write("Analyze and filter Coop store data.")

    # Step 1: Load the data
    file_path = "Coop Locations_EB_Edit_LatLong.csv"
    if file_path:
        df = ro.load_data(file_path)
        st.write("Data Loaded:")
        st.dataframe(df.head())

        # Step 2: Clean the data
        # df = ro.clean_data(df)
        # st.write("Cleaned Data:")
        # st.dataframe(df.head())

        # Step 2: Perform basic analysis
        if st.checkbox("Show Basic Analysis"):
            summary_stats = ro.basic_analysis(df)
            st.write("Summary Statistics:")
            st.json(summary_stats)

        # Step 3: Filter the data
        analysis_options = [
            "Store Type Distribution",
            "Solution Type Distribution",
            "Crosstab of Store Type vs. Solution Type"
        ]
        selected_option = st.selectbox("Select Analysis Type:", analysis_options)

        # Show the selected analysis
        if selected_option == "Store Type Distribution":
            st.write("Store Type Distribution:")
            st.bar_chart(ro.store_type_distribution(df))

        elif selected_option == "Solution Type Distribution":
            st.write("Solution Type Distribution:")
            st.bar_chart(ro.solution_type_distribution(df))

        elif selected_option == "Crosstab of Store Type vs. Solution Type":
            crosstab = ro.store_solution_crosstab(df)
            st.write("Crosstab:")
            st.dataframe(crosstab)

        # Step 2: Show idle time analysis by store type
        idle_time_results = ro.calculate_idle_time_by_store_type(df)
        st.write("Idle Time Analysis by Store Type:")
        st.dataframe(idle_time_results)

        # Visualization
        st.write("Visualizations:")

        # Bar chart for total cleaning hours and idle time
        # Create a bar chart using Plotly
        fig = px.bar(
            idle_time_results,
            x='Store Type',
            y=['Total Cleaning Hours', 'Idle Time'],
            title="Total Cleaning Hours vs Idle Time by Store Type",
            labels={'value': 'Hours', 'variable': 'Type'}
        )

        # Update the layout to increase the height
        fig.update_layout(
            height=500,  # Set the height of the chart
            width=800  # Optional: Set the width of the chart
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)

        # Pie chart for idle time percentage
        st.write("Idle Time Percentage by Store Type:")

        # Bar chart for total cleaning hours, idle time, and total available hours
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=idle_time_results['Store Type'],
            y=idle_time_results['Total Cleaning Hours'],
            name='Total Cleaning Hours',
            marker_color='blue'
        ))
        fig.add_trace(go.Bar(
            x=idle_time_results['Store Type'],
            y=idle_time_results['Idle Time'],
            name='Idle Time',
            marker_color='red'
        ))

        # Adding total available hours as a line for reference
        fig.add_trace(go.Scatter(
            x=idle_time_results['Store Type'],
            y=idle_time_results['Total Available Hours'],
            mode='lines+markers',
            name='Total Available Hours',
            line=dict(color='green', width=2)
        ))

        fig.update_layout(
            title='Total Cleaning Hours vs Idle Time by Store Type',
            xaxis_title='Store Type',
            yaxis_title='Hours',
            barmode='group'
        )

        st.plotly_chart(fig)

    if st.button("Compare Fixed vs Mobile"):
        comparison_results = ro.compare_fixed_vs_mobile(df)
        st.write("Comparison Results:")
        st.json(comparison_results)

    comparison_results = ro.compare_fixed_vs_mobile(df)
    st.plotly_chart(visualize_cost_savings_waterfall(comparison_results))

    #Clusters
    if st.checkbox("Show Store Locations on Map"):
        ro.plot_store_locations(df)
        # Get clustering parameters
        eps_miles = st.slider("Set the distance threshold (miles) for clustering", min_value=0.1, max_value=50.0,
                              value=10.0, step=0.1)

        # Perform clustering
        labels = ro.perform_clustering(df, eps_miles)
        df['Cluster'] = labels  # Add cluster labels to the DataFrame

        # Plot clusters
        st.write(f"Clusters based on a {eps_miles} mile threshold:")
        ro.plot_clusters(df, labels)

    # Optimal path
    # Convert columns to numeric if needed
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    if st.button("Show Approximate Cleaning Path for 100 Stores"):
        # Get the optimal path using nearest neighbor for the first 100 stores
        optimal_path = ro.nearest_neighbor_tsp(df, start_index=0, num_stores=100)
        # Visualize the path using Folium
        optimal_path_map = ro.visualize_optimal_path_folium(df, optimal_path)
        # Display the map in Streamlit
        folium_static(optimal_path_map)

    if st.button("Show Optimal Cleaning Path with Road Routes"):
        # Get the optimal path (for simplicity, let's assume Nearest Neighbor is used)
        optimal_path = ro.nearest_neighbor_tsp(df, start_index=0, num_stores=100)
        # Visualize the path using Folium
        road_route_map = ro.visualize_route_on_map(df, optimal_path)
        # Display the map in Streamlit
        folium_static(road_route_map)

# Run the app
if __name__ == "__main__":
    main()
