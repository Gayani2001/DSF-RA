import pandas as pd
import numpy as np
from geopy.distance import geodesic

dfSFA_GPSDataCopy = dfSFA_GPSData.copy()
# Convert 'RecievedDate' to datetime format
dfSFA_GPSDataCopy['RecievedDate'] = pd.to_datetime(dfSFA_GPSDataCopy['RecievedDate'])

# Sort the dataframe by 'UserCode' and 'RecievedDate'
dfSFA_GPSDataCopy.sort_values(['UserCode', 'RecievedDate'], inplace=True)

# Calculate the distance and time difference between successive rows for each 'UserCode'
dfSFA_GPSDataCopy['ShiftedLatitude'] = dfSFA_GPSDataCopy.groupby('UserCode')['Latitude'].shift()
dfSFA_GPSDataCopy['ShiftedLongitude'] = dfSFA_GPSDataCopy.groupby('UserCode')['Longitude'].shift()
dfSFA_GPSDataCopy['Distance'] = dfSFA_GPSDataCopy.apply(lambda row: geodesic((row['ShiftedLatitude'], row['ShiftedLongitude']), (row['Latitude'], row['Longitude'])).meters if pd.notnull(row['ShiftedLatitude']) else np.nan, axis=1)
dfSFA_GPSDataCopy['TimeDiff'] = dfSFA_GPSDataCopy.groupby('UserCode')['RecievedDate'].diff().dt.total_seconds()

# Calculate the speed
dfSFA_GPSDataCopy['Speed'] = dfSFA_GPSDataCopy['Distance'] / dfSFA_GPSDataCopy['TimeDiff']

# Plot the speed for each 'UserCode'
for user in dfSFA_GPSDataCopy['UserCode'].unique():
    plt.figure(figsize=(10, 6))
    dfSFA_GPSDataCopy[dfSFA_GPSDataCopy['UserCode'] == user]['Speed'].plot()
    plt.title(f'Average Speed for UserCode {user}')
    plt.xlabel('Time')
    plt.ylabel('Speed (m/s)')
    plt.show()

import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt

dfSFA_GPSDataCopy = dfSFA_GPSData.copy()

# Convert 'RecievedDate' to datetime format
dfSFA_GPSDataCopy['RecievedDate'] = pd.to_datetime(dfSFA_GPSDataCopy['RecievedDate'])

# Extract date from 'RecievedDate'
dfSFA_GPSDataCopy['Date'] = dfSFA_GPSDataCopy['RecievedDate'].dt.date

# Sort the dataframe by 'UserCode', 'Date' and 'RecievedDate'
dfSFA_GPSDataCopy.sort_values(['UserCode', 'Date', 'RecievedDate'], inplace=True)

# Calculate the distance and time difference between successive rows for each 'UserCode' on each day
dfSFA_GPSDataCopy[['ShiftedLatitude', 'ShiftedLongitude']] = dfSFA_GPSDataCopy.groupby(['UserCode', 'Date'])[['Latitude', 'Longitude']].shift()
dfSFA_GPSDataCopy['Distance'] = dfSFA_GPSDataCopy.apply(lambda row: geodesic((row['ShiftedLatitude'], row['ShiftedLongitude']), (row['Latitude'], row['Longitude'])).meters if pd.notnull(row['ShiftedLatitude']) else np.nan, axis=1)
dfSFA_GPSDataCopy['TimeDiff'] = dfSFA_GPSDataCopy.groupby(['UserCode', 'Date'])['RecievedDate'].diff().dt.total_seconds()

# Calculate the speed
dfSFA_GPSDataCopy['Speed'] = dfSFA_GPSDataCopy['Distance'] / dfSFA_GPSDataCopy['TimeDiff']

# Get the first day for each 'UserCode'
first_day = dfSFA_GPSDataCopy.groupby('UserCode')['Date'].min()

# Filter the data to include only the first day for each 'UserCode'
first_day_data = dfSFA_GPSDataCopy[dfSFA_GPSDataCopy['Date'].isin(first_day)]

# Calculate the average speed for each 'UserCode' on the first day
average_speed_first_day = first_day_data.groupby('UserCode')['Speed'].mean()

# Plot the average speed for each 'UserCode' on the first day
plt.figure(figsize=(10, 6))
average_speed_first_day.plot(kind='bar')
plt.title('Average Speed for Each UserCode on the First Day')
plt.xlabel('UserCode')
plt.ylabel('Average Speed (m/s)')
plt.show()

# Get the first day for each 'UserCode'
first_day = dfSFA_GPSDataCopy.groupby('UserCode')['Date'].min()

# For each 'UserCode', plot the speed variation on the first day
for user in dfSFA_GPSDataCopy['UserCode'].unique():
    # Filter the data to include only the first day for the current 'UserCode'
    user_first_day_data = dfSFA_GPSDataCopy[(dfSFA_GPSDataCopy['UserCode'] == user) & (dfSFA_GPSDataCopy['Date'] == first_day[user])]
    
    # Plot the speed variation for the current 'UserCode' on the first day
    plt.figure(figsize=(10, 6))
    user_first_day_data['Speed'].plot()
    plt.title(f'Speed Variation for BPO {user} on the First Day')
    plt.xlabel('Time')
    plt.ylabel('Speed (m/s)')
    x = str(user) + " 's speed_variation_on_the_FirstDay.png"
    plt.savefig(x)  # Save the figure
    plt.show()

import pandas as pd

# Assuming dfSFA_GPSData is the DataFrame and 'Latitude' and 'Longitude' are your column names
latitude_list = dfSFA_GPSData['Latitude'].tolist()
longitude_list = dfSFA_GPSData['Longitude'].tolist()

# Find the optimum epsilon value for DBSCAN

import math

def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance
    distance = R * c *1000
    return distance

class StayPointDetector:
    KMS_PER_RADIAN = 6371.0088  # Earth's radius in kilometers

    def __init__(self, df):
        """Initialize the processor with a dataframe."""
        self.df = df

    def perform_dbscan(self, epsilon=0.005, min_samples=1, algorithm='ball_tree', metric='haversine'):
        coords = np.radians(self.df[['Latitude', 'Longitude']].values)
        epsilon = epsilon / self.KMS_PER_RADIAN

        db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm=algorithm, metric=metric)
        db.fit(coords)

        self.df['Cluster'] = db.labels_

        cluster_means = self.df.groupby('Cluster')[['Latitude', 'Longitude']].transform('mean')
        self.df['Constant Latitude'] = cluster_means['Latitude']
        self.df['Constant Longitude'] = cluster_means['Longitude']
        # self.df = self.df.join(cluster_means, on='Cluster', rsuffix=' Mean')
        
    def calculate_distance(self, row):
        if pd.notnull(row['Next Latitude']) and pd.notnull(row['Next Longitude']):
            return geodesic((row['Constant Latitude'], row['Constant Longitude']), 
                            (row['Next Latitude'], row['Next Longitude'])).meters
        else:
            return np.nan

    def process_dbscan(self, distance_threshold=25,Group_time_difference_threshold=2, time_difference_threshold=5 ):
        self.df['RecievedDate'] = pd.to_datetime(self.df['RecievedDate'])
        self.df['Date'] = self.df['RecievedDate'].dt.date
        self.df['Location Change'] = (self.df['Constant Latitude'].diff().ne(0) | self.df['Constant Longitude'].diff().ne(0)).cumsum()
        grouped = self.df.groupby(['UserCode', 'Date', 'Location Change'])
        
        # grouped = self.df.groupby(['UserCode', 'Date', 'Constant Latitude', 'Constant Longitude'])
        self.df['First Time'] = grouped['RecievedDate'].transform('first')
        self.df['Last Time'] = grouped['RecievedDate'].transform('last')
        self.df['Spending Time (minutes)'] = (self.df['Last Time'] - self.df['First Time']).dt.total_seconds() / 60

        mask = self.df['Spending Time (minutes)'] >= Group_time_difference_threshold
        self.df = self.df[mask]

        self.df = self.df.drop_duplicates(subset=['UserCode', 'Date', 'Constant Latitude', 'Constant Longitude'])

        self.df['Next Latitude'] = self.df['Constant Latitude'].shift(-1)
        self.df['Next Longitude'] = self.df['Constant Longitude'].shift(-1)
        self.df['Distance'] = self.df.apply(self.calculate_distance, axis=1)
        self.df['Distance'] = self.df['Distance'].shift(1)

        self.df['Next Spending Time (minutes)'] = self.df['Spending Time (minutes)'].shift(-1)
        mask = (self.df['Distance'] <= distance_threshold) & (self.df['Next Spending Time (minutes)'] >= time_difference_threshold)
        
        self.df.loc[mask, 'Mean Latitude'] = self.df.loc[mask, ['Constant Latitude', 'Next Latitude']].mean(axis=1)
        self.df.loc[mask, 'Mean Longitude'] = self.df.loc[mask, ['Constant Longitude', 'Next Longitude']].mean(axis=1)
        # self.df.loc[mask, 'Time Difference (minutes)'] = self.df.loc[mask, ['Spending Time (minutes)']].sum(axis=1)
        self.df.loc[mask, 'Time Difference (minutes)'] = self.df.loc[mask, ['Spending Time (minutes)', 'Next Spending Time (minutes)']].sum(axis=1)
        
        # Fill NaN values in 'Mean Latitude' and 'Mean Longitude' with 'Constant Latitude' and 'Constant Longitude'
        self.df['Mean Latitude'] = self.df['Mean Latitude'].fillna(self.df['Constant Latitude'])
        self.df['Mean Longitude'] = self.df['Mean Longitude'].fillna(self.df['Constant Longitude'])
        
        # When the condition is not met, set 'Time Difference (minutes)' to be the 'Spending Time (minutes)' of the current row
        self.df.loc[~mask, 'Time Difference (minutes)'] = self.df.loc[~mask, 'Spending Time (minutes)']
        self.df = self.df.reset_index(drop=True)

    def process(self, epsilon=0.005, distance_threshold=25, Group_time_difference_threshold=2, time_difference_threshold=5):
        self.perform_dbscan(epsilon=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
        self.process_dbscan(distance_threshold=distance_threshold, time_difference_threshold=time_difference_threshold)

    def test_perform_dbscan(self):
        self.perform_dbscan()
        assert 'Cluster' in self.df.columns
        assert 'Constant Latitude' in self.df.columns
        assert 'Constant Longitude' in self.df.columns

class GeoSpatialAnalyzer:
    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0088  

    def __init__(self, df):
        """Initialize the processor with a dataframe."""
        self.df = df.copy()

    @staticmethod
    def calculate_distances(df1, df2):
        """
        Calculate haversine distances between two dataframes.
        
        Parameters:
        df1, df2: DataFrames containing 'Latitude' and 'Longitude' columns.
        
        Returns:
        distances: A 2D numpy array containing haversine distances.
        """
        customer_coords = np.radians(df1[['Latitude', 'Longitude']].values)
        cluster_coords = np.radians(df2[['Mean Latitude', 'Mean Longitude']].drop_duplicates().values)

        distances = haversine_distances(customer_coords, cluster_coords) * GeoSpatialAnalyzer.EARTH_RADIUS_KM

        return distances

    @staticmethod
    def assign_closest_usercode(df1, df2, distances):
        """
        Assign the closest user code to each row in df1 based on the distances.
        
        Parameters:
        df1, df2: DataFrames containing 'Latitude' and 'Longitude' columns.
        distances: A 2D numpy array containing haversine distances between df1 and df2.
        
        Returns:
        df1: DataFrame with an additional column 'Closet UserCode'.
        closest_clusters_index: Indices of the closest user codes.
        unique_clusters: DataFrame of unique clusters.
        """
        closest_clusters_index = np.argmin(distances, axis=1)
        unique_clusters = df2[['Mean Latitude', 'Mean Longitude', 'UserCode']].drop_duplicates().reset_index(drop=True)

        df1['Closet UserCode'] = unique_clusters.loc[closest_clusters_index, 'UserCode'].values

        return df1, closest_clusters_index, unique_clusters

    @staticmethod
    def assign_visited_status(df, distances, threshold=0.1):
        """
        Assign a visited status to each row in df based on the distances.
        
        Parameters:
        df: DataFrame containing 'Latitude' and 'Longitude' columns.
        distances: A 2D numpy array containing haversine distances.
        threshold: A threshold for the distance to consider a location as visited.
        
        Returns:
        df: DataFrame with an additional column 'Visited'.
        """
        min_distances = np.min(distances, axis=1)
        df['Visited'] = min_distances <= threshold

        return df

    @staticmethod
    def count_visited_status(df):
        """
        Count the visited status in df.
        
        Parameters:
        df: DataFrame containing a 'Visited' column.
        
        Returns:
        visited_counts: A Series containing counts of visited status.
        """
        visited_counts = df['Visited'].value_counts()

        return visited_counts

    @staticmethod
    def count_false_per_user(df):
        """
        Count the number of 'False' in the 'Visited' column for each user.
        
        Parameters:
        df: DataFrame containing 'Visited' and 'Closet UserCode' columns.
        
        Returns:
        user_false_counts: A Series containing counts of 'False' for each user.
        """
        user_false_counts = df.groupby('Closet UserCode')['Visited'].apply(lambda x: (x == False).sum())

        return user_false_counts

    @staticmethod
    def assign_visited_usercode(df1, df2, distances, threshold=0.1):
        """
        Assign a visited user code to each row in df1 based on the distances.
        
        Parameters:
        df1, df2: DataFrames containing 'Latitude' and 'Longitude' columns.
        distances: A 2D numpy array containing haversine distances.
        threshold: A threshold for the distance to consider a location as visited.
        
        Returns:
        df1: DataFrame with an additional column 'Visited UserCode'.
        """
        closest_clusters_index = np.argmin(distances, axis=1)
        unique_clusters = df2[['Mean Latitude', 'Mean Longitude', 'UserCode']].drop_duplicates().reset_index(drop=True)

        min_distances = np.min(distances, axis=1)
        df1['Visited UserCode'] = np.where(min_distances <= threshold, unique_clusters.loc[closest_clusters_index, 'UserCode'].values, np.nan)

        return df1

    @staticmethod
    def get_DistributorNo_for_unvisited_ClosetUserCode(df):
        """
        Get the distributor number for unvisited closet user code.
        
        Parameters:
        df: DataFrame containing 'Visited' and 'Closet UserCode' columns.
        
        Returns:
        no_values: A Series containing distributor numbers for unvisited closet user code.
        """
        unvisited_df = df[df['Visited'] == False]
        no_values = unvisited_df.groupby('Closet UserCode')['No'].apply(lambda x: list(set(x)))
        return no_values

    @staticmethod
    def check_DistributorNo_in_other_ClosetUsercode(no_values):
        """
        Check if there are common distributor numbers in other closet user codes.
        
        Parameters:
        no_values: A Series containing distributor numbers for unvisited closet user code.
        """
        common_found = False
        for usercode, no_list in no_values.items():
            other_no_values = no_values.drop(usercode).values
            other_no_list = [item for sublist in other_no_values for item in sublist]
            common_no_values = set(no_list).intersection(other_no_list)
            if len(common_no_values) != 0:
                common_found = True
                print(f'Common "No" values in "{usercode}" and other UserCodes: {common_no_values}')
                print(f'Count of common "No" values: {len(common_no_values)}')
        if not common_found:
            print("No Common Distributors in Nearest BPOs'")

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
import pandas as pd

class GeoSpatialEvaluator:
    def __init__(self, df, coords_cols=['Latitude', 'Longitude'], cluster_col='Cluster', visited_col='Visited'):
        self.df = df
        self.coords_cols = coords_cols
        self.cluster_col = cluster_col
        self.visited_col = visited_col

    def evaluate_performance(self, true_labels):
        """
        Evaluate the performance of the geospatial analysis.
        """
        # Predicted labels
        pred_labels = self.df[self.visited_col]

        # Precision, Recall, F1 Score
        precision = precision_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

        # Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)
        print('Confusion Matrix:')
        print(cm)

        # Average Distance to Nearest Cluster for 'Visited' Locations
        visited_df = self.df[self.df[self.visited_col] == True]
        avg_distance = visited_df['Distance'].mean()
        print(f'Average Distance to Nearest Cluster for "Visited" Locations: {avg_distance}')

    @staticmethod
    def check_location_within_clusters(dfGeo, dfGPS, epsilon=0.05):
        """
        Check if each location in dfGeoCustomer is within a certain distance (defined by epsilon) from any cluster in dfSFA_GPSDataDBSCAN1.
        """
        # Convert Latitude and Longitude into radians for haversine_distances
        customer_coords_eval = np.radians(dfGeo[['Latitude', 'Longitude']].values)
        cluster_coords_eval = np.radians(dfGPS[['Latitude', 'Longitude']].drop_duplicates().values)

        # Calculate haversine_distances and convert to kilometers
        kms_per_radian = 6371.0088 # Earth's radius in kms
        distances = haversine_distances(customer_coords_eval, cluster_coords_eval) * kms_per_radian

        #dfGeoCustomerCopy = dfGeoCustomer.copy()
        # Check if the minimum distance is within the error tolerance (epsilon)
        dfGeo['Visited'] = np.min(distances, axis=1) <= epsilon

        dfGeo_True = dfGeo[dfGeo['Visited'] == True]

        return dfGeo_True

    def calculate_silhouette_score(self):
        labels = self.df[self.cluster_col]
        coords = np.radians(self.df[self.coords_cols].values)
        silhouette = silhouette_score(coords, labels, metric='euclidean')
        return silhouette

    def calculate_davies_bouldin_index(self):
        labels = self.df[self.cluster_col]
        coords = np.radians(self.df[self.coords_cols].values)
        db_score = davies_bouldin_score(coords, labels)
        return db_score

# import pandas as pd
# import numpy as np
# from geopy.distance import geodesic
# from datetime import datetime


# # 1.1 Normal working...
def process_result(df):
    # Convert RecievedDate to datetime
    df['RecievedDate'] = pd.to_datetime(df['RecievedDate'])

    # Shift the Latitude and Longitude columns to get the next coordinates
    df['Next Latitude'] = df['Constant Latitude'].shift(-1)
    df['Next Longitude'] = df['Constant Longitude'].shift(-1)

    # Calculate the distance between each point and the next one
    df['Distance'] = df.apply(lambda row: geodesic((row['Constant Latitude'], row['Constant Longitude']), 
                                                   (row['Next Latitude'], row['Next Longitude'])).meters if pd.notnull(row['Next Latitude']) and pd.notnull(row['Next Longitude']) else np.nan, axis=1)

    # Shift the Distance column down to put the distance in the ending row
    df['Distance'] = df['Distance'].shift(1)

    # Calculate the time difference between each point and the next one
    df['Time Difference (minutes)'] = df['RecievedDate'].diff().dt.total_seconds() / 60

    # Create a mask where the distance is less than or equal to 10 meters and the time difference is greater than or equal to 2 minutes
    mask = (df['Distance'] <= 10) & (df['Time Difference (minutes)'] >= 2)

    # Assign a Mean Latitude, Mean Longitude value for the rows where the mask is True
    df.loc[mask, 'Mean Latitude'] = df.loc[mask, 'Constant Latitude']
    df.loc[mask, 'Mean Longitude'] = df.loc[mask, 'Constant Longitude']

    # Define aggregation functions for each column
    agg_funcs = {col: 'first' for col in df.columns if col not in ['Mean Latitude', 'Mean Longitude']}
    agg_funcs['Time Difference (minutes)'] = 'sum'

    # Group the dataframe by the date, Mean Latitude, and Mean Longitude and apply the aggregation functions
    df = df[mask].groupby([df['RecievedDate'].dt.date, 'Mean Latitude', 'Mean Longitude'], as_index=False).agg(agg_funcs)

    # Return the dataframe
    return df

dfSFA_GPSDataDBSCAN11 = dfSFA_GPSDataDBSCAN1.copy()
# Call the function to process the result of DBSCAN clustering
new_df11 = process_result(dfSFA_GPSDataDBSCAN11)

# Print the first 10 rows of the new dataframe
print(new_df11.head(10))


def process_result(df):
    # Convert RecievedDate to datetime
    df['RecievedDate'] = pd.to_datetime(df['RecievedDate'])

    # Create a new column 'Date' that contains only the date part of 'RecievedDate'
    df['Date'] = df['RecievedDate'].dt.date

    # Shift the Latitude and Longitude columns to get the next coordinates
    df['Next Latitude'] = df['Constant Latitude'].shift(-1)
    df['Next Longitude'] = df['Constant Longitude'].shift(-1)

    # Calculate the distance between each point and the next one
    df['Distance'] = df.apply(lambda row: geodesic((row['Constant Latitude'], row['Constant Longitude']), 
                                                   (row['Next Latitude'], row['Next Longitude'])).meters if pd.notnull(row['Next Latitude']) and pd.notnull(row['Next Longitude']) else np.nan, axis=1)

    # Shift the Distance column down to put the distance in the ending row
    df['Distance'] = df['Distance'].shift(1)

    # Calculate the time difference between each point and the next one
    df['Time Difference (minutes)'] = df['RecievedDate'].diff().dt.total_seconds() / 60

    # Create a mask where the distance is less than or equal to 10 meters and the time difference is greater than or equal to 2 minutes
    mask = (df['Distance'] <= 10) & (df['Time Difference (minutes)'] >= 2)

    # Assign a Mean Latitude, Mean Longitude value for the rows where the mask is True
    df.loc[mask, 'Mean Latitude'] = df.loc[mask, 'Constant Latitude']
    df.loc[mask, 'Mean Longitude'] = df.loc[mask, 'Constant Longitude']

    # Define aggregation functions for each column
    agg_funcs = {col: 'first' for col in df.columns if col not in ['Date', 'Mean Latitude', 'Mean Longitude']}
    agg_funcs['Time Difference (minutes)'] = 'sum'

    # Group the dataframe by the Date, Mean Latitude, and Mean Longitude and apply the aggregation functions
    df = df[mask].groupby(['Date', 'Mean Latitude', 'Mean Longitude'], as_index=False).agg(agg_funcs)

    # Drop duplicates based on Date, Mean Latitude, and Mean Longitude
    df = df.drop_duplicates(subset=['Date', 'Mean Latitude', 'Mean Longitude'])

    # Return the dataframe
    return df

dfSFA_GPSDataDBSCAN111 = dfSFA_GPSDataDBSCAN1.copy()
# Call the function to process the result of DBSCAN clustering
new_df111 = process_result(dfSFA_GPSDataDBSCAN111)

# Print the first 10 rows of the new dataframe
print(new_df111.head(10))

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Initialize the geolocator
geolocator = Nominatim(user_agent="poi_search")

# Add a delay between requests to avoid being blocked
geocode = RateLimiter(geolocator.reverse, min_delay_seconds=0.05)

def get_location(row):
    return geocode((row['Latitude'], row['Longitude']))

def get_address_component(loc, component):
    keys = ['city', 'town', 'residential', 'hamlet', 'suburb', 'village', 'amenity']
    if component in keys:
        for key in keys:
            component_value = loc.raw.get('address', {}).get(key)
            if component_value is not None:
                return component_value
    else:
        return loc.raw.get('address', {}).get(component, None)

# Create a location column in the DataFrame
dfGeoCustomer['location'] = dfGeoCustomer.apply(get_location, axis=1)

# Get city or town
dfGeoCustomer['city'] = dfGeoCustomer['location'].apply(lambda loc: get_address_component(loc, 'city'))

# Get district and province
dfGeoCustomer['district'] = dfGeoCustomer['location'].apply(lambda loc: get_address_component(loc, 'state_district'))
dfGeoCustomer['province'] = dfGeoCustomer['location'].apply(lambda loc: get_address_component(loc, 'state'))


from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import StandardScaler

# Assuming dfSFA_GPSData is the original DataFrame
dfSFA_GPSDataKMeans = dfSFA_GPSData.copy()

# Convert Latitude and Longitude into radians for KMeans
coords = dfSFA_GPSDataKMeans[['Latitude', 'Longitude']].values

# Using KMeans to cluster the GPS data points
kmeans = KMeans(n_clusters=600)
dfSFA_GPSDataKMeans['Cluster'] = kmeans.fit_predict(StandardScaler().fit_transform(coords))

# Calculating Constant Latitude and Longitude for each Cluster by taking mean value of each cluster points.
dfSFA_GPSDataKMeans['Constant Latitude'] = dfSFA_GPSDataKMeans.groupby('Cluster')['Latitude'].transform('mean')
dfSFA_GPSDataKMeans['Constant Longitude'] = dfSFA_GPSDataKMeans.groupby('Cluster')['Longitude'].transform('mean')

# Now, let's check if a particular location from dfGeoCustomer falls within any of these clusters
# Convert Latitude and Longitude into radians for haversine_distances
customer_coords = dfGeoCustomer[['Latitude', 'Longitude']].values
cluster_coords = dfSFA_GPSDataKMeans[['Constant Latitude', 'Constant Longitude']].drop_duplicates().values

# Calculate haversine_distances and convert to kilometers
distances = haversine_distances(np.radians(customer_coords), np.radians(cluster_coords)) * 6371

# Check if the minimum distance is within the error tolerance (epsilon)
dfGeoCustomer['Visited'] = np.min(distances, axis=1) <= 0.05

dfGeoCustomer_True = dfGeoCustomer[dfGeoCustomer['Visited'] == True]
print(dfGeoCustomer_True)

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import haversine_distances
from sklearn.preprocessing import StandardScaler

dfSFA_GPSDataMBKMEANS1 = dfSFA_GPSData.copy()
# Convert Latitude and Longitude into radians for MiniBatchKMeans
coords = dfSFA_GPSDataMBKMEANS1[['Latitude', 'Longitude']].values

# Using MiniBatchKMeans to cluster the GPS data points
mbk = MiniBatchKMeans(n_clusters=600, batch_size=10000)
dfSFA_GPSDataMBKMEANS1['Cluster'] = mbk.fit_predict(StandardScaler().fit_transform(coords))

# Calculating Constant Latitude and Longitude for each Cluster by taking mean value of each cluster points.
dfSFA_GPSDataMBKMEANS1['Constant Latitude'] = dfSFA_GPSDataMBKMEANS1.groupby('Cluster')['Latitude'].transform('mean')
dfSFA_GPSDataMBKMEANS1['Constant Longitude'] = dfSFA_GPSDataMBKMEANS1.groupby('Cluster')['Longitude'].transform('mean')

# Now, let's check if a particular location from dfGeoCustomer falls within any of these clusters
# Convert Latitude and Longitude into radians for haversine_distances
customer_coords = dfGeoCustomer[['Latitude', 'Longitude']].values
cluster_coords = dfSFA_GPSDataMBKMEANS1[['Constant Latitude', 'Constant Longitude']].drop_duplicates().values

# Calculate haversine_distances and convert to kilometers
distances = haversine_distances(np.radians(customer_coords), np.radians(cluster_coords)) * 6371

# Check if the minimum distance is within the error tolerance (epsilon)
dfGeoCustomer['Visited'] = np.min(distances, axis=1) <= 0.100


dfGeoCustomer_True = dfGeoCustomer[dfGeoCustomer['Visited'] == True]
print(dfGeoCustomer_True)


dfSFA_GPSDataMBKMEANS1.head()


import hdbscan

dfSFA_GPSDataHDBSCAN1 = dfSFA_GPSData.copy()
# Convert Latitude and Longitude into radians for HDBSCAN
coords = np.radians(dfSFA_GPSDataHDBSCAN1[['Latitude', 'Longitude']].values)

# Using HDBSCAN to cluster the GPS data points
hdb = hdbscan.HDBSCAN(min_cluster_size=10, metric='haversine')
hdb.fit(coords)

cluster_labels  = hdb.labels_
num_clusters = len(set(cluster_labels))

# Adding Cluster Number to DataFrame
dfSFA_GPSDataHDBSCAN1['Cluster'] = cluster_labels

# Calculating Constant Latitude and Longitude for each Cluster by taking mean value of each cluster points.
dfSFA_GPSDataHDBSCAN1['Constant Latitude'] = dfSFA_GPSDataHDBSCAN1.groupby('Cluster')['Latitude'].transform('mean')
dfSFA_GPSDataHDBSCAN1['Constant Longitude'] = dfSFA_GPSDataHDBSCAN1.groupby('Cluster')['Longitude'].transform('mean')

print(dfSFA_GPSDataHDBSCAN1[['Constant Latitude', 'Constant Longitude']])

