#!/usr/bin/env python
# coding: utf-8

# # VacationPy
# ---
# 
# ## Starter Code to Import Libraries and Load the Weather and Coordinates Data

# In[1]:


# Dependencies and Setup
import hvplot.pandas
import pandas as pd
import requests

# Import API key
from api_key import geoapify_key


# In[2]:


# Load the CSV file created in Part 1 into a Pandas DataFrame
city_data_df = pd.read_csv("Resources/cities.csv")

# Display sample data
city_data_df.head()


# ---
# 
# ### Step 1: Create a map that displays a point for every city in the `city_data_df` DataFrame. The size of the point should be the humidity in each city.

# In[3]:


get_ipython().run_cell_magic('capture', '--no-display', "\n# Configure the map plot\nmap_plot = city_data_df.hvplot.points(\n    x='Longitude', \n    y='Latitude',   \n    size='Humidity',  \n    geo=True,\n    tiles='OSM',  # OpenStreetMap\n    title='City Humidity Map'\n)\n# Display the map\nmap_plot\n")


# ### Step 2: Narrow down the `city_data_df` DataFrame to find your ideal weather condition

# In[21]:


# Narrow down cities that fit criteria and drop any results with null values
ideal_temp_min = 20
ideal_temp_max = 45
ideal_humidity_max = 90

ideal_weather_df = city_data_df[
    (city_data_df['Max Temp'] >= ideal_temp_min) &
    (city_data_df['Max Temp'] <= ideal_temp_max) &
    (city_data_df['Humidity'] < ideal_humidity_max)
]

# Drop any rows with null values
ideal_weather_df_clean = ideal_weather_df.dropna()

# Display sample data
ideal_weather_df_clean


# ### Step 3: Create a new DataFrame called `hotel_df`.

# In[18]:


# Use the Pandas copy function to create DataFrame called hotel_df to store the city, country, coordinates, and humidity
hotel_df = ideal_weather_df[['City Name', 'Country', 'Latitude', 'Longitude', 'Humidity']].copy()

# Add an empty column, "Hotel Name," to the DataFrame so you can store the hotel found using the Geoapify API
hotel_df['Hotel Name'] = ''

# Display sample data
hotel_df


# ### Step 4: For each city, use the Geoapify API to find the first hotel located within 10,000 metres of your coordinates.

# In[57]:


# Set parameters to search for a hotel
radius = 10000
params = { 
    "limit": 20,  # Limit to the first hotel found
    "apiKey": geoapify_key,
    "categories": "accommodation.hotel",
}

# Print a message to follow up the hotel search
print("Starting hotel search")

# Iterate through the hotel_df DataFrame
for index, row in hotel_df.iterrows():
    # get latitude, longitude from the DataFrame
    latitude = row["Latitude"]  # Replace with the actual column name for latitude
    longitude = row['Longitude']  # Replace with the actual column name for longitude
    
    print(f"{longitude},{latitude} -LATLONGS")
    
    # Add the current city's latitude and longitude to the params dictionary
    params["filter"]=f"circle:{longitude},{latitude},{radius}"
    params["bias"] = f"proximity:{longitude},{latitude}"


    # Set base URL
    base_url = "https://api.geoapify.com/v2/places"

    # Make and API request using the params dictionary
    name_address = "https://api.geoapify.com/v2/places"
    response = requests.get(base_url, params=params)
    
    # Convert the API response to JSON format
    if response.status_code == 200:
        data = response.json()  # Convert the response to JSON format
    # Now you can process the JSON data as needed
        if data['features']:
            hotel_info = data['features'][0]['properties']
            name_address = {
                "name": hotel_info.get('name'),  # Get the hotel name
                "address": hotel_info.get('address_line2')  # Get the address
            }
            print(name_address)  # Display the hotel name and address
        else:
            print("No hotel found.")
    else:
        print("Error:", response.status_code)

    # Grab the first hotel from the results and store the name in the hotel_df DataFrame
    try:
        hotel_df.loc[index, "Hotel Name"] = name_address["features"][0]["properties"]["name"]
    except Exception as e:
        print(f"Key error: {e}")
        # If no hotel is found, set the hotel name as "No hotel found".
        hotel_df.loc[index, "Hotel Name"] = "No hotel found"

    # Log the search results
    print(f"{hotel_df.loc[index, 'City Name']} - nearest hotel: {hotel_df.loc[index, 'Hotel Name']}")

# Display sample data
hotel_df


# ### Step 5: Add the hotel name and the country as additional information in the hover message for each city in the map.

# In[36]:


get_ipython().run_cell_magic('capture', '--no-display', '\n# Configure the map plot\nmap_plot = hotel_df.hvplot.points(\n    "Latitude",\n    "Longitude",\n    geo=True,\n    tiles="OSM",\n    frame_width=800,\n    frame_height=600,\n    hover_cols=["Hotel Name", "Country"]\n)\n# Display the map\nmap_plot\n')


# In[ ]:




