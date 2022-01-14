import rasterio
from pystac_client import Client
from odc.stac import stac_load

# Open the stac catalogue
deafrica_stac_address = 'https://explorer.digitalearth.africa/stac'
catalog = Client.open(deafrica_stac_address)

# Construct a config dictionary for desired product
# Use definition from https://explorer.digitalearth.africa/products/rainfall_chirps_daily

product_name = 'rainfall_chirps_daily'
measurement_name = 'rainfall'
measurement_dtype = 'float32'
measurement_nodata = -9999
measurement_unit = 'mm'

config = {
    product_name: {
        "assets": {
            measurement_name: {
                "data_type": measurement_dtype,
                "nodata": measurement_nodata, 
                "unit": measurement_unit,
            }
        }
    }
}

def load_chirps_data(lat,lon, start_date, end_date):
    """returns a dataset with precipitation values 
    for selected area of interest
    """
    buffer_lat, buffer_lon = 0.02, 0.02

    #join lat,lon,buffer to get bounding box
    xmin, xmax = (lon - buffer_lon, lon + buffer_lon)
    ymin, ymax = (lat + buffer_lat, lat - buffer_lat)
    
    # Construct a bounding box to search over
    # [xmin, ymin, xmax, ymax] in latitude and longitude
    bbox = [xmin, ymin, xmax, ymax]
   
    # Construct a time range to search over
    timerange = f'{start_date}/{end_date}'

    # Choose the product/s to load
    products = ['rainfall_chirps_daily']

    # Identify all data matching the above:
    query = catalog.search(
        bbox=bbox,
        collections=products,
        datetime=timerange
    )

    items = list(query.get_items())
    print(f"Found: {len(items):d} datasets")

    # Load the items found, using native CRS and Resolution from explorer definiton
    # Loads as a dask array
    crs = 'EPSG:4326'
    resolution = 0.05

    data = stac_load(
        items,
        crs=crs,
        resolution=resolution,
        chunks={},
        groupby="solar_day",
        stac_cfg=config,
    )

    # Subset the data
    subset = data.sel(longitude=slice(bbox[0], bbox[2]), latitude=slice(bbox[1], bbox[3]))

    # Load into memory, masking no-data values
    with rasterio.Env(AWS_S3_ENDPOINT='s3.af-south-1.amazonaws.com', AWS_NO_SIGN_REQUEST='YES'):
        loaded_data = subset.where(subset.rainfall != measurement_nodata).persist()
        #loaded_data = data.where(data.rainfall != measurement_nodata).persist()

    return loaded_data
