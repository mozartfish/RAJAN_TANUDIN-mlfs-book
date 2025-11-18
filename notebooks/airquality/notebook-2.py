# %%
import sys
from pathlib import Path
import os


def is_google_colab() -> bool:
    if "google.colab" in str(get_ipython()):
        return True
    return False


def clone_repository() -> None:
    !git clone https://github.com/featurestorebook/mlfs-book.git
    %cd mlfs-book


def install_dependencies() -> None:
    !pip install --upgrade uv
    !uv pip install --all-extras --system --requirement pyproject.toml


if is_google_colab():
    clone_repository()
    install_dependencies()
    root_dir = str(Path().absolute())
    print("Google Colab environment")
else:
    root_dir = Path().absolute()
    # Strip ~/notebooks/ccfraud from PYTHON_PATH if notebook started in one of these subdirectories
    if root_dir.parts[-1:] == ('airquality',):
        root_dir = Path(*root_dir.parts[:-1])
    if root_dir.parts[-1:] == ('notebooks',):
        root_dir = Path(*root_dir.parts[:-1])
    root_dir = str(root_dir)
    print("Local environment")

# Add the root directory to the `PYTHONPATH` to use the `recsys` Python module from the notebook.
if root_dir not in sys.path:
    sys.path.append(root_dir)
print(f"Added the following directory to the PYTHONPATH: {root_dir}")

# Set the environment variables from the file <root_dir>/.env
from mlfs import config

if os.path.exists(f"{root_dir}/.env"):
    settings = config.HopsworksSettings(_env_file=f"{root_dir}/.env")

# %% [markdown]
# <span style="font-width:bold; font-size: 3rem; color:#333;">- Part 02: Daily Feature Pipeline for Air Quality (aqicn.org) and weather (openmeteo)</span>
# 
# ## 🗒️ This notebook is divided into the following sections:
# 1. Download and Parse Data
# 2. Feature Group Insertion
# 
# 
# __This notebook should be scheduled to run daily__
# 
# In the book, we use a GitHub Action stored here:
# [.github/workflows/air-quality-daily.yml](https://github.com/featurestorebook/mlfs-book/blob/main/.github/workflows/air-quality-daily.yml)
# 
# However, you are free to use any Python Orchestration tool to schedule this program to run daily.

# %% [markdown]
# ### <span style='color:#ff5f27'> 📝 Imports

# %%
import datetime
import time
import requests
import pandas as pd
import hopsworks
from mlfs.airquality import util
from mlfs import config
import json
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# ## <span style='color:#ff5f27'> 🌍 Get the Sensor URL, Country, City, Street names from Hopsworks </span>
# 
# __Update the values in the cell below.__
# 
# __These should be the same values as in notebook 1 - the feature backfill notebook__
# 

# %%
project = hopsworks.login()
fs = project.get_feature_store()
secrets = hopsworks.get_secrets_api()

# This line will fail if you have not registered the AQICN_API_KEY as a secret in Hopsworks
AQICN_API_KEY = secrets.get_secret("AQICN_API_KEY").value
location_str = secrets.get_secret("SENSOR_LOCATION_JSON").value
location = json.loads(location_str)

country = location['country']
city = location['city']
street = location['street']
aqicn_url = location['aqicn_url']
latitude = location['latitude']
longitude = location['longitude']

today = datetime.date.today()

location_str

# %% [markdown]
# ### <span style="color:#ff5f27;"> 🔮 Get references to the Feature Groups </span>

# %%
# Retrieve feature groups
air_quality_fg = fs.get_feature_group(
    name='air_quality',
    version=1,
)
weather_fg = fs.get_feature_group(
    name='weather',
    version=1,
)

# %% [markdown]
# ---

# %% [markdown]
# ## <span style='color:#ff5f27'> 🌫 Retrieve Today's Air Quality data (PM2.5) from the AQI API</span>
# 

# %%
import requests
import pandas as pd

aq_today_df = util.get_pm25(aqicn_url, country, city, street, today, AQICN_API_KEY)
aq_today_df

# %%
aq_today_df.info()

# %% [markdown]
# ## <span style='color:#ff5f27'> 🌦 Get Weather Forecast data</span>

# %%
hourly_df = util.get_hourly_weather_forecast(city, latitude, longitude)
hourly_df = hourly_df.set_index('date')

# We will only make 1 daily prediction, so we will replace the hourly forecasts with a single daily forecast
# We only want the daily weather data, so only get weather at 12:00
daily_df = hourly_df.between_time('11:59', '12:01')
daily_df = daily_df.reset_index()
daily_df['date'] = pd.to_datetime(daily_df['date']).dt.date
daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df['city'] = city
daily_df

# %%
daily_df.info()

# %% [markdown]
# ## <span style="color:#ff5f27;">⬆️ Uploading new data to the Feature Store</span>

# %%
# Insert new data
air_quality_fg.insert(aq_today_df)

# %%
# Insert new data
weather_fg.insert(daily_df, wait=True)

# %% [markdown]
# ## <span style="color:#ff5f27;">⏭️ **Next:** Part 03: Training Pipeline
#  </span> 
# 
# In the following notebook you will read from a feature group and create training dataset within the feature store
# 


