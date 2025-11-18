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
# # <span style="font-width:bold; font-size: 3rem; color:#333;">Training Pipeline</span>
# 
# ## 🗒️ This notebook is divided into the following sections:
# 
# 1. Select features for the model and create a Feature View with the selected features
# 2. Create training data using the feature view
# 3. Train model
# 4. Evaluate model performance
# 5. Save model to model registry

# %% [markdown]
# ### <span style='color:#ff5f27'> 📝 Imports

# %%
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, r2_score
import hopsworks
from mlfs.airquality import util
import json

import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# ## <span style="color:#ff5f27;"> 📡 Connect to Hopsworks Feature Store </span>

# %%
# Check if HOPSWORKS_API_KEY env variable is set or if it is set in ~/.env
if settings.HOPSWORKS_API_KEY is not None:
    api_key = settings.HOPSWORKS_API_KEY.get_secret_value()
    os.environ['HOPSWORKS_API_KEY'] = api_key
project = hopsworks.login()
fs = project.get_feature_store()

secrets = hopsworks.get_secrets_api()
location_str = secrets.get_secret("SENSOR_LOCATION_JSON").value
location = json.loads(location_str)
country = location['country']
city = location['city']
street = location['street']

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
# 
# ## <span style="color:#ff5f27;"> 🖍 Feature View Creation and Retrieving </span>

# %%
# Select features for training data.
selected_features = air_quality_fg.select(['pm25', 'date']).join(weather_fg.select_features(), on=['city'])

# %% [markdown]
# ### Feature Views
# 
# `Feature Views` are selections of features from different **Feature Groups** that make up the input and output API (or schema) for a model. A **Feature Views** can create **Training Data** and also be used in Inference to retrieve inference data.
# 
# The Feature Views allows a schema in form of a query with filters, defining a model target feature/label and additional transformation functions (declarative feature encoding).
# 
# In order to create Feature View we can use `FeatureStore.get_or_create_feature_view()` method.
# 
# You can specify the following parameters:
# 
# - `name` - name of a feature group.
# 
# - `version` - version of a feature group.
# 
# - `labels`- our target variable.
# 
# - `transformation_functions` - declarative feature encoding (not used here)
# 
# - `query` - selected features/labels for the model 

# %%
feature_view = fs.get_or_create_feature_view(
    name='air_quality_fv',
    description="weather features with air quality as the target",
    version=1,
    labels=['pm25'],
    query=selected_features,
)

# %% [markdown]
# ## <span style="color:#ff5f27;">🪝 Split the training data into train/test data sets </span>
# 
# We use a time-series split here, with training data before this date `start_date_test_data` and test data after this date

# %%
start_date_test_data = "2025-05-01"
# Convert string to datetime object
test_start = datetime.strptime(start_date_test_data, "%Y-%m-%d")

# %%
X_train, X_test, y_train, y_test = feature_view.train_test_split(test_start=test_start)

# %%
X_train

# %%
X_features = X_train.drop(columns=['date'])
X_test_features = X_test.drop(columns=['date'])

# %%
y_train

# %% [markdown]
# The `Feature View` is now saved in Hopsworks and you can retrieve it using `FeatureStore.get_feature_view(name='...', version=1)`.

# %% [markdown]
# ---

# %% [markdown]
# ## <span style="color:#ff5f27;">🧬 Modeling</span>
# 
# We will train a regression model to predict pm25 using our 4 features (wind_speed, wind_dir, temp, precipitation)

# %%
# Creating an instance of the XGBoost Regressor
xgb_regressor = XGBRegressor()

# Fitting the XGBoost Regressor to the training data
xgb_regressor.fit(X_features, y_train)


# %%
# Predicting target values on the test set
y_pred = xgb_regressor.predict(X_test_features)

# Calculating Mean Squared Error (MSE) using sklearn
mse = mean_squared_error(y_test.iloc[:, 0], y_pred)
print("MSE:", mse)

# Calculating R squared using sklearn
r2 = r2_score(y_test.iloc[:, 0], y_pred)
print("R squared:", r2)

# %%
df = y_test
df['predicted_pm25'] = y_pred

# %%
df['date'] = X_test['date']
df = df.sort_values(by=['date'])
df.head(5)

# %%
# Creating a directory for the model artifacts if it doesn't exist
model_dir = "air_quality_model"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
images_dir = model_dir + "/images"
if not os.path.exists(images_dir):
    os.mkdir(images_dir)

# %%
file_path = images_dir + "/pm25_hindcast.png"
plt = util.plot_air_quality_forecast(city, street, df, file_path, hindcast=True)
plt.show()

# %%
# Plotting feature importances using the plot_importance function from XGBoost
plot_importance(xgb_regressor)
feature_importance_path = images_dir + "/feature_importance.png"
plt.savefig(feature_importance_path)
plt.show()

# %% [markdown]
# ---

# %% [markdown]
# ## <span style='color:#ff5f27'>🗄 Model Registry</span>
# 
# One of the features in Hopsworks is the model registry. This is where you can store different versions of models and compare their performance. Models from the registry can then be served as API endpoints.

# %%
# Saving the XGBoost regressor object as a json file in the model directory
xgb_regressor.save_model(model_dir + "/model.json")

# %%
res_dict = {
    "MSE": str(mse),
    "R squared": str(r2),
}

# %%
mr = project.get_model_registry()

# Creating a Python model in the model registry named 'air_quality_xgboost_model'

aq_model = mr.python.create_model(
    name="air_quality_xgboost_model",
    metrics=res_dict,
    feature_view=feature_view,
    description="Air Quality (PM2.5) predictor",
)

# Saving the model artifacts to the 'air_quality_model' directory in the model registry
aq_model.save(model_dir)

# %% [markdown]
# ---
# ## <span style="color:#ff5f27;">⏭️ **Next:** Part 04: Batch Inference</span>
# 
# In the following notebook you will use your model for Batch Inference.
# 

# %%



