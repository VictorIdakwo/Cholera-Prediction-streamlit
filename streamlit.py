import streamlit as st
import geopandas as gpd
import joblib
import folium
from folium import Choropleth
from sklearn.preprocessing import StandardScaler
from streamlit_folium import st_folium
from PIL import Image  # For loading images

# Paths to model, scaler, and shapefile
model_path = "models/random-forest-model.joblib"
scaler_path = "models/scaler_rf.joblib"
data_path = "Population_Cholera.shp"
logo_path = "static/eHA-logo-blue_320x132.png"  # Company logo

# Load trained model and scaler
trained_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load the shapefile (GeoDataFrame)
prediction_data = gpd.read_file(data_path)

# Ensure CRS is set to EPSG:4326 (WGS84)
if prediction_data.crs.to_epsg() != 4326:
    prediction_data = prediction_data.to_crs(epsg=4326)

# Features
all_features = ['Aspect', 'Elevatn', 'builtupr', 'LST', 'LULCC', 'NDVI', 'NDWI', 'PopDnsty', 'Poverty', 'Prcpittn', 'Slope', 'rwi']
display_features = ['Aspect', 'Elevation', 'Built-up Area', 'LST', 'Land use/Cover', 'NDVI', 'NDWI', 'Pop Density', 'Poverty', 'Precipitation', 'Slope', 'Relative Wealth Index']

# Base year
base_year = 2024

# Adjust features for future years
def adjust_for_future(X_pred, year):
    """Modifies features based on the selected future year."""
    year_difference = year - base_year
    if year_difference > 0:
        X_pred['PopDnsty'] *= (1 + 0.02 * year_difference)  # 2% yearly growth
        X_pred['Prcpittn'] *= (1 + 0.01 * year_difference)  # 1% yearly increase
        X_pred['LST'] += 0.5 * year_difference  # Temperature increase
    return X_pred

# Generate prediction map
def generate_map(selected_features, year):
    """Creates and returns a Folium map with predictions."""
    X_pred = prediction_data[all_features].copy()

    # Set unselected features to zero
    for feature in all_features:
        if feature not in selected_features:
            X_pred[feature] = 0  

    # Adjust for future projections
    X_pred = adjust_for_future(X_pred, year)

    # Handle missing values
    X_pred = X_pred.fillna(X_pred.mean())

    # Scale and predict cases
    X_pred_scaled = scaler.transform(X_pred)
    prediction_data['pred_cases'] = trained_model.predict(X_pred_scaled).astype(int)

    # Create Folium map
    center = prediction_data.geometry.centroid.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=8)

    # Add choropleth layer
    Choropleth(
        geo_data=prediction_data,
        data=prediction_data,
        columns=['ward_name', 'pred_cases'],
        key_on='feature.properties.ward_name',
        fill_color='YlOrRd',
        fill_opacity=1.0,
        line_opacity=0.1,
        legend_name=f'Predicted Cases in {year}'
    ).add_to(m)

    # Add tooltips
    for _, row in prediction_data.iterrows():
        folium.GeoJson(
            row.geometry,
            tooltip=folium.Tooltip(
                f"Predicted Cases: {row['pred_cases']}<br>"
                f"Ward: {row['ward_name']}<br>"
                f"LGA: {row['lga_name']}<br>"
                f"Year: {year}"
            ),
        ).add_to(m)

    return m

# Streamlit UI
st.set_page_config(page_title="Cholera Prediction", layout="wide")

# Sidebar
with st.sidebar:
    # Display company logo at the top left inside the sidebar
    image = Image.open(logo_path)
    st.image(image, use_container_width=True)  # ✅ Logo at the top left

    st.header("Feature Selection")
    selected_features = st.multiselect("Select Features", display_features, default=display_features)

    st.header("Year Selection")
    year = st.slider("Select Year", 2024, 2035, 2024)

# Main Page
st.title("Cholera Case Prediction")

# Generate and display the map
st.subheader(f"Predicted Cholera Cases in {year}")
cholera_map = generate_map(selected_features, year)
st_folium(cholera_map, width=1000, height=600)
