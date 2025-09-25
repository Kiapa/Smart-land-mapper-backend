import ee
import math
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import google.auth
from google.oauth2.service_account import Credentials

# --- NEW: Initialize Google Earth Engine with a Service Account ---
try:
    # 1. Define the path to your service account key file
    KEY_FILE_PATH = 'gen-lang-client-0427725945-dc7fe2faa8ab.json'

    # 2. Authenticate using the key file
    # This automatically handles the credentials for the ee.Initialize() call
    credentials, project_id = google.auth.load_credentials_from_file(KEY_FILE_PATH)
    scoped_credentials = credentials.with_scopes([
        'https://www.googleapis.com/auth/cloud-platform',
        'https://www.googleapis.com/auth/earthengine'
    ])

    ee.Initialize(credentials=scoped_credentials)
    print("Google Earth Engine initialized successfully using Service Account.")

except Exception as e:
    print(f"Error initializing Google Earth Engine with Service Account: {e}")
    print(
        "Please ensure 'service-account-key.json' is in the correct path and the service account is registered with GEE.")
    # Exit if we can't initialize, as the API will not work.
    exit()


# --- Pydantic Models for Data Validation (No changes below this line) ---

class Point(BaseModel):
    """Represents a single coordinate point from the mobile app."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class AnalysisRequest(BaseModel):
    """The request model for an analysis, containing the polygon points."""
    points: List[Point] = Field(..., min_items=4)


# Define models for the structured response
class SlopeStats(BaseModel):
    mean: float
    min: float
    max: float


class RoadDistanceStats(BaseModel):
    min: float
    mean: float
    max: float


class ShapeMetrics(BaseModel):
    area_sqm: float
    perimeter_m: float
    regularity_score: float


class AnalysisResult(BaseModel):
    """The detailed result of the parcel analysis to be sent back to the app."""
    slope_stats: SlopeStats
    vegetation_cover_percent: float
    road_distance_stats: RoadDistanceStats
    building_distance_m: float
    shape_metrics: ShapeMetrics
    suitability_score: float  # We'll add a scoring logic


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Real Parcel Analysis API (GEE)",
    description="Uses Google Earth Engine to perform geospatial analysis on land parcels.",
    version="1.0.0"
)


# --- Your GEE Analysis Functions (Adapted for API) ---

def analyze_slope(roi: ee.Geometry) -> dict:
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    slope_roi = slope.clip(roi)
    stats = slope_roi.reduceRegion(
        reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(), sharedInputs=True),
        geometry=roi,
        scale=30,
        maxPixels=1e9
    ).getInfo()
    return {
        'mean': stats.get('slope_mean'),
        'min': stats.get('slope_min'),
        'max': stats.get('slope_max')
    }


def analyze_vegetation_cover(roi: ee.Geometry) -> float:
    # UPDATED: Use the harmonized Sentinel-2 dataset as recommended by GEE logs
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(roi) \
        .filterDate("2024-01-01", "2024-12-31") \
        .select(['B8', 'B4']) \
        .median()
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    veg = ndvi.gt(0.6).rename('veg')
    pixel_area = ee.Image.pixelArea()
    total_area_img = pixel_area.updateMask(ee.Image.constant(1).clip(roi))
    total_area = total_area_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=10, maxPixels=1e13).get(
        'area').getInfo()
    veg_area_img = pixel_area.updateMask(veg)
    veg_area = veg_area_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=10, maxPixels=1e13).get(
        'area').getInfo()
    return 100 * veg_area / total_area if total_area is not None and total_area > 0 else 0


def analyze_distance_to_roads(roi: ee.Geometry) -> dict:
    roads = ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/Africa")
    distance_to_roads = roads.distance(ee.Number(10000))
    dist_roi = distance_to_roads.clip(roi)
    stats = dist_roi.reduceRegion(
        reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(), sharedInputs=True),
        geometry=roi,
        scale=30,
        maxPixels=1e13
    ).getInfo()
    return {
        'min': stats.get('distance_min'),
        'mean': stats.get('distance_mean'),
        'max': stats.get('distance_max')
    }


# --- OPTIMIZED: New, much faster function for building distance ---
def analyze_distance_to_buildings(roi: ee.Geometry) -> float:
    # 1. Define a reasonable search radius (e.g., 5000 meters)
    search_radius_m = 5000

    # 2. Create a buffer around the user's ROI to define a small search area.
    search_area = roi.buffer(search_radius_m)

    # 3. Load the building dataset and filter it to our small search area FIRST.
    nearby_buildings = ee.FeatureCollection("projects/sat-io/open-datasets/MSBuildings/Kenya").filterBounds(search_area)

    # 4. Check if any buildings were found nearby to avoid errors.
    nearby_building_count = nearby_buildings.size().getInfo()
    if nearby_building_count == 0:
        print("   -> No buildings found within 5km.")
        # Return a large number as no buildings are close.
        return search_radius_m

        # 5. Calculate distance from the parcel's center to the NEAREST of the found buildings.
    # This is much faster than creating a distance raster for the whole country.
    roi_centroid = roi.centroid(1)

    # Add a 'distance' property to each nearby building from the centroid
    nearby_with_distance = nearby_buildings.map(lambda f: f.set('distance', roi_centroid.distance(f.geometry())))

    # Sort by that distance and take the first (closest) one.
    nearest_building = nearby_with_distance.sort('distance').first()

    # Get the distance value.
    distance = nearest_building.get('distance').getInfo()

    return distance


def analyze_shape_metrics(roi: ee.Geometry) -> dict:
    area = roi.area(1).getInfo()
    perimeter = roi.perimeter(1).getInfo()
    pa_ratio = perimeter / area if area and area > 0 else 0
    shape_index = (4 * math.pi * area) / (perimeter ** 2) if perimeter and perimeter > 0 else 0
    bounds = roi.bounds().coordinates().getInfo()[0]
    xs = [pt[0] for pt in bounds]
    ys = [pt[1] for pt in bounds]
    length = (max(xs) - min(xs)) * 111000
    width = (max(ys) - min(ys)) * 111000
    elongation = max(length, width) / min(length, width) if min(length, width) > 0 else float('inf')
    elong_score = 1 / (1 + abs(elongation - 1)) if elongation != float('inf') else 0
    shape_score = min(shape_index / 0.78, 1.0) if shape_index > 0 else 0
    pa_square = 4 * math.sqrt(area) / area if area > 0 else 0
    pa_score = min(pa_square / pa_ratio, 1.0) if pa_ratio > 0 else 0
    regularity_score = 100 * (0.6 * elong_score + 0.2 * shape_score + 0.2 * pa_score)
    return {"area_sqm": area, "perimeter_m": perimeter, "regularity_score": regularity_score}


def calculate_suitability_score(slope_stats, road_stats, veg_cover):
    """A simple scoring model based on the analysis results."""
    score = 100
    # Penalize for steep slope
    if slope_stats.get('mean', 0) > 15:
        score -= (slope_stats.get('mean', 0) - 15) * 2
    # Penalize for being far from roads
    if road_stats.get('min', 0) > 500:
        score -= (road_stats.get('min', 0) - 500) / 20
    # Reward for good vegetation, penalize for none or too much (e.g., protected forest)
    if 30 <= veg_cover <= 80:
        score += 5
    else:
        score -= 5
    return max(0, min(100, score))


# --- API Endpoint ---
@app.post("/analyze", response_model=AnalysisResult)
def analyze_parcel(request: AnalysisRequest):
    """
    Receives polygon coordinates from the app and returns a full GEE analysis.
    """
    try:
        # --- NEW: ADDED DIAGNOSTIC PRINT STATEMENTS ---
        print("\n--- [START] New Analysis Request Received ---")
        coords = [[p.longitude, p.latitude] for p in request.points]
        roi = ee.Geometry.Polygon(coords)

        print("1. Starting slope analysis...")
        slope_data = analyze_slope(roi)
        print("   ...Slope analysis complete.")

        print("2. Starting vegetation cover analysis...")
        veg_cover = analyze_vegetation_cover(roi)
        print(f"   ...Vegetation analysis complete. Result: {veg_cover:.2f}%")

        print("3. Starting road distance analysis...")
        road_data = analyze_distance_to_roads(roi)
        print("   ...Road distance analysis complete.")

        print("4. Starting building distance analysis...")
        building_dist = analyze_distance_to_buildings(roi)
        print("   ...Building distance analysis complete.")

        print("5. Starting shape metrics analysis...")
        shape_data = analyze_shape_metrics(roi)
        print("   ...Shape metrics analysis complete.")

        print("6. Calculating final score...")
        score = calculate_suitability_score(slope_data, road_data, veg_cover)
        print("   ...Score calculation complete.")

        # 4. Assemble the final, structured result
        result = AnalysisResult(
            slope_stats=SlopeStats(**slope_data),
            vegetation_cover_percent=veg_cover,
            road_distance_stats=RoadDistanceStats(**road_data),
            building_distance_m=building_dist,
            shape_metrics=ShapeMetrics(**shape_data),
            suitability_score=score
        )
        print("--- [SUCCESS] Analysis complete. Sending response. ---\n")
        return result

    except Exception as e:
        # If any GEE or other error occurs, return a server error to the app
        print(f"--- [ERROR] An exception occurred during analysis: {e} ---")
        raise HTTPException(status_code=500, detail=f"Server error during analysis: {e}")

