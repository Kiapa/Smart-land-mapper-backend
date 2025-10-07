import ee
import math
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import google.auth
from google.oauth2.service_account import Credentials

# --- GEE Initialization (No changes) ---
try:
    KEY_FILE_PATH = 'gen-lang-client-0427725945-dc7fe2faa8ab.json'
    credentials, project_id = google.auth.load_credentials_from_file(KEY_FILE_PATH)
    scoped_credentials = credentials.with_scopes([
        'https://www.googleapis.com/auth/cloud-platform',
        'https://www.googleapis.com/auth/earthengine'
    ])
    ee.Initialize(credentials=scoped_credentials)
    print("Google Earth Engine initialized successfully using Service Account.")
except Exception as e:
    print(f"Error initializing GEE: {e}")
    exit()

# --- Pydantic Models ---
class Point(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

class AnalysisRequest(BaseModel):
    points: List[Point] = Field(..., min_items=4)

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

# UPDATED: Add land_use_zone to the response model
class AnalysisResult(BaseModel):
    slope_stats: SlopeStats
    vegetation_cover_percent: float
    road_distance_stats: RoadDistanceStats
    building_distance_m: float
    shape_metrics: ShapeMetrics
    suitability_score: float
    land_use_zone: Optional[str] = "N/A" # Optional with a default value

app = FastAPI(title="Real Parcel Analysis API (GEE)")

# --- GEE Analysis Functions ---
def analyze_slope(roi: ee.Geometry) -> dict:
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    slope_roi = slope.clip(roi)
    stats = slope_roi.reduceRegion(
        reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(), sharedInputs=True),
        geometry=roi, scale=30, maxPixels=1e9
    ).getInfo()
    return {'mean': stats.get('slope_mean'), 'min': stats.get('slope_min'), 'max': stats.get('slope_max')}

def analyze_vegetation_cover(roi: ee.Geometry) -> float:
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate("2024-01-01", "2024-12-31").select(['B8', 'B4']).median()
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    veg = ndvi.gt(0.6).rename('veg')
    pixel_area = ee.Image.pixelArea()
    total_area_img = pixel_area.updateMask(ee.Image.constant(1).clip(roi))
    total_area = total_area_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=10, maxPixels=1e13).get('area').getInfo()
    veg_area_img = pixel_area.updateMask(veg)
    veg_area = veg_area_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=10, maxPixels=1e13).get('area').getInfo()
    return 100 * veg_area / total_area if total_area is not None and total_area > 0 else 0

def analyze_distance_to_roads(roi: ee.Geometry) -> dict:
    roads = ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/Africa")
    distance_to_roads = roads.distance(10000)
    dist_roi = distance_to_roads.clip(roi)
    stats = dist_roi.reduceRegion(
        reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(), sharedInputs=True),
        geometry=roi, scale=30, maxPixels=1e13
    ).getInfo()
    return {'min': stats.get('distance_min'), 'mean': stats.get('distance_mean'), 'max': stats.get('distance_max')}

def analyze_distance_to_buildings(roi: ee.Geometry) -> float:
    search_radius_m = 5000
    search_area = roi.buffer(search_radius_m)
    nearby_buildings = ee.FeatureCollection("projects/sat-io/open-datasets/MSBuildings/Kenya").filterBounds(search_area)
    if nearby_buildings.size().getInfo() == 0:
        return search_radius_m
    roi_centroid = roi.centroid(1)
    nearby_with_distance = nearby_buildings.map(lambda f: f.set('distance', roi_centroid.distance(f.geometry())))
    nearest_building = nearby_with_distance.sort('distance').first()
    return nearest_building.get('distance').getInfo()

def analyze_shape_metrics(roi: ee.Geometry) -> dict:
    area = roi.area(1).getInfo()
    perimeter = roi.perimeter(1).getInfo()
    pa_ratio = perimeter / area if area and area > 0 else 0
    shape_index = (4 * math.pi * area) / (perimeter ** 2) if perimeter and perimeter > 0 else 0
    bounds = roi.bounds().coordinates().getInfo()[0]
    xs, ys = [pt[0] for pt in bounds], [pt[1] for pt in bounds]
    length, width = (max(xs) - min(xs)) * 111000, (max(ys) - min(ys)) * 111000
    elongation = max(length, width) / min(length, width) if min(length, width) > 0 else float('inf')
    elong_score = 1 / (1 + abs(elongation - 1)) if elongation != float('inf') else 0
    shape_score = min(shape_index / 0.78, 1.0) if shape_index > 0 else 0
    pa_square = 4 * math.sqrt(area) / area if area > 0 else 0
    pa_score = min(pa_square / pa_ratio, 1.0) if pa_ratio > 0 else 0
    regularity_score = 100 * (0.6 * elong_score + 0.2 * shape_score + 0.2 * pa_score)
    return {"area_sqm": area, "perimeter_m": perimeter, "regularity_score": regularity_score}

# --- NEW: Integrated your land use analysis function ---
def analyze_land_use(roi):
    """Determines the dominant land use category of the drawn ROI using the digitized Nyeri land use layer."""
    landuse_fc = ee.FeatureCollection("projects/gen-lang-client-0427725945/assets/nyeri_land_use")
    intersected = landuse_fc.filterBounds(roi)
    count = intersected.size().getInfo()

    if count == 0:
        return "Not classified"

    # Add intersection area attribute
    def calculate_intersection(f):
        return f.set("intersect_area", f.geometry().intersection(roi, 1).area(1))

    area_stats = intersected.map(calculate_intersection)

    # Find land use with max overlap
    dominant = area_stats.sort('intersect_area', False).first() # Sort descending

    landuse_value = dominant.get("landuse").getInfo()

    return landuse_value

def calculate_suitability_score(slope_stats, road_stats, veg_cover):
    score = 100
    if (slope_stats.get('mean') or 0) > 15:
        score -= ((slope_stats.get('mean') or 0) - 15) * 2
    if (road_stats.get('min') or 0) > 500:
        score -= ((road_stats.get('min') or 0) - 500) / 20
    if 30 <= veg_cover <= 80:
        score += 5
    else:
        score -= 5
    return max(0, min(100, score))

# --- API Endpoint ---
@app.post("/analyze", response_model=AnalysisResult)
def analyze_parcel(request: AnalysisRequest):
    try:
        print("\n--- [START] New Analysis Request Received ---")
        coords = [[p.longitude, p.latitude] for p in request.points]
        roi = ee.Geometry.Polygon(coords)

        print("1. Analyzing slope...")
        slope_data = analyze_slope(roi)
        print("2. Analyzing vegetation...")
        veg_cover = analyze_vegetation_cover(roi)
        print("3. Analyzing road distance...")
        road_data = analyze_distance_to_roads(roi)
        print("4. Analyzing building distance...")
        building_dist = analyze_distance_to_buildings(roi)
        print("5. Analyzing shape metrics...")
        shape_data = analyze_shape_metrics(roi)

        # --- ADDED: Calling the new land use function ---
        print("6. Analyzing land use zone...")
        land_use = analyze_land_use(roi)
        print(f"   ...Land use found: {land_use}")

        print("7. Calculating final score...")
        score = calculate_suitability_score(slope_data, road_data, veg_cover)

        result = AnalysisResult(
            slope_stats=SlopeStats(**slope_data),
            vegetation_cover_percent=veg_cover,
            road_distance_stats=RoadDistanceStats(**road_data),
            building_distance_m=building_dist,
            shape_metrics=ShapeMetrics(**shape_data),
            suitability_score=score,
            land_use_zone=land_use # Added the new data to the response
        )
        print("--- [SUCCESS] Analysis complete. ---")
        return result

    except Exception as e:
        print(f"--- [ERROR] An exception occurred: {e} ---")
        raise HTTPException(status_code=500, detail=f"Server error during analysis: {e}")