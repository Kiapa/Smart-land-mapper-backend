import ee
import math
import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import firebase_admin
from firebase_admin import credentials, auth, firestore
from fastapi.security import OAuth2PasswordBearer
import datetime

# --- Firebase & GEE Initialization ---
# This uses the same service account key for both Google services
try:
    KEY_FILE_PATH = 'gen-lang-client-0427725945-dc7fe2faa8ab.json'
    cred = credentials.Certificate(KEY_FILE_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase Admin SDK initialized successfully.")

    # Initialize GEE using the same credentials object
    ee.Initialize(cred)
    print("Google Earth Engine initialized successfully.")

except Exception as e:
    print(f"Error during initialization: {e}")
    exit()


# --- Pydantic Models ---
class Point(BaseModel):
    latitude: float
    longitude: float


class AnalysisResult(BaseModel):
    slope_stats: Dict[str, Any]
    vegetation_cover_percent: float
    road_distance_stats: Dict[str, Any]
    building_distance_m: float
    shape_metrics: Dict[str, Any]
    suitability_score: float
    land_use_zone: Optional[str] = "N/A"


class SaveReportRequest(BaseModel):
    parcelName: str
    reportData: AnalysisResult
    polygonCoords: List[Point]


app = FastAPI(title="SmartLand Mapper API")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# --- Dependency to verify Firebase ID Token ---
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Verifies the Firebase ID token and returns the user's data."""
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication credentials: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# --- Analysis Functions ---
def analyze_slope(roi):
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    stats = slope.clip(roi).reduceRegion(reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), True), geometry=roi,
                                         scale=30, maxPixels=1e9).getInfo()
    return {'mean': stats.get('slope_mean'), 'min': stats.get('slope_min'), 'max': stats.get('slope_max')}


def analyze_vegetation_cover(roi):
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate("2024-01-01",
                                                                                        "2024-12-31").select(
        ['B8', 'B4']).median()
    ndvi = s2.normalizedDifference(['B8', 'B4'])
    # Using .gt(0.2) for general vegetation, as 0.6 is very dense
    veg_mask = ndvi.gt(0.2)
    pixel_area = ee.Image.pixelArea()
    veg_area = pixel_area.updateMask(veg_mask).reduceRegion(reducer=ee.Reducer.sum(), geometry=roi, scale=10,
                                                            maxPixels=1e13).get('area').getInfo()
    total_area = roi.area(1).getInfo()
    return (veg_area / total_area) * 100 if total_area and total_area > 0 else 0


def analyze_distance_to_roads(roi):
    roads = ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/Africa")
    stats = roads.distance(10000).clip(roi).reduceRegion(reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), True),
                                                         geometry=roi, scale=30, maxPixels=1e13).getInfo()
    return {'min': stats.get('distance_min'), 'mean': stats.get('distance_mean'), 'max': stats.get('distance_max')}


def analyze_distance_to_buildings(roi):
    search_area = roi.buffer(5000)
    nearby_buildings = ee.FeatureCollection("projects/sat-io/open-datasets/MSBuildings/Kenya").filterBounds(search_area)
    if nearby_buildings.size().getInfo() == 0: return 5000
    nearest = nearby_buildings.map(lambda f: f.set('distance', roi.centroid(1).distance(f.geometry()))).sort(
        'distance').first()
    return nearest.get('distance').getInfo()


# --- UPDATED: Full shape metrics analysis ---
def analyze_shape_metrics(roi):
    area = roi.area(1).getInfo()
    perimeter = roi.perimeter(1).getInfo()

    if not area or not perimeter or area == 0 or perimeter == 0:
        return {"area_sqm": 0, "perimeter_m": 0, "regularity_score": 0}

    # Shape Index: Compares the shape to a circle (most compact shape)
    shape_index = (4 * math.pi * area) / (perimeter ** 2)

    # Elongation: Ratio of length to width
    bounds = roi.bounds().coordinates().getInfo()[0]
    xs = [pt[0] for pt in bounds]
    ys = [pt[1] for pt in bounds]
    length = (max(xs) - min(xs)) * 111000  # Approximate conversion from degrees to meters
    width = (max(ys) - min(ys)) * 111000
    elongation = max(length, width) / min(length, width) if min(length, width) > 0 else float('inf')

    # Scoring: Normalize values to a 0-1 scale
    elong_score = 1 / (1 + abs(elongation - 1)) if elongation != float('inf') else 0
    shape_score = min(shape_index / 0.78, 1.0)  # 0.78 is a common threshold for "regular"

    # Final weighted score
    regularity_score = 100 * (0.7 * elong_score + 0.3 * shape_score)

    return {"area_sqm": area, "perimeter_m": perimeter, "regularity_score": regularity_score}


def analyze_land_use(roi):
    landuse_fc = ee.FeatureCollection("projects/theta-messenger-442807-s5/assets/nyeri_land_use")
    intersected = landuse_fc.filterBounds(roi)
    if intersected.size().getInfo() == 0: return "Not classified"
    dominant = intersected.map(lambda f: f.set("iarea", f.geometry().intersection(roi, 1).area(1))).sort('iarea',
                                                                                                         False).first()
    return dominant.get("landuse").getInfo()


# --- UPDATED: Full suitability score model ---
def calculate_suitability_score(slope_stats, road_stats, shape_metrics, veg_cover):
    """Calculates a weighted suitability score based on multiple factors."""

    # --- Define weights for each factor (must sum to 1.0) ---
    WEIGHTS = {
        "slope": 0.40,
        "roads": 0.25,
        "shape": 0.20,
        "vegetation": 0.15
    }

    # --- Score each factor on a scale of 0 to 100 ---

    # 1. Slope Score (lower is better)
    mean_slope = slope_stats.get('mean') or 0
    slope_score = max(0, 100 - (mean_slope * 5))  # Harsh penalty for steep slopes

    # 2. Road Proximity Score (closer is better)
    min_dist_road = road_stats.get('min') or 5000
    if min_dist_road < 100:
        road_score = 100
    elif min_dist_road > 2000:
        road_score = 0
    else:
        road_score = 100 * (1 - (min_dist_road - 100) / 1900)

    # 3. Shape Score (uses the pre-calculated regularity score)
    shape_score = shape_metrics.get('regularity_score') or 0

    # 4. Vegetation Score (an optimal range is best)
    if 10 <= veg_cover <= 60:
        veg_score = 100  # Ideal range for development/agriculture
    elif veg_cover < 10:
        veg_score = 60  # Potentially arid
    else:
        veg_score = 40  # Potentially dense forest, costly to clear

    # --- Calculate Final Weighted Score ---
    final_score = (
            (slope_score * WEIGHTS["slope"]) +
            (road_score * WEIGHTS["roads"]) +
            (shape_score * WEIGHTS["shape"]) +
            (veg_score * WEIGHTS["vegetation"])
    )

    return max(0, min(100, final_score))  # Ensure score is between 0 and 100


# --- API Endpoints ---
@app.post("/analyze")
async def analyze_parcel(request: dict):
    try:
        coords = request.get("points")
        roi = ee.Geometry.Polygon([[p['longitude'], p['latitude']] for p in coords])

        slope_data = analyze_slope(roi)
        veg_cover = analyze_vegetation_cover(roi)
        road_data = analyze_distance_to_roads(roi)
        building_dist = analyze_distance_to_buildings(roi)
        shape_data = analyze_shape_metrics(roi)
        land_use = analyze_land_use(roi)

        # --- UPDATED: Call the real suitability score function ---
        score = calculate_suitability_score(slope_data, road_data, shape_data, veg_cover)

        return {
            "slope_stats": slope_data, "vegetation_cover_percent": veg_cover,
            "road_distance_stats": road_data, "building_distance_m": building_dist,
            "shape_metrics": shape_data, "suitability_score": score,
            "land_use_zone": land_use
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/savereport")
def save_report(request: SaveReportRequest, user: dict = Depends(get_current_user)):
    try:
        user_uid = user.get("uid")
        report_doc = {
            "parcelName": request.parcelName,
            "createdAt": datetime.datetime.now(datetime.timezone.utc),
            "analysisData": request.reportData.dict(),
            "polygon": [p.dict() for p in request.polygonCoords]
        }
        db.collection('users').document(user_uid).collection('reports').add(report_doc)
        return {"status": "success", "message": "Report saved successfully."}
    except Exception as e:
        print(f"--- [ERROR] Could not save report for UID {user_uid}: {e} ---")
        raise HTTPException(status_code=500, detail=f"Failed to save report: {e}")

