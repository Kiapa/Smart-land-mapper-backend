import ee
import math
import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import firebase_admin
from firebase_admin import credentials, auth
from google.cloud import firestore as gcloud_firestore
from google.oauth2 import service_account as gcloud_sa
from fastapi.security import OAuth2PasswordBearer
import datetime
import json

# --- Firebase & GEE Initialization ---
try:
    KEY_FILE_PATH = 'kiapa.json'

    # Load service account info
    with open(KEY_FILE_PATH, 'r') as f:
        service_account_info = json.load(f)

    service_account = service_account_info['client_email']
    project_id = service_account_info['project_id']

    print(f"→ Initializing with project: {project_id}")
    print(f"→ Service account: {service_account}")

    # Initialize Firebase Admin SDK (for authentication)
    cred = credentials.Certificate(KEY_FILE_PATH)
    firebase_admin.initialize_app(cred)
    print("✓ Firebase Admin SDK initialized successfully.")

    # Initialize Firestore with explicit credentials
    # This fixes the "403 Missing or insufficient permissions" error
    firestore_creds = gcloud_sa.Credentials.from_service_account_file(KEY_FILE_PATH)
    db = gcloud_firestore.Client(
        project=project_id,
        credentials=firestore_creds
    )
    print(f"✓ Firestore initialized with explicit credentials for project: {project_id}")

    # Initialize Google Earth Engine
    credentials_gee = ee.ServiceAccountCredentials(service_account, KEY_FILE_PATH)
    ee.Initialize(credentials_gee)
    print("✓ Google Earth Engine initialized successfully.")

except Exception as e:
    print(f"✗ Error during initialization: {e}")
    import traceback

    traceback.print_exc()
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
    try:
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token.get('uid')
        email = decoded_token.get('email', 'unknown')
        print(f"✓ Token verified for user: {uid} ({email})")
        return decoded_token
    except Exception as e:
        print(f"✗ Token verification failed: {e}")
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication credentials: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# --- Analysis Functions ---
def analyze_slope(roi):
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    stats = slope.clip(roi).reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
        geometry=roi,
        scale=30,
        maxPixels=1e9
    ).getInfo()
    return {
        'mean': stats.get('slope_mean'),
        'min': stats.get('slope_min'),
        'max': stats.get('slope_max')
    }


def analyze_vegetation_cover(roi):
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(roi) \
        .filterDate("2024-01-01", "2024-12-31") \
        .select(['B8', 'B4']) \
        .median()
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    veg_mask = ndvi.gt(0.2)
    stats = veg_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=30,
        maxPixels=1e9
    )
    percentage_fraction = stats.get('NDVI').getInfo() or 0
    return percentage_fraction * 100


def analyze_distance_to_roads(roi):
    search_radius_m = 10000
    search_area = roi.buffer(search_radius_m)
    nearby_roads = ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/Africa") \
        .filterBounds(search_area)

    if nearby_roads.size().getInfo() == 0:
        return {'min': search_radius_m, 'mean': search_radius_m, 'max': search_radius_m}

    distance_image = nearby_roads.distance(search_radius_m)
    stats = distance_image.clip(roi).reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
        geometry=roi,
        scale=30,
        maxPixels=1e13
    ).getInfo()
    return {
        'min': stats.get('distance_min'),
        'mean': stats.get('distance_mean'),
        'max': stats.get('distance_max')
    }


def analyze_distance_to_buildings(roi):
    search_area = roi.buffer(5000)
    nearby_buildings = ee.FeatureCollection("projects/sat-io/open-datasets/MSBuildings/Kenya") \
        .filterBounds(search_area)

    if nearby_buildings.size().getInfo() == 0:
        return 5000

    nearest = nearby_buildings.map(
        lambda f: f.set('distance', roi.centroid(1).distance(f.geometry()))
    ).sort('distance').first()

    return nearest.get('distance').getInfo()


def analyze_shape_metrics(roi):
    area = roi.area(1).getInfo()
    perimeter = roi.perimeter(1).getInfo()

    if not area or area == 0 or not perimeter or perimeter == 0:
        return {"area_sqm": 0, "perimeter_m": 0, "regularity_score": 0}

    shape_index = (4 * math.pi * area) / (perimeter ** 2)
    bounds = roi.bounds().coordinates().getInfo()[0]
    xs, ys = [pt[0] for pt in bounds], [pt[1] for pt in bounds]
    length, width = (max(xs) - min(xs)) * 111000, (max(ys) - min(ys)) * 111000
    elongation = max(length, width) / min(length, width) if min(length, width) > 0 else float('inf')
    elong_score = 1 / (1 + abs(elongation - 1)) if elongation != float('inf') else 0
    shape_score = min(shape_index / 0.78, 1.0)
    regularity_score = 100 * (0.7 * elong_score + 0.3 * shape_score)

    return {
        "area_sqm": area,
        "perimeter_m": perimeter,
        "regularity_score": regularity_score
    }


def analyze_land_use(roi):
    landuse_fc = ee.FeatureCollection("projects/gen-lang-client-0427725945/assets/nyeri_land_use")
    intersected = landuse_fc.filterBounds(roi)

    if intersected.size().getInfo() == 0:
        return "Not classified"

    dominant = intersected.map(
        lambda f: f.set("iarea", f.geometry().intersection(roi, 1).area(1))
    ).sort('iarea', False).first()

    return dominant.get("landuse").getInfo()


def calculate_suitability_score(slope_stats, road_stats, shape_metrics, veg_cover):
    WEIGHTS = {"slope": 0.40, "roads": 0.25, "shape": 0.20, "vegetation": 0.15}

    mean_slope = slope_stats.get('mean') or 0
    slope_score = max(0, 100 - (mean_slope * 5))

    min_dist_road = road_stats.get('min') or 5000
    road_score = 100 if min_dist_road < 100 else (max(0, 100 * (1 - (min_dist_road - 100) / 1900)))

    shape_score = shape_metrics.get('regularity_score') or 0
    veg_score = 100 if 10 <= veg_cover <= 60 else (60 if veg_cover < 10 else 40)

    final_score = ((slope_score * WEIGHTS["slope"]) +
                   (road_score * WEIGHTS["roads"]) +
                   (shape_score * WEIGHTS["shape"]) +
                   (veg_score * WEIGHTS["vegetation"]))

    return max(0, min(100, final_score))


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
        score = calculate_suitability_score(slope_data, road_data, shape_data, veg_cover)

        return {
            "slope_stats": slope_data,
            "vegetation_cover_percent": veg_cover,
            "road_distance_stats": road_data,
            "building_distance_m": building_dist,
            "shape_metrics": shape_data,
            "suitability_score": score,
            "land_use_zone": land_use
        }
    except Exception as e:
        print(f"✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/savereport")
async def save_report(request: SaveReportRequest, user: dict = Depends(get_current_user)):
    try:
        user_uid = user.get("uid")
        user_email = user.get("email", "unknown")

        print(f"→ Saving report for user: {user_uid} ({user_email})")
        print(f"→ Parcel name: {request.parcelName}")
        print(f"→ Writing to Firestore path: users/{user_uid}/reports")

        report_doc = {
            "parcelName": request.parcelName,
            "createdAt": datetime.datetime.now(datetime.timezone.utc),
            "analysisData": request.reportData.dict(),
            "polygon": [p.dict() for p in request.polygonCoords]
        }

        # Save to Firestore using the explicitly initialized client
        doc_ref = db.collection('users').document(user_uid).collection('reports').add(report_doc)
        report_id = doc_ref[1].id

        print(f"✓ Report saved successfully with ID: {report_id}")

        return {
            "status": "success",
            "message": "Report saved successfully.",
            "report_id": report_id
        }
    except Exception as e:
        print(f"✗ Could not save report for UID {user_uid}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save report: {e}")