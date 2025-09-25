import ee
import geemap
import math

Map = geemap.Map()
ee.authenticate(project='theta-messenger-442807-s5')
ee.Initialize()


# Ensure a map is available for drawing
if 'Map' not in locals() or not isinstance(Map, geemap.Map):
    print("Please run the map initialization cell first (e.g., the one that displays an interactive map).")
    # Assuming 'nyeri' is loaded in a previous cell
    gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
    nyeri = gaul.filter(ee.Filter.eq('ADM0_NAME', 'Kenya')) \
                .filter(ee.Filter.eq('ADM2_NAME', 'Nyeri'))
else:
    # Assuming 'nyeri' is loaded in a previous cell if Map is available
    if 'nyeri' not in locals():
        gaul = ee.FeatureCollection("FAO/GAUL/2015/level2")
        nyeri = gaul.filter(ee.Filter.eq('ADM0_NAME', 'Kenya')) \
                    .filter(ee.Filter.eq('ADM2_NAME', 'Nyeri'))


def analyze_slope(roi):
    """Analyzes slope characteristics within the given region of interest."""
    print("\n--- Slope Analysis ---")
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    slope_roi = slope.clip(roi)
    slope_stats = slope_roi.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.minMax(),
            sharedInputs=True
        ),
        geometry=roi,
        scale=30,
        maxPixels=1e9
    ).getInfo()
    print("Slope Statistics:")
    print(f"  Mean: {slope_stats.get('slope_mean', 'N/A'):.2f}")
    print(f"  Min: {slope_stats.get('slope_min', 'N/A'):.2f}")
    print(f"  Max: {slope_stats.get('slope_max', 'N/A'):.2f}")
    return slope_stats

def analyze_vegetation_cover(roi):
    """Analyzes vegetation cover percentage within the given region of interest."""
    print("\n--- Vegetation Cover Analysis ---")
    s2 = ee.ImageCollection("COPERNICUS/S2_SR") \
            .filterBounds(roi) \
            .filterDate("2024-01-01", "2024-12-31") \
            .select(['B8', 'B4']) \
            .median()
    ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI')
    veg = ndvi.gt(0.6).rename('veg')
    pixel_area = ee.Image.pixelArea()
    total_area_img = pixel_area.updateMask(ee.Image.constant(1).clip(roi))
    total_area = total_area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=10,
        maxPixels=1e13
    ).get('area').getInfo()
    veg_area_img = pixel_area.updateMask(veg)
    veg_area = veg_area_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=10,
        maxPixels=1e13
    ).get('area').getInfo()
    veg_pct = 100 * veg_area / total_area if total_area is not None and total_area > 0 else 0
    print(f"🌱 Vegetation cover: {veg_pct:.2f}% of the selected parcel")
    return veg_pct

def analyze_distance_to_roads(roi):
    """Analyzes distance to the nearest road within the given region of interest."""
    print("\n--- Distance to Roads ---")
    roads = ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/Africa")
    distance_to_roads = roads.distance(ee.Number(10000))
    dist_roi = distance_to_roads.clip(roi)
    road_stats = dist_roi.reduceRegion(
        reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.minMax(), sharedInputs=True),
        geometry=roi,
        scale=30,
        maxPixels=1e13
    ).getInfo()
    print("Distance to Roads (meters):")
    if road_stats:
        print(f"  Closest point: {road_stats.get('distance_min', float('inf')):.2f} m")
        print(f"  Average distance: {road_stats.get('distance_mean', float('nan')):.2f} m")
        print(f"  Furthest point: {road_stats.get('distance_max', float('inf')):.2f} m")
    else:
        print("  Could not calculate distance to roads.")
    return road_stats

def analyze_distance_to_buildings(roi, nyeri_boundary):
    """Analyzes distance to the nearest building within the given region of interest (ROI or parcels)."""
    print("\n--- Distance to Buildings ---")

    # --- Step 1: Load Microsoft Building Footprints (Africa)
    buildings = ee.FeatureCollection("projects/sat-io/open-datasets/MSBuildings/Kenya")

    # --- Step 2: Filter to Nyeri
    buildings_nyeri = buildings.filterBounds(nyeri_boundary)

    # --- Step 3: Create distance raster (meters)
    dist_img = buildings_nyeri.distance(20000).rename('dist_m')

    # --- Step 4: Clip to ROI for visualization
    dist_buildings_roi = dist_img.clip(roi)

    # --- Step 5: Handle ROI type
    if isinstance(roi, ee.FeatureCollection):
        # For multiple parcels → centroid for each feature
        roi_centroids = roi.map(
            lambda f: ee.Feature(f.geometry().centroid(1)).copyProperties(f)
        )

        distances_fc = dist_img.reduceRegions(
            collection=roi_centroids,
            reducer=ee.Reducer.first(),
            scale=30
        )

        print("🏠 Distances to nearest building (meters) for first 5 parcels:")
        try:
            print(distances_fc.limit(5).getInfo())
        except Exception as e:
            print(f"Could not retrieve distances for parcels: {e}")
        result = distances_fc

    else:
        # For a single ROI geometry or Feature → use centroid
        roi_centroid = roi.centroid(1) if isinstance(roi, ee.Geometry) else roi.geometry().centroid(1)

        distance_at_centroid = dist_img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=roi_centroid,
            scale=30,
            maxPixels=1e13
        ).get('dist_m').getInfo()

        print("🏠 Distance to nearest building (meters) at ROI centroid:")
        if distance_at_centroid is not None:
            print(f"  Distance: {distance_at_centroid:.2f} m")
        else:
            print("  No building distance found for ROI.")
        result = distance_at_centroid

    # --- Step 6: Optional visualization
    Map.addLayer(dist_buildings_roi, {'min': 0, 'max': 2000}, 'Distance to Buildings')
    Map.addLayer(buildings_nyeri.limit(500), {'color': 'blue'}, 'Buildings sample')

    return result



def analyze_shape_metrics(roi):
    """Analyzes shape characteristics of the given region of interest."""
    print("\n--- Shape Metrics ---")
    area = roi.area(1).getInfo()
    perimeter = roi.perimeter(1).getInfo()
    pa_ratio = perimeter / area if area and area > 0 else 0
    shape_index = (4 * math.pi * area) / (perimeter ** 2) if perimeter and perimeter > 0 else 0
    bounds = roi.bounds().coordinates().getInfo()[0]
    xs = [pt[0] for pt in bounds]
    ys = [pt[1] for pt in bounds]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    length = (xmax - xmin) * 111000
    width = (ymax - ymin) * 111000
    if min(length, width) == 0:
        elongation = float('inf')
    else:
         elongation = max(length, width) / min(length, width)
    elong_score = 1 / (1 + abs(elongation - 1)) if elongation != float('inf') else 0
    shape_score = min(shape_index / 0.78, 1.0) if shape_index > 0 else 0
    pa_square = 4 * math.sqrt(area) / area if area > 0 else 0
    pa_score = min(pa_square / pa_ratio, 1.0) if pa_ratio > 0 else 0
    regularity_score = 100 * (0.6 * elong_score + 0.2 * shape_score + 0.2 * pa_score)

    print("Shape Metrics:")
    print(f"  Area: {area:.2f} m²")
    print(f"  Perimeter: {perimeter:.2f} m")
    print(f"  Shape Index: {shape_index:.4f}")
    print(f"  Elongation Ratio: {elongation:.2f}")
    print(f"  Regularity Score : {regularity_score:.1f} %")
    return {"area": area, "perimeter": perimeter, "shape_index": shape_index, "elongation": elongation, "regularity_score": regularity_score}

# Example of how to use the functions with the current ROI
if 'Map' in locals() and isinstance(Map, geemap.Map):
    current_roi = Map.user_roi if Map.user_roi else nyeri.geometry()
    if current_roi is not None:
        analyze_slope(current_roi)
        analyze_vegetation_cover(current_roi)
        analyze_distance_to_roads(current_roi)
        analyze_distance_to_buildings(current_roi, nyeri) # Pass nyeri boundary to the function
        analyze_shape_metrics(current_roi)
    else:
        print("👉 Please draw a parcel on the map above or ensure the Nyeri boundary is loaded.")
else:
     print("Map is not initialized. Please run the map initialization cell first.")