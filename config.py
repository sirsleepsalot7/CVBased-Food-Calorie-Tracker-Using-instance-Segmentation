# # Physical diameter of standard Golden/Brass 5 Rupee coin in centimeters
# COIN_REAL_DIAMETER_CM = 2.30

# # Comprehensive Nutritional & Geometric Database
# # density: g/cm^3 | kcal_per_g: kcal/g
# NUTRITION_DB = {
#     # --- Original Base Classes ---
#     "banana": {"density": 0.94, "kcal_per_g": 0.89, "model": "cylinder"},
#     "apple": {"density": 0.82, "kcal_per_g": 0.52, "model": "ellipsoid"},
#     "orange": {"density": 0.88, "kcal_per_g": 0.47, "model": "ellipsoid"},
#     "guava": {"density": 0.88, "kcal_per_g": 0.68, "model": "ellipsoid"},
#     "sandwich": {"density": 0.45, "kcal_per_g": 2.50, "model": "slab", "height_cm": 3.0},
#     "pizza": {"density": 0.50, "kcal_per_g": 2.66, "model": "slab", "height_cm": 1.2},
#     "donut": {"density": 0.35, "kcal_per_g": 4.52, "model": "slab", "height_cm": 2.5},
#     "cake": {"density": 0.40, "kcal_per_g": 3.71, "model": "slab", "height_cm": 4.0},

#     # --- Requested Additions ---
#     "bread": {"density": 0.28, "kcal_per_g": 2.65, "model": "slab", "height_cm": 1.2},
#     "egg": {"density": 1.04, "kcal_per_g": 1.55, "model": "ellipsoid"},
#     "rice": {"density": 0.72, "kcal_per_g": 1.30, "model": "slab", "height_cm": 2.5},

#     # --- 20 Everyday Foods & Staples ---
#     "roti": {"density": 0.60, "kcal_per_g": 2.97, "model": "slab", "height_cm": 0.3},
#     "paratha": {"density": 0.75, "kcal_per_g": 3.25, "model": "slab", "height_cm": 0.5},
#     "dosa": {"density": 0.45, "kcal_per_g": 1.68, "model": "slab", "height_cm": 0.4},
#     "idli": {"density": 0.78, "kcal_per_g": 1.32, "model": "ellipsoid"},
#     "samosa": {"density": 0.65, "kcal_per_g": 3.08, "model": "slab", "height_cm": 3.5},
#     "boiled_potato": {"density": 0.85, "kcal_per_g": 0.87, "model": "ellipsoid"},
#     "tomato": {"density": 0.95, "kcal_per_g": 0.18, "model": "ellipsoid"},
#     "cucumber": {"density": 0.96, "kcal_per_g": 0.15, "model": "cylinder"},
#     "carrot": {"density": 0.97, "kcal_per_g": 0.41, "model": "cylinder"},
#     "mango": {"density": 0.92, "kcal_per_g": 0.60, "model": "ellipsoid"},
#     "watermelon_slice": {"density": 0.92, "kcal_per_g": 0.30, "model": "slab", "height_cm": 2.5},
#     "cookie": {"density": 0.55, "kcal_per_g": 4.80, "model": "slab", "height_cm": 0.8},
#     "paneer_cube": {"density": 1.05, "kcal_per_g": 2.65, "model": "slab", "height_cm": 2.0},
#     "chicken_breast": {"density": 1.06, "kcal_per_g": 1.65, "model": "slab", "height_cm": 2.2},
#     "french_fries": {"density": 0.48, "kcal_per_g": 3.12, "model": "slab", "height_cm": 1.5},
#     "burger": {"density": 0.52, "kcal_per_g": 2.54, "model": "slab", "height_cm": 5.0},
#     "croissant": {"density": 0.32, "kcal_per_g": 4.06, "model": "cylinder"},
#     "muffin": {"density": 0.42, "kcal_per_g": 3.77, "model": "ellipsoid"},
#     "gulab_jamun": {"density": 1.10, "kcal_per_g": 3.00, "model": "ellipsoid"},
#     "vada": {"density": 0.62, "kcal_per_g": 2.75, "model": "slab", "height_cm": 2.2},

#     # --- Robust Fallback ---
#     "default": {"density": 0.60, "kcal_per_g": 1.50, "model": "slab", "height_cm": 2.0}
# }

# Reference Scale Calibration (cm)
# Scale Calibration (cm)
COIN_DIAMETER_CM = 2.5
CALIBRATION_MARKER_SIZE_CM = 4.0

# Phone IP Webcam URL
# Replace 192.168.x.x:8080 with the exact IPv4 displayed at the bottom of the IP Webcam app
PHONE_IP_URL = "http://192.0.0.4:8080/video"

# Set to True to use your phone, False to use your laptop webcam
USE_PHONE_CAM = True
CAMERA_INDEX = 0
# Comprehensive Nutrition & Density Database (g/cm^3 and per 100g values)
NUTRITION_DB = {
    # --- Custom / Previous Items ---
    "guava": {
        "density_g_cm3": 0.82,
        "calories_per_100g": 68,
        "protein_per_100g": 2.6,
        "carbs_per_100g": 14.3,
        "fat_per_100g": 1.0,
    },
    "roti": {
        "density_g_cm3": 0.60,
        "calories_per_100g": 297,
        "protein_per_100g": 9.0,
        "carbs_per_100g": 51.0,
        "fat_per_100g": 3.7,
    },
    "samosa": {
        "density_g_cm3": 0.65,
        "calories_per_100g": 262,
        "protein_per_100g": 5.0,
        "carbs_per_100g": 38.0,
        "fat_per_100g": 10.0,
    },
    "rice": {
        "density_g_cm3": 0.75,
        "calories_per_100g": 130,
        "protein_per_100g": 2.7,
        "carbs_per_100g": 28.0,
        "fat_per_100g": 0.3,
    },

    # --- Trained Model Classes (Roboflow Fruits Dataset) ---
    "apple": {
        "density_g_cm3": 0.85,
        "calories_per_100g": 52,
        "protein_per_100g": 0.3,
        "carbs_per_100g": 13.8,
        "fat_per_100g": 0.2,
    },
    "banana": {
        "density_g_cm3": 0.94,
        "calories_per_100g": 89,
        "protein_per_100g": 1.1,
        "carbs_per_100g": 22.8,
        "fat_per_100g": 0.3,
    },
    "cherry": {
        "density_g_cm3": 0.92,
        "calories_per_100g": 50,
        "protein_per_100g": 1.0,
        "carbs_per_100g": 12.0,
        "fat_per_100g": 0.3,
    },
    "cucumber": {
        "density_g_cm3": 0.96,
        "calories_per_100g": 15,
        "protein_per_100g": 0.7,
        "carbs_per_100g": 3.6,
        "fat_per_100g": 0.1,
    },
    "grapes": {
        "density_g_cm3": 0.96,
        "calories_per_100g": 69,
        "protein_per_100g": 0.7,
        "carbs_per_100g": 18.1,
        "fat_per_100g": 0.2,
    },
    "kiwi": {
        "density_g_cm3": 0.98,
        "calories_per_100g": 61,
        "protein_per_100g": 1.1,
        "carbs_per_100g": 14.7,
        "fat_per_100g": 0.5,
    },
    "lemon": {
        "density_g_cm3": 0.92,
        "calories_per_100g": 29,
        "protein_per_100g": 1.1,
        "carbs_per_100g": 9.3,
        "fat_per_100g": 0.3,
    },
    "mango": {
        "density_g_cm3": 1.02,
        "calories_per_100g": 60,
        "protein_per_100g": 0.8,
        "carbs_per_100g": 15.0,
        "fat_per_100g": 0.4,
    },
    "orange": {
        "density_g_cm3": 0.90,
        "calories_per_100g": 47,
        "protein_per_100g": 0.9,
        "carbs_per_100g": 11.8,
        "fat_per_100g": 0.1,
    },
    "pinapple": {  # Preserved dataset spelling
        "density_g_cm3": 0.95,
        "calories_per_100g": 50,
        "protein_per_100g": 0.5,
        "carbs_per_100g": 13.1,
        "fat_per_100g": 0.1,
    },
    "pineapple": {  # Standard spelling alias
        "density_g_cm3": 0.95,
        "calories_per_100g": 50,
        "protein_per_100g": 0.5,
        "carbs_per_100g": 13.1,
        "fat_per_100g": 0.1,
    },
    "tomato": {
        "density_g_cm3": 0.95,
        "calories_per_100g": 18,
        "protein_per_100g": 0.9,
        "carbs_per_100g": 3.9,
        "fat_per_100g": 0.2,
    },
    "tomatoe": {  # Dataset label variant
        "density_g_cm3": 0.95,
        "calories_per_100g": 18,
        "protein_per_100g": 0.9,
        "carbs_per_100g": 3.9,
        "fat_per_100g": 0.2,
    },
    "watermelon": {
        "density_g_cm3": 0.92,
        "calories_per_100g": 30,
        "protein_per_100g": 0.6,
        "carbs_per_100g": 7.6,
        "fat_per_100g": 0.2,
    }
}