import logging
import cv2
import numpy as np
import joblib
from flask import Flask, request, jsonify
from PIL import Image
from rembg import remove

# Maps for Variety
variety_map = {
    0: "Sweet Baby",
    1: "Rocky 475",
    2: "Kinaree 457"
}

# Maps for Sweetness, Flesh Colour, and Flesh Firmness
sweetness_map = {
    0: "Less Sweet",
    1: "Moderately Sweet",
    2: "High Sweet"
}

flesh_colour_map = {
    0: "Pale Red",
    1: "Red",
    2: "Deep Red"
}

flesh_firmness_map = {
    0: "Hard",
    1: "Moderately Firm",
    2: "Soft"
}


# Configure logging with a timestamp, log level, and message.
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Load the pre-trained model once at startup.
MODEL_PATH = "RandomForestClassifier_model.pkl"
SCALAER_MODEL_PATH = "StandardScaler.pkl"
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Failed to load model: %s", e)
    raise


def extract_features_from_image(image, weight):
    logging.debug("Starting feature extraction.")
    image_original = image.copy()
    # Apply Gaussian blur and adjust brightness/contrast.
    alpha = 2.3  # Contrast
    beta = 28    # Brightness
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Convert to PIL image, remove background using rembg, and convert back.
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_pil = remove(image_pil)
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Create a binary mask.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        logging.error("No contours found in image during feature extraction.")
        return None, "No contours found in image."

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Fit an ellipse to compute eccentricity.
    ellipse = cv2.fitEllipse(largest_contour)
    major_axis_length = ellipse[1][1]  # Major axis
    minor_axis_length = ellipse[1][0]  # Minor axis
    eccentricity_value = 1 - (minor_axis_length**2 / major_axis_length**2)
    eccentricity = np.sqrt(max(0, eccentricity_value))

    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    roundness = (4 * area) / (np.pi * major_axis_length**2)
    circularity = (4 * np.pi * area) / (perimeter**2)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    original_image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    original_image_gray = original_image_gray[mask == 255]

    # Calculate image entropy.
    entropy = -np.sum(
        (original_image_gray / 255.0) *
        np.log2((original_image_gray / 255.0) + 1e-10)
    )

    masked_image = cv2.bitwise_and(image_original, image_original, mask=mask)
    mean_values = cv2.mean(image_original, mask=mask)
    mean_r, mean_g, mean_b = mean_values[2], mean_values[1], mean_values[0]

    hsv_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
    mean_hue, mean_sat, mean_val = cv2.mean(hsv_image, mask=mask)[:3]
    std_dev = np.std(original_image_gray)

    height = h
    width = w
    aspect_ratio = width / height if height != 0 else 0
    pixel_area = height * width

    feature_row = {
        "Pixel Area": pixel_area,
        "Mean R": round(mean_r, 2),
        "Mean G": round(mean_g, 2),
        "Mean B": round(mean_b, 2),
        "Standard Deviation": round(std_dev, 4),
        "Entropy": round(entropy, 2),
        "Aspect Ratio": round(aspect_ratio, 4),
        "Roundness": round(roundness, 4),
        "Circularity": round(circularity, 4),
        "Eccentricity": round(eccentricity, 4),
        "Mean Hue": round(mean_hue, 2),
        "Mean Saturation": round(mean_sat, 2),
        "Mean Value": round(mean_val, 2),
    }

    # Insert weight as the first feature.
    features = [weight] + [float(value) for value in feature_row.values()]
    logging.debug("Feature extraction completed. Features: %s", features)
    return features, None


def reshape_features(features):
    scaler = joblib.load(SCALAER_MODEL_PATH)
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    logging.debug("Feature reshping completed. Features: %s", features)
    return features


@app.route('/analyze', methods=['POST'])
def analyze():
    logging.info("Received request at /analyze endpoint.")
    if 'image' not in request.files:
        logging.error("No image file provided in request.")
        return jsonify({'error': 'No image file provided.'}), 400
    image_file = request.files['image']

    weight_str = request.form.get('weight')
    if weight_str is None:
        logging.error("Weight parameter is missing in request.")
        return jsonify({'error': 'Weight parameter is missing.'}), 400
    try:
        weight = float(weight_str)
    except ValueError:
        logging.error("Invalid weight value provided: %s", weight_str)
        return jsonify({'error': 'Weight must be a numeric value.'}), 400

    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        logging.error("Invalid image file.")
        return jsonify({'error': 'Invalid image file.'}), 400

    logging.info("Starting feature extraction for the uploaded image.")
    features, error = extract_features_from_image(image, weight)
    if error:
        logging.error("Feature extraction failed: %s", error)
        return jsonify({'error': error}), 400

    scaled_features = reshape_features(features)

    logging.info("Extracted features successfully. Running prediction.")

    prediction = model.predict(scaled_features)[0]
    logging.info("Prediction completed: %s", prediction)

    # ---------------------------------------------
    # 1) Convert numeric predictions to Python ints
    #    in case they are numpy.int64 or similar.
    # 2) Map each int to a descriptive label.
    # ---------------------------------------------
    sweetness_num = int(prediction[0])
    flesh_colour_num = int(prediction[1])
    flesh_firmness_num = int(prediction[2])
    variety_num = int(prediction[3])

    sweetness_label = sweetness_map.get(sweetness_num, "Unknown Sweetness")
    flesh_colour_label = flesh_colour_map.get(
        flesh_colour_num, "Unknown Colour")
    flesh_firmness_label = flesh_firmness_map.get(
        flesh_firmness_num, "Unknown Firmness")
    variety_label = variety_map.get(variety_num, "Unknown Variety")

    # Convert features list elements from numpy types to native Python
    features_native = [
        x.item() if isinstance(x, np.generic) else x
        for x in features
    ]

    response = {
        "variety": variety_label,
        "sweetness": sweetness_label,
        "flesh_colour": flesh_colour_label,
        "flesh_firmness": flesh_firmness_label,
        "features": features_native
    }

    logging.info("Returning response: %s", response)
    return jsonify(response)


if __name__ == '__main__':
    logging.info("Starting Flask server on port 5001.")
    app.run(host='0.0.0.0', port=5001, debug=True)
