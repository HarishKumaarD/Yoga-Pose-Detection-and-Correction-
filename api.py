from flask import Flask, request, jsonify
from model import process_pose
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/api/detect_pose', methods=['POST'])
def detect_pose_api():
    """
    API endpoint to process an uploaded image for yoga pose detection.

    Returns:
        JSON response with detected pose and corrections.
    """
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image file provided"}), 400
    
    # Read the image
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Process the image
    processed_image, corrections, pose_type = process_pose(image)
    
    # Return JSON with feedback and pose type
    return jsonify({
        "pose": pose_type,
        "corrections": corrections
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=True)
