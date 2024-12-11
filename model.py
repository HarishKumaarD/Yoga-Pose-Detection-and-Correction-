import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points a, b, and c.
    
    Args:
        a (tuple): The first point (x, y).
        b (tuple): The second point (x, y) - the joint.
        c (tuple): The third point (x, y).

    Returns:
        float: The angle in degrees between the three points.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def classify_pose(landmarks):
    """
    Classify the yoga pose based on the joint angles.
    
    Args:
        landmarks (list): List of landmarks from MediaPipe Pose model.

    Returns:
        tuple: Pose type and feedback for corrections.
    """
    # Extract key landmarks
    left_shoulder = landmarks[11]
    left_elbow = landmarks[13]
    left_wrist = landmarks[15]
    left_hip = landmarks[23]
    left_knee = landmarks[25]
    left_ankle = landmarks[27]
    right_shoulder = landmarks[12]
    right_elbow = landmarks[14]
    right_wrist = landmarks[16]

    # Calculate angles for arms and legs
    left_arm_angle = calculate_angle([left_shoulder.x, left_shoulder.y], [left_elbow.x, left_elbow.y], [left_wrist.x, left_wrist.y])
    left_leg_angle = calculate_angle([left_hip.x, left_hip.y], [left_knee.x, left_knee.y], [left_ankle.x, left_ankle.y])
    right_arm_angle = calculate_angle([right_shoulder.x, right_shoulder.y], [right_elbow.x, right_elbow.y], [right_wrist.x, right_wrist.y])
    
    # Pose classification and feedback logic
    pose_type = "Unknown Pose"
    feedback = []
    
    if 160 < left_arm_angle < 180 and 90 < left_leg_angle < 120:
        pose_type = "Warrior Pose"
        if left_arm_angle < 170:
            feedback.append("Straighten your left arm.")
        if left_leg_angle < 100 or left_leg_angle > 120:
            feedback.append("Adjust your left leg angle.")
    
    elif 170 < left_arm_angle < 180 and 160 < left_leg_angle < 180:
        pose_type = "Tree Pose"
        if left_arm_angle < 175:
            feedback.append("Raise your left arm fully.")
        if left_leg_angle < 160:
            feedback.append("Lift your left leg higher.")
    
    elif 160 < left_arm_angle < 180 and 160 < right_arm_angle < 180 and 160 < left_leg_angle < 180:
        pose_type = "Downward Dog"
        if left_arm_angle < 170:
            feedback.append("Straighten your left arm.")
        if right_arm_angle < 170:
            feedback.append("Straighten your right arm.")
        if left_leg_angle < 160:
            feedback.append("Straighten your left leg.")
    
    elif 150 < left_arm_angle < 180 and 150 < right_arm_angle < 180 and 160 < left_leg_angle < 180:
        pose_type = "Triangle Pose"
        if left_arm_angle < 160:
            feedback.append("Extend your left arm further.")
        if right_arm_angle < 160:
            feedback.append("Extend your right arm further.")
    
    elif 160 < left_arm_angle < 180 and 160 < right_arm_angle < 180 and 160 < left_leg_angle < 180:
        pose_type = "Mountain Pose"
        if left_arm_angle < 170 or right_arm_angle < 170:
            feedback.append("Raise your arms fully.")
    
    elif 160 < left_arm_angle < 180 and 160 < right_arm_angle < 180 and left_leg_angle > 160:
        pose_type = "Cobra Pose"
        if left_arm_angle < 170 or right_arm_angle < 170:
            feedback.append("Straighten your arms more.")
        if left_leg_angle < 160:
            feedback.append("Relax your legs and keep them straight.")
    
    return pose_type, feedback

def process_pose(image):
    """
    Process the image to detect poses and provide feedback.
    
    Args:
        image (numpy.array): Input image.
    
    Returns:
        tuple: Processed image, feedback, and pose type.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    feedback = []
    pose_type = "Unknown Pose"
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        pose_type, feedback = classify_pose(landmarks)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    return image, feedback, pose_type
