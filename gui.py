import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from model import process_pose

class YogaApp:
    """
    Tkinter GUI for Yoga Pose Detection and Real-Time Feedback.
    """
    def __init__(self, root):
        """
        Initialize the YogaApp with the root window.
        
        Args:
            root (tk.Tk): The main Tkinter window.
        """
        self.root = root
        self.root.title("Yoga Pose Detection and Correction")
        self.root.configure(bg="#f7f7f7")
        
        self.is_detecting = False
        self.cap = None
        
        # Panel to display image
        self.panel = tk.Label(self.root, bg="#f7f7f7")
        self.panel.pack(pady=20)

        # Feedback label
        self.feedback_label = tk.Label(self.root, text="", font=("Arial", 14), fg="red", bg="#f7f7f7")
        self.feedback_label.pack(pady=10)

        # Pose label
        self.pose_label = tk.Label(self.root, text="Pose Detected: Unknown", font=("Arial", 14, "bold"), fg="blue", bg="#f7f7f7")
        self.pose_label.pack(pady=10)

        # Buttons
        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_and_detect, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", relief="raised", bd=5)
        self.upload_button.pack(pady=10)
        
        self.start_button = tk.Button(self.root, text="Start Real-Time Detection", command=self.real_time_detection, font=("Arial", 12, "bold"), bg="#2196F3", fg="white", relief="raised", bd=5)
        self.start_button.pack(pady=10)
        
        self.stop_button = tk.Button(self.root, text="Stop Real-Time Detection", command=self.stop_detection, font=("Arial", 12, "bold"), bg="#F44336", fg="white", relief="raised", bd=5)
        self.stop_button.pack(pady=10)

    def upload_and_detect(self):
        """
        Handle image upload, pose detection, and display feedback.
        """
        file_path = filedialog.askopenfilename()
        
        # Check if a file was selected
        if not file_path:
            return
        
        image = cv2.imread(file_path)
        
        if image is None:
            return
        
        processed_image, corrections, pose_type = process_pose(image)
        
        # Convert processed image for tkinter
        image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.update_image(img_tk, corrections, pose_type)

    def real_time_detection(self):
        """
        Start real-time pose detection using webcam.
        """
        self.is_detecting = True
        self.cap = cv2.VideoCapture(0)
        
        while self.cap.isOpened() and self.is_detecting:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            processed_frame, corrections, pose_type = process_pose(frame)
            
            image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(image_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            self.update_image(img_tk, corrections, pose_type)
            
            self.root.update_idletasks()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def stop_detection(self):
        """
        Stop real-time webcam detection.
        """
        self.is_detecting = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def update_image(self, img_tk, corrections, pose_type):
        """
        Update the GUI with the new image and feedback.
        
        Args:
            img_tk (PhotoImage): Image to display in the Tkinter label.
            corrections (list): List of correction suggestions.
            pose_type (str): The detected yoga pose.
        """
        self.panel.configure(image=img_tk)
        self.panel.image = img_tk
        self.feedback_label.configure(text="\n".join(corrections))
        self.pose_label.configure(text=f"Pose Detected: {pose_type}")
