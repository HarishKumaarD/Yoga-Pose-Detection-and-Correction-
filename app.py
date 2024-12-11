import threading
from tkinter import Tk
from gui import YogaApp
from api import app as flask_app

def run_flask():
    flask_app.run(debug=True, use_reloader=False, threaded=True)

if __name__ == "__main__":
    # Run Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Start Tkinter GUI
    root = Tk()
    app = YogaApp(root)
    root.mainloop()
