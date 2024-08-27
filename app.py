import signal
import sys
from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.logger import logging
from src.exception import CustomizedException

app = Flask(__name__)

def graceful_shutdown(signum, frame):
    print("Received signal", signum)
    print("Shutting down gracefully...")
    # Perform any cleanup here
    # e.g., closing database connections, etc.
    sys.exit(0)
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    elif request.method == "POST":
        try:
            data = CustomData(
                AmbTemp_C_Avg=request.form.get("AmbTemp_C_Avg"),
                WindSpeedAve_ms=request.form.get("WindSpeedAve_ms"),
                WindDirAve_deg=request.form.get("WindDirAve_deg"),
                RTD_C_Avg_Mean=request.form.get("RTD_C_Avg_Mean"),
                Minute=request.form.get("Minute"),
            )
            
            # Correct method name
            processed_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            
            result = predict_pipeline.predict(processed_df)[0]
            
            # Pass back form data to preserve inputs
            return render_template(
                "home.html",
                result=result,
                AmbTemp_C_Avg=data.AmbTemp_C_Avg,
                WindSpeedAve_ms=data.WindSpeedAve_ms,
                WindDirAve_deg=data.WindDirAve_deg,
                RTD_C_Avg_Mean=data.RTD_C_Avg_Mean,
                Minute=data.Minute,
            )
        except Exception as e:
            logging.error(f"An error occurred: {CustomizedException(e)}")
            return render_template("home.html", result="An error occurred during prediction.")


if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, graceful_shutdown)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, graceful_shutdown)  # Handle termination signals

    app.run(host="0.0.0.0", port=8080,debug=False)