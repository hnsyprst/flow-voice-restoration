import os
import subprocess


def get_mlflow_tracking_uri_and_authenticate():
    """
    Set environment variables for MLflow. Requires authentication with gcloud.
    """

    tracking_uri = "http://34.124.203.102/"
    command = "gcloud secrets versions access latest --secret='mlflow_tracking_password'"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        password = result.stdout
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = "bandlab"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    else:
        raise RuntimeError(f"Error occurred while getting password from gcloud: {result.stderr}")
    return tracking_uri
