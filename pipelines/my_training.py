
import dotenv
dotenv.load_dotenv()

import os

import logging

PACKAGES = {
    "keras": "3.8.0",
    "scikit-learn": "1.6.1",
    "mlflow": "2.20.0",
}

def packages(*names: str):
    """Return a dictionary of the specified packages and their corresponding version.

    This function is useful to set up the different pipelines while keeping the
    package versions consistent and centralized in a single location.

    Any packages that should be locked to a specific version will be part of the
    `PACKAGES` dictionary. If a package is not present in the dictionary, it will be
    installed using the latest version available.
    """
    return {name: PACKAGES.get(name, "") for name in names}

from metaflow import (
    FlowSpec,
    card,
    step,
    Parameter,
    project,
    conda_base,
    current
)


@project(name="penguins")
@conda_base(
    python="3.12.8",
    packages=packages(
        "scikit-learn",
        "pandas",
        "numpy",
        "keras",
        "jax[cpu]",
        "boto3",
        "mlflow",
        "python-dotenv"
    ),
)
class MyTraining(FlowSpec):
    """
        My custom training pipeline
    """

    mlflow_tracking_uri = Parameter(
            "mlflow-tracking-uri",
            help="MLFlow server location",
            default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )
    
    @card
    @step
    def start(self):
        import mlflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logging.info("MLFLOW tracking server: %s", self.mlflow_tracking_uri)
        self.mode = "development"
        # Modify the previous depending on the mode
        logging.info("Running flow in %s mode.", self.mode)
  
        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLFLow server {self.mlflow_tracking_uri}"
            raise RuntimeError(message) from e

        self.next(self.end)

    @step
    def end(self):
        logging.info("The pipeline finished successfully")

if __name__ == "__main__":
    MyTraining()
