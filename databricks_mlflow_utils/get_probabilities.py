import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
import pandas as pd
import mlflow.pyfunc

from dependency_checker import DependencyChecker

class PyFuncWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Load the model using mlflow.sklearn.load_model.
        """
        import mlflow.sklearn
        # Load the model from the logged sklearn model
        self.model = mlflow.sklearn.load_model(context.artifacts["sklearn_model"])
        
    def predict(self, context, model_input):
        """
        Predict both class labels and probabilities.
        """
        # Make predictions
        predictions = self.model.predict(model_input)
        probabilities = self.model.predict_proba(model_input)
        
        # Return both predictions and probabilities as a DataFrame
        frame = pd.DataFrame({
            "predictions": predictions,
            "probabilities": [list(prob) for prob in probabilities]
        })
        return frame

class ConvertToPyFuncForProbability():
    def __init__(self, model_uri):
        self.model_uri = model_uri
        # Initialize DependencyChecker and check/install dependencies
        dependency_checker = DependencyChecker(model_uri)
        if not dependency_checker.check_python_version():
            raise RuntimeError("Python version mismatch. Please run the script with the required Python version.")
        dependency_checker.check_and_install_packages()
        
        # Proceed with loading the scikit-learn model
        self.model = mlflow.sklearn.load_model(model_uri)
        
        # Download the model artifacts to a local path
        local_model_path = mlflow.artifacts.download_artifacts(model_uri)
        
        # Load the PyFuncModel from the local path
        pyfunc_model = mlflow.pyfunc.load_model(local_model_path)
        model_metadata = pyfunc_model.metadata
        
        # Use the local path to load the input example
        self.input_example = model_metadata.load_input_example(local_model_path)
        if self.input_example is None:
            raise ValueError("Input example is not available in the model metadata.")
        else:
            if not isinstance(self.input_example, pd.DataFrame):
                # Convert input_example to DataFrame if necessary
                self.input_example = pd.DataFrame(self.input_example)
        
    def get_signature(self):
        # Use the original model to generate outputs for signature inference
        input_example = self.input_example
        predictions = self.model.predict(input_example)
        probabilities = self.model.predict_proba(input_example)
        output = pd.DataFrame({
            "predictions": predictions,
            "probabilities": [list(prob) for prob in probabilities]
        })
        return infer_signature(input_example, output)
        
    def get_experiment_id(self):
        client = MlflowClient()
        run_id = self.model_uri.split("/")[-2]
        run = client.get_run(run_id)
        experiment_id = run.info.experiment_id
        return experiment_id
        
    def convert(self):
        import os
        import databricks_mlflow_utils
        # Get the module path dynamically
        module_path = os.path.abspath(databricks_mlflow_utils.__file__)

        # If the module is part of a package, use the directory
        if os.path.basename(module_path) == "__init__.py":
            # Module is a package, use the directory
            code_paths = [os.path.dirname(module_path)]
        else:
            # Module is a single file
            code_paths = [module_path]
        # Define artifacts
        artifacts = {
            "sklearn_model": self.model_uri
        }
        # Retrieve the pip requirements
        pip_requirements = mlflow.pyfunc.get_model_dependencies(self.model_uri)
        with mlflow.start_run(experiment_id=self.get_experiment_id()) as mlflow_run:
            # Log the PyFunc model
            mlflow.pyfunc.log_model(
                python_model=PyFuncWrapper(),
                artifact_path="pyfunc_model_with_signature",
                artifacts=artifacts,
                signature=self.get_signature(),
                input_example=self.input_example,
                pip_requirements=pip_requirements,
                code_path=code_paths
            )
        print("Model converted successfully")
        return {"converted_model_uri": f"runs:/{mlflow_run.info.run_id}/pyfunc_model_with_signature"}