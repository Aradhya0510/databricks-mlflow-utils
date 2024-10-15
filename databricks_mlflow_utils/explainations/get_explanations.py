import mlflow
import pandas as pd
import mlflow.pyfunc
import os

from databricks_mlflow_utils.dependency_checker import DependencyChecker

class PyFuncWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Load the model and the explainer.
        """
        import mlflow
        import shap

        # Load the model from the logged model
        model_flavor = context.artifacts["model_flavor"]
        model_uri = context.artifacts["model_uri"]

        if model_flavor == "sklearn":
            self.model = mlflow.sklearn.load_model(model_uri)
        elif model_flavor == "xgboost":
            self.model = mlflow.xgboost.load_model(model_uri)
        elif model_flavor == "lightgbm":
            self.model = mlflow.lightgbm.load_model(model_uri)
        elif model_flavor == "tensorflow":
            self.model = mlflow.tensorflow.load_model(model_uri)
        elif model_flavor == "pytorch":
            self.model = mlflow.pytorch.load_model(model_uri)
        else:
            self.model = mlflow.pyfunc.load_model(model_uri)

        # Load the explainer using mlflow.shap
        explainer_uri = context.artifacts["explainer_uri"]
        self.explainer = mlflow.shap.load_explainer(explainer_uri)
        
    def predict(self, context, model_input):
        """
        Predict and provide explanations.
        """
        import numpy as np

        # Ensure input is a DataFrame
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # Make predictions
        predictions = self.model.predict(model_input)

        # Get SHAP values
        shap_values = self.explainer.shap_values(model_input)

        # Handle multiclass classification (shap_values is a list of arrays)
        if isinstance(shap_values, list) and all(isinstance(sv, np.ndarray) for sv in shap_values):
            shap_values_serializable = [sv.tolist() for sv in shap_values]
        else:
            shap_values_serializable = shap_values.tolist()

        # Return predictions and explanations as a DataFrame
        frame = pd.DataFrame({
            "predictions": predictions,
            "shap_values": [shap_values_serializable] * len(predictions)  # Repeat for each prediction
        })
        return frame


    def predict(self, context, model_input):
        """
        Predict and provide explanations.
        """
        # Ensure input is a DataFrame
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # Make predictions
        predictions = self.model.predict(model_input)

        # Get SHAP values
        shap_values = self.explainer.shap_values(model_input)

        # Return predictions and explanations as a DataFrame
        frame = pd.DataFrame({
            "predictions": predictions,
            "shap_values": [list(sv) for sv in shap_values]  # Convert to list for serialization
        })
        return frame

class ConvertToPyFuncForExplanation():
    def __init__(self, model_uri, explainer_type='shap'):
        self.model_uri = model_uri
        self.explainer_type = explainer_type.lower()
        self.explainer = None

        # Initialize DependencyChecker and check/install dependencies
        dependency_checker = DependencyChecker(model_uri)
        if not dependency_checker.check_python_version():
            raise RuntimeError("Python version mismatch. Please run the script with the required Python version.")
        dependency_checker.check_and_install_packages()

        # Proceed with loading the model
        self.model, self.model_flavor = self.load_model_and_flavor(model_uri)

        # Download the model artifacts to a local path
        local_model_path = mlflow.artifacts.download_artifacts(model_uri)

        # Load the PyFuncModel from the local path
        pyfunc_model = mlflow.pyfunc.load_model(local_model_path)
        model_metadata = pyfunc_model.metadata

        # Load the input example (optional)
        self.input_example = model_metadata.load_input_example(local_model_path)
        if self.input_example is not None and not isinstance(self.input_example, pd.DataFrame):
            self.input_example = pd.DataFrame(self.input_example)

    def load_model_and_flavor(self, model_uri):
        import mlflow
        from mlflow.models import Model

        model_path = mlflow.artifacts.download_artifacts(model_uri)
        model_conf = Model.load(os.path.join(model_path, "MLmodel"))
        flavors = model_conf.flavors

        if 'sklearn' in flavors:
            model = mlflow.sklearn.load_model(model_uri)
            model_flavor = 'sklearn'
        elif 'xgboost' in flavors:
            model = mlflow.xgboost.load_model(model_uri)
            model_flavor = 'xgboost'
        elif 'lightgbm' in flavors:
            model = mlflow.lightgbm.load_model(model_uri)
            model_flavor = 'lightgbm'
        elif 'tensorflow' in flavors:
            model = mlflow.tensorflow.load_model(model_uri)
            model_flavor = 'tensorflow'
        elif 'pytorch' in flavors:
            model = mlflow.pytorch.load_model(model_uri)
            model_flavor = 'pytorch'
        else:
            model = mlflow.pyfunc.load_model(model_uri)
            model_flavor = 'pyfunc'
        return model, model_flavor

    def create_explainer(self, data):
        if self.explainer_type == 'shap':
            self.create_shap_explainer(data)
        elif self.explainer_type == 'lime':
            self.create_lime_explainer(data)
        else:
            raise ValueError(f"Unsupported explainer type: {self.explainer_type}")

    def create_lime_explainer(self, data):
        from lime.lime_tabular import LimeTabularExplainer
        # Ensure data is a NumPy array
        data_np = data.values if isinstance(data, pd.DataFrame) else data
        # Create LIME explainer
        self.explainer = LimeTabularExplainer(
            training_data=data_np,
            feature_names=data.columns.tolist(),
            class_names=['prediction'],
            mode='regression' if self.is_regression_model() else 'classification'
        )

    def is_regression_model(self):
        # Implement logic to determine if the model is for regression
        # For example, check if model has 'predict_proba' method
        return not hasattr(self.model, 'predict_proba')

    
    def create_shap_explainer(self, data):
        import shap
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # Create the appropriate explainer based on model flavor
        if self.model_flavor == 'sklearn':
            self.explainer = shap.KernelExplainer(self.model.predict, data)
        elif self.model_flavor in ['xgboost', 'lightgbm']:
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_flavor == 'tensorflow':
            self.explainer = shap.DeepExplainer(self.model, data)
        elif self.model_flavor == 'pytorch':
            self.explainer = shap.DeepExplainer(self.model, data)
        else:
            self.explainer = shap.KernelExplainer(self.model.predict, data)

    def get_signature(self):
        # Use the input example or raise an error if not available
        if self.input_example is None:
            raise ValueError("Input example is not available. Please provide an input example.")
        input_example = self.input_example
        predictions = self.model.predict(input_example)
        # Generate a sample explanation
        shap_values = self.explainer.shap_values(input_example)
        output = pd.DataFrame({
            "predictions": predictions,
            "shap_values": [list(sv) for sv in shap_values]
        })
        return mlflow.infer_signature(input_example, output)

    def get_experiment_id(self):
        client = mlflow.MlflowClient()
        run_id = self.model_uri.split("/")[-2]
        run = client.get_run(run_id)
        experiment_id = run.info.experiment_id
        return experiment_id

    def convert(self):
        if self.explainer is None:
            raise RuntimeError("Explainer has not been created. Please call 'create_explainer' with appropriate data before converting.")

        import os
        import mlflow
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


        with mlflow.start_run(experiment_id=self.get_experiment_id()) as mlflow_run:
            # Log the explainer
            if self.explainer_type == 'shap':
                explainer_artifact_path = "explainer"
                mlflow.shap.log_explainer(self.explainer, explainer_artifact_path)
                explainer_uri = f"runs:/{mlflow_run.info.run_id}/{explainer_artifact_path}"
            elif self.explainer_type == 'lime':
                explainer_artifact_path = "lime_explainer.pkl"
                # Save the LIME explainer manually
                import pickle
                with open(explainer_artifact_path, 'wb') as f:
                    pickle.dump(self.explainer, f)
                mlflow.log_artifact(explainer_artifact_path, artifact_path="explainer")
                explainer_uri = f"runs:/{mlflow_run.info.run_id}/explainer/{explainer_artifact_path}"

            # Define artifacts
            artifacts = {
                "model_flavor": self.model_flavor,
                "model_uri": self.model_uri,
                "explainer_uri": explainer_uri,
                "explainer_type": self.explainer_type
            }

            # Retrieve the pip requirements
            pip_requirements = mlflow.pyfunc.get_model_dependencies(self.model_uri)
            # Add SHAP to the requirements if not already present
            if 'shap' not in pip_requirements:
                pip_requirements.append('shap')
            # Update pip requirements
            if self.explainer_type == 'lime':
                pip_requirements.append('lime')

            # Log the PyFunc model
            mlflow.pyfunc.log_model(
                python_model=PyFuncWrapper(),
                artifact_path="pyfunc_model_with_explanation",
                artifacts=artifacts,
                signature=self.get_signature(),
                input_example=self.input_example,
                pip_requirements=pip_requirements,
                code_path=code_paths
            )
            
        print("Model converted successfully")
        return {"converted_model_uri": f"runs:/{mlflow_run.info.run_id}/pyfunc_model_with_explanation"}
