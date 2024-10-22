import mlflow
import pandas as pd
import mlflow.pyfunc
import os
import tempfile
import logging
from databricks_mlflow_utils.dependency_checker import DependencyChecker

# class PyFuncWrapper(mlflow.pyfunc.PythonModel):
#     def __init__(self, model_flavor, model_uri):
#         self.model_flavor = model_flavor
#         self.model_uri = model_uri
#         self.model = None
#         self.explainer = None

#     def load_context(self, context):
#         import shap
#         import dill

#         # Load the model
#         self.model = self._load_model()

#         # Load the explainer from artifacts
#         explainer_path = context.artifacts.get("explainer")
#         if explainer_path:
#             with open(explainer_path, 'rb') as f:
#                 self.explainer = dill.load(f)
#         else:
#             raise FileNotFoundError("Explainer artifact not found in the context.")

#     def _load_model(self):
#         # Load the model based on its flavor
#         flavor_loader = {
#             'sklearn': mlflow.sklearn.load_model,
#             'xgboost': mlflow.xgboost.load_model,
#             'lightgbm': mlflow.lightgbm.load_model,
#             'tensorflow': mlflow.tensorflow.load_model,
#             'pytorch': mlflow.pytorch.load_model,
#             'prophet': getattr(mlflow, 'prophet', None),
#             'arima': getattr(mlflow, 'arima', None),
#             'pyfunc': mlflow.pyfunc.load_model
#         }
#         loader = flavor_loader.get(self.model_flavor)
#         if loader is None:
#             raise ValueError(f"Model flavor '{self.model_flavor}' is not supported.")
#         return loader(self.model_uri)

#     def predict(self, context, model_input):
#         # Ensure model_input is a DataFrame
#         if not isinstance(model_input, pd.DataFrame):
#             model_input = pd.DataFrame(model_input)

#         # Make predictions
#         predictions = self.model.predict(model_input)

#         # Get SHAP values
#         if self.explainer is None:
#             raise RuntimeError("SHAP explainer not loaded.")
#         shap_values = self.explainer(model_input)

#         # Process SHAP values
#         output_df = pd.DataFrame({"predictions": predictions})
#         if len(shap_values.values.shape) > 1 and shap_values.values.shape[1] > 1:
#             # Multi-class classification
#             output_df["shap_values"] = list(shap_values.values)
#         else:
#             # Regression or binary classification
#             output_df["shap_values"] = shap_values.values.tolist()

#         return output_df

# class ConvertToPyFuncForExplanation:
#     def __init__(self, model_uri):
#         self.model_uri = model_uri
#         self.model = None
#         self.model_flavor = None
#         self.input_example = None
#         self.explainer = None

#         # Initialize DependencyChecker and check/install dependencies
#         dependency_checker = DependencyChecker(model_uri)
#         if not dependency_checker.check_python_version():
#             raise RuntimeError("Python version mismatch. Please run the script with the required Python version.")
#         dependency_checker.check_and_install_packages()

#         # Load the model and its flavor
#         self.model, self.model_flavor = self.load_model_and_flavor()

#         # Load the input example if available
#         self.load_input_example()

#     def load_model_and_flavor(self):
#         from mlflow.models import Model

#         model_path = mlflow.artifacts.download_artifacts(self.model_uri)
#         model_conf = Model.load(os.path.join(model_path, "MLmodel"))
#         flavors = model_conf.flavors

#         flavor_priority = [
#             'xgboost', 'lightgbm', 'sklearn',
#             'pytorch', 'tensorflow', 'prophet',
#             'arima', 'pyfunc'
#         ]
#         for flavor in flavor_priority:
#             if flavor in flavors:
#                 if flavor == 'prophet' and not hasattr(mlflow, 'prophet'):
#                     continue  # Skip if mlflow.prophet is not available
#                 if flavor == 'arima' and not hasattr(mlflow, 'arima'):
#                     continue  # Skip if mlflow.arima is not available
#                 model_loader = getattr(mlflow, flavor).load_model
#                 model = model_loader(self.model_uri)
#                 return model, flavor

#         raise ValueError("No supported model flavor found in the MLmodel file.")

#     def load_input_example(self):
#         try:
#           # Download the model artifacts to a local path
#           local_model_path = mlflow.artifacts.download_artifacts(self.model_uri)
#           # Load the PyFuncModel from the local path
#           pyfunc_model = mlflow.pyfunc.load_model(local_model_path)
#           model_metadata = pyfunc_model.metadata
#           # Load the input example (optional)
#           self.input_example = model_metadata.load_input_example(local_model_path)
#           if self.input_example is not None and not isinstance(self.input_example, pd.DataFrame):
#               self.input_example = pd.DataFrame(self.input_example)
              
#         except Exception as e:
#             logging.warning(f"input exmaple loading failed: {e}")
#             self.input_example = None
        
#     def create_explainer(self, data):
#         import shap

#         # Ensure data is a DataFrame
#         if not isinstance(data, pd.DataFrame):
#             data = pd.DataFrame(data)

#         model_type = type(self.model).__name__

#         # Handle models not supported by SHAP
#         unsupported_models = ['Prophet', 'ARIMA', 'AutoARIMA']
#         if model_type in unsupported_models:
#             raise NotImplementedError(f"SHAP explanations are not supported for {model_type} models.")

#         # Use shap.Explainer with a callable function
#         try:
#             # For sklearn Pipelines or any other models
#             self.explainer = self._select_explainer(data, model_type)
#         except Exception as e:
#             logging.warning(f"shap.Explainer failed: {e}")
#             # self.explainer = self._select_explainer(data, model_type)

#     def _select_explainer(self, data, model_type):
#         import shap
#         print("we are here! this should create explainer")
#         # Fallback to specific explainers based on model type
#         if self.model_flavor in ['xgboost', 'lightgbm'] or 'Tree' in model_type or 'Forest' in model_type:
#             return shap.TreeExplainer(self.model)
#         elif 'Linear' in model_type or self.model_flavor == 'sklearn':
#             # For sklearn Pipelines, use the predict function
#             predict_function = lambda x: self.model.predict_proba(pd.DataFrame(x, columns=data.columns)) if hasattr(self.model, 'predict_proba') else self.model.predict(pd.DataFrame(x, columns=data.columns))
#             return shap.KernelExplainer(predict_function, data)
#         elif self.model_flavor in ['tensorflow', 'pytorch']:
#             return shap.DeepExplainer(self.model, data)
#         else:
#             # Default to KernelExplainer
#             predict_function = lambda x: self.model.predict_proba(pd.DataFrame(x, columns=data.columns)) if hasattr(self.model, 'predict_proba') else self.model.predict(pd.DataFrame(x, columns=data.columns))
#             return shap.KernelExplainer(predict_function, data)

#     def get_signature(self):
#         from mlflow.models.signature import infer_signature

#         if self.input_example is None:
#             raise ValueError("Input example is not available for inferring signature.")

#         predictions = self.model.predict(self.input_example)
#         shap_values = self.explainer(self.input_example)

#         output_df = pd.DataFrame({"predictions": predictions})
#         if len(shap_values.values.shape) > 1 and shap_values.values.shape[1] > 1:
#             output_df["shap_values"] = list(shap_values.values)
#         else:
#             output_df["shap_values"] = shap_values.values.tolist()

#         return infer_signature(self.input_example, output_df)

#     def convert(self, experiment_id=None):
#         import mlflow
#         import dill

#         if self.explainer is None:
#             raise RuntimeError("Explainer has not been created. Call 'create_explainer' first.")

#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Save the explainer using dill
#             explainer_path = os.path.join(temp_dir, "explainer.dill")
#             with open(explainer_path, 'wb') as f:
#                 dill.dump(self.explainer, f)

#             # Define artifacts
#             artifacts = {"explainer": explainer_path}

#             # Get pip requirements
#             pip_requirements = self.get_pip_requirements()

#             # Ensure required packages are included
#             required_packages = ['shap', 'dill']
#             for package in required_packages:
#                 if not any(req.startswith(package) for req in pip_requirements):
#                     pip_requirements.append(package)

#             # Create an instance of PyFuncWrapper
#             python_model = PyFuncWrapper(model_flavor=self.model_flavor, model_uri=self.model_uri)

#             with mlflow.start_run(experiment_id=experiment_id) as mlflow_run:
#                 # Log the PyFunc model
#                 mlflow.pyfunc.log_model(
#                     python_model=python_model,
#                     artifact_path="pyfunc_model_with_explanation",
#                     artifacts=artifacts,
#                     signature=self.get_signature(),
#                     input_example=self.input_example,
#                     pip_requirements=pip_requirements,
#                     metadata={'model_flavor': self.model_flavor, 'model_uri': self.model_uri},
#                 )

#             print("Model converted successfully.")
#             return {"converted_model_uri": f"runs:/{mlflow_run.info.run_id}/pyfunc_model_with_explanation"}

#     def get_pip_requirements(self):
#         from mlflow.utils.environment import _get_pip_deps

#         # Try to get pip requirements from the model's conda.yaml or requirements.txt
#         try:
#             model_path = mlflow.artifacts.download_artifacts(self.model_uri)
#             conda_env_path = os.path.join(model_path, 'conda.yaml')
#             if os.path.exists(conda_env_path):
#                 pip_requirements = _get_pip_deps(conda_env_path)
#             else:
#                 pip_requirements = []
#         except Exception:
#             pip_requirements = []

#         return pip_requirements

import mlflow
import pandas as pd
import mlflow.pyfunc
import os
import tempfile
import logging
import numpy as np

class PyFuncWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_flavor, model_uri, NLE=False):
        self.model_flavor = model_flavor
        self.model_uri = model_uri
        self.model = None
        self.explainer = None
        self.NLE = NLE
        self.llm_params = None  # Will be set in load_context if NLE is True

    def load_context(self, context):
        import shap
        import dill
        import os

        # Load the model
        self.model = self._load_model()

        # Load the explainer from artifacts
        explainer_path = context.artifacts.get("explainer")
        if explainer_path:
            with open(explainer_path, 'rb') as f:
                self.explainer = dill.load(f)
        else:
            raise FileNotFoundError("Explainer artifact not found in the context.")

        # If NLE is True, ensure NaturalLanguageExplainer is imported and get llm_params
        if self.NLE:
            from .natural_language_explainer import NaturalLanguageExplainer
            # Read llm_params from environment variables
            api_key = os.environ.get('LLM_API_KEY')
            base_url = os.environ.get('LLM_BASE_URL')
            model_name = os.environ.get('LLM_MODEL_NAME')
            if api_key and base_url and model_name:
                self.llm_params = {
                    'api_key': api_key,
                    'base_url': base_url,
                    'model': model_name
                }
            else:
                self.llm_params = None

    def _load_model(self):
        # Load the model based on its flavor
        flavor_loader = {
            'sklearn': mlflow.sklearn.load_model,
            'xgboost': mlflow.xgboost.load_model,
            'lightgbm': mlflow.lightgbm.load_model,
            'tensorflow': mlflow.tensorflow.load_model,
            'pytorch': mlflow.pytorch.load_model,
            'prophet': getattr(mlflow, 'prophet', None),
            'arima': getattr(mlflow, 'arima', None),
            'pyfunc': mlflow.pyfunc.load_model
        }
        loader = flavor_loader.get(self.model_flavor)
        if loader is None:
            raise ValueError(f"Model flavor '{self.model_flavor}' is not supported.")
        return loader(self.model_uri)

    def predict(self, context, model_input):
        import pandas as pd

        # Ensure model_input is a DataFrame
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # Make predictions
        predictions = self.model.predict(model_input)

        # Get SHAP values
        if self.explainer is None:
            raise RuntimeError("SHAP explainer not loaded.")
        shap_values = self.explainer(model_input)

        # Process SHAP values
        output_df = pd.DataFrame({"predictions": predictions})
        # Include probabilities if applicable
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(model_input)
            output_df["probabilities"] = probabilities.tolist()

        if len(shap_values.values.shape) == 3:
            # Multi-class classification
            output_df["shap_values"] = [v.tolist() for v in shap_values.values]
        else:
            # Regression or binary classification
            output_df["shap_values"] = shap_values.values.tolist()

        # If NLE is True, generate natural language explanations
        if self.NLE:
            from .natural_language_explainer import NaturalLanguageExplainer

            nle = NaturalLanguageExplainer(self.model_uri, llm_params=self.llm_params)
            explanations = []
            for idx, row in model_input.iterrows():
                explanation = nle.generate_individual_explanation(row)
                explanations.append(explanation)
            output_df["explanations"] = explanations

        return output_df

class ConvertToPyFuncForExplanation:
    def __init__(self, model_uri, NLE=False):
        self.model_uri = model_uri
        self.NLE = NLE
        self.model = None
        self.model_flavor = None
        self.input_example = None
        self.explainer = None

        #Initialize DependencyChecker and check/install dependencies
        dependency_checker = DependencyChecker(model_uri)
        if not dependency_checker.check_python_version():
            raise RuntimeError("Python version mismatch. Please run the script with the required Python version.")
        dependency_checker.check_and_install_packages()

        # Load the model and its flavor
        self.model, self.model_flavor = self.load_model_and_flavor()

        # Load the input example if available
        self.load_input_example()

    def load_model_and_flavor(self):
        from mlflow.models import Model

        model_path = mlflow.artifacts.download_artifacts(self.model_uri)
        model_conf = Model.load(os.path.join(model_path, "MLmodel"))
        flavors = model_conf.flavors

        flavor_priority = [
            'xgboost', 'lightgbm', 'sklearn',
            'pytorch', 'tensorflow', 'prophet',
            'arima', 'pyfunc'
        ]
        for flavor in flavor_priority:
            if flavor in flavors:
                if flavor == 'prophet' and not hasattr(mlflow, 'prophet'):
                    continue  # Skip if mlflow.prophet is not available
                if flavor == 'arima' and not hasattr(mlflow, 'arima'):
                    continue  # Skip if mlflow.arima is not available
                model_loader = getattr(mlflow, flavor).load_model
                model = model_loader(self.model_uri)
                return model, flavor

        raise ValueError("No supported model flavor found in the MLmodel file.")

    def load_input_example(self):
        try:
            # Download the model artifacts to a local path
            local_model_path = mlflow.artifacts.download_artifacts(self.model_uri)
            # Load the PyFuncModel from the local path
            pyfunc_model = mlflow.pyfunc.load_model(local_model_path)
            model_metadata = pyfunc_model.metadata
            # Load the input example (optional)
            self.input_example = model_metadata.load_input_example(local_model_path)
            if self.input_example is not None and not isinstance(self.input_example, pd.DataFrame):
                self.input_example = pd.DataFrame(self.input_example)
        except Exception:
            self.input_example = None

    def create_explainer(self, data):
        import shap

        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        model_type = type(self.model).__name__

        # Handle models not supported by SHAP
        unsupported_models = ['Prophet', 'ARIMA', 'AutoARIMA']
        if model_type in unsupported_models:
            raise NotImplementedError(f"SHAP explanations are not supported for {model_type} models.")

        # Use shap.Explainer with a callable function
        try:
            # For sklearn Pipelines or any other models
            self.explainer = self._select_explainer(data, model_type)
        except Exception as e:
            logging.warning(f"shap.Explainer failed: {e}")
            # self.explainer = self._select_explainer(data, model_type)

    def _select_explainer(self, data, model_type):
        import shap

        # Fallback to specific explainers based on model type
        if self.model_flavor in ['xgboost', 'lightgbm'] or 'Tree' in model_type or 'Forest' in model_type:
            return shap.TreeExplainer(self.model)
        elif 'Linear' in model_type or self.model_flavor == 'sklearn':
            # For sklearn Pipelines, use the predict function
            predict_function = lambda x: self.model.predict_proba(pd.DataFrame(x, columns=data.columns)) if hasattr(self.model, 'predict_proba') else self.model.predict(pd.DataFrame(x, columns=data.columns))
            return shap.KernelExplainer(predict_function, data)
        elif self.model_flavor in ['tensorflow', 'pytorch']:
            return shap.DeepExplainer(self.model, data)
        else:
            # Default to KernelExplainer
            predict_function = lambda x: self.model.predict_proba(pd.DataFrame(x, columns=data.columns)) if hasattr(self.model, 'predict_proba') else self.model.predict(pd.DataFrame(x, columns=data.columns))
            return shap.KernelExplainer(predict_function, data)

    def get_signature(self):
        from mlflow.models.signature import infer_signature

        if self.input_example is None:
            raise ValueError("Input example is not available for inferring signature.")

        # Get predictions
        predictions = self.model.predict(self.input_example)

        # Initialize output_df with predictions
        output_df = pd.DataFrame({"predictions": predictions})

        # Get probabilities if model supports predict_proba
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(self.input_example)
            output_df["probabilities"] = probabilities.tolist()

        # Get shap_values
        shap_values = self.explainer(self.input_example)
        if len(shap_values.values.shape) == 3:
            # Multi-class classification
            output_df["shap_values"] = [v.tolist() for v in shap_values.values]
        else:
            # Regression or binary classification
            output_df["shap_values"] = shap_values.values.tolist()

        # If NLE is True, generate explanations
        if self.NLE:
            from .natural_language_explainer import NaturalLanguageExplainer

            # Initialize llm_params (assuming environment variables are set)
            api_key = os.environ.get('LLM_API_KEY')
            base_url = os.environ.get('LLM_BASE_URL')
            model_name = os.environ.get('LLM_MODEL_NAME')
            if api_key and base_url and model_name:
                llm_params = {
                    'api_key': api_key,
                    'base_url': base_url,
                    'model': model_name
                }
            else:
                llm_params = None  # or handle accordingly

            nle = NaturalLanguageExplainer(self.model, self.explainer, llm_params=llm_params)
            explanations = []
            for idx, row in self.input_example.iterrows():
                explanation = nle.generate_individual_explanation(row)
                explanations.append(explanation)
            output_df["explanations"] = explanations

            # Generate global explanation
            global_explanation = nle.generate_global_explanation(self.input_example)
            # Include global explanation in the output DataFrame
            output_df["global_explanation"] = [global_explanation] * len(output_df)

        # Now infer the signature
        return infer_signature(self.input_example, output_df)


    def convert(self, experiment_id=None):
        import mlflow
        import dill

        if self.explainer is None:
            raise RuntimeError("Explainer has not been created. Call 'create_explainer' first.")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the explainer using dill
            explainer_path = os.path.join(temp_dir, "explainer.dill")
            with open(explainer_path, 'wb') as f:
                dill.dump(self.explainer, f)

            # Define artifacts
            artifacts = {"explainer": explainer_path}

            # If NLE is True, generate the global explanation and save it as an artifact
            if self.NLE:
                from .natural_language_explainer import NaturalLanguageExplainer

                # Initialize llm_params (assuming environment variables are set)
                api_key = os.environ.get('LLM_API_KEY')
                base_url = os.environ.get('LLM_BASE_URL')
                model_name = os.environ.get('LLM_MODEL_NAME')
                if api_key and base_url and model_name:
                    llm_params = {
                        'api_key': api_key,
                        'base_url': base_url,
                        'model': model_name
                    }
                else:
                    llm_params = None  # or handle accordingly

                nle = NaturalLanguageExplainer(self.model, self.explainer, llm_params=llm_params)
                # Generate global explanation
                global_explanation = nle.generate_global_explanation(self.input_example)
                # Save the global explanation to a file
                global_explanation_path = os.path.join(temp_dir, "global_explanation.txt")
                with open(global_explanation_path, 'w') as f:
                    f.write(global_explanation)
                # Add to artifacts
                artifacts["global_explanation"] = global_explanation_path

            # Get pip requirements
            pip_requirements = self.get_pip_requirements()

            # Ensure required packages are included
            required_packages = ['shap', 'dill']
            if self.NLE:
                required_packages.append('openai')
            for package in required_packages:
                if not any(req.startswith(package) for req in pip_requirements):
                    pip_requirements.append(package)

            # Create an instance of PyFuncWrapper
            python_model = PyFuncWrapper(model_flavor=self.model_flavor, model_uri=self.model_uri, NLE=self.NLE)

            with mlflow.start_run(experiment_id=experiment_id) as mlflow_run:
                # Log the PyFunc model
                mlflow.pyfunc.log_model(
                    python_model=python_model,
                    artifact_path="pyfunc_model_with_explanation",
                    artifacts=artifacts,
                    signature=self.get_signature(),
                    input_example=self.input_example,
                    pip_requirements=pip_requirements,
                    code_path=['.'],  # Include the current directory
                    metadata={'model_flavor': self.model_flavor, 'model_uri': self.model_uri},
                )

            print("Model converted successfully.")
            return {"converted_model_uri": f"runs:/{mlflow_run.info.run_id}/pyfunc_model_with_explanation"}

    def get_pip_requirements(self):
        from mlflow.utils.environment import _get_pip_deps

        # Try to get pip requirements from the model's conda.yaml or requirements.txt
        try:
            model_path = mlflow.artifacts.download_artifacts(self.model_uri)
            conda_env_path = os.path.join(model_path, 'conda.yaml')
            if os.path.exists(conda_env_path):
                pip_requirements = _get_pip_deps(conda_env_path)
            else:
                pip_requirements = []
        except Exception:
            pip_requirements = []

        # Ensure required packages are included
        required_packages = ['shap', 'dill']
        if self.NLE:
            required_packages.append('openai')

        for package in required_packages:
            if not any(req.startswith(package) for req in pip_requirements):
                pip_requirements.append(package)

        return pip_requirements
