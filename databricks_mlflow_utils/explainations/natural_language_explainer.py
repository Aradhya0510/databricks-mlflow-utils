import mlflow
import pandas as pd
import numpy as np
import shap
import dill
import os
import joblib

class NaturalLanguageExplainer:
    def __init__(self, model_uri):
        """
        Initialize the NaturalLanguageExplainer.

        Parameters:
        - model_uri: The MLflow model URI that contains both the model and the explainer.
        """
        self.model_uri = model_uri
        self.model = self._load_model()
        self.explainer = self._load_explainer()
        self.is_regression = self._determine_if_regression()
        self.expected_value = self._get_expected_value()

    def _load_model(self):
        """
        Load the MLflow model from the model URI.

        Returns:
        - model: The loaded model.
        """
        model = mlflow.pyfunc.load_model(self.model_uri)
        return model

    def _load_explainer(self):
        """
        Load the SHAP explainer from the model's artifacts.

        Returns:
        - explainer: The loaded SHAP explainer.
        """
        # Download the artifacts to a local directory
        artifacts_path = mlflow.artifacts.download_artifacts(artifact_uri=self.model_uri)
        explainer_path = os.path.join(artifacts_path, 'artifacts/explainer.dill')
        if explainer_path:
            with open(explainer_path, 'rb') as f:
                explainer = dill.load(f)
        else:
            raise FileNotFoundError("Explainer artifact not found in the context.")
        
        return explainer

    def _determine_if_regression(self):
        """
        Determine if the model is for regression or classification.

        Returns:
        - True if regression, False if classification.
        """
        # Try to check if the model has a predict_proba method
        # Since we're using a PyFunc model, we may need to access the underlying model
        try:
            # Access the underlying model if possible
            if hasattr(self.model._model_impl, 'predict_proba'):
                return False  # Classification
            else:
                return True   # Regression
        except AttributeError:
            # Default to regression if predict_proba is not available
            return True

    def _get_expected_value(self):
        """
        Get the expected value (base value) from the explainer.

        Returns:
        - The expected value as a scalar.
        """
        # For SHAP, the expected_value can be a scalar or an array
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            return np.mean(self.explainer.expected_value)
        else:
            return self.explainer.expected_value

    def generate_individual_explanation(self, instance, top_k=3):
        """
        Generate a natural language explanation for a single instance.

        Parameters:
        - instance: A pandas Series or DataFrame row representing the instance.
        - top_k: The number of top features to include in the explanation.

        Returns:
        - explanation: A string containing the natural language explanation.
        """
        return self._generate_shap_individual_explanation(instance, top_k)

    def _generate_shap_individual_explanation(self, instance, top_k):
        """
        Generate a SHAP-based natural language explanation for an individual instance.

        Parameters:
        - instance: A pandas Series or DataFrame row representing the instance.
        - top_k: The number of top features to include in the explanation.

        Returns:
        - explanation: A string containing the natural language explanation.
        """
        # Ensure instance is a DataFrame
        if isinstance(instance, pd.Series):
            instance = instance.to_frame().T
        elif isinstance(instance, pd.DataFrame):
            if len(instance) != 1:
                raise ValueError("Instance should be a single sample (one row).")
        else:
            raise TypeError("Instance must be a pandas Series or DataFrame.")

        # Get prediction
        prediction_output = self.model.predict(instance)

        if self.is_regression:
            prediction = prediction_output[0]
        else:
            # For classification, the model's predict method may return a DataFrame
            if isinstance(prediction_output, pd.DataFrame):
                prediction = prediction_output['predictions'][0]
                probabilities = prediction_output['probabilities'][0]
                # Assuming probabilities is a list of probabilities corresponding to classes
                predicted_class_idx = np.argmax(probabilities)
                predicted_class_prob = probabilities[predicted_class_idx]
                predicted_class_label = prediction
            else:
                # Handle other output formats if necessary
                predicted_class_label = prediction_output[0]
                predicted_class_prob = None  # Set to None if not available

            prediction = predicted_class_label

        # Get SHAP values
        shap_values = self.explainer(instance)
        if self.is_regression:
            shap_values_array = shap_values.values[0]
        else:
            # For classification, shap_values.values has shape (1, num_classes, num_features)
            # We select the predicted class
            shap_values_array = shap_values.values[0, predicted_class_idx, :]

        # Create a DataFrame of feature contributions
        feature_contributions = pd.DataFrame({
            'feature': instance.columns,
            'value': instance.iloc[0].values,
            'shap_value': shap_values_array
        })

        # Sort features by absolute SHAP value
        feature_contributions['abs_shap_value'] = feature_contributions['shap_value'].abs()
        feature_contributions = feature_contributions.sort_values(by='abs_shap_value', ascending=False)

        # Select top_k features
        top_features = feature_contributions.head(top_k)

        # Build explanation
        if self.is_regression:
            explanation = f"The model predicted a value of {prediction:.2f} because "
        else:
            if predicted_class_prob is not None:
                explanation = f"The model predicted class '{prediction}' with probability {predicted_class_prob:.2f} because "
            else:
                explanation = f"The model predicted class '{prediction}' because "

        for idx, row in top_features.iterrows():
            feature_name = row['feature']
            feature_value = row['value']
            shap_value = row['shap_value']
            if shap_value > 0:
                contribution = f"increasing the prediction by {shap_value:.2f}"
            else:
                contribution = f"decreasing the prediction by {abs(shap_value):.2f}"
            explanation += f"{feature_name} = {feature_value} {contribution}, "

        explanation = explanation.rstrip(', ') + '.'

        return explanation

    def generate_global_explanation(self, data, top_k=5):
        """
        Generate a global natural language explanation summarizing the model behavior.

        Parameters:
        - data: A pandas DataFrame used to compute the global explanations.
        - top_k: The number of top features to include in the explanation.

        Returns:
        - explanation: A string containing the global natural language explanation.
        """
        return self._generate_shap_global_explanation(data, top_k)

    def _generate_shap_global_explanation(self, data, top_k):
        """
        Generate a SHAP-based global natural language explanation.

        Parameters:
        - data: A pandas DataFrame used to compute the global explanations.
        - top_k: The number of top features to include in the explanation.

        Returns:
        - explanation: A string containing the global natural language explanation.
        """
        # Compute SHAP values
        shap_values = self.explainer(data)

        if self.is_regression:
            # Regression
            shap_values_array = shap_values.values
            mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)
        else:
            # Classification
            # shap_values_array has shape (num_samples, num_classes, num_features)
            # We compute the mean over samples and classes
            shap_values_array = shap_values.values
            mean_abs_shap = np.mean(np.abs(shap_values_array), axis=(0,1))

        feature_importance = pd.DataFrame({
            'feature': data.columns,
            'mean_abs_shap': mean_abs_shap
        })

        # Sort features by mean absolute SHAP value
        feature_importance = feature_importance.sort_values(by='mean_abs_shap', ascending=False)

        # Select top_k features
        top_features = feature_importance.head(top_k)

        # Build explanation
        explanation = "Overall, the most important features affecting the model predictions are: "

        for idx, row in top_features.iterrows():
            feature_name = row['feature']
            mean_contribution = row['mean_abs_shap']
            explanation += f"{feature_name} (average impact {mean_contribution:.2f}), "

        explanation = explanation.rstrip(', ') + '.'

        return explanation
