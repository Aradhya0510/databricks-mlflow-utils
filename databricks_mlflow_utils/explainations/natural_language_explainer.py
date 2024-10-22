import mlflow
import pandas as pd
import numpy as np

# class NaturalLanguageExplainer:
#     def __init__(self, model_uri):
#         """
#         Initialize the NaturalLanguageExplainer.

#         Parameters:
#         - model_uri: The MLflow model URI that contains both the model and the explainer.
#         """
#         self.model_uri = model_uri
#         self.model = self._load_model()
#         self.explainer = self._load_explainer()
#         self.underlying_model = self._get_underlying_model()
#         self.is_regression = self._determine_if_regression()
#         self.expected_value = self._get_expected_value()

#     def _load_model(self):
#         """
#         Load the MLflow model from the model URI.

#         Returns:
#         - model: The loaded model.
#         """
#         model = mlflow.pyfunc.load_model(self.model_uri)
#         return model

#     def _load_explainer(self):
#         """
#         Load the SHAP explainer from the model's implementation.

#         Returns:
#         - explainer: The loaded SHAP explainer.
#         """
#         try:
#             explainer = self.model._model_impl.python_model.explainer
#             if explainer is None:
#                 raise RuntimeError("SHAP explainer not loaded.")
#             return explainer
#         except AttributeError:
#             raise RuntimeError("Explainer not found in the model implementation.")

#     def _get_underlying_model(self):
#         """
#         Get the underlying model from the PyFunc model.

#         Returns:
#         - The underlying model object.
#         """
#         try:
#             return self.model._model_impl.python_model.model
#         except AttributeError:
#             raise RuntimeError("Underlying model not found in the model implementation.")

#     def _determine_if_regression(self):
#         """
#         Determine if the model is for regression or classification.

#         Returns:
#         - True if regression, False if classification.
#         """
#         if hasattr(self.underlying_model, 'predict_proba'):
#             return False  # Classification
#         else:
#             return True   # Regression

#     def _get_expected_value(self):
#         """
#         Get the expected value (base value) from the explainer.

#         Returns:
#         - The expected value as a scalar.
#         """
#         # For SHAP, the expected_value can be a scalar or an array
#         if isinstance(self.explainer.expected_value, (list, np.ndarray)):
#             return np.mean(self.explainer.expected_value)
#         else:
#             return self.explainer.expected_value

#     def generate_individual_explanation(self, instance, top_k=3):
#         """
#         Generate a natural language explanation for a single instance.

#         Parameters:
#         - instance: A pandas Series or DataFrame row representing the instance.
#         - top_k: The number of top features to include in the explanation.

#         Returns:
#         - explanation: A string containing the natural language explanation.
#         """
#         return self._generate_shap_individual_explanation(instance, top_k)

#     def _generate_shap_individual_explanation(self, instance, top_k):
#         """
#         Generate a SHAP-based natural language explanation for an individual instance.

#         Parameters:
#         - instance: A pandas Series or DataFrame row representing the instance.
#         - top_k: The number of top features to include in the explanation.

#         Returns:
#         - explanation: A string containing the natural language explanation.
#         """

#         # Ensure instance is a DataFrame
#         if isinstance(instance, pd.Series):
#             instance = instance.to_frame().T
#         elif isinstance(instance, pd.DataFrame):
#             if len(instance) != 1:
#                 raise ValueError("Instance should be a single sample (one row).")
#         else:
#             raise TypeError("Instance must be a pandas Series or DataFrame.")
        
#         # Get prediction
#         prediction_output = self.model.predict(instance)
#         print("Debug: Prediction output:", prediction_output)
        
#         if isinstance(prediction_output, pd.DataFrame):
#             prediction = prediction_output['predictions'].iloc[0]
#         elif isinstance(prediction_output, np.ndarray):
#             prediction = prediction_output[0]
#         else:
#             prediction = prediction_output

#         print("Debug: Final prediction:", prediction)

#         # Get SHAP values
#         try:
#             shap_values = self.explainer(instance)
#         except Exception as e:
#             print("Debug: Error in SHAP calculation:", str(e))
#             raise

#         # Handle multi-class scenario
#         if len(shap_values.values.shape) == 3:
#             # For multi-class, we'll use the SHAP values for the predicted class
#             # Assuming the order of classes in SHAP values matches the order of classes in the model
#             classes = self.underlying_model.classes_
#             class_index = list(classes).index(prediction)
#             shap_values_array = shap_values.values[0, :, class_index]
#         else:
#             shap_values_array = shap_values.values[0]

#         # Create a DataFrame of feature contributions
#         feature_contributions = pd.DataFrame({
#             'feature': instance.columns,
#             'value': instance.iloc[0].values,
#             'shap_value': shap_values_array
#         })

#         # Sort features by absolute SHAP value
#         feature_contributions['abs_shap_value'] = feature_contributions['shap_value'].abs()
#         feature_contributions = feature_contributions.sort_values(by='abs_shap_value', ascending=False)

#         # Select top_k features
#         top_features = feature_contributions.head(top_k)

#         # Build explanation
#         explanation = f"The model predicted class '{prediction}' because "

#         contributions = []
#         for idx, row in top_features.iterrows():
#             feature_name = row['feature']
#             feature_value = row['value']
#             shap_value = row['shap_value']
#             if shap_value > 0:
#                 contribution = f"{feature_name} (value: {feature_value:.2f}) increased the likelihood of this class by {shap_value:.2f}"
#             else:
#                 contribution = f"{feature_name} (value: {feature_value:.2f}) decreased the likelihood of this class by {abs(shap_value):.2f}"
#             contributions.append(contribution)

#         explanation += '; '.join(contributions) + '.'

#         return explanation

#     def generate_global_explanation(self, data, top_k=5):
#         """
#         Generate a global natural language explanation summarizing the model behavior.

#         Parameters:
#         - data: A pandas DataFrame used to compute the global explanations.
#         - top_k: The number of top features to include in the explanation.

#         Returns:
#         - explanation: A string containing the global natural language explanation.
#         """
#         return self._generate_shap_global_explanation(data, top_k)

#     def _generate_shap_global_explanation(self, data, top_k):
#         """
#         Generate a SHAP-based global natural language explanation.

#         Parameters:
#         - data: A pandas DataFrame used to compute the global explanations.
#         - top_k: The number of top features to include in the explanation.

#         Returns:
#         - explanation: A string containing the global natural language explanation.
#         """
#         # Compute SHAP values
#         shap_values = self.explainer(data)
#         shap_values_array = shap_values.values

#         if len(shap_values_array.shape) == 3:
#             # For multi-class classification, average over samples and classes
#             mean_abs_shap = np.mean(np.abs(shap_values_array), axis=(0, 2))
#         else:
#             # For regression or binary classification
#             mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)

#         feature_importance = pd.DataFrame({
#             'feature': data.columns,
#             'mean_abs_shap': mean_abs_shap
#         })

#         # Sort features by mean absolute SHAP value
#         feature_importance = feature_importance.sort_values(by='mean_abs_shap', ascending=False)

#         # Select top_k features
#         top_features = feature_importance.head(top_k)

#         # Build explanation
#         explanation = "Overall, the most important features affecting the model predictions are: "

#         contributions = []
#         for idx, row in top_features.iterrows():
#             feature_name = row['feature']
#             mean_contribution = row['mean_abs_shap']
#             contributions.append(f"{feature_name} (average impact {mean_contribution:.2f})")

#         explanation += '; '.join(contributions) + '.'

#         return explanation

import pandas as pd
import numpy as np 
class NaturalLanguageExplainer:
    def __init__(self, model, explainer, llm_params=None):
        """
        Initialize the NaturalLanguageExplainer.

        Parameters:
        - model: The underlying model (e.g., sklearn model).
        - explainer: The SHAP explainer.
        - llm_params: Dictionary containing 'api_key', 'base_url', and 'model' for LLM.
        """
        self.model = model
        self.explainer = explainer
        self.llm_params = llm_params
        self.is_regression = self._determine_if_regression()
        self.expected_value = self._get_expected_value()

        # Initialize LLM client if llm_params are provided
        if self.llm_params:
            self.llm_client = self._initialize_llm_client()
        else:
            self.llm_client = None

    def _initialize_llm_client(self):
        from openai import OpenAI
        api_key = self.llm_params.get('api_key')
        base_url = self.llm_params.get('base_url')
        model_name = self.llm_params.get('model')
        if not api_key or not base_url or not model_name:
            raise ValueError("LLM parameters 'api_key', 'base_url', and 'model' must be provided.")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.llm_model_name = model_name
        return client

    def _determine_if_regression(self):
        """
        Determine if the model is for regression or classification.

        Returns:
        - True if regression, False if classification.
        """
        if hasattr(self.model, 'predict_proba'):
            return False  # Classification
        else:
            return True   # Regression

    def _get_expected_value(self):
        """
        Get the expected value (base value) from the explainer.

        Returns:
        - The expected value as a scalar.
        """
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
        if isinstance(prediction_output, pd.DataFrame):
            prediction = prediction_output['predictions'].iloc[0]
        elif isinstance(prediction_output, np.ndarray):
            prediction = prediction_output[0]
        else:
            prediction = prediction_output

        # Get SHAP values
        shap_values = self.explainer(instance)

        # Handle multi-class scenario
        if len(shap_values.values.shape) == 3:
            # For multi-class, we'll use the SHAP values for the predicted class
            classes = self.model.classes_
            class_index = list(classes).index(prediction)
            shap_values_array = shap_values.values[0, :, class_index]
        else:
            shap_values_array = shap_values.values[0]

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
        if self.llm_client:
            # Use LLM to generate explanation
            shap_info = top_features[['feature', 'value', 'shap_value']].to_dict('records')
            prompt = f"Explain the model's prediction based on the following SHAP values:\n{shap_info}"

            response = self.llm_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant that explains machine learning model predictions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.llm_model_name,
                max_tokens=256
            )
            explanation = response.choices[0].message.content.strip()
        else:
            explanation = f"The model predicted class '{prediction}' because "

            contributions = []
            for idx, row in top_features.iterrows():
                feature_name = row['feature']
                feature_value = row['value']
                shap_value = row['shap_value']
                if shap_value > 0:
                    contribution = f"{feature_name} (value: {feature_value:.2f}) increased the likelihood of this class by {shap_value:.2f}"
                else:
                    contribution = f"{feature_name} (value: {feature_value:.2f}) decreased the likelihood of this class by {abs(shap_value):.2f}"
                contributions.append(contribution)

            explanation += '; '.join(contributions) + '.'

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
        """
        # Compute SHAP values
        shap_values = self.explainer(data)
        shap_values_array = shap_values.values

        if len(shap_values_array.shape) == 3:
            # For multi-class classification, average over samples and classes
            mean_abs_shap = np.mean(np.abs(shap_values_array), axis=(0, 2))
        else:
            # For regression or binary classification
            mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)

        feature_importance = pd.DataFrame({
            'feature': data.columns,
            'mean_abs_shap': mean_abs_shap
        })

        # Sort features by mean absolute SHAP value
        feature_importance = feature_importance.sort_values(by='mean_abs_shap', ascending=False)

        # Select top_k features
        top_features = feature_importance.head(top_k)

        # Build explanation
        if self.llm_client:
            # Use LLM to generate explanation
            shap_info = top_features[['feature', 'mean_abs_shap']].to_dict('records')
            prompt = f"Provide a global explanation of the model based on the following feature importances:\n{shap_info}"

            response = self.llm_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant that explains machine learning models."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.llm_model_name,
                max_tokens=256
            )
            explanation = response.choices[0].message.content.strip()
        else:
            explanation = "Overall, the most important features affecting the model predictions are: "

            contributions = []
            for idx, row in top_features.iterrows():
                feature_name = row['feature']
                mean_contribution = row['mean_abs_shap']
                contributions.append(f"{feature_name} (average impact {mean_contribution:.2f})")

            explanation += '; '.join(contributions) + '.'

        return explanation
