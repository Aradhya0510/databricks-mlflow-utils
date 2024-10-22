import mlflow
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
        self.max_tokens=self.llm_params.get('max_tokens', 256),  # Adjusted to accommodate the word limit
        self.temperature=self.llm_params.get('temprature', 0.5),  # Adjusted for creativity balance
        self.top_p=self.llm_params.get('top_p', 1),
        # frequency_penalty=0,
        # presence_penalty=0
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
            top_features = feature_contributions.head(top_k)
            shap_info = top_features[['feature', 'value', 'shap_value']].to_dict('records')
            
            # Improved prompt
            prompt = (
                "As an AI assistant, generate a concise and structured explanation for the model's prediction. "
                "Use the following SHAP values to identify the top contributing features:\n"
                f"{shap_info}\n"
                "Instructions:\n"
                "- Begin with a summary sentence stating the predicted class and its significance.\n"
                "- List the top contributing features in order of importance.\n"
                "- For each feature, mention its name, its value, and how it influenced the prediction (increase or decrease).\n"
                "- Limit the explanation to a maximum of 150 words.\n"
                "- Do not include any additional information beyond the features provided.\n"
                "Please provide the explanation in plain language suitable for a non-technical audience."
            )

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
                max_tokens=self.max_tokens,  # Adjusted to accommodate the word limit
                temperature=self.temperature,  # Adjusted for creativity balance
                top_p=self.top_p,
                # frequency_penalty=0,
                # presence_penalty=0
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
            
            # Improved prompt
            prompt = (
                "As an AI assistant, generate a concise and structured explanation for the model's prediction. "
                "Use the following SHAP values to identify the top contributing features:\n"
                f"{shap_info}\n"
                "Instructions:\n"
                "- Begin with a summary sentence stating the predicted class and its significance.\n"
                "- List the top contributing features in order of importance.\n"
                "- For each feature, mention its name, its value, and how it influenced the prediction (increase or decrease).\n"
                "- Limit the explanation to a maximum of 150 words.\n"
                "- Do not include any additional information beyond the features provided.\n"
                "Please provide the explanation in plain language suitable for a non-technical audience."
            )

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
                max_tokens=self.max_tokens,  # Adjusted to accommodate the word limit
                temperature=self.temperature,  # Adjusted for creativity balance
                top_p=self.top_p,
                # frequency_penalty=0,
                # presence_penalty=0
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
