Here's an updated README incorporating the new module for explanations:

---

# MLflow Model Converter for Predictions, Probabilities, and SHAP Explanations

A Python package that simplifies the conversion of existing MLflow models to return both predictions and probabilities, along with generating SHAP explanations. It ensures consistency of model dependencies during repackaging, making it particularly useful in scenarios like Databricks AutoML, where the best run models typically return only predictions by default.

## Table of Contents

- [Features](#features)
- [Why Use This Package](#why-use-this-package)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [ConvertToPyFuncForProbability Class](#converttopyfuncforprobability-class)
  - [ConvertToPyFuncForExplanation Class](#converttopyfuncforexplanation-class)
  - [DependencyChecker Class](#dependencychecker-class)
- [Example Scenario: Databricks AutoML](#example-scenario-databricks-automl)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Convert Existing MLflow Models**: Transform any existing MLflow model to return both predictions and probabilities without retraining.
- **Generate SHAP Explanations**: Add SHAP-based explanations for model predictions using a seamless API.
- **Dependency Consistency**: Automatically checks and aligns the model's dependencies to ensure consistency with the original environment.
- **Simple API**: Provides an easy-to-use interface that requires only a few lines of code.
- **Seamless Integration**: Designed to work smoothly with MLflow models generated by tools like Databricks AutoML.

## Why Use This Package

In real-world applications, it's essential to have access to both the predicted classes and their associated probabilities, along with understanding why certain predictions were made (explainability). However, some automated machine learning tools, such as Databricks AutoML, log the best run models in MLflow to return only predictions by default. Modifying these models to include probabilities and explanations can be complex due to the need to handle:

- Multiple components that need adjustment.
- Ensuring environment consistency during repackaging.
- Managing dependencies and versions to match the original training environment.
- Generating SHAP explanations, which often require custom code.

This package simplifies the entire process, allowing you to convert existing MLflow models to return both predictions and probabilities, and generate SHAP explanations effortlessly.

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/Aradhya0510/databricks-mlflow-utils.git
```

## Usage

### Convert a model to return predictions and probabilities:

```python
from databricks_mlflow_utils import get_probabilities

# Replace with your actual model URI
model_uri = "runs:/<run_id>/model"

# Create an instance of the converter
converter = get_probabilities.ConvertToPyFuncForProbability(model_uri)

# Perform the conversion
result = converter.convert()

# Output the URI of the converted model
print("Converted model URI:", result["converted_model_uri"])
```

### Convert a model to include SHAP explanations:

```python
from databricks_mlflow_utils.explainations import get_explanations

# Assuming you have a model logged in MLflow and you have its model URI
model_uri = "runs:/<run_id>/model"

# Create an instance of the converter
converter = get_explanations.ConvertToPyFuncForExplanation(model_uri)

# Inspect the model pipeline to get the expected columns
from sklearn.pipeline import Pipeline
from databricks.automl_runtime.sklearn.column_selector import ColumnSelector

# Check if the model is a Pipeline
if isinstance(converter.model, Pipeline):
    for name, step in converter.model.named_steps.items():
        if isinstance(step, ColumnSelector):
            expected_columns = step.cols
            print(f"Columns expected by the model: {expected_columns}")
else:
    print("Model is not a scikit-learn Pipeline.")

# Prepare the dataset for the explainer
# Define the catalog, schema, and table names
catalog = "your_catalog"
schema = "your_schema"
table = "your_table"

# Read the table from Unity Catalog
spark_df = spark.read.table(f"{catalog}.{schema}.{table}")

# Convert the Spark DataFrame to a Pandas DataFrame
pandas_df = spark_df.toPandas()

# Ensure the DataFrame has the expected columns
data_for_explainer = pandas_df[expected_columns]

# Ensure the DataFrame is indexed correctly
data_for_explainer = data_for_explainer.reset_index(drop=True)
print(f"Dataset columns: {data_for_explainer.columns.tolist()}")

# Create the explainer with your dataset
converter.create_explainer(data_for_explainer)

# Now you can convert the model
result = converter.convert()

# The result contains the URI of the converted model
converted_model_uri = result["converted_model_uri"]
print(f"Converted model URI: {converted_model_uri}")

import mlflow

# Load the converted model
wrapped_model = mlflow.pyfunc.load_model(converted_model_uri)

# Prepare some input data (make sure it matches the model's expected input)
input_data = data_for_explainer.sample(5)  # Sample 5 rows for testing

# Get predictions and explanations
output = wrapped_model.predict(input_data)

print("Predictions and Explanations:")
print(output)

```

**Notes:**

- The `model_uri` should point to the MLflow model you wish to convert.
- The converted model will be logged to MLflow and can be accessed using the returned URI.

## How It Works

### ConvertToPyFuncForProbability Class

This class handles the conversion of an existing MLflow model to return both predictions and probabilities:

- **Loads the Original Model**: Retrieves the model from the specified MLflow run.
- **Ensures Dependency Consistency**: Uses the `DependencyChecker` class to verify and install required dependencies, ensuring the model runs in an environment consistent with the original training environment.
- **Generates Signature and Input Example**: Infers the model's input and output signature for accurate predictions.
- **Logs the New Model**: Creates a new MLflow model that wraps the original model and modifies the `predict` method to return both predictions and probabilities.

### ConvertToPyFuncForExplanation Class

This class extends functionality by adding SHAP-based explanations to the model:

- **Loads the Original Model and Explainer**: Retrieves the model and explainer from MLflow artifacts.
- **Creates SHAP Explainer**: Automatically selects the appropriate SHAP explainer based on the model type and flavor.
- **Generates Signature and Input Example**: Infers the model's input and output signature, including SHAP values.
- **Logs the New Model**: Logs a new model in MLflow that returns both predictions and SHAP explanations when queried.

### DependencyChecker Class

This class ensures that the dependencies required by the model are consistent with those in the original training environment:

- **Retrieves Required Packages**: Extracts the list of dependencies from the original MLflow model.
- **Checks Python Version**: Verifies that the Python version matches the one used during training.
- **Installs Missing Packages**: Automatically installs any missing packages or versions to match the original environment.

## Example Scenario: Databricks AutoML

When using Databricks AutoML, the best run models are logged in MLflow but return only predictions by default. In many cases, especially in classification tasks, you need both the predicted classes and the probabilities for each class, as well as explanations to make informed decisions.

Manually modifying the logged model to include probabilities and explanations involves several complex steps:

- **Creating a PyFunc Wrapper**: You need to write a custom Python function (pyfunc) wrapper that modifies the predict method to return both predictions and probabilities, as well as explanations.
- **Re-logging the Model with Updated Artifacts**: After wrapping, you must re-log the model to MLflow, ensuring that all artifacts are correctly updated.
- **Ensuring Dependency Consistency**: The environment in which the model was trained may have specific versions of libraries. You need to replicate this environment to avoid compatibility issues.
- **Handling Version Conflicts**: Different versions of libraries (e.g., scikit-learn, pandas) can lead to runtime errors if not properly managed.
- **Advanced MLflow Manipulations**: Modifying and repackaging MLflow models requires a deep understanding of MLflow's APIs and best practices.
- **Potential Slowdowns in Productionization**: For users unfamiliar with these advanced concepts, the process can be time-consuming and may delay the deployment of models to production.

This package automates the entire process:

- **Simplifies Repackaging**: With just a few lines of code, you can convert the model to return both predictions, probabilities, and SHAP explanations.
- **Maintains Environment Consistency**: The `DependencyChecker` ensures that the model's dependencies are consistent with the original environment, avoiding runtime errors.
- **Integrates Seamlessly**: Works out of the box with models generated by Databricks AutoML and other MLflow-compatible tools.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

