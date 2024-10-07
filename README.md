# MLflow Model Converter for Predictions and Probabilities

A Python package that simplifies the conversion of existing MLflow models to return both predictions and probabilities. It ensures consistency of model dependencies during repackaging, making it particularly useful in scenarios like Databricks AutoML, where the best run model logs only return predictions by default.

## Table of Contents

- [Features](#features)
- [Why Use This Package](#why-use-this-package)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
  - [ConvertToPyFuncForProbability Class](#converttopyfuncforprobability-class)
  - [DependencyChecker Class](#dependencychecker-class)
- [Example Scenario: Databricks AutoML](#example-scenario-databricks-automl)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Convert Existing MLflow Models**: Transform any existing MLflow model to return both predictions and probabilities without retraining.
- **Dependency Consistency**: Automatically checks and aligns the model's dependencies to ensure consistency with the original environment.
- **Simple API**: Provides an easy-to-use interface that requires only a few lines of code.
- **Seamless Integration**: Designed to work smoothly with MLflow models generated by tools like Databricks AutoML.

## Why Use This Package

In many real-world applications, it's essential to have access to both the predicted classes and the associated probabilities for each class. However, some automated machine learning tools, such as Databricks AutoML, log the best run models in MLflow to return only predictions by default. Modifying these models to include probabilities can be complex due to the need to handle:

- Multiple components that need adjustment.
- Ensuring environment consistency during repackaging.
- Managing dependencies and versions to match the original training environment.

This package simplifies the entire process, allowing you to convert existing MLflow models to return both predictions and probabilities effortlessly.

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/Aradhya0510/databricks-mlflow-utils.git
```

## Usage

Here's how you can use the package to convert an existing MLflow model:

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

### DependencyChecker Class

This class ensures that the dependencies required by the model are consistent with those in the original training environment:

- **Retrieves Required Packages**: Extracts the list of dependencies from the original MLflow model.
- **Checks Python Version**: Verifies that the Python version matches the one used during training.
- **Installs Missing Packages**: Automatically installs any missing packages or versions to match the original environment.

## Example Scenario: Databricks AutoML

When using Databricks AutoML, the best run models are logged in MLflow but return only predictions by default. In many cases, especially in classification tasks, you need both the predicted classes and the probabilities for each class to make informed decisions.

Manually modifying the logged model to include probabilities involves:

- Adjusting the model's `predict` method.
- Ensuring that all dependencies and the environment remain consistent.
- Handling potential version conflicts and compatibility issues.

This package automates the entire process:

- **Simplifies Repackaging**: With just a few lines of code, you can convert the model to return both predictions and probabilities.
- **Maintains Environment Consistency**: The `DependencyChecker` ensures that the model's dependencies are consistent with the original environment, avoiding runtime errors.
- **Integrates Seamlessly**: Works out of the box with models generated by Databricks AutoML and other MLflow-compatible tools.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
