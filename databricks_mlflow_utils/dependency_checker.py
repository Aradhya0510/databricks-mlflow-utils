import mlflow
import subprocess
import pkg_resources
import sys
import mlflow.pyfunc

class DependencyChecker:
    def __init__(self, model_uri):
        self.model_uri = model_uri
        self.required_packages = self.get_required_packages()
        self.required_python_version = self.get_required_python_version()

    def get_required_packages(self):
        # Retrieve the pip requirements from the model
        pip_requirements = mlflow.pyfunc.get_model_dependencies(self.model_uri)
        # Check if pip_requirements is a path to a file
        if isinstance(pip_requirements, str):
            # It's a path to a file
            with open(pip_requirements, 'r') as f:
                requirements = f.readlines()
            requirements = [line.strip() for line in requirements if line.strip()]
            return requirements
        elif isinstance(pip_requirements, list):
            return pip_requirements
        else:
            return []

    def get_required_python_version(self):
        # Get the Python version from the MLmodel file
        local_path = mlflow.artifacts.download_artifacts(self.model_uri)
        model_conf = mlflow.models.Model.load(local_path)
        python_version = model_conf.flavors['python_function'].get('python_version')
        return python_version

    def check_python_version(self):
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if self.required_python_version and not self.required_python_version.startswith(current_version):
            print(f"Python version mismatch: model requires Python {self.required_python_version}, current version is {current_version}")
            return False
        return True

    def check_and_install_packages(self):
        if isinstance(self.required_packages, str):
            # It's a path to requirements.txt
            print("Installing required packages from requirements.txt...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", self.required_packages])
            print("Packages installed.")
        elif isinstance(self.required_packages, list):
            # Existing logic for handling a list of packages
            installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
            packages_to_install = []
            for requirement in self.required_packages:
                pkg_name = requirement.strip()
                if pkg_name.startswith("-"):
                    # Skip options like '-e .'
                    continue
                elif "==" in pkg_name:
                    name, version = pkg_name.split("==")
                    if name.lower() in installed_packages:
                        if installed_packages[name.lower()] != version:
                            print(f"Version mismatch for {name}: installed {installed_packages[name.lower()]}, required {version}")
                            packages_to_install.append(pkg_name)
                    else:
                        print(f"{name} not installed.")
                        packages_to_install.append(pkg_name)
                else:
                    if pkg_name.lower() not in installed_packages:
                        print(f"{pkg_name} not installed.")
                        packages_to_install.append(pkg_name)
            if packages_to_install:
                print("Installing required packages...")
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages_to_install)
                print("Packages installed.")
            else:
                print("All required packages are already installed.")
        else:
            print("No required packages found.")