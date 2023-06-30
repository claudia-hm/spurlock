"""Utils module that can be used by any subfolder."""
import yaml  # type: ignore


def readYML(filename):
    """Read fairness yaml file."""
    # Load the YAML file
    with open(filename, "r") as file:
        yaml_data = yaml.safe_load(file)

    return yaml_data
