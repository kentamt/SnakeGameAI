class BaseObject:
    """Define a base class for all objects in the YAML file"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Recursive function to create an instance of the appropriate Python class for each level of nesting in the YAML file
def create_object(data):
    # If the data is a mapping, create an instance of BaseObject
    if isinstance(data, dict):
        obj = BaseObject(**data)

        # Recursively create instances of the appropriate Python classes for each nested object
        for key, value in data.items():
            setattr(obj, key, create_object(value))

        return obj
    # If the data is a list, create a list of instances of the appropriate Python classes for each item in the list
    elif isinstance(data, list):
        return [create_object(item) for item in data]
    # Otherwise, return the data as-is
    else:
        return data


class EnvInfo:
    def __init__(self, env_name):
        self.name = env_name


class LogCfg:
    def __init__(self, agent, curr_time, env_name, cfg_path):
        self.agent = agent
        self.curr_time = curr_time
        self.env_name = env_name
        self.cfg_path = cfg_path
