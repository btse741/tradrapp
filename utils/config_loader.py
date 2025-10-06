import os
import yaml

def load_db_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
    config_path = os.path.join(project_root, "config.yml")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    params = config['database']
    conn_params = {
        'dbname': params['dbname'],
        'user': params['user'],
        'password': params['password'],
        'host': params['host'],
        'port': params['port']
    }
    return conn_params
