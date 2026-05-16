import os
import yaml

def load_yaml_config(config_path: str) -> dict:
    """Load and parse a YAML configuration file, resolving relative paths.
    
    Supports both root-level path keys (interface workflow) and 'paths' block 
    keys (surface workflow).
    """
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    
    config_dir = os.path.dirname(os.path.abspath(config_path))
    
    # 1. Resolve root-level paths (Interface style)
    root_keys = ["sub_path", "film_path", "structure_path", "output_dir"]
    for key in root_keys:
        val = cfg.get(key)
        if val and isinstance(val, str) and not os.path.isabs(val):
            candidate = os.path.join(config_dir, val)
            if os.path.exists(candidate):
                cfg[key] = candidate
                
    # 2. Resolve 'paths' block paths (Surface style)
    paths = cfg.get("paths", {})
    if isinstance(paths, dict):
        path_keys = ["precursor", "inhibitor", "substrate_bulk", "input_structure", "output_prefix"]
        for key in path_keys:
            val = paths.get(key)
            if val and isinstance(val, str) and not os.path.isabs(val):
                # Try relative to config_dir first
                candidate = os.path.join(config_dir, val)
                if os.path.exists(candidate):
                    paths[key] = candidate
                # Legacy fallback check in current working directory handled by os.path.exists(val) check in original code
                # but we prefer absolute paths.
    
    return cfg
