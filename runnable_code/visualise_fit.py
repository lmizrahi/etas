import numpy as np
import json
import uuid
import os

from etas.plots import ETASFitVisualisation


if __name__ == '__main__':
    
    with open("../config/visualisation_config.json", 'r') as f:
        visualisation_config = json.load(f)
    
    with open(visualisation_config["fn_parameters"], 'r') as f:
        etas_output = json.load(f)
        
    store_path = f"{visualisation_config['data_path']}fits_{etas_output['id']}/"
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    metadata = {
        "fn_catalog": etas_output["fn_catalog"],
        "fn_pij": etas_output["fn_pij"],
        "delta_m": etas_output["delta_m"],
        "mc": etas_output["m_ref"],
        "parameters": etas_output["final_parameters"],
        "label": etas_output["name"],
        "magnitude_list": np.arange(
            visualisation_config["magnitudes"]["lower"],
            visualisation_config["magnitudes"]["upper"],
            etas_output["delta_m"]
        ),
        "store_path": store_path,
        "comparison_parameters": visualisation_config["comparison_parameters"]
    }

    fit_vis = ETASFitVisualisation(metadata)
    fit_vis.all_plots()