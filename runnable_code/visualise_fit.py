import numpy as np
import json
import uuid
import os

from etas.plots import ETASFitVisualisation


if __name__ == '__main__':
    
    with open("../config/visualisation_config.json", 'r') as f:
        visualisation_config = json.load(f)
    
    with open(visualisation_config["output_etas"], 'r') as f:
        etas_output = json.load(f)
        
    store_path = f"../output_data/fits_{uuid.uuid4()}/"
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    metadata = {
        "fn_catalog": visualisation_config["fn_catalog"],
        "fn_Pij": visualisation_config["fn_Pij"],
        "delta_m": etas_output["delta_m"],
        "mc": etas_output["m_ref"],
        "parameters": etas_output["final_parameters"],
        "label": visualisation_config["label"],
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