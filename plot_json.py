from datetime import time
from matplotlib.transforms import Bbox
import pandas as pd
import json
import pdb

time_data = json.load(open("all_server_data.json", 'r'))

# For server-wise plots
for gpu in time_data.keys():
    data = pd.DataFrame(time_data[gpu])
    data.plot(kind="bar").get_figure().savefig(f"server_wise_plots/{gpu}.png")

# Averaging over different deep models 
model_wise = {}
for gpu in time_data.keys():
    if gpu == "auriga_Tesla_P100-PCIE-16GB_2_gpus":
        continue
    model_wise[gpu] = pd.DataFrame(time_data[gpu]).mean().to_dict()

data = pd.DataFrame(model_wise)
plot = data.plot(kind="bar", title = "Avg time taken for one iteration")
plot.set_ylabel("time (ms)")
plot.set_xlabel("modes (precision and training/inference)")
fig = plot.get_figure()
fig.savefig(f"with_bell_G_time_taken_vs_experiment_modes for different GPUs.png", bbox_inches='tight')