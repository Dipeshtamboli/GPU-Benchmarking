from datetime import time
import pandas as pd
import json
import pdb

time_data = json.load(open("all_server_data.json", 'r'))
# print(time_data)

for gpu in time_data.keys():
    data = pd.DataFrame(time_data[gpu])
    # pdb.set_trace()
    data.plot(kind="bar").get_figure().savefig(f"plots/{gpu}.png")