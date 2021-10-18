import pandas as pd
import os,time
import glob
import json
import torchvision.models as models
import pdb

MODEL_LIST = {
    'mnasnet':models.mnasnet.__all__[1:],
    'resnet': models.resnet.__all__[1:],
    'densenet': models.densenet.__all__[1:],
    'squeezenet': models.squeezenet.__all__[1:],
    'vgg': models.vgg.__all__[1:],
    'mobilenet':models.mobilenet.__all__[1:],
    'shufflenetv2':models.shufflenetv2.__all__[1:]
}

columes=[]
for key,values in MODEL_LIST.items():
    for i in values:
        if i in ["mobilenet_v2", "MobileNetV3"]:
            continue
        columes.append((key,i))

folder_name='*_results/'
csv_list=glob.glob(folder_name+'/*.csv')
combine_csv = {}
for csv in csv_list:
    df=pd.read_csv(csv)
    df.columns = pd.MultiIndex.from_tuples(columes)
    df.groupby(level=0,axis=1).mean().mean()
    title=csv.split('/')[1].split('_benchmark')[0]
    title=title.replace(' ','_')
    mode = title.split("__")[-1].split('_')[-1]
    precisn = title.split("__")[-1].split('_')[0]
    server = csv.split('/')[0].split('_')[0]
    gpu_name = title.split("__")[0]
    if not f"{server}_{gpu_name}" in combine_csv.keys(): combine_csv[f"{server}_{gpu_name}"]={};
    combine_csv[f"{server}_{gpu_name}"][f"{precisn}_{mode}"] = df.groupby(level=0,axis=1).mean().mean().to_frame().to_dict()[0]
comb_df = pd.DataFrame(combine_csv)
json.dump(combine_csv, open("all_server_data.json",'w'))