import pandas as pd
import os,time
import glob
import plotly.offline
# plotly.offline.init_notebook_mode()
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=True, world_readable=True)
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
folder_name='auriga_result/'

csv_list=glob.glob(folder_name+'/*.csv')
columes=[]
for key,values in MODEL_LIST.items():
    for i in values:
        if i in ["mobilenet_v2", "MobileNetV3"]:
            continue
        columes.append((key,i))

for csv in csv_list:
    df=pd.read_csv(csv)
    # pdb.set_trace()
    df.columns = pd.MultiIndex.from_tuples(columes)
    df.groupby(level=0,axis=1).mean().mean()
#     print(csv)
    title=csv.split('/')[1].split('_benchmark')[0]
    title=title.replace(' ','_')
    plot_data = df.groupby(level=0,axis=1).mean().mean().to_frame()
    pdb.set_trace()
    plot = plot_data.plot()
    fig = plot.get_figure()
    fig.savefig("output.png")
    # plot_data.plot(x=plot_data.keys(), y=plot_data,kind='scatter',mode='markers',title=title,yTitle='time(ms)',xTitle='models',asImage=True,filename=title)
    # df.groupby(level=0,axis=1).mean().mean().to_frame().plot(kind='scatter',mode='markers',title=title,yTitle='time(ms)',xTitle='models',asImage=True,filename=title)
    for model in MODEL_LIST.keys():
        df.mean()[model].iplot(kind='scatter',mode='markers',title=model+"_"+title,yTitle='time(ms)',xTitle='models',asImage=True,filename=model+"_"+title)
        time.sleep(1)