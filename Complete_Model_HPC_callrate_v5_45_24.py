#full model set for hpc (includes attempts to send to gpu)
print('importing packages')
# Install dependencies before running:
#   pip install -r requirements.txt
import os
import os.path as osp
import torch
from torch_geometric.data import Dataset
import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops
import time
import pickle
#from torch_geometric.nn import SAGPooling  #not needed because I have a custom sagpool
import datetime
import pandas as pd
from torch.nn.functional import softmax
import torchmetrics
import torch.nn as nn
import matplotlib.pyplot as plt
from src.data.dataset import SCDataset
from src.models.gat import GAT
from src.train import run_training

print('package importing done except for sag_pool_custom')

print('setting notes and pre-parameters')

model_notes = '''
#checklist for running as of 2.19.16:
1. Set synth_data to true if running synth data
2. If running real data:
    a. Check the graph_idx. Is it what you want? (exictatory cells, etc)
    b. check the edge index. Is it what you want? (trim_50kb_w_self_edge_index, etc)
    c. check num_excitatory cells. Is it what you want? (len(graph_idx_list))
3. if running synth data
    a. check synth_data_stamp_name is for the dataset you want.
4. set hpc_run to true if running on hpc
5. set num_epochs
6. set batch_size


#version notes
#v0.2.1 you're going to load only the first 10 of the excitatory cells in the dataset class. 
#v0.3.1 added sag pool, extracted summed the sagpool scores to scoresum, added model save function with timestamp
#v0.4.1 added custom sag pool. See week 12 work notes for how to use. key feature: it returns an extra tensor with the complete sagpool scores.
#problem: pre_drop_score_sum collects scores from all epochs during trainging. We really only want the scores from the last epoch.
#v0.4.2 added total run time measurer
#0.5.2 testing last-epoch only score collector
#0.5.3 changed epoch runtime 
#0.6.3 added the scoresum collector
#0.6.4 fixed scoresum failing when run batch size != set batch_size (ie because num graphs not evenly divisible by batch_size)
#0.6.5 fixed epoch counter bug
#0.7.5 extracted scores with metadata and saved to csv
#0.7.5 fixed tensor size alignment issue with scoresum extraction tensor.size [n] != size [1,n]
#1.7.5 FULL RUN COMPLETE!
#1.8.5 added call rate counter, included it in model results, included weighted score in model results, renamed model results csv. Saved the 
#2.8.5 added two more convolution layers, two normalization layers, a dropout layer, and a second pooling layer. also softmaxxed the attention scores and took their product to output as the total graph attention score. 
#2.9.5 added a notes feature
#2.9.6 fixed misshape feed into sagpool caused by 2x attention heads and 2x hidden layer
#2.10.6 activated all changes from 1.8.5 to 2.9.6.
#2.11.6 added precision and f1 metrics. PROBLEM: IT ALWAYS PREDICTS THE SAME CLASS, sometimes 1, sometimes 0.
#2.11.7 changed adam lr to .0001 from .01.
#2.12.7 added the to.device() to accommodate gpu usage on hpc. model still runs on laptop
#2.12.9 changed a bunch of slashes around to keep the hpc happy
#2.12.11 more hpc grammar tweaks. script can't find pts/noedge_idx_0.pt
#2.13.11 added device checker that's called for main model only and goes to model_notes. 
#2.13.12 global variable for model_notes in the check_device function.
#2.13.13 more signposting print statements for hpc
#2.14.13 attempted to fix non-gpu tensor error by sending errant tensors to gpu and moving the todevice function to the top
#2.15.13 fixed send to device error on line 208 in the node call rate calculator. 
#2.16.13 added even more send to device errors
#2.17.13 now you have to send all the outputs to cpu before you can save them.
#2.18.14 added a parallelizer using nn.DataParallel. hopefully this will prevent crashing with 1 gpu. update: I stopped it early, but 99% confident it wont
#2.19.15 manually set num_classes = 2 instead of using the dataset.num_classes in the lin layer. This method loads all data objects to gpu, resulting in gpu death. 
#2.19.16 fixed batch size to 100, added print statements in more places, hopefully this will show up in o file
#2.19.21 added a first pass of archetecture for synth data. running test now. 
2.19.22 solved some issues with non-sparse synth data
2.23.22 finally hit the directory name size limit. moved everything to C:/Users/username/OneDrive - Imperial College London/0_Imperial_main_asof_1.19.23/0Thesis_Project/0MAIN
3.24.22 Synth works! however, the model can't really distinguish between 
3.25.22 sucks on synth data 1 gene above mean sets outcome. radically simplifying model to 1 conv and one sagpool followed by meanpool
3.26.22 added timestamp to model notes-- needed to connect job number on model runs to written output. 
3.27.22 removed the step printing. 
3.28.22 made learning rate a parameter and set from .0001 to .01
3.30.22 added metrics by epoch csv output and predropscores by epoch csv output. graph them to see how ht emodel learns over time. 
3.31.22 fixed call rate with category vector in synth data. note: call rate for real data with category vector will still be wrong. 
3.38.22 Unsoftmaxing the attention score. This will undo normalization, but allow negative values to shine. also doing two convolutions before the pool isntead of 1. 
3.40.22 send the tensors for predrop by epoch list to cpu. check line 496. unknown if this will work
3.41.22 added num_heads as a parameter and increased from 2 to 8. 
3.42.22 num_heads bug fixed
4.42.22: added call rate to feature matrix
callrate 5.46.24: the callrate works on batch 1, cell count 100 on hpc. performance unknown yet. 


#testing notes
#t1: 10 cells loaded in dataset class with 50kb edge index w self loops. 
# Dataset definition bottleneck solved. You shouldn't include raw file names you can't use. 
# Performance is poor: only .4 acc on the test set.
#t2: v0.5.2, 100 cells loaded w 50kb self loop edge index
# runtime: ~5m per epoch, terminated after two epochs
# performance on two epochs: train acc: .6 test acc: .3
#t3: v1.8.5. 600s/epoch, 20ish mins(?) of class instantiation time
#t4: v1.8.5 all excitatory cells, left to run overnight. failed because of memory overallocation
#t5: v1.8.5 tried 20000 excitatory cells. failed because of memory overallocation. always step 1 of forward pass
#t6: v1.8.5 tried 10000 excitatory cells. failed because of memory allocation.
#t7: v1.8.5 tried 5000 excitatory cells. failed mem alloc. I think it's the batch size of 100, not the cell count. 
#t8: v1.8.5 tried 1000 exctatory cells, batch soze = 10. ran fine. total time 14k sec. test acc .63. scores ar weird-- top 1000 all sam score. 
#t9: v1.8.5 tried all excitatory cells, batch size = 10. 
#t10: v2.10.6 tried 1000 excitatory cells, batch size = 10. computer froze.
#t11: v2.10.6 rerun of t10
#t12: v2.11.6 run of 100 cells, batch = 10. need some data for the visualizer. cpu crash. lowering batchs size to 2. 
#t12: v2.11.6 run of 100 cells, batch = 2. no crash. still pred all the same values. 
#t12: v2.11.7 run of 1000, batch = 5. new adam value .0001 from .01. see if it fixes the pred same prob. 
#t13 v2.17.13 run on hpc. all cells, batch size = 30. ran out of memory and crashed, >20gb used. Reducing batch size, but that may not be hte issue.
#t14 v3.28.22 on synth data 20230802_222843. IT WORKS! YOU NEEDED TO ESTABLISH NODE TYPE!!!! 
'''
script_start_time = time.time()

#pre-parameters are now loaded from config.py so paths are not hard coded
from config import CONFIG

num_excitatory_cells = CONFIG.num_excitatory_cells
hpc_run = CONFIG.hpc_run
os.chdir(CONFIG.base_path)
synth_data = CONFIG.synth_data
synth_data_stamp_name = CONFIG.synth_data_stamp_name
if synth_data == True:
    with open(f"synth_datasets/dataset_{synth_data_stamp_name}/dataset_{synth_data_stamp_name}_graph_idx_list.pkl", 'rb') as f:
        synth_graph_idx_list = pickle.load(f)
    synth_edge_index = torch.load(f"synth_datasets/dataset_{synth_data_stamp_name}/dataset_{synth_data_stamp_name}_edge_index.pt")
    global_edge_index = synth_edge_index
    graph_idx_list = synth_graph_idx_list
    root_dir = f"synth_datasets/dataset_{synth_data_stamp_name}"
    #open the dataset notes and append to model notes
    with open(f"synth_datasets/dataset_{synth_data_stamp_name}/dataset_{synth_data_stamp_name}_notes.txt", 'r') as f:
        dataset_notes = f.read()
    model_notes += f"synth data used. {dataset_notes}"

    print("loaded graph idx pickle, loaded edge torch, appended dataset notes")

else:
    #import pickle with excitatory graph_idx_list
    with open(r"graph_idx/excitatory_graph_idx_list.pkl", 'rb') as f:
        excitatory_graph_idx_list = pickle.load(f)
    #load .pt with desired edge index
    trim_50kb_w_self_edge_index = torch.load(r"edge_indices/trim_50kb_w_self_edge_index.pt")
    global_edge_index = trim_50kb_w_self_edge_index
    graph_idx_list = excitatory_graph_idx_list[:num_excitatory_cells]
    root_dir = r"pts"
    print("loaded graph idx pickle and edge torch")

num_graphs = len(graph_idx_list)
split_point_for_traintest = int(len(graph_idx_list)*.8) #cell count used for training, rest for testing. 
num_epochs = CONFIG.num_epochs
batch_size =  1 
pooling_keep_count = int(187215) #the number of nodes kept in pooling. 187215 is the node count, menaing none are dropped. 
learning_rate = .001 
num_classes = 2
num_heads = 8
torch.manual_seed(12345)

print('parameters set')
#you can't import this above because you need to have set the working directory first.
from sag_pool_custom import SAGPoolingCustom

#append the model notes with the parameters
model_notes += f"""
#parameters
synth_data = {synth_data}
synth_data_stamp_name = {synth_data_stamp_name} #only meaningful if synth_data = True
hp_run = {hpc_run}
num_graphs = {num_graphs}
split_point_for_traintest = {split_point_for_traintest}
num_epochs = {num_epochs}
batch_size = {batch_size}
pooling_keep_count = {pooling_keep_count}
root_dir = {root_dir}
learning_rate = {learning_rate}
"""

#debugging tools
#load up the feature matrix
#def load_pickle_file(file_path):
#    with open(file_path, 'rb') as f:
#        data = pickle.load(f)
#    return data
#data = load_pickle_file(r"GSE214979_data_7_4_23_v3.2.0.pkl")
def tensors_have_same_values(x, x_hat):
    # Compare the tensors element-wise
    comparison = torch.eq(x, x_hat)
    
    # Check if all elements are the same (all True)
    all_same = comparison.all()
    
    return all_same

#you don't use these pickles for synth data
if synth_data == False:
    # Load the dictionaries from pickle files
    with open(r"dic_pickles/feature_to_id.pkl", 'rb') as f:
        feature_to_id = pickle.load(f)

    with open(r"dic_pickles/id_to_feature.pkl", 'rb') as f:
        id_to_feature = pickle.load(f)

    with open(r"dic_pickles/id_to_node_metadata.pkl", 'rb') as f:
        id_to_node_metadata = pickle.load(f)


print('pickles loaded')

#setting device expresses a preference for gpu (cuda) if avaliable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device checker to see if you're really on gpu. note: don't call it on every tensor or the messages will be crazy long
def check_device(pytorch_object):
    global model_notes
    if hasattr(pytorch_object, 'device'):
        print(f'The object is on {pytorch_object.device}.')
        model_notes += f'The object is on {pytorch_object.device}.'
    elif hasattr(pytorch_object, 'parameters'):
        parameters = list(pytorch_object.parameters())
        if parameters:
           print(f'The model is on {parameters[0].device}.')
           model_notes += f'The model is on {parameters[0].device}.'
        else:
            print('The model has no parameters.')
            model_notes += 'The model has no parameters.'
    else:
        print('The object is not a PyTorch tensor or model.')
        model_notes += 'The object is not a PyTorch tensor or model.'


#define the dataset
#t1


#model starts here
dataset = SCDataset(root_dir, global_edge_index, graph_idx_list, device, synth_data)

print('dataset defined')
print('calculating call rate')

#calculate the call rate
node_call_rate = dataset.calculate_call_rate_by_node() #you're calculating this for the second time, but here it's global and can get refernced later. 
print('call rate calculated again')

datafirst = dataset[0]  # Get the first graph object. You use this later for extracting feature matrix size. don't delete. 

#explore
print()
print(datafirst)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {datafirst.num_nodes}')
print(f'Number of edges: {datafirst.num_edges}')
print(f'Average node degree: {datafirst.num_edges / datafirst.num_nodes:.2f}')
print(f'Has isolated nodes: {datafirst.has_isolated_nodes()}')
print(f'Has self-loops: {datafirst.has_self_loops()}')
print(f'Is undirected: {datafirst.is_undirected()}')

#post-parameters
dataset = dataset.shuffle()
train_dataset = dataset[:split_point_for_traintest]
test_dataset = dataset[split_point_for_traintest:num_graphs] 
pre_drop_score_sum = torch.zeros(dataset[0].num_nodes).to(device) #define empty tensor to accumulate sagpool scores using first graph. this breaks if you start varying the number of nodes across graphs.
#pre_drop_score_sum_not_final_epoch = torch.zeros(dataset[0].num_nodes).to(device) #redundant

#set dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()
"""

#altnerative new model, lots of changes. let's see if it works better

print('GAT defined')



#going from here to training for loop took an hour or two. no idea why
model = GAT(dataset.num_node_features, hidden_channels=64, num_heads=num_heads, pooling_keep_count=pooling_keep_count, num_classes=num_classes)

#distribute across gpus
if torch.cuda.device_count() > 1:
    print("you're using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

model.to(device) #sending the model to gpu
check_device(model)
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

print('model defined')


run_training()
