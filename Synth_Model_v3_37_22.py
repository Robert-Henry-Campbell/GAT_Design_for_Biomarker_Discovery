#full model set for hpc (includes attempts to send to gpu)
print('importing packages')
import subprocess
import sys
def install_package(package): #install packages with this tool | Example usage: install_package("torchmetrics")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
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

#pre-parameters are now loaded from config.py
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
batch_size =  100 
pooling_keep_count = int(187215) #the number of nodes kept in pooling. 187215 is the node count, menaing none are dropped. 
learning_rate = .01 
num_classes = 2
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
class SCDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(SCDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.root = root  # add this if you want to specify the root directory while creating an object of your dataset
        self.edge_index = global_edge_index #the edge index is defined here and passed to data objects as they are loaded

    @property
    def raw_file_names(self):
        # Since your files are named "noedgeidx_<value>.pt", we can generate this list using a list comprehension
        return [f'noedgeidx_{i}.pt' for i in graph_idx_list]  # adjust the graph idx list above

    @property
    def processed_file_names(self):
        # If you're not doing any preprocessing, you can just reuse your raw_file_names here
        return self.raw_file_names

    def len(self):
        return len(self.raw_file_names)
    
    def get(self, idx):
        # You can just load the raw file here since you're not preprocessing the data
        data = torch.load(osp.join(self.root, self.raw_file_names[idx]))
        data.edge_index = self.edge_index.to(device) #you need this line to give it an edge index
        if data.x.device != device:
            data = data.to(device)
        return data

    
    def calculate_call_rate_by_node(self):
        graph_count = self.len()  # total number of graphs
        node_nonzero_count = torch.zeros(self.get(0).num_nodes, dtype=torch.float).to(device)  # non-zero count per node
        if synth_data == False:
            for idx in range(graph_count):
                data = self.get(idx)
                if data.x.is_sparse:  
                    # If sparse, directly add the number of non-zero values for each node
                    node_indices = data.x._indices()[0]  # Get the indices of non-zero elements
                    node_nonzero_count.index_add_(0, node_indices.to(device), torch.ones(node_indices.size(0), device=device)) #this line is the replacement for the one below it v2.15.13
                    #node_nonzero_count.index_add_(0, node_indices, torch.ones(node_indices.size(0))).to(device) #device error in this line
                else: 
                    input(f"calculating call rate, but graph {idx} isn't sparse! ")
            call_rate = node_nonzero_count / graph_count  # calculate call rate
            return call_rate
        else:
            
            ''' #this version assumes feature matrix has no one hot encoding (ie one value only)
            # Calculate call rate for synthetic data, which is not sparse
            for idx in range(graph_count):
                data = self.get(idx)
                # If dense, manually get the indices of non-zero elements
                node_indices = (data.x != 0).nonzero(as_tuple=True)[0]
                node_nonzero_count.index_add_(0, node_indices.to(device), torch.ones(node_indices.size(0), device=device))
            '''
            #this untested gpt version checks call rate on zero element of each node only (ie the true feature)
            for idx in range(graph_count):
                data = self.get(idx)
                # Check where this element is not equal to zero
                node_indices = (data.x[:, 0] != 0).nonzero(as_tuple=True)[0] # the [:, 0] selects first column (ie the feature)
                node_nonzero_count.index_add_(0, node_indices.to(device), torch.ones(node_indices.size(0), device=device))
            
            call_rate = node_nonzero_count / graph_count  # calculate call rate
            return call_rate


#model starts here
dataset = SCDataset(root_dir)

print('dataset defined')
print('calculating call rate')

#calculate the call rate
node_call_rate = dataset.calculate_call_rate_by_node()
print('call rate calculated')

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
class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(dataset.num_node_features, hidden_channels, heads=2)
        self.conv2 = GATConv(hidden_channels*2, hidden_channels, heads=2)
        self.pool1 = SAGPoolingCustom(hidden_channels*2, min_score=None, multiplier=1.0, ratio = pooling_keep_count)
        self.lin = Linear(hidden_channels*2, num_classes) #I had dataset.num_classes here, but it loads every data object at once to find out how many classes there are which caused GPU death. instead made a global variable num_classes 
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels*2)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels*2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index) 
        x = x.relu()
        #x = self.batch_norm1(x)
        #x, edge_index, _, batch, perm, score, pre_drop_score_one = self.pool1(x, edge_index, None, batch) # pooling after the conv layer
        #x = self.conv2(x, edge_index)
        #x = x.relu()
        #x = self.batch_norm2(x)
        x, edge_index, _, batch, perm, score, pre_drop_score_two = self.pool1(x, edge_index, None, batch) # pooling after the conv layer

        x = global_mean_pool(x, batch)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        #pre_drop_score_one = softmax(pre_drop_score_one, dim=0) #softmax to normalize across graphs
        pre_drop_score_two = softmax(pre_drop_score_two, dim=0) #softmax to normalize across graphs

        #pre_drop_score = pre_drop_score_one * pre_drop_score_two #the product of the two scores is the total feature 
        pre_drop_score = pre_drop_score_two #test redefinition for model tuning
        return x, pre_drop_score

print('GAT defined')



#going from here to training for loop took an hour or two. no idea why
model = GAT(hidden_channels=64)

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

def train():
    model.train()

    for data in train_loader:
        #try sending everything to device
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)
        data.y = data.y.to(device)         

        #actual training stuff here 
        out,pre_drop_score = model(data.x, data.edge_index, data.batch)  
        loss = criterion(out, data.y) 
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()  

predrop_by_epoch_list = []
#alternative new test function
def test(loader):
    model.eval()
    
    metric_precision = torchmetrics.Precision(task = 'binary', average='macro').to(device)
    metric_f1 = torchmetrics.F1Score(task = 'binary', average='macro', num_classes=2).to(device)

    correct = 0
    total = 0

    for data in loader:
        #try sending everything to device
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)
        data.y = data.y.to(device)
        
        
        out, pre_drop_score = model(data.x, data.edge_index, data.batch)
        
        #if it's the last epoch accumulate the final epoch pre_drop_scores.
        if epoch == num_epochs: #ie if it's the last epoch. 
            global pre_drop_score_sum  # Needed to modify global copy of pre_drop_score_sum
            pre_drop_score_sum = pre_drop_score_sum.to(device) #this is probably redundant. 

            #all scores are batched together in mega-feature matrix. must cut them up into <batchsize> graphs
            batched_pre_drop_scores = pre_drop_score.detach()  # Detach the score from the current computation graph
            num_graphs_in_batch = batched_pre_drop_scores.size()[0]/datafirst.num_nodes
            
            #check the num graphs in batch is a whole number, suggesting the correct number of nodes in each graph
            assert (batched_pre_drop_scores.size()[0]/datafirst.num_nodes).is_integer(), f"num nodes in batch = {num_graphs_in_batch}, not evenly divisble by {datafirst.num_nodes}"
            
            #once checked, you need to turn it from float to int to satisfy view function. Can't do this before checking because it would conceal badly shaped tensors
            num_graphs_in_batch = int(num_graphs_in_batch)

            #re-shape the tensor to separate graphs by rows
            unbatched_pre_drop_scores = batched_pre_drop_scores.view(num_graphs_in_batch, datafirst.num_nodes)
            split_pre_drop_scores = torch.split(unbatched_pre_drop_scores, split_size_or_sections=1, dim=0)

            #check again the shape is right, add to the score sum
            for i, graph_pre_drop_score in enumerate(split_pre_drop_scores):
                assert graph_pre_drop_score.shape == (1,datafirst.num_nodes), f"graph_pre_drop_score {i} has incorrect shape {graph_pre_drop_score.shape}"
                pre_drop_score_sum += graph_pre_drop_score.squeeze() #the squeeze takes it from size [1,x] to size [x]. no data lost, but size match acheived. 

            #check that all the values were used by checking the len of the split score list is equal to batch size
            assert len(split_pre_drop_scores) == num_graphs_in_batch, f"Not all values from the original tensor were used, expected {batch_size} tensors, got {len(split_pre_drop_scores)}"

        #if it's not the final epoch, calculate predrop sum for the epoch and add it to the list of dictionaries of predropscoresums by epoch
        else:
            global predrop_by_epoch_list
            single_batch_scoresum = torch.zeros(dataset[0].num_nodes).to(device)
            #all scores are batched together in mega-feature matrix. must cut them up into <batchsize> graphs
            batched_pre_drop_scores = pre_drop_score.detach()  # Detach the score from the current computation graph
            num_graphs_in_batch = batched_pre_drop_scores.size()[0]/datafirst.num_nodes
            
            #check the num graphs in batch is a whole number, suggesting the correct number of nodes in each graph
            assert (batched_pre_drop_scores.size()[0]/datafirst.num_nodes).is_integer(), f"num nodes in batch = {num_graphs_in_batch}, not evenly divisble by {datafirst.num_nodes}"
            
            #this line replaces the commented code below
            single_batch_scoresum = torch.sum(batched_pre_drop_scores.view(int(num_graphs_in_batch), datafirst.num_nodes), dim=0)
           
            """
            #once checked, you need to turn it from float to int to satisfy view function. Can't do this before checking because it would conceal badly shaped tensors
            num_graphs_in_batch = int(num_graphs_in_batch)

            #re-shape the tensor to separate graphs by rows
            unbatched_pre_drop_scores = batched_pre_drop_scores.view(num_graphs_in_batch, datafirst.num_nodes)
            split_pre_drop_scores = torch.split(unbatched_pre_drop_scores, split_size_or_sections=1, dim=0)


            #check again the shape is right, add to the score sum
            for i, graph_pre_drop_score in enumerate(split_pre_drop_scores):
                assert graph_pre_drop_score.shape == (1,datafirst.num_nodes), f"graph_pre_drop_score {i} has incorrect shape {graph_pre_drop_score.shape}"
                single_batch_scoresum += graph_pre_drop_score.squeeze() #the squeeze takes it from size [1,x] to size [x]. no data lost, but size match acheived. 

            #check that all the values were used by checking the len of the split score list is equal to batch size
            assert len(split_pre_drop_scores) == num_graphs_in_batch, f"Not all values from the original tensor were used, expected {batch_size} tensors, got {len(split_pre_drop_scores)}"
            """
            predrop_by_epoch_list.append({epoch: single_batch_scoresum.numpy()})



        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
        
        metric_precision.update(pred, data.y)
        metric_f1.update(pred, data.y)

    accuracy = correct / total
    precision = metric_precision.compute()
    f1 = metric_f1.compute()
    return accuracy, precision, f1


print('train and test defined')
print('starting the epochs...')
#eochs with 4 graphs takes a minute total with 4 cells, no self edge 50kb
epoch_start_time = time.time()
metrics_list = []
for epoch in range(1, num_epochs+1):
    train()
    train_acc, train_prec, train_f1 = test(train_loader)
    test_acc, test_prec, test_f1 = test(test_loader)
    metrics = (f' \n Epoch: {epoch:03d} | epoch_time: {round(time.time()-epoch_start_time,1)} | Train Acc: {train_acc:.4f} | Train Prec: {train_prec:.4f} | Train F1: {train_f1:.4f} | Test Acc: {test_acc:.4f} | Test Prec: {test_prec:.4f} | Test F1: {test_f1:.4f}' )
    print(metrics)
    model_notes += metrics

    metrics_list.append({
        'Epoch': epoch,
        'Train_Acc': train_acc,
        'Train_Prec': train_prec.item(),
        'Train_F1': train_f1.item(),
        'Test_Acc': test_acc,
        'Test_Prec': test_prec.item(),
        'Test_F1': test_f1.item()
    })

    epoch_start_time = time.time()

print('training complete, moving to saving results')

# Function to generate the timestamp
def get_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    return timestamp

#set single timestamp for matching score and model names
timestamp = get_timestamp()
model_notes += f"timestamp : {timestamp}"

#make a pd dataframe of the scores
# Create a DataFrame with node IDs and their corresponding scores
model_results_df = pd.DataFrame({
    'node_id': range(len(pre_drop_score_sum.cpu())),
    'score': pre_drop_score_sum.cpu().numpy(),
    'call_rate': node_call_rate.cpu().numpy(),
    #'weighted_score': pre_drop_score_sum.cpu().numpy() % node_call_rate.cpu().numpy() #look up how to do tensor division
})

#set the path where results will be saved. 
if synth_data == True:
    outputs_folder = "synth_models"
else:
    outputs_folder = "models"


#if it's a true model, you add node metadata to results 
if synth_data == False:
    #for each node, we want to add the metadata
    #this takes about 20 mins on the laptop: huge time sink. You could pre-make the metadata part of the df to speed it up
    for node_id in model_results_df['node_id']:
        metadata_series = id_to_node_metadata[node_id][1]
        for column, value in metadata_series.items():
            model_results_df.loc[model_results_df['node_id'] == node_id, column] = value
    # Set the node_id as the index of the DataFrame
    model_results_df.set_index('node_id', inplace=True)
    #save results to a csv
    model_results_df.to_csv(f"{outputs_folder}/Model_{timestamp}_results.csv", index = False)
else:  #save model results without appending node metadata
    #save results to a csv
    model_results_df.to_csv(f"{outputs_folder}/Model_{timestamp}_results.csv")

#save call rate to a pt
call_rate_filename = f"{outputs_folder}/Model_{timestamp}_call_rate.pt"
torch.save(node_call_rate.cpu(), call_rate_filename)

#save notes to a txt file
with open(f"{outputs_folder}/Model_{timestamp}_notes.txt",'w') as notes_file:
    notes_file.write(model_notes)

# Create an metrics by epoch DataFrame from list
metrics_by_epoch_df = pd.DataFrame(metrics_list)
# Save metrics by epoch to csv
metrics_by_epoch_filename = f"{outputs_folder}/Model_{timestamp}_metrics_by_epoch.csv"
metrics_by_epoch_df.to_csv(metrics_by_epoch_filename, index=False)

#plot the training metrics by epoch
metrics_by_epoch_df.plot(x='Epoch', y=['Train_Acc', 'Train_Prec', 'Train_F1'],kind='line', title='Training Performance', grid=True)
plt.xlabel('Epoch')
plt.ylabel('Performance')
plt.savefig(f"{outputs_folder}/Model_{timestamp}_train_metrics_by_epoch.png",dpi=300)
plt.clf()

#plot the test metrics by epoch
metrics_by_epoch_df.plot(x='Epoch', y=['Test_Acc', 'Test_Prec', 'Test_F1'],kind='line', title='Test Performance', grid=True)
plt.xlabel('Epoch')
plt.ylabel('Performance')
plt.savefig(f"{outputs_folder}/Model_{timestamp}_test_metrics_by_epoch.png",dpi=300)
plt.clf()

#train vs test by epoch
metrics_by_epoch_df.plot(x='Epoch', y=['Test_F1', 'Train_F1'],kind='line', title='Test vs Train F1', grid=True)
plt.xlabel('Epoch')
plt.ylabel('Performance')
plt.savefig(f"{outputs_folder}/Model_{timestamp}_train_test_loss_by_epoch.png",dpi=300)
plt.clf()

#aggreaget the predrop by epoch scores to be summed across epochs, not batches
# Initialize an aggregation dictionary
aggregated_tensors = {}

# Loop and aggregate
for d in predrop_by_epoch_list:
    for key, tensor in d.items():
        if key in aggregated_tensors:
            # Add to the existing tensor
            aggregated_tensors[key] += tensor
        else:
            # Create a new entry
            aggregated_tensors[key] = tensor

# Convert the aggregation dictionary to the desired list of dictionaries format
predrop_by_epoch_list = [{k: v} for k, v in aggregated_tensors.items()]



#create a predropbyepoch dataframe from list
#start by swapping the list of dictionaries to a dictionary of lists
# Convert list of dictionaries into dictionary of lists
dict_of_lists = {}
for d in predrop_by_epoch_list:
    for key, value in d.items():
        if key not in dict_of_lists:
            dict_of_lists[key] = []
        dict_of_lists[key].extend(value)

predrop_by_epoch = dict_of_lists
predrop_by_epoch_df = pd.DataFrame(predrop_by_epoch)
#save to csv
predrop_by_epoch_filename = f"{outputs_folder}/Model_{timestamp}_predrop_scores_by_epoch.csv"
predrop_by_epoch_df.to_csv(predrop_by_epoch_filename, index=False)

#plot the predrop by epoch
predrop_by_epoch_df.T.plot(figsize=(10, 6))
plt.title('Scores by Epoch')
plt.xlabel('Epoch')
plt.ylabel('Epoch Score Sum')
plt.legend(title='Nodes', loc='upper right', labels=predrop_by_epoch_df.columns)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{outputs_folder}/Model_{timestamp}_scores_by_epoch.png",dpi=300)
plt.clf()


#save model
model_filename = f"{outputs_folder}/Model_{timestamp}.pth"
torch.save(model.state_dict(), model_filename)
fin_string = f"finished. total run time: {time.time() - script_start_time} | everything saved but notes. Now saving notes..."
print(fin_string)
model_notes += f"finished. total run time: {time.time() - script_start_time} | everything saved but notes. Now saving notes..."

print('model run finished, printing model notes')
print(f'{model_notes}')

pass