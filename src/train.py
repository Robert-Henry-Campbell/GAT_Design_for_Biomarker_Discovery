import time
import datetime
import pandas as pd
import torch
import torchmetrics
import matplotlib.pyplot as plt
from config import CONFIG

num_epochs = CONFIG.num_epochs


def train(model, loader, criterion, optimizer, device):
    """Train ``model`` for one epoch using ``loader``."""
    model.train()

    for data in loader:
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)
        data.y = data.y.to(device)

        out, _ = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

predrop_by_epoch_list = []


def test(model, loader, epoch, num_epochs, device, num_nodes, pre_drop_score_sum, predrop_by_epoch_list, batch_size):
    """Evaluate ``model`` using ``loader`` and update score collections."""
    model.eval()

    metric_precision = torchmetrics.Precision(task='binary', average='macro').to(device)
    metric_f1 = torchmetrics.F1Score(task='binary', average='macro', num_classes=2).to(device)

    correct = 0
    total = 0

    for data in loader:
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.batch = data.batch.to(device)
        data.y = data.y.to(device)

        out, pre_drop_score = model(data.x, data.edge_index, data.batch)

        if epoch == num_epochs:
            pre_drop_score_sum = pre_drop_score_sum.to(device)
            batched_pre_drop_scores = pre_drop_score.detach()
            num_graphs_in_batch = batched_pre_drop_scores.size()[0] / num_nodes
            assert (
                batched_pre_drop_scores.size()[0] / num_nodes
            ).is_integer(), (
                f"num nodes in batch = {num_graphs_in_batch}, not evenly divisble by {num_nodes}"
            )
            num_graphs_in_batch = int(num_graphs_in_batch)
            unbatched_pre_drop_scores = batched_pre_drop_scores.view(num_graphs_in_batch, num_nodes)
            split_pre_drop_scores = torch.split(unbatched_pre_drop_scores, split_size_or_sections=1, dim=0)

            for i, graph_pre_drop_score in enumerate(split_pre_drop_scores):
                assert graph_pre_drop_score.shape == (
                    1,
                    num_nodes,
                ), f"graph_pre_drop_score {i} has incorrect shape {graph_pre_drop_score.shape}"
                pre_drop_score_sum += graph_pre_drop_score.squeeze()

            assert len(split_pre_drop_scores) == num_graphs_in_batch, (
                f"Not all values from the original tensor were used, expected {batch_size} tensors, got {len(split_pre_drop_scores)}"
            )
        else:
            single_batch_scoresum = torch.zeros(num_nodes).to(device)
            batched_pre_drop_scores = pre_drop_score.detach()
            num_graphs_in_batch = batched_pre_drop_scores.size()[0] / num_nodes
            assert (
                batched_pre_drop_scores.size()[0] / num_nodes
            ).is_integer(), (
                f"num nodes in batch = {num_graphs_in_batch}, not evenly divisble by {num_nodes}"
            )
            single_batch_scoresum = torch.sum(
                batched_pre_drop_scores.view(int(num_graphs_in_batch), num_nodes), dim=0
            )
            predrop_by_epoch_list.append({epoch: single_batch_scoresum.cpu().numpy()})

        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

        metric_precision.update(pred, data.y)
        metric_f1.update(pred, data.y)

    accuracy = correct / total
    precision = metric_precision.compute()
    f1 = metric_f1.compute()
    return accuracy, precision, f1


def run_training():
    print('train and test defined')
    print('starting the epochs...')
    epoch_start_time = time.time()
    metrics_list = []
    num_nodes = dataset[0].num_nodes
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, criterion, optimizer, device)
        train_acc, train_prec, train_f1 = test(
            model,
            train_loader,
            epoch,
            num_epochs,
            device,
            num_nodes,
            pre_drop_score_sum,
            predrop_by_epoch_list,
            batch_size,
        )
        test_acc, test_prec, test_f1 = test(
            model,
            test_loader,
            epoch,
            num_epochs,
            device,
            num_nodes,
            pre_drop_score_sum,
            predrop_by_epoch_list,
            batch_size,
        )
        metrics = (
            f" \n Epoch: {epoch:03d} | epoch_time: {round(time.time()-epoch_start_time,1)} | "
            f"Train Acc: {train_acc:.4f} | Train Prec: {train_prec:.4f} | Train F1: {train_f1:.4f} | "
            f"Test Acc: {test_acc:.4f} | Test Prec: {test_prec:.4f} | Test F1: {test_f1:.4f}"
        )
        print(metrics)
        model_notes += metrics

        metrics_list.append(
            {
                'Epoch': epoch,
                'Train_Acc': train_acc,
                'Train_Prec': train_prec.item(),
                'Train_F1': train_f1.item(),
                'Test_Acc': test_acc,
                'Test_Prec': test_prec.item(),
                'Test_F1': test_f1.item(),
            }
        )

        epoch_start_time = time.time()

    print('training complete, moving to saving results')

    def get_timestamp():
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        return timestamp

    timestamp = get_timestamp()
    model_notes += f"timestamp : {timestamp}"

    dense_node_call_rate = node_call_rate.to_dense()

    model_results_df = pd.DataFrame(
        {
            'node_id': range(len(pre_drop_score_sum.cpu())),
            'score': pre_drop_score_sum.cpu().numpy(),
            'call_rate': dense_node_call_rate.cpu().numpy(),
        }
    )

    if synth_data:
        outputs_folder = 'synth_models'
    else:
        outputs_folder = 'models'

    if not synth_data:
        for node_id in model_results_df['node_id']:
            metadata_series = id_to_node_metadata[node_id][1]
            for column, value in metadata_series.items():
                model_results_df.loc[model_results_df['node_id'] == node_id, column] = value
        model_results_df.set_index('node_id', inplace=True)
        model_results_df.to_csv(f"{outputs_folder}/Model_{timestamp}_results.csv", index=False)
    else:
        model_results_df.to_csv(f"{outputs_folder}/Model_{timestamp}_results.csv")

    call_rate_filename = f"{outputs_folder}/Model_{timestamp}_call_rate.pt"
    torch.save(dense_node_call_rate.cpu(), call_rate_filename)

    with open(f"{outputs_folder}/Model_{timestamp}_notes.txt", 'w') as notes_file:
        notes_file.write(model_notes)

    metrics_by_epoch_df = pd.DataFrame(metrics_list)
    metrics_by_epoch_filename = f"{outputs_folder}/Model_{timestamp}_metrics_by_epoch.csv"
    metrics_by_epoch_df.to_csv(metrics_by_epoch_filename, index=False)

    metrics_by_epoch_df.plot(x='Epoch', y=['Train_Acc', 'Train_Prec', 'Train_F1'], kind='line', title='Training Performance', grid=True)
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.savefig(f"{outputs_folder}/Model_{timestamp}_train_metrics_by_epoch.png", dpi=300)
    plt.clf()

    metrics_by_epoch_df.plot(x='Epoch', y=['Test_Acc', 'Test_Prec', 'Test_F1'], kind='line', title='Test Performance', grid=True)
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.savefig(f"{outputs_folder}/Model_{timestamp}_test_metrics_by_epoch.png", dpi=300)
    plt.clf()

    metrics_by_epoch_df.plot(x='Epoch', y=['Test_F1', 'Train_F1'], kind='line', title='Test vs Train F1', grid=True)
    plt.xlabel('Epoch')
    plt.ylabel('Performance')
    plt.savefig(f"{outputs_folder}/Model_{timestamp}_train_test_loss_by_epoch.png", dpi=300)
    plt.clf()

    aggregated_tensors = {}
    for d in predrop_by_epoch_list:
        for key, tensor in d.items():
            if key in aggregated_tensors:
                aggregated_tensors[key] += tensor
            else:
                aggregated_tensors[key] = tensor
    predrop_by_epoch_list[:] = [{k: v} for k, v in aggregated_tensors.items()]

    dict_of_lists = {}
    for d in predrop_by_epoch_list:
        for key, value in d.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].extend(value)

    predrop_by_epoch = dict_of_lists
    predrop_by_epoch_df = pd.DataFrame(predrop_by_epoch)
    predrop_by_epoch_filename = f"{outputs_folder}/Model_{timestamp}_predrop_scores_by_epoch.csv"
    predrop_by_epoch_df.to_csv(predrop_by_epoch_filename, index=False)

    predrop_by_epoch_df.T.plot(figsize=(10, 6))
    plt.title('Scores by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Epoch Score Sum')
    plt.legend(title='Nodes', loc='upper right', labels=predrop_by_epoch_df.columns)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outputs_folder}/Model_{timestamp}_scores_by_epoch.png", dpi=300)
    plt.clf()

    model_filename = f"{outputs_folder}/Model_{timestamp}.pth"
    torch.save(model.state_dict(), model_filename)
    fin_string = f"finished. total run time: {time.time() - script_start_time} | everything saved but notes. Now saving notes..."
    print(fin_string)
    model_notes += fin_string

    print('model run finished, printing model notes')
    print(f'{model_notes}')
