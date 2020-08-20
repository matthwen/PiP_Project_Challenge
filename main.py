# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import torch.utils.data as tud
import datasets
from utils import plot
from architectures import SimpleCNN
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import tqdm
import utils
from time import strftime, localtime
import shutil
import dill as pickle


def main(data_path, results_path="Results", network_config: dict = None, learningrate: int = 1e-3,
         weight_decay: float = 1e-5,
         n_updates: int = 100, device: torch.device = torch.device("cuda:0"), testrun=False,
         memory_debug=False, batch_size=1,
         num_workers=1):
    dt_suffix = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    if results_path is not None:
        results_path = os.path.join(results_path, dt_suffix)
        plotpath = os.path.join(results_path, 'plots')
        os.makedirs(plotpath, exist_ok=True)
        # copy config
        # shutil.copy2(os.path.dirname(sys.argv[0]) + "/working_config.json", results_path)
        shutil.copy2("working_config.json", results_path)
    data = datasets.ChallengeImagesReducedBatched(data_folder=data_path)
    data_len = len(data)

    train_set = tud.Subset(data, indices=np.arange(int(data_len * (3 / 5))))
    val_set = tud.Subset(data, indices=np.arange(int(data_len * (3 / 5)), int(data_len * (4 / 5))))
    test_set = tud.Subset(data, indices=np.arange(int(data_len * (4 / 5)), data_len))

    collate_fn = datasets.collate_fn if batch_size > 1 else None
    train_loader = tud.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  collate_fn=collate_fn)
    val_loader = tud.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                collate_fn=collate_fn)
    test_loader = tud.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 collate_fn=collate_fn)
    if network_config is not None:
        net = SimpleCNN(**network_config)
    else:
        net = SimpleCNN()
    net.to(device)

    mse = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=learningrate, weight_decay=weight_decay)

    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))

    print_stats_at = 1e2
    plot_at = 1e4
    validate_at = 5e3
    update = 0
    best_validation_loss = np.inf
    update_progess_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar

    # Save initial model as "best" model (will be overwritten later)
    torch.save(net, os.path.join(results_path, 'best_model.pt'))

    while update < n_updates:
        for data in train_loader:
            inputs, crop_array, targets, means, stds = data
            inputs = inputs.to(device)
            targets = targets.to(device).float()

            optimizer.zero_grad()
            inputs = inputs.unsqueeze(1).float()

            outputs = net(inputs)
            outputs = outputs.squeeze(1)
            target_masks = crop_array.to(dtype=torch.bool)

            # loss = mse(predictions, targets.reshape((-1,)))

            if batch_size == 1:
                prediction = outputs[0, target_masks[0]]
                loss = mse(prediction, targets[0].reshape((-1,)))
            else:
                predictions = [outputs[i, target_masks[i]] for i in range(len(outputs))]
                losses = torch.stack(
                    [mse(prediction, target.reshape((-1,))) for prediction, target in zip(predictions, targets)])
                loss = losses.mean()

            loss.backward()
            optimizer.step()

            if update % print_stats_at == 0 and update > 0:
                writer.add_scalar(tag="training/loss",
                                  scalar_value=loss.cpu(),
                                  global_step=update)

            # Plot output
            if update % plot_at == 0:
                plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                     plotpath, update)

            # Evaluate model on validation set
            if update % validate_at == 0 and update > 0:
                val_loss = evaluate_model(net, dataloader=val_loader, device=device, testrun=testrun)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss.cpu(), global_step=update)
                # Add weights as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(),
                                         global_step=update)
                # Add gradients as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/gradients_{i}',
                                         values=param.grad.cpu(),
                                         global_step=update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    best_validation_loss = val_loss
                    torch.save(net, os.path.join(results_path, 'best_model.pt'))

            update_progess_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progess_bar.update()

            if memory_debug is True:
                utils.debug_memory()

            # Increment update counter, exit if maximum number of updates is reached
            update += 1
            if update >= n_updates:
                break

            torch.cuda.empty_cache()

    update_progess_bar.close()
    print('Finished Training!')

    print(f"Computing scores for best model")
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    test_loss = evaluate_model(net, dataloader=test_loader, device=device, testrun=testrun)
    val_loss = evaluate_model(net, dataloader=val_loader, device=device, testrun=testrun)
    train_loss = evaluate_model(net, dataloader=train_loader, device=device, testrun=testrun)

    print(f"Scores:")
    print(f"test loss: {test_loss}")
    print(f"validation loss: {val_loss}")
    print(f"training loss: {train_loss}")

    # Write result to file
    with open(os.path.join(results_path, 'results.txt'), 'w') as fh:
        print(f"Scores:", file=fh)
        print(f"test loss: {test_loss}", file=fh)
        print(f"validation loss: {val_loss}", file=fh)
        print(f"training loss: {train_loss}", file=fh)

    ################################


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, testrun=True,
                   denormalize=False):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`"""
    mse = torch.nn.MSELoss()
    loss = torch.tensor(0., device=device)
    with torch.no_grad():
        count = 0
        for data in tqdm.tqdm(dataloader, desc="scoring", position=0):
            count += 1
            if testrun is True and count > 10:
                break
            inputs, crop_array, targets, means, stds = data
            predictions, _ = predict(device, model, denormalize, inputs, crop_array, targets, means, stds)

            loss += (torch.stack(
                [mse(prediction, target.reshape((-1,))) for prediction, target in zip(predictions, targets)]).sum()
                     / len(dataloader.dataset))
    return loss


def predict(device, model, denormalize, inputs, crop_array, targets, means, stds):
    inputs = inputs.to(device)
    targets = targets.to(device)
    inputs = inputs.unsqueeze(1).float()
    outputs = model(inputs)

    outputs = outputs.squeeze(1)
    target_masks = crop_array.to(dtype=torch.bool)
    predictions = [outputs[i, target_masks[i]] for i in range(len(outputs))]
    return predictions, outputs


def score(target_file: str, model_path: str, prediction_file: str):
    score_predict(target_file, model_path, prediction_file)
    #scoring broken, also pointless since there are no targets
    #loss = score_pickles(prediction_file, target_file)
    #print(loss)
    return


def score_predict(target_file: str, model_path: str, prediction_file: str):
    with open(target_file, 'rb') as tfh:
        scoring_data = pickle.load(tfh)
    outputs = []
    model = torch.load(model_path)
    scoring_set = datasets.ChallengeImagesScoring(scoring_data)

    # collate_fn = datasets.collate_fn
    score_loader = tud.DataLoader(scoring_set, batch_size=1, shuffle=False, num_workers=1)

    for data in tqdm.tqdm(score_loader, desc="scoring", position=0):
        inputs, crop_array, targets, means, stds, size, center = data
        prediction, output = predict(torch.device("cuda:0"), model, True, inputs, crop_array, targets, means, stds)
        output = output.cpu()
        output = output * stds + means
        output = output[0].to(dtype=torch.uint8).cpu().numpy()

        outputs_cropped = utils.crop(output, size, center)

        outputs.append(outputs_cropped[2])
    with open(prediction_file, 'wb') as f:
        pickle.dump(outputs, f)
    return


def score_pickles(prediction_file: str, target_file: str):
    with open(prediction_file, 'rb') as pfh:
        predictions = pickle.load(pfh)
    if not isinstance(predictions, list):
        raise TypeError(f"Expected a list of numpy arrays as pickle file. "
                        f"Got {type(predictions)} object in pickle file instead.")

    if not all([isinstance(prediction, np.ndarray) and np.uint8 == prediction.dtype
                for prediction in predictions]):
        raise TypeError("List of predictions contains elements which are not numpy arrays of dtype uint8")

    # Load targets
    with open(target_file, 'rb') as tfh:
        targets = pickle.load(tfh)
    if len(targets) != len(predictions):
        raise IndexError(f"list of targets has {len(targets)} elements "
                         f"but list of submitted predictions has {len(predictions)} elements.")

    # Compute MSE for each sample
    mses = [mse(target, prediction) for target, prediction in zip(targets, predictions)]

    return np.mean(mses)


def mse(target_array, prediction_array):
    if prediction_array.shape != target_array.shape:
        raise IndexError(f"Target shape is {target_array.shape} but prediction shape is {prediction_array.shape}")
    prediction_array, target_array = np.asarray(prediction_array, np.float64), np.asarray(target_array, np.float64)
    return np.mean((prediction_array - target_array) ** 2)


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to config file', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as fh:
        config = json.load(fh)
    if config["scoring"] is True:
        score(config["target_file"], config["model_path"], config["prediction_file"])
    else:
        main(**config)
# python3 -m tensorboard.main --logdir=/home/matthias/git/PiP_Project_Challenge/results/tensorboard
