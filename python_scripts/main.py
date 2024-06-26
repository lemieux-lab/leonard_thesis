#!/usr/bin/env python
import torch
from tqdm import tqdm
import pdb
import numpy as np
from torch.autograd import Variable
import os
import argparse
import datasets
import models
import pickle
import time
import monitoring
import pdb
import h5py 

## build data in .npy format
# datafile = h5py.File("GDC_processed/TCGA_TPM_lab.h5", "r")
# tcga_data = datafile["data"][:,:].T
# labs = np.array(datafile["labels"][:], dtype = str)
# patients =  np.array(datafile["samples"][:], dtype = str)
# genes =  np.array(datafile["genes"][:], dtype = str)
# biotypes = np.array(datafile["biotypes"][:], dtype = str)
# datafile.close()
# np.save("data/tgca_data.npy", tcga_data)
#
def build_parser():
    parser = argparse.ArgumentParser(description="")

    ### Hyperparameter options
    parser.add_argument('--epoch', default=10, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=260389, type=int, help='Seed for random initialization and stuff.')
    parser.add_argument('--batch-size', default=40000, type=int, help="The batch size.")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    

    ### Dataset specific options
    parser.add_argument('--data-dir', default='./data/', help='The folder contaning the dataset.')
    parser.add_argument('--data-file', default='.', help='The data file with the dataset.')
    parser.add_argument('--dataset', choices=['gene', 'domaingene', 'impute', 'fedomains', 'doubleoutput'], default='gene', help='Which dataset to use.')
    parser.add_argument('--mask', type=int, default=0, help="percentage of masked values")
    parser.add_argument('--missing', type=int, default=0, help="number of held out combinations for FE domains")
    parser.add_argument('--data-domain', default='.', help='Number of domains in the data for triple factemb')
    parser.add_argument('--transform', default=True,help='log10(exp+1)')
    
    # Model specific options
    parser.add_argument('--layers-size', default=[250, 75, 50, 25, 10], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--emb_size', default=2, type=int, help='The size of the embeddings.')
    parser.add_argument('--set-gene-emb', default='.', help='Starting points for gene embeddings.')
    parser.add_argument('--warm_pca', default='.', help='Datafile to use as a PCA warm start for the sample embeddings')

    parser.add_argument('--weight-decay', default=1e-7, type=float, help='The size of the embeddings.')
    parser.add_argument('--model', choices=['factor', 'triple', 'multiple','doubleoutput', 'choybenchmark'], default='factor', help='Which model to use.')
    parser.add_argument('--cpu', action='store_true', help='If we want to run on cpu.') # TODO: should probably be cpu instead.
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")
    parser.add_argument('--gpu-selection', type=int, default=0, help="selectgpu")


    # Monitoring options
    parser.add_argument('--save-error', action='store_true', help='If we want to save the error for each tissue and each gene at every epoch.')
    parser.add_argument('--make-grid', default=True, type=bool,  help='If we want to generate fake patients on a meshgrid accross the patient embedding space')
    parser.add_argument('--nb-gridpoints', default=50, type=int, help='Number of points on each side of the meshgrid')
    parser.add_argument('--load-folder', help='The folder where to load and restart the training.')
    parser.add_argument('--save-dir', default='./testing123/', help='The folder where everything will be saved.')

    return parser

def parse_args(argv):

    if type(argv) == list or argv is None:
        opt = build_parser().parse_args(argv)
    else:
        opt = argv

    return opt

def main(argv=None):

    opt = parse_args(argv)
    # TODO: set the seed
    seed = opt.seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    exp_dir = opt.load_folder
    if exp_dir is None: # we create a new folder if we don't load.
        exp_dir = monitoring.create_experiment_folder(opt)

    # creating the dataset
    print ("Getting the dataset...")
    dataset = datasets.get_dataset(opt,exp_dir)
    
    # Creating a model
    print ("Getting the model...")

    my_model, optimizer, epoch, opt = monitoring.load_checkpoint(exp_dir, opt, dataset.dataset.input_size(), dataset.dataset.additional_info())

    # Training optimizer and stuff
    criterion = torch.nn.MSELoss()

    if not opt.cpu:
        print ("Putting the model on gpu...")
        my_model.cuda(opt.gpu_selection)

    # The training.
    print ("Start training.")
    
    #monitoring and predictions
    predictions =np.zeros((dataset.dataset.nb_patient,dataset.dataset.nb_gene))
    indices_patients = np.arange(dataset.dataset.nb_patient)
    indices_genes = np.arange(dataset.dataset.nb_gene)
    xdata = np.transpose([np.tile(indices_genes, len(indices_patients)),
                          np.repeat(indices_patients, len(indices_genes))])
    progress_bar_modulo = len(dataset)/100




    monitoring_dic = {}
    monitoring_dic['train_loss'] = []
    computing_times = []
    for t in range(epoch, opt.epoch):

        start_timer = time.time()

        thisepoch_trainloss = []

        with tqdm(dataset, unit="batch") as tepoch:
            for mini in tepoch:
                tepoch.set_description(f"Epoch {t}")


                inputs, targets = mini[0], mini[1]

                inputs = Variable(inputs, requires_grad=False).float()
                targets = Variable(targets, requires_grad=False).float()

                if not opt.cpu:
                    inputs = inputs.cuda(opt.gpu_selection)
                    targets = targets.cuda(opt.gpu_selection)

                # Forward pass: Compute predicted y by passing x to the model
                y_pred = my_model(inputs).float()
                y_pred = y_pred.squeeze()

                targets = torch.reshape(targets,(targets.shape[0],))
                # Compute and print loss

                loss = criterion(y_pred, targets)
                to_list = loss.cpu().data.numpy().reshape((1, ))[0]
                thisepoch_trainloss.append(to_list)
                tepoch.set_postfix(loss=loss.item())

                # np.save(os.path.join(exp_dir, 'pixel_epoch_{}'.format(t)),my_model.emb_1.weight.cpu().data.numpy() )
                # np.save(os.path.join(exp_dir,'digit_epoch_{}'.format(t)),my_model.emb_2.weight.cpu().data.numpy())

                # Zero gradients, perform a backward pass, and update the weights.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                elapsed_time = time.time() - start_timer
                computing_times.append(np.array([t, elapsed_time, float(loss)]))
                np.save(os.path.join(exp_dir, "computing_times.npy"), np.array(computing_times))

        monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)
        monitoring_dic['train_loss'].append(np.mean(thisepoch_trainloss))
        np.save(f'{exp_dir}/train_loss.npy',monitoring_dic['train_loss'])

if __name__ == '__main__':
    main()