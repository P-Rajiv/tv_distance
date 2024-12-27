import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import softmax
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import wandb
import metrics as metrics
from timeit import default_timer as timer
import argparse
import timm
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import logging

def create_logger(name):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform

    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[1:-1]
                imageLabel = [int(i) for i in imageLabel]
                # if sum(imageLabel) > 0:
                #     imageLabel.append(0)
                # else:
                #     imageLabel.append(1)    
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)  
            
        
            
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        
        if self.transform != None: imageData = self.transform(imageData)
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #-------------------------------------------------------------------------------- 
    

#Hyperparameters
# lr = 2e-4

threshold = 0.5


def save_model(models, optimizer, scheduler, epoch, args, folder="saved_models/", name="best"):

    if not os.path.exists(folder):
        os.makedirs(folder)

    state = {'epoch': epoch + 1,
             'model_rep': models.state_dict(),
             'optimizer_state': optimizer.state_dict(),
             'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
             'args': vars(args)}

    run_name = "debug" if args.debug else wandb.run.name
    torch.save(state, f"{folder}{args.label}_{run_name}_{name}_model.pkl")


def train(args, debug = False):
    """Training Function"""
    device = args.device
    NIH_CLASS_CNT = args.class_cnt
    IMG_SIZE = 512
    tasks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]     
    
    logger = create_logger('Main')

    # create a timm model from timm
    model = timm.create_model('inception_v3', pretrained=False, num_classes = NIH_CLASS_CNT)


    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.7, patience=5)
    scheduler = ReduceLROnPlateau(optimizer, patience=4, verbose=True, factor=0.2)
    criterion = nn.BCEWithLogitsLoss()

    if not debug:
        wandb.init(project="nih_benchmarks", group=args.label, config=args, reinit=True)
        dropout_str = "" if not args.dropout else "-dropout"
        wandb.run.name = f"{args.model}{dropout_str}-lr:{args.lr}-wd:{args.weight_decay}_" + wandb.run.name

    metric, aggregators, model_saver = metrics.get_metrics(args.dataset, tasks)
    # accum_iter = 64

    # if pretrained:
    #     config = torch.load(model_path)
    #     model.load_state_dict(config['state_dict'])
    #     optimizer.load_state_dict(config['optimizer_state_dict'])


    nih_classes = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'Normal']


    nih_path_train = '/data6/rajivporana_scratch/nih_data/nih_train_data/training_images/train'
    nih_path_val = '/data6/rajivporana_scratch/nih_data/nih_train_data/val/images'
    nih_path_test = '/data6/rajivporana_scratch/nih_data/test_images/nih_test/images'

    # m4_pathFileTrain = "../datasets/train_1_reordered_balanced_exp1_loss1.txt"
    nih_pathFileTrain = '/data5/home/rajivporana/nih_data/train_only.txt' # args.train_txt
    nih_pathFileVal = '/data5/home/rajivporana/nih_data/val_only.txt' # args.val_txt
    nih_pathFileTest = '/data5/home/rajivporana/nih_data/nih_test.txt' # args.test_txt


    model_storage = "saved_models/"
    
    nih_trBatchSize = args.batch_size
    nih_valBatchSize = args.batch_size

    if debug:
        print("M4: NIH dataset: Loading from dir: ", nih_path_train,
                " using training txt: ", nih_pathFileTrain,
                " and validation txt: ", nih_pathFileVal)
    m4_train_data = DatasetGenerator(pathImageDirectory = nih_path_train,
                                        pathDatasetFile = nih_pathFileTrain,
                                        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
    m4_train_data_2 = DatasetGenerator(pathImageDirectory = nih_path_val,
                                        pathDatasetFile = nih_pathFileVal,
                                        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))

    m4_train_data = ConcatDataset([m4_train_data, m4_train_data_2])

    if debug:
        print("M4: NIH dataset: Initializing for validation")
    m4_valid_data = DatasetGenerator(pathImageDirectory = nih_path_test,
                                        pathDatasetFile = nih_pathFileTest,
                                        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=(IMG_SIZE,IMG_SIZE)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]))
    
    if debug:
        print("M4: NIH dataset: Dataloader for training")
    nih_dataLoaderTrain = DataLoader(dataset = m4_train_data,
                                    batch_size = nih_trBatchSize,
                                    shuffle = True,
                                    num_workers = 8,
                                    pin_memory = True,
                                    drop_last=True)
    if debug:
        print("M4: NIH dataset: Dataloader for validation")
    nih_dataLoaderVal = DataLoader(dataset = m4_valid_data,
                                    batch_size = nih_valBatchSize,
                                    shuffle = True,
                                    num_workers = 8,
                                    pin_memory = True,
                                    drop_last=True)

    if debug:
        print("m4 data loaded")       
    
    best_result = {k: -float("inf") for k in model_saver}  # Something to maximize.

    
    iter_epochs = tqdm(range(args.num_epochs))
    for epoch_num in iter_epochs:
        train_losses = 0
        valid_losses = 0
        train_correct = 0
        valid_correct = 0
        # training mode
        model.train()
        for batch_no, (images, labels) in enumerate(tqdm(nih_dataLoaderTrain)):
            start = timer()
            images = images.to(device)
            labels = labels.to(device)
            # zeroing the optimizer
            optimizer.zero_grad()
            
            outputs = model(images)
            prediction = (outputs >= threshold).to(torch.float32)
        
            loss = criterion(outputs, labels)
            train_losses += loss.item()
            # calculating the gradients
            loss.backward()
            optimizer.step()

            if debug:
                print("Check for valid gradients")
                for param in model.parameters():
                    print(param.grad)
            #if ((batch_no + 1) % accum_iter == 0) or (batch_no + 1 == len(train_dataloader)):
               # scheduler.step()
               # scheduler.zero_grad()
            #
            #prediction = prediction.squeeze(axis = 1)
            # labels = labels.squeeze(axis = 1)
            # Correct predictions
            train_correct += (prediction == labels).sum()

        train_accuracy = train_correct.item() / len(nih_dataLoaderTrain) 
        train_loss = train_losses / len(nih_dataLoaderTrain)

        model.eval()
        for batch_no, (images, labels) in enumerate(tqdm(nih_dataLoaderVal)):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            prediction = (outputs >= threshold).to(torch.float32)
            loss = criterion(outputs, labels)
            
            valid_losses += loss.item()
            # Correct predictions
            # prediction = prediction.squeeze(axis = 1)
            # labels = labels.squeeze(axis = 1)    
            valid_correct += (prediction == labels).sum()
            
            for t in tasks:
                metric[t].update(outputs[:,t], labels[:,t])

        valid_accuracy = valid_correct.item() / len(nih_dataLoaderVal)
        valid_loss = valid_losses / len(nih_dataLoaderVal)

        
        iter_epochs.set_description(desc = 'Train Loss {} Validation : Loss {}, Accuracy {}'.format(train_loss, valid_loss, valid_accuracy))
        
        scheduler.step(valid_loss)
       
        epoch_stats = {}
        epoch_stats['train_loss'] = train_loss
        epoch_stats['validation_loss'] = valid_loss
        # Print the stored (averaged across batches) validation losses and metrics, per task.
        clog = "epochs {}/{}:".format(epoch_num, args.num_epochs)
        metric_results = {}
        for t in tasks:
            metric_results[t] = metric[t].get_result()
            metric[t].reset()
            for metric_key in metric_results[t]:
                clog += ' val metric-{} {} = {:5.4f}'.format(metric_key, t, metric_results[t][metric_key])
            clog += " |||"

        # Store aggregator metrics (e.g., avg) as well
        for agg_key in aggregators:
            clog += ' val metric-{} = {:5.4f}'.format(agg_key, aggregators["avg"](metric_results))

        logger.info(clog)
        for i, t in enumerate(tasks):
            for metric_key in metric_results[t]:
                epoch_stats[f"val_metric_{metric_key}_{t}"] = metric_results[t][metric_key]

        # Store aggregator metrics (e.g., avg) as well
        for agg_key in aggregators:
            epoch_stats[f"val_metric_{agg_key}"] = aggregators[agg_key](metric_results)
        if not args.debug:
            wandb.log(epoch_stats, step=epoch_num)
        

        # Any time one of the model_saver metrics is improved upon, store a corresponding model.
        c_saver_metric = {k: model_saver[k](metric_results) for k in model_saver}
        for k in c_saver_metric:
            if c_saver_metric[k] >= best_result[k]:
                best_result[k] = c_saver_metric[k]
                # Evaluate the model on the test set and store relative results.
                # test_evaluator(args, test_loader, tasks, DEVICE, model, loss_fn, metric, aggregators, logger, k, epoch)
                if args.store_models:
                    # Save (overwriting) any model that improves the average metric
                    save_model(model, optimizer, scheduler, epoch_num, args,
                               folder=model_storage, name=k)

        end = timer()
        print('Epoch ended in {}s'.format(end - start))

    # Save training/validation results.
    if args.store_models and (not args.time_measurement_exp):
        # Save last model.
        save_model(model, optimizer, scheduler, epoch_num, args, folder=model_storage,
                   name="last")    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnext', help='model to train')
    parser.add_argument('--label', type=str, default='', help='wandb group')
    parser.add_argument('--dataset', type=str, default='nih', help='which dataset to use',
                        choices=['nih'])
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--p', type=float, default=0.1, help='Task dropout probability')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=25, help='Epochs to train for.')
    
    parser.add_argument('--debug', action='store_true', help='Debug mode: disables wandb.')
    parser.add_argument('--store_models', action='store_true', help='Whether to store  models at fixed frequency.')
    parser.add_argument('--decay_lr', action='store_true', help='Whether to decay the lr with the epochs.')
    parser.add_argument('--dropout', action='store_true', help='Whether to use additional dropout in training.')
    parser.add_argument('--no_dropout', action='store_true', help='Whether to not use dropout at all.')
    parser.add_argument('--store_convergence_stats', action='store_true',
                        help='Whether to store the squared norm of the unitary scalarization gradient at that point')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2 regularization.')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of experiment repetitions.')
    parser.add_argument('--random_seed', type=int, default=1, help='Start random seed to employ for the run.')
    
    parser.add_argument('--baseline_losses_weights', type=int, nargs='+',
                        help='Weights to use for losses. Be sure that the ordering is correct! (ordering defined as in config for losses.')
    parser.add_argument('--time_measurement_exp', action='store_true',
                        help="whether to only measure time (does not log training/validation losses/metrics)")
    parser.add_argument('--class_cnt', type=int, help='Number of classes in the dataset')
    parser.add_argument('--device', type = str, help = 'device to use cpu/gpu')
    parser.add_argument('--train_txt', type = str, help = 'train txt file')
    parser.add_argument('--val_txt', type = str, help = 'val txt file')
    parser.add_argument('--test_txt', type = str, help = 'test txt file')
    
    args = parser.parse_args()
    

    train(args, args.debug)
