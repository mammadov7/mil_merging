import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict
from pathlib import Path
import mils
import abmil
from clam import CLAM_SB

import dsmil 


def ties_merge_key(fine_tuned_models, init_model, k=0.5, lam=1.0):
    n = len(fine_tuned_models)
    # Compute task vectors: difference between fine-tuned and initial weights.
    task_vectors = [model - init_model for model in fine_tuned_models]
    # Step 1: Trim redundant parameters.
    trimmed_vectors = []
    for tv in task_vectors:
        flat_tv = tv.view(-1)
        # Compute threshold so that only the top k fraction (by absolute value) is kept.
        threshold = torch.quantile(flat_tv.abs(), 1 - k)
        # Keep values with absolute value >= threshold; set the rest to 0.
        trimmed = torch.where(tv.abs() >= threshold, tv, torch.tensor(0.0, device=tv.device))
        trimmed_vectors.append(trimmed)
    # Stack trimmed vectors: shape (n, ...) same as model weights.
    trimmed_stack = torch.stack(trimmed_vectors, dim=0)
    # Step 2: Elect final sign.
    # Sum the trimmed vectors across models and take the sign.
    sum_trimmed = torch.sum(trimmed_stack, dim=0)
    elected_sign = torch.sign(sum_trimmed)  # Elementwise sign (+1, -1, or 0)
    # Step 3: Disjoint merge.
    # For each parameter, average only the trimmed values with sign matching the elected sign.
    # Create a mask for matching signs.
    sign_matches = torch.stack([torch.eq(torch.sign(tv), elected_sign) for tv in trimmed_vectors], dim=0)
    selected_values = torch.where(sign_matches, trimmed_stack, torch.tensor(0.0, device=trimmed_stack.device))
    # Count the number of models contributing per parameter.
    count = torch.sum(sign_matches.float(), dim=0)
    # Avoid division by zero.
    avg_selected = torch.where(count > 0, torch.sum(selected_values, dim=0) / count, torch.tensor(0.0, device=count.device))
    merged_task_vector = avg_selected
    # Compute the merged model by adding the scaled merged task vector to the initial weights.
    merged_model = init_model + lam * merged_task_vector
    return merged_model

def ties_merge(state_dicts, init_dict):
    merged={}
    for key in init_dict.keys():
        merged[key]=ties_merge_key([state_dict[key] for state_dict in state_dicts], init_dict[key])
    return merged

def average_state_dicts(state_dicts, weights=None):
    num_dicts = len(state_dicts)
    if weights is None:
        weights = [1.0 / num_dicts] * num_dicts
    else:
        # Normalize the weights so they sum to 1.
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    # Initialize the averaged state dictionary.
    avg_state_dict = {}
    # Assume all state_dicts have the same keys.
    for key in state_dicts[0].keys():
        # Sum the weighted parameters.
        avg_state_dict[key] = sum(w * sd[key] for w, sd in zip(weights, state_dicts))
    return avg_state_dict


def activation(args,bag_prediction,max_prediction):
    if args.num_classes==1:
        bag_res = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
        max_res = torch.sigmoid(max_prediction).squeeze().cpu().numpy()
    else:
        bag_res = torch.softmax(bag_prediction.squeeze(),dim=0).squeeze().cpu().numpy()
        max_res = torch.softmax(max_prediction.squeeze(),dim=0).squeeze().cpu().numpy()
    if args.average:
        return [0.5*max_res+0.5*bag_res]
    else:
        return [bag_res]


def get_bag_feats(csv_file_df, args, random_state=1):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df, random_state=random_state).reset_index(drop=True) # UNCOMMENT AFTER STD EXPERIMENT
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
    return label, feats


# def train(train_df, milnet, criterion, optimizer, args, random_state):
#     milnet.train()
#     total_loss = 0
#     grad_acc = 16
#     Tensor = torch.cuda.FloatTensor
#     optimizer.zero_grad()
#     for i in range(len(train_df)):
#         label, feats = get_bag_feats(train_df.iloc[i], args, random_state)
#         feats = dropout_patches(feats, args.dropout_patch)
#         bag_label = Variable(Tensor(np.array([label])))
#         bag_feats = Variable(Tensor(np.array([feats])))
#         bag_feats = bag_feats.view(-1, args.feats_size)
#         ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
#         max_prediction, _ = torch.max(ins_prediction, 0)  
#         bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
#         max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
#         loss = 0.5*bag_loss + 0.5*max_loss
#         total_loss = total_loss + loss.item()
#         loss = loss / grad_acc
#         loss.backward()
#         if (i + 1) % grad_acc == 0 or (i + 1) == len(train_df):
#             optimizer.step()
#             optimizer.zero_grad()
#     return total_loss / len(train_df)

def train(train_df, milnet, criterion, optimizer, args, random_state):
    milnet.train()
    csvs = shuffle(train_df).reset_index(drop=True)
    total_loss = 0
    bc = 0
    Tensor = torch.cuda.FloatTensor
    for i in range(len(train_df)):
        optimizer.zero_grad()
        label, feats = get_bag_feats(train_df.iloc[i], args, random_state)
        bag_label = Variable(Tensor(np.array([label])))
        bag_feats = Variable(Tensor(np.array([feats])))
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0)   
        # print(max_prediction,bag_prediction)     
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        # sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)


def dropout_patches(feats, p):
    if len(feats) < 1000 or p == 0:
        return feats
    num_rows = len(feats)
    num_rows_to_select = int(num_rows * p)
    random_indices = torch.randperm(num_rows)[:num_rows_to_select]
    selected_rows = feats[random_indices]
    return selected_rows

def test(test_df, milnet, criterion, optimizer, args, random_state):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i in range(len(test_df)):
            label, feats = get_bag_feats(test_df.iloc[i], args, random_state)
            bag_label = Variable(Tensor(np.array([label])))
            bag_feats = Variable(Tensor(np.array([feats])))
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            test_labels.extend([label])
            test_predictions.extend(activation(args, bag_prediction, max_prediction))

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal


def real_test(test_df, milnet, criterion, optimizer, args, thresholds_optimal,random_state):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i in range(len(test_df)):
            label, feats = get_bag_feats(test_df.iloc[i], args, random_state)
            bag_label = Variable(Tensor(np.array([label])))
            bag_feats = Variable(Tensor(np.array([feats])))
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            # sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([label])
            test_predictions.extend(activation(args, bag_prediction, max_prediction))

    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, _ = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    return total_loss / len(test_df), avg_score, auc_value

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=384, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='automil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--q_n', default=128, type=int, help='Number of hiddent parameters')
    parser.add_argument('--init_seed', default=2, type=int, help='Seed for shuffle')
    parser.add_argument('--nb_models', default=10, type=int, help='Seed for init')
    parser.add_argument('--nb_epochs', default=5, type=int, help='Seed for init')
    parser.add_argument('--init_type', default='uniform', type=str, help='uniform, xavier, switch')
    parser.add_argument('--average', type=bool, default=True, help='Average the score of max-pooling and bag aggregating')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    mils, criterions, optimizers, schedulers = [],[],[],[]
    seeds = np.random.randint(1, 1000, size=10)
    for i in range(1,args.nb_models+1):
        torch.manual_seed(args.init_seed)
        torch.cuda.manual_seed(args.init_seed)
        torch.cuda.manual_seed_all(args.init_seed)  
        i_classifier = dsmil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = dsmil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity,q_n=args.q_n).cuda()
        milnet = dsmil.MILNet(i_classifier, b_classifier).cuda()
        mils.append(milnet)
        criterions.append(nn.BCEWithLogitsLoss())
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        optimizers.append(optimizer)
        schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005))
    init_dict = milnet.state_dict()

    # milnet = get_model(args).cuda()
    
    if args.num_classes > 1:
        if args.dataset[:5] == 'bracs':
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(np.array([0.64860427, 2.53205128, 0.94047619]),dtype=torch.float).cuda())
        else:
            criterion = nn.CrossEntropyLoss()
    else :
        criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    print(args.dataset,args.q_n,args.non_linearity)
    print(args.lr)
    print(args.weight_decay)

    
    train_csv, val_csv = [], []
    if args.dataset == 'TCGA-lung-default':
        bags_csv = 'datasets/tcga-dataset/TCGA.csv'
    else:
        bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')
        train_csv = os.path.join('datasets', args.dataset, 'train.csv')
        val_csv = os.path.join('datasets', args.dataset, 'val.csv')
    
    bags_path = pd.read_csv(bags_csv)
    train_csv = pd.read_csv(train_csv)
    val_csv = pd.read_csv(val_csv)

    train_csv = [ Path(x).name for x in train_csv['path'].tolist() ]
    val_csv = [ Path(x).name for x in val_csv['path'].tolist() ]
    
    train_path = bags_path[bags_path.apply(lambda x: Path(x['path']).name in train_csv, axis=1)]
    val_path = bags_path[bags_path.apply(lambda x: Path(x['path']).name in val_csv, axis=1)]
    try:
        test_path = pd.read_csv(os.path.join('datasets', args.dataset, 'test.csv'))
    except:
        test_path=[]

    best_score = 0

    save_path = os.path.join('weights/'+args.model, datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))
    print(save_path)
    os.makedirs(save_path, exist_ok=True)
    run = len(glob.glob(os.path.join(save_path, '*.pth')))
    best_mil=[]
    patience = 0
    thresholds_optimal=0.0
    mil_ids=np.arange(args.nb_models)
    early_aucs=[]
    M, N = len(mil_ids), 5


    for i in mil_ids:
        random_state = np.random.RandomState(seeds[i])
        scores=[] 
        for epoch in range(1, N+1):
            train_path = shuffle(train_path, random_state=random_state).reset_index(drop=True)
            val_path = shuffle(val_path, random_state=random_state).reset_index(drop=True)
            train_loss_bag = train(train_path, mils[i], criterions[i], optimizers[i], args,random_state) # iterate all bags
            test_loss_bag, avg_score, aucs, thresholds_optimal = test(val_path, mils[i], criterions[i], optimizers[i], args,random_state)
            print('Model: '+str(i))
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                      (epoch, N, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
            schedulers[i].step()
            # current_score = test_loss_bag
            scores.append(sum(aucs))
        early_aucs.append(scores)

    # early_aucs = np.array(early_aucs)[:,4]
    early_aucs = np.array(early_aucs)
    ema_early_auc = np.average(early_aucs, weights=[0, 0, 0, 1, 1], axis=1)
    bests=np.argsort(ema_early_auc)[-3:]
    best=np.argsort(ema_early_auc)[-1]
    print('############################ BEST MODEL SELECTION #####################################')
    print(early_aucs) 
    print(bests)
    print('#######################################################################################')

    dict1 = ties_merge([mils[bests[0]].state_dict(),
                        mils[bests[1]].state_dict(), 
                        mils[bests[2]].state_dict()], 
                        init_dict)

    del mils
    del criterions
    del optimizers
    del schedulers
        
    random_state = np.random.RandomState(seeds[best])
    i_classifier = dsmil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
    b_classifier = dsmil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity,q_n=args.q_n).cuda()
    milnet = dsmil.MILNet(i_classifier, b_classifier).cuda()
    milnet.load_state_dict(dict1, strict=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)

    for epoch in range(1, args.num_epochs):
        patience+=1
        train_path = shuffle(train_path, random_state=random_state).reset_index(drop=True)
        val_path = shuffle(val_path, random_state=random_state).reset_index(drop=True)
        train_loss_bag = train(train_path, milnet, criterion, optimizer, args,random_state) # iterate all bags
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(val_path, milnet, criterion, optimizer, args,random_state)
        if args.dataset=='TCGA-lung':
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
        else:
            print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, AUC: ' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        scheduler.step()
        # current_score = test_loss_bag
        current_score = sum(aucs)
        if current_score > best_score:
            
            best_score = current_score
            save_name = os.path.join(save_path, str(run+1)+'.pth')
            torch.save(milnet.state_dict(), save_name)
            if args.dataset=='TCGA-lung':
                print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
            else:
                print('Best model saved at: ' + save_name)
                print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
    
            if len(test_path) == 0:
                continue
            print('\nBest Test Set with val threshold')
            test_loss_bag, avg_score, aucs = real_test(test_path, milnet, criterion, optimizer, args, thresholds_optimal,random_state)
            print('Test loss: %.4f, Test average score: %.4f, Test AUC: ' % (test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
            patience=0
            
            # print('\nBest Test Set with test threshold')
            # test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_path, milnet, criterion, optimizer, args)
            # print('Test loss: %.4f, Test average score: %.4f, Test AUC: ' % (test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
        # if patience == 20:
        #     break
    test_loss_bag, avg_score, aucs = real_test(test_path, milnet, criterion, optimizer, args, thresholds_optimal,random_state)
    print('Last model Test')
    print('Test loss: %.4f, Test average score: %.4f, Test AUC: ' % (test_loss_bag, avg_score) + '|'.join('class-{}>>{}'.format(*k) for k in enumerate(aucs))) 
    save_name = os.path.join(save_path, 'last.pth')
    torch.save(milnet.state_dict(), save_name)

if __name__ == '__main__':
    main()