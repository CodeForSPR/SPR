import argparse
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import warnings
import datetime

from utils import str2bool, load_data, calculate_class_weights
from model import SPR

warnings.filterwarnings("ignore")

# ================================================================================= #

parser = argparse.ArgumentParser("Implementation for SPR framework")
# input configuration
parser.add_argument('--cuda_available', type=bool, default=True)
parser.add_argument('--cuda_device', type=int, default=0)  # -1 for cpu;
parser.add_argument('--net_train', type=str, required=True)
parser.add_argument('--features_train', type=str, required=True)
parser.add_argument('--net_test', type=str, required=False, default=None)
parser.add_argument('--feature_dim', type=int, default=64)

# training details
parser.add_argument('--epoches', type=int, default=2000)  # bitcoinAlpha, bitcoinOtc: 2000; Slashdot, Epinions: 4000
parser.add_argument('--interval', type=int, default=10)  # bitcoinAlpha, bitcoinOtc: 10; Slashdot, Epinions: 200
parser.add_argument('--batch_size', type=int, default=5000)

# saving paths
parser.add_argument('--modify_input_features', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--model_path', type=str, default="model_{}.pkl".format((datetime.datetime.now()).strftime("%Y%m%d%H%M%S")))
parser.add_argument('--embedding_path', type=str, default="embedding_{}.pkl".format((datetime.datetime.now()).strftime("%Y%m%d%H%M%S")))

# model parameters
parser.add_argument('--seed', type=int, default=2020)  # random.randint(0, 1e10), we fix the temporary seed as 2020 in the released code.
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--regularize', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--nheads', type=int, default=4)
parser.add_argument('--loss_gamma', type=float, default=1.0)
parser.add_argument('--loss_lambda', type=float, default=4.0)

parameters = parser.parse_args()
print(parameters)
args = {}
for arg in vars(parameters):
    args[arg] = getattr(parameters, arg)
# ================================================================================= #


rnd_seed = args['seed']
np.random.seed(rnd_seed)
random.seed(rnd_seed)
torch.manual_seed(rnd_seed)

cuda = args['cuda_available']
if args['cuda_device'] == -1:
    cuda = False
if cuda:
    print("Using {} CUDA!!!".format(args['cuda_device']))
    torch.cuda.set_device(args['cuda_device'])
    torch.cuda.manual_seed(rnd_seed)
else:
    print("Using CPU!!!")

num_nodes, num_edges, train_pos, train_neg, train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir, num_feats, input_features = load_data(args)
args['class_weights'] = calculate_class_weights(num_edges[0], num_edges[1])

features = nn.Embedding(num_nodes, num_feats)
if args['modify_input_features']:
    features.weight = nn.Parameter(torch.FloatTensor(input_features), requires_grad=True)
else:
    features.weight = nn.Parameter(torch.FloatTensor(input_features), requires_grad=False)

if cuda:
    features.cuda()

##################################################################


spr = SPR(num_nodes=num_nodes,
          num_feats=num_feats,
          features=features,
          train_pos=train_pos,
          train_neg=train_neg,
          train_pos_dir=train_pos_dir,
          train_neg_dir=train_neg_dir,
          test_pos_dir=test_pos_dir,
          test_neg_dir=test_neg_dir,
          cuda_available=cuda,
          alpha=args['alpha'],
          dropout=args['dropout'],
          nheads=args['nheads'],
          class_weights=args['class_weights'],
          criterion_lambda=args['loss_lambda'],
          criterion_gamma=args['loss_gamma'],
          )

if cuda:
    spr.cuda()

optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, spr.parameters()), lr=args['lr'], weight_decay=args['regularize'])

train_list = list(np.random.permutation(list(range(0, num_nodes))))
interval = args['interval']
total_batches, batch_size = args['epoches'], args['batch_size']
batch_start, batch_end = 0, batch_size
minimal_loss, minimal_batch = 1e9, 0

for batch in range(total_batches):
    spr.train()
    if batch_end > len(train_list):
        batch_start, batch_end = 0, batch_size
        random.shuffle(train_list)
    batch_center_nodes = sorted(train_list[batch_start:batch_end])
    batch_start = batch_end
    batch_end += batch_size
    # ----------------------------------- #
    optimizer.zero_grad()
    loss = spr.loss(batch_center_nodes)

    if loss < minimal_loss:
        minimal_loss, minimal_batch = loss, batch
        torch.save(spr.state_dict(), args["model_path"])

    loss.backward()
    optimizer.step()

    print('batch {} loss: {}'.format(batch, loss))
    # ----------------------------------- #
    if (batch + 1) % interval == 0 or batch == total_batches - 1:
        spr.eval()
        optimizer.zero_grad()
        metrics = spr.test_func()
        if batch != total_batches - 1:
            print(batch, 'Test F1MT: {}, F1MA: {}, F1WT: {}, F1WT: {}, AUC: {}'.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))
        elif batch == total_batches - 1:
            print(batch, 'LAST F1MT: {}, F1MA: {}, F1WT: {}, F1WT: {}, AUC: {}'.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))
    sys.stdout.flush()
print("Training done...", "\n", "-" * 50)
print('Loading model at {}th epoch'.format(minimal_batch))
spr.load_state_dict(torch.load(args["model_path"]))
spr.eval()
optimizer.zero_grad()
print("Saving node representations at {} epoch!!!".format(minimal_batch))
metrics = spr.test_func(last_epoch=True, path=args['embedding_path'])
print('BEST F1MT: {}, F1MA: {}, F1WT: {}, F1BI: {}, AUC: {})'.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]))
