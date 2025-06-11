from __future__ import print_function
import argparse
import torch.optim as optim
from utils import *
from basenet import *
import os
from utils_HSI import sample_gt, metrics
from datasets import get_dataset, HyperX
import torch.utils.data as data
import scipy.io as io
from net import SSSE
from loss_helper import *

# Training settings
parser = argparse.ArgumentParser(description='TAADA')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epoch', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=3, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--name', type=str, default='board', metavar='B',
                    help='board dir')

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--patch_size', type=int, default=7,
                    help="Size of the spatial neighbourhood (optional, if "
                         "absent will be set by the model)")
parser.add_argument('--training_sample', type=int, default=0.8,
                    help="The proportion of training source domain samples")
parser.add_argument('--training_tar_sample', type=int, default=0.5,
                    help="The proportion of training target domain samples(no labels)")
parser.add_argument('--large_num', type=int, default=3,
                    help="The data augmentation factor")
parser.add_argument('--num_trials', type=int, default=1,
                    help='the number of epoch')

# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                      help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=True,
                      help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                      help="Random mixes between spectra")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
DEV = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
num_epoch = args.num_epoch
batch_size = args.batch_size
lr = args.lr
num_k = args.num_k
large_num = args.large_num
use_gpu = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if __name__ == '__main__':

    acc_test_list1 = np.zeros([args.num_trials, args.num_trials])
    acc_test_list2 = np.zeros([args.num_trials, args.num_trials])

for fla in range(args.num_trials):
    best_acc1 = 0
    best_acc2 = 0
    FLAG = 1
    if FLAG == 1:
        acc_test_list, acc_maxval_test_list = np.zeros_like(args.lr), np.zeros_like(args.lr)
        source_name = 'Houston13'
        target_name = 'Houston18'
        FOLDER = './data/Houston/'
        result_dir = './Result_comparison/'

    if FLAG == 2:
        acc_test_list, acc_maxval_test_list = np.zeros_like(args.lr), np.zeros_like(args.lr)
        source_name = 'Dioni'
        target_name = 'Loukia'
        FOLDER = './data/HyRANK/'
        result_dir = './Result_comparison/'

    if FLAG == 3:
        acc_test_list, acc_maxval_test_list = np.zeros_like(args.lr), np.zeros_like(args.lr)
        source_name = 'Hangzhou'
        target_name = 'Shanghai'
        FOLDER = './data/DataCube/'
        result_dir = './Result_comparison/'

    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(source_name,
                                                                                        FOLDER)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(target_name,
                                                                                        FOLDER)

    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams = vars(args)
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': DEV, 'center_pixel': False, 'supervision': 'full'})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    # padding补边
    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    train_gt_src, val_gt_src, training_set, valing_set = sample_gt(gt_src, args.training_sample, mode='random')
    test_gt_tar, _, tesing_set, _ = sample_gt(gt_tar, 1, mode='random')
    train_gt_tar, _, _, _ = sample_gt(gt_tar, args.training_tar_sample, mode='random')
    img_src_con, img_tar_con, train_gt_src_con, train_gt_tar_con = img_src, img_tar, train_gt_src, train_gt_tar
    for i in range(large_num):
        # numpy.concatenate((a1,a2,...), axis=0)
        img_src_con = np.concatenate((img_src_con, img_src))
        train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))

    # Generate the dataset
    hyperparams_train = hyperparams.copy()
    hyperparams_train.update(
        {'flip_augmentation': True, 'radiation_augmentation': True, 'mixture_augmentation': False})
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=hyperparams['batch_size'],
                                   pin_memory=True,
                                   # num_workers=4,
                                   shuffle=True,
                                   drop_last=False)
    val_dataset = HyperX(img_src, val_gt_src, **hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                 pin_memory=True,
                                 # num_workers=4,
                                 batch_size=hyperparams['batch_size'],
                                 shuffle=True,
                                 drop_last=False)
    train_tar_dataset = HyperX(img_tar_con, train_gt_tar_con, **hyperparams)
    train_tar_loader = data.DataLoader(train_tar_dataset,
                                       pin_memory=True,
                                       # num_workers=4,
                                       batch_size=hyperparams['batch_size'],
                                       shuffle=True,
                                       drop_last=False)
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                  pin_memory=True,
                                  # num_workers=4,
                                  batch_size=hyperparams['batch_size'],
                                  shuffle=False,
                                  drop_last=False)
    len_src_loader = len(train_loader)
    len_tar_train_loader = len(train_tar_loader)
    len_src_dataset = len(train_loader.dataset)
    len_tar_train_dataset = len(train_tar_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    len_tar_loader = len(test_loader)
    len_val_dataset = len(val_loader.dataset)
    len_val_loader = len(val_loader)
    print(hyperparams)

    G = SSSE(input_channels=N_BANDS,
             patch_size=7,
             n_classes=num_classes).to(DEV)
    F1 = ResClassifier(num_classes)
    F2 = ResClassifier(num_classes)
    F1.apply(weights_init)
    F2.apply(weights_init)
    lr = args.lr
    if args.cuda:
        G.cuda()
        F1.cuda()
        F2.cuda()
    if args.optimizer == 'momentum':
        optimizer_g = optim.SGD(list(G.parameters()), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=args.lr,
                                weight_decay=0.0005)
    elif args.optimizer == 'adam':
        optimizer_g = optim.Adam(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
    else:
        optimizer_g = optim.Adadelta(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
        optimizer_f = optim.Adadelta(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)


    def train(ep, train_loader, train_tar_loader):
        iter_source, iter_target = iter(train_loader), iter(train_tar_loader)
        criterion = nn.CrossEntropyLoss().cuda()
        G.train()
        F1.train()
        F2.train()
        num_iter = len_src_loader
        for batch_idx in range(1, num_iter):
            if batch_idx % len(train_tar_loader) == 0:
                iter_target = iter(train_tar_loader)
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            label_source = label_source - 1
            if args.cuda:
                data1, target1 = data_source.cuda(), label_source.cuda()
                data2 = data_target.cuda()
            # when pretraining network source only
            eta = 1.0
            data = Variable(torch.cat((data1, data2), 0))
            target1 = Variable(target1)

            # Step A train all networks to minimize loss on source
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = G(data)
            output1 = F1(output)
            output2 = F2(output)

            output_s1 = output1[:batch_size, :]
            output_s2 = output2[:batch_size, :]
            output_t1 = output1[batch_size:, :]
            output_t2 = output2[batch_size:, :]
            output_t1 = F.softmax(output_t1)
            output_t2 = F.softmax(output_t2)

            entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
            entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))

            loss1 = criterion(output_s1, target1.long())
            loss2 = criterion(output_s2, target1.long())
            all_loss = loss1 + loss2 + 0.01 * entropy_loss
            all_loss.backward()
            optimizer_g.step()
            optimizer_f.step()

            # Step B train classifier to maximize discrepancy
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            output = G(data)
            output1 = F1(output)
            output2 = F2(output)
            output_s1 = output1[:batch_size, :]
            output_s2 = output2[:batch_size, :]
            output_t1 = output1[batch_size:, :]
            output_t2 = output2[batch_size:, :]
            output_t1 = F.softmax(output_t1)
            output_t2 = F.softmax(output_t2)
            loss1 = criterion(output_s1, target1.long())
            loss2 = criterion(output_s2, target1.long())

            entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
            entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
            loss_dis = torch.mean(torch.abs(output_t1 - output_t2))
            F_loss = loss1 + loss2 - eta * loss_dis + 0.01 * entropy_loss
            F_loss.backward()
            optimizer_f.step()
            # Step C train genrator to minimize discrepancy
            for i in range(num_k):
                optimizer_g.zero_grad()
                output = G(data)
                output1 = F1(output)
                output2 = F2(output)

                output_s1 = output1[:batch_size, :]
                output_s2 = output2[:batch_size, :]
                output_t1 = output1[batch_size:, :]
                output_t2 = output2[batch_size:, :]

                loss1 = criterion(output_s1, target1.long())
                loss2 = criterion(output_s2, target1.long())
                output_t1 = F.softmax(output_t1)
                output_t2 = F.softmax(output_t2)
                loss_dis = torch.mean(torch.abs(output_t1 - output_t2))
                entropy_loss = -torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
                entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))

                loss_dis.backward()
                optimizer_g.step()
            if batch_idx % args.log_interval == 0:
                print(
                    'Train Ep: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} Entropy: {:.6f}'.format(
                        ep, batch_idx * len(data), 256. * len(train_loader),
                            100. * batch_idx / len(train_loader), loss1.item(), loss2.item(), loss_dis.item(),
                        entropy_loss.item()))

            if batch_idx == 1 and ep > 1:
                G.train()
                F1.train()
                F2.train()


    def val(val_loader):
        G.eval()
        F1.eval()
        F2.eval()
        test_loss = 0
        correct = 0
        correct2 = 0
        size = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEV), target.to(DEV)
                target2 = target - 1
                data1, target1 = Variable(data), Variable(target2)
                output = G(data1)
                output1 = F1(output)
                output2 = F2(output)
                test_loss += F.nll_loss(output1, target1.long()).item()

                pred1 = output1.data.max(1)[1]  # get the index of the max log-probability
                correct += pred1.eq(target1.data).cpu().sum()
                pred2 = output2.data.max(1)[1]  # get the index of the max log-probability
                correct2 += pred2.eq(target1.data).cpu().sum()
                k = target1.data.size()[0]

                size += k
            val_loss = test_loss
            val_loss /= len(val_loader)  # loss function already averages over batch size
            print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%)\n'.format(
                val_loss, correct, size, 100. * correct / size, 100. * correct2 / size))
        return correct


    def test(test_loader):
        G.eval()
        F1.eval()
        F2.eval()
        test_loss = 0
        correct = 0
        correct2 = 0
        size = 0
        pred1_list, pred2_list, label_list, outputdata = [], [], [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEV), target.to(DEV)
                target2 = target - 1
                data1, target1 = Variable(data), Variable(target2)
                output = G(data1)
                outputdata.append(output.cpu().numpy())
                output1 = F1(output)
                output2 = F2(output)
                test_loss += F.nll_loss(output1, target1.long()).item()

                pred1 = output1.data.max(1)[1]  # get the index of the max log-probability
                correct += pred1.eq(target1.data).cpu().sum()
                pred2 = output2.data.max(1)[1]  # get the index of the max log-probability
                correct2 += pred2.eq(target1.data).cpu().sum()
                k = target1.data.size()[0]
                pred1_list.append(pred1.cpu().numpy())
                pred2_list.append(pred2.cpu().numpy())
                label_list.append(target2.cpu().numpy())
                size += k
                acc1 = 100. * float(correct) / float(size)
                acc2 = 100. * float(correct2) / float(size)

            test_loss = test_loss
            test_loss /= len(test_loader)  # loss function already averages over batch size
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) ({:.2f}%)\n'.format(
                test_loss, correct, len_tar_dataset,
                100. * correct / len_tar_dataset, 100. * correct2 / len_tar_dataset))
            # if 100. * correct / size > 67 or 100. * correct2 / size > 67:
            value = max(100. * correct / len_tar_dataset, 100. * correct2 / len_tar_dataset)
        return value, pred1_list, pred2_list, label_list, acc1, acc2, outputdata, label_list


    def traindata(train_loader):
        G.eval()
        F1.eval()
        F2.eval()
        test_loss = 0
        correct = 0
        correct2 = 0
        size = 0
        pred1_list, pred2_list, label_list, outputdata = [], [], [], []
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(DEV), target.to(DEV)
                target2 = target - 1
                data1, target1 = Variable(data), Variable(target2)
                output = G(data1)
                outputdata.append(output.cpu().numpy())
                output1 = F1(output)
                output2 = F2(output)
                test_loss += F.nll_loss(output1, target1.long()).item()

                pred1 = output1.data.max(1)[1]  # get the index of the max log-probability
                correct += pred1.eq(target1.data).cpu().sum()
                pred2 = output2.data.max(1)[1]  # get the index of the max log-probability
                correct2 += pred2.eq(target1.data).cpu().sum()
                pred1_list.append(pred1.cpu().numpy())
                pred2_list.append(pred2.cpu().numpy())
                label_list.append(target2.cpu().numpy())


            test_loss = test_loss
            test_loss /= len(train_loader)  # loss function already averages over batch size
            print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) ({:.2f}%)\n'.format(
                test_loss, correct, len_src_dataset,
                100. * correct / len_src_dataset, 100. * correct2 / len_src_dataset))
            # if 100. * correct / size > 67 or 100. * correct2 / size > 67:
            value = max(100. * correct / len_src_dataset, 100. * correct2 / len_src_dataset)
        return outputdata, label_list

    for ep in range(1, num_epoch + 1):
        train(ep, train_loader, train_tar_loader)
        correct = val(val_loader)
        # 5 epoch test
        if ep % args.log_interval == 0:
            value, pred1_list, pred2_list, label_list, acc1, acc2, outputdata_target, target_label = test(test_loader)

            if acc1 > best_acc1:
                best_acc1 = acc1
                results1 = metrics(np.concatenate(pred1_list), np.concatenate(label_list),
                               ignored_labels=hyperparams['ignored_labels'], n_classes=gt_src.max())
                io.savemat(os.path.join(result_dir, source_name + '_results1_' + target_name + str(fla) + '.mat'),
                {'lr': args.lr, 'k': args.num_k, 'results1': results1})
                print('current best acc1:', best_acc1)
            if acc2 > best_acc2:
                best_acc2 = acc2
                results2 = metrics(np.concatenate(pred2_list), np.concatenate(label_list),
                               ignored_labels=hyperparams['ignored_labels'], n_classes=gt_src.max())
                io.savemat(os.path.join(result_dir, source_name + '_results2_' + target_name + str(fla) + '.mat'),
                {'lr': args.lr, 'k': args.num_k, 'results2': results2})
                print('current best acc2:', best_acc2)

    print('current best acc1:', best_acc1)
    print('current best acc2:', best_acc2)
