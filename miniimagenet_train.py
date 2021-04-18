from torch.utils.data import DataLoader
from MiniImagenet import MiniImagenet
from meta import Meta
from tqdm import tqdm
import numpy as np
import scipy.stats
import argparse
import torch
import wandb


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def evaluate(model, dataset, device, desc="Eval Test"):
    db = DataLoader(dataset, 1, shuffle=True, num_workers=1, pin_memory=True)
    all_accs, losses = [], []

    eval_bar = tqdm(db, desc=desc, total=len(db), leave=False)
    for x_spt, y_spt, x_qry, y_qry in eval_bar:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                        x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
        loss, accs = model.finetunning(x_spt, y_spt, x_qry, y_qry)
        all_accs.append(accs)
        losses.append(loss)

    accs = list(map(lambda a: a[-1], all_accs))
    acc = np.array(accs).mean(axis=0).astype(np.float16)
    loss = np.array(losses).mean(axis=0).astype(np.float16)

    return acc, loss


def getOutputDims(args):
    imgSize = args.imgsz
    for _ in range(args.vvs_depth + 2):
        imgSize = (imgSize - args.kernel_size) + 1
        if args.max_pool == "yes":
            imgSize = imgSize // 2
    return int(32 * (imgSize ** 2))


def defineModel(args):
    
    # Add Retina Net
    config = [
        ('conv2d', [32, 3, args.kernel_size, args.kernel_size, 1, 0]),
        ('relu', [True]),
        ('bn', [32])
    ]
    if args.max_pool == "yes": 
        config += [('max_pool2d', [2, 2, 0])]
    config += [
        ('conv2d', [args.ret_channels, 32, args.kernel_size, args.kernel_size, 1, 0]),
        ('relu', [True]),
        ('bn', [args.ret_channels])
    ]
    if args.max_pool == "yes": 
        config += [('max_pool2d', [2, 2, 0])]
    
    # Add First VVS Layer
    config += [
        ('conv2d', [32, args.ret_channels, args.kernel_size, args.kernel_size, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
    ]
    if args.max_pool == "yes": 
        config += [('max_pool2d', [2, 2, 0])]
    
    # Add VVS Net
    for _ in range(args.vvs_depth-1):
        config += [
            ('conv2d', [32, 32, args.kernel_size, args.kernel_size, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
        ]
        if args.max_pool == "yes":
            config += [('max_pool2d', [2, 2, 0])]
    config += [
        ('flatten', []),
        ('linear', [args.n_way, getOutputDims(args)])
    ]
    return config


def main():
    # Manually seed torch and numpy for reproducible results
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Open csv file to write for metric logging
    try:
        f = open("results.csv", "w")
    except FileNotFoundError:
        f = open("results.csv", "x")
    f.write("Steps,tr_loss,tr_acc,val_loss,val_acc,te_loss,te_acc\n")


    # Choose PyTorch device and create the model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Meta(args, defineModel(args)).to(device)

    # Setup Weights and Biases logger, config hyperparams and watch model
    wandb.init(project="Meta-SGD")
    wandb.config.update(args)
    wandb.watch(model)

    # Print additional information on the model
    if args.verbose:
        tmp = filter(lambda x: x.requires_grad, model.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        print(args)
        print(model)
        print('Total trainable tensors:', num)

    # Create datasets
    # batchsz here means total episode number
    print("\nGathering Datasets:")
    mini = MiniImagenet('miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry, batchsz=10000, resize=args.imgsz)
    mini_train_eval = MiniImagenet('miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                                   k_query=args.k_qry, batchsz=args.eval_steps, resize=args.imgsz)
    mini_val = MiniImagenet('miniimagenet/', mode='val', n_way=args.n_way, k_shot=args.k_spt,
                            k_query=args.k_qry, batchsz=args.eval_steps, resize=args.imgsz)
    mini_test = MiniImagenet('miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry, batchsz=args.eval_steps, resize=args.imgsz)

    print("\nMeta-Training:")
    pruning_factor = args.pruning
    best_tr_acc, best_val_acc, best_te_acc = 0, 0, 0
    epoch_bar = tqdm(range(args.epoch//10000), desc="Training", total=len(range(args.epoch//10000)))
    for epoch in epoch_bar:
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        task_bar = tqdm(enumerate(db), desc=f"Epoch {epoch}", total=len(db), leave=False)
        for step, (x_spt, y_spt, x_qry, y_qry) in task_bar:
            total_steps = len(db) * epoch + step + 1

            # Perform training for each task
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            model(x_spt, y_spt, x_qry, y_qry)                                

            if total_steps % args.save_summary_steps == 0:  # evaluation
                # Get evaluation metrics
                tr_acc, tr_loss = evaluate(model, mini_train_eval, device, "Eval Train")
                val_acc, val_loss = evaluate(model, mini_val, device, "Eval Val")
                te_acc, te_loss = evaluate(model, mini_test, device)

                # Update Task tqdm bar
                metrics = {
                    'tr acc': tr_acc,
                    'val_acc': val_acc,
                    'te_acc': te_acc
                }
                task_bar.set_postfix(metrics)
                metrics['tr_loss'] = tr_loss
                metrics['val_loss'] = val_loss
                metrics['te_loss'] = te_loss
                wandb.log(metrics)
                f.write(f"{total_steps},{tr_loss},{tr_acc},{val_loss},{val_acc},{te_loss},{te_acc}\n")

                # Update best metrics
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    pruning_factor = args.pruning
                else:
                    pruning_factor -= 1
                best_te_acc = max(te_acc, best_te_acc)
                best_tr_acc = max(tr_acc, best_tr_acc)

                if pruning_factor == 0: break

                # Update tqdm 
                epoch_bar.set_postfix({
                    'b_tr_acc': best_tr_acc,
                    'b_val_acc': best_val_acc,
                    'b_te_acc': best_te_acc,
                    'prune': pruning_factor
                })

        if pruning_factor == 0: break

    f.close()


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=500000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--lr_type', type=str, help='scalar, vector or matrix (for learning rate)', default="vector")
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=1)
    argparser.add_argument('--ret_channels', type=int, help='number of channels at Retina Output', default=32)
    argparser.add_argument('--vvs_depth', type=int, help='number of conv layers for VVSNet', default=8)
    argparser.add_argument('--kernel_size', type=int, help='size of the convolutional kernels', default=9)
    argparser.add_argument('--max_pool', type=str, help='use max-pooling for each conv layer', default="no")
    argparser.add_argument('--eval_steps', type=int, help='number of batches to iterate in test mode', default=200)
    argparser.add_argument('--save_summary_steps', type=int, help='frequence to log model evaluation metrics', default=250)
    argparser.add_argument('--pruning', type=int, help='stop the training after this number of evaluations without accuracy increase', default=12)
    argparser.add_argument('--verbose', type=int, help='print additional information', default=0)
    argparser.add_argument('--seed', type=int, help='seed for reproducible results', default=42)

    args = argparser.parse_args()

    main()
