from torch.utils.data import DataLoader
from MiniImagenet import MiniImagenet
from meta import Meta
from tqdm import tqdm
import numpy as np
import scipy.stats
import argparse
import torch


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    try:
        f = open("results.csv", "w")
    except FileNotFoundError:
        f = open("results.csv", "x")

    print(args)
    f.write("Steps,loss,acc,te_loss,te_acc\n")

    config = [
        ('conv2d', [32, 3, args.kernel_size, args.kernel_size, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('conv2d', [args.ret_channels, 32, args.kernel_size, args.kernel_size, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('conv2d', [32, args.ret_channels, args.kernel_size, args.kernel_size, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
    ]
    for _ in range(args.vvs_depth-1):
        config += [
            ('conv2d', [32, 32, args.kernel_size, args.kernel_size, 1, 0]),
            ('relu', [True]),
            ('bn', [32]),
        ]
    config += [
        ('flatten', []),
        ('linear', [args.n_way, 32 * (68 - 2 * (args.kernel_size // 2) * args.vvs_depth) ** 2])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = MiniImagenet('miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    best_train_loss = 0
    pruning_factor = 0
    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        num_steps = len(db)
        print ("\n\nStarting Training...")
        t = tqdm(enumerate(db), desc=f"Epoch {epoch}", total=num_steps)
        for step, (x_spt, y_spt, x_qry, y_qry) in t:

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            loss, accs = maml(x_spt, y_spt, x_qry, y_qry)

            if step % args.save_summary_steps == 0:  # evaluation
                f.write(f"{epoch*num_steps + step},{loss},{accs[-1]},")

                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test, losses = [], []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    loss, accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    accs_all_test.append(accs)
                    losses.append(loss)

                # [b, update_step+1]
                # accs_log = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                accs_list = list(map(lambda a: a[-1], accs_all_test))
                te_acc = np.array(accs_list).mean(axis=0).astype(np.float16)
                losses = np.array(losses).mean(axis=0).astype(np.float16)
                t.set_postfix({'step': epoch*num_steps + step, 'tr acc': accs[-1], 'te_loss': losses, 'te_acc': te_acc})
                f.write(f"{losses},{te_acc}\n")

                if te_acc > best_train_loss:
                    best_train_loss = te_acc
                    pruning_factor = 0
                else:
                    pruning_factor += 1

                if pruning_factor == args.pruning_factor:
                    break

        if pruning_factor == args.pruning_factor:
            break

    f.close()


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=1)
    argparser.add_argument('--ret_channels', type=int, help='number of channels at Retina Output', default=32)
    argparser.add_argument('--vvs_depth', type=int, help='number of conv layers for VVSNet', default=4)
    argparser.add_argument('--kernel_size', type=int, help='size of the convolutional kernels', default=9)
    argparser.add_argument('--save_summary_steps', type=int, help='frequence to log model evaluation metrics', default=1000)
    argparser.add_argument('--pruning_factor', type=int, help='stop the training after this number of evaluations without accuracy increase', default=10)

    args = argparser.parse_args()

    main()
