import argparse
import itertools
import os
import warnings

import torch
from torch.optim import lr_scheduler

from model import CRHS
from data_utils import *
from configure import get_default_config

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='100', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='10', help='number of test times')
parser.add_argument('--city', type=str, default='xa')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()


def main():
    # Environments
    warnings.filterwarnings("ignore")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device)

    config = get_default_config()
    config['city'] = args.city
    config['print_num'] = args.print_num
    config['seed'] = args.seed

    seed = config['seed']
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)
    torch.backends.cudnn.deterministic = True

    # Build the model
    crhs = CRHS(config)
    print(crhs.autoencoder_a)
    optimizer = torch.optim.Adam(
        itertools.chain(crhs.autoencoder_a.parameters(), crhs.autoencoder_s.parameters(),
                        crhs.autoencoder_d.parameters(),
                        crhs.adv.parameters(),
                        ), lr=config[config['city']]['lr'], weight_decay=1e-6)

    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=200)

    crhs.to_device(device)

    # Load data
    redata = RegionData(config)
    xs_raw = [redata.a_m, redata.s_m, redata.d_m]
    xs = []
    for view in range(len(xs_raw)):
        xs.append(torch.from_numpy(xs_raw[view]).float().to(device))

    # Training
    crhs.train(config, xs, optimizer, scheduler, device)


if __name__ == '__main__':
    main()
