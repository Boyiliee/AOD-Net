import os
import torch
import torch.backends.cudnn
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from utils import logger, weight_init
from config import get_config
from model import AODnet, AOD_pono_net
from data import HazeDataset


@logger
def load_data(cfg):
    data_transform = transforms.Compose([
        transforms.Resize([480, 640]),
        transforms.ToTensor()
    ])
    train_haze_dataset = HazeDataset(cfg.ori_data_path, cfg.haze_data_path, data_transform)
    train_loader = torch.utils.data.DataLoader(train_haze_dataset, batch_size=cfg.batch_size, shuffle=True,
                                               num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    val_haze_dataset = HazeDataset(cfg.val_ori_data_path, cfg.val_haze_data_path, data_transform)
    val_loader = torch.utils.data.DataLoader(val_haze_dataset, batch_size=cfg.val_batch_size, shuffle=False,
                                             num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

    return train_loader, len(train_loader), val_loader, len(val_loader)


@logger
def save_model(epoch, path, net, optimizer, net_name):
    if not os.path.exists(os.path.join(path, net_name)):
        os.mkdir(os.path.join(path, net_name))
    torch.save({'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()},
               f=os.path.join(path, net_name, '{}_{}.pkl'.format('AOD', epoch)))


@logger
def load_network(device):
    net = AOD_pono_net().to(device)
    net.apply(weight_init)
    return net


@logger
def load_optimizer(net, cfg):
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer


@logger
def loss_func(device):
    criterion = torch.nn.MSELoss().to(device)
    return criterion


@logger
def load_summaries(cfg):
    summary = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.net_name), comment='')
    return summary


def main(cfg):
    # -------------------------------------------------------------------
    # basic config
    print(cfg)
    if cfg.gpu > -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # -------------------------------------------------------------------
    # load summaries
    summary = load_summaries(cfg)
    # -------------------------------------------------------------------
    # load data
    train_loader, train_number, val_loader, val_number = load_data(cfg)
    # -------------------------------------------------------------------
    # load loss
    criterion = loss_func(device)
    # -------------------------------------------------------------------
    # load network
    network = load_network(device)
    # -------------------------------------------------------------------
    # load optimizer
    optimizer = load_optimizer(network, cfg)
    # -------------------------------------------------------------------
    # start train
    print('Start train')
    network.train()
    for epoch in range(cfg.epochs):
        for step, (ori_image, haze_image) in enumerate(train_loader):
            count = epoch * train_number + (step + 1)
            ori_image, haze_image = ori_image.to(device), haze_image.to(device)
            dehaze_image = network(haze_image)
            loss = criterion(dehaze_image, ori_image)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            summary.add_scalar('loss', loss.item(), count)
            if step % cfg.print_gap == 0:
                summary.add_image('DeHaze_Images', make_grid(dehaze_image[:4].data, normalize=True, scale_each=True),
                                  count)
                summary.add_image('Haze_Images', make_grid(haze_image[:4].data, normalize=True, scale_each=True), count)
                summary.add_image('Origin_Images', make_grid(ori_image[:4].data, normalize=True, scale_each=True),
                                  count)
            print('Epoch: {}/{}  |  Step: {}/{}  |  lr: {:.6f}  | Loss: {:.6f}'
                  .format(epoch + 1, cfg.epochs, step + 1, train_number,
                          optimizer.param_groups[0]['lr'], loss.item()))
        # -------------------------------------------------------------------
        # start validation
        print('Epoch: {}/{} | Validation Model Saving Images'.format(epoch + 1, cfg.epochs))
        network.eval()
        for step, (ori_image, haze_image) in enumerate(val_loader):
            if step > 10:   # only save image 10 times
                break
            ori_image, haze_image = ori_image.to(device), haze_image.to(device)
            dehaze_image = network(haze_image)
            torchvision.utils.save_image(
                torchvision.utils.make_grid(torch.cat((haze_image, dehaze_image, ori_image), 0),
                                            nrow=ori_image.shape[0]),
                os.path.join(cfg.sample_output_folder, '{}_{}.jpg'.format(epoch + 1, step)))
        network.train()
        # -------------------------------------------------------------------
        # save per epochs model
        save_model(epoch, cfg.model_dir, network, optimizer, cfg.net_name)
    # -------------------------------------------------------------------
    # train finish
    summary.close()


if __name__ == '__main__':
    config_args, unparsed_args = get_config()
    main(config_args)
