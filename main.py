import csv
import os
import time

import torch
import torch.optim

import criteria
import utils
from dataloaders.make3d_dataloader import Make3DDataset
from dataloaders.nyu_dataloader import NYUDataset
from metrics import AverageMeter, Result
from models import ResNet, FCDenseNet57

args = utils.parse_command()
print(args)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
              'delta1', 'delta2', 'delta3',
              'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()

cuda_enabled = torch.cuda.is_available()
device = torch.device("cuda" if cuda_enabled else "cpu")


def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    datadir = os.path.join('data', args.data)
    traindir = os.path.join(datadir, 'train')
    valdir = os.path.join(datadir, 'val')

    # todo totally remove sparsification
    if args.data == 'nyudepthv2':
        if not args.evaluate:
            train_dataset = NYUDataset(traindir, type='train',
                                       modality=args.modality, sparsifier=sparsifier)
        val_dataset = NYUDataset(valdir, type='val',
                                 modality=args.modality, sparsifier=sparsifier)
    elif args.data == 'kitti':
        from dataloaders.kitti_dataloader import KITTIDataset
        if not args.evaluate:
            train_dataset = KITTIDataset(traindir, type='train',
                                         modality=args.modality, sparsifier=sparsifier)
        val_dataset = KITTIDataset(valdir, type='val',
                                   modality=args.modality, sparsifier=sparsifier)
    elif args.data == 'make3d':
        if not args.evaluate:
            train_dataset = Make3DDataset(datadir, train=True, full_size=(args.img_output_size, args.img_output_size))
        val_dataset = NotImplementedError
    else:
        raise RuntimeError('Dataset not found.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    train_loader = None
    if not args.evaluate:
        if args.train_size is not None:
            inputs, targets = zip(*[train_dataset[i] for i in range(args.train_size)])
            train_dataset = torch.utils.data.TensorDataset(torch.stack(inputs), torch.stack(targets))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   pin_memory=True)

    print("=> data loaders created.")
    return train_loader, val_loader


def main():
    global args, best_result, output_directory, train_csv, test_csv

    # evaluation mode
    start_epoch = 0
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
            "=> no best model found at '{}'".format(args.evaluate)
        print("=> loading best model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        output_directory = os.path.dirname(args.evaluate)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        _, val_loader = create_data_loaders(args)
        args.evaluate = True
        validate(val_loader, model, checkpoint['epoch'], write_to_file=False)
        return

    # optionally resume from a checkpoint
    elif args.resume:
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), \
            "=> no checkpoint found at '{}'".format(chkpt_path)
        print("=> loading checkpoint '{}'".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        output_directory = os.path.dirname(os.path.abspath(chkpt_path))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        train_loader, val_loader = create_data_loaders(args)
        args.resume = True

    # create new model
    else:
        train_loader, val_loader = create_data_loaders(args)
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
        in_channels = len(args.modality)
        if args.arch == 'resnet50':
            model = ResNet(layers=50, decoder=args.decoder, output_size=224,
                           in_channels=in_channels, pretrained=args.pretrained)
        elif args.arch == 'resnet18':
            model = ResNet(layers=18, decoder=args.decoder, output_size=args.img_output_size,
                           in_channels=in_channels, pretrained=args.pretrained, output_channels=args.output_channels)
        elif args.arch == 'densenet57':
            # todo add depth to args+number of outputs
            model = FCDenseNet57(out_channels=args.output_channels)
        else:
            raise NotImplementedError(f"Haven't implemented architecture {args.arch}.")
        print("=> model created.")

        optim = torch.optim.Adam if args.weight_decay == 0 else torch.optim.AdamW
        optimizer = optim(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        model = model.to(device)

    # define loss function (criterion) and optimizer
    if args.criterion == 'l2':
        if args.data == 'make3d':
            criterion = criteria.Make3DMaskedMSELoss()
        else:
            criterion = criteria.MaskedMSELoss()
    elif args.criterion == 'l1':
        if args.data == 'make3d':
            criterion = criteria.Make3DMaskedL1Loss()
        else:
            # todo see if this difference is even needed
            criterion = criteria.MaskedL1Loss()
    else:
        raise NotImplementedError(f"Haven't implemented criterion {args.criterion}.")
    criterion = criterion.to(device)

    # create results folder, if not already exists
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train.csv')
    test_csv = os.path.join(output_directory, 'test.csv')
    best_txt = os.path.join(output_directory, 'best.txt')

    # create new csv files with only header
    if not args.resume:
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for epoch in range(start_epoch, args.epochs):
        # utils.adjust_learning_rate(optimizer, epoch, args.lr)
        train(train_loader, model, criterion, optimizer, epoch)  # train for one epoch
        # result, img_merge = validate(val_loader, model, epoch)  # evaluate on validation set

        # remember best rmse and save checkpoint
        # is_best = result.rmse < best_result.rmse
        # if is_best:
        #     best_result = result
        #     with open(best_txt, 'w') as txtfile:
        #         txtfile.write(
        #             "epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
        #             format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1,
        #                    result.gpu_time))
        #     if img_merge is not None:
        #         img_filename = output_directory + '/comparison_best.png'
        #         utils.save_image(img_merge, img_filename)

        # utils.save_checkpoint({
        #     'args': args,
        #     'epoch': epoch,
        #     'arch': args.arch,
        #     'model': model,
        #     'best_result': best_result,
        #     'optimizer': optimizer,
        # }, is_best, epoch, output_directory)


def train(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        input, target = input.to(device), target.to(device)
        if cuda_enabled:
            torch.cuda.synchronize()

        data_time = time.time() - end

        end = time.time()

        pred = model(input)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if cuda_enabled:
            torch.cuda.synchronize()

        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data, make3d=args.data == 'make3d')
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                epoch, i + 1, len(train_loader), data_time=data_time,
                gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                         'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                         'gpu_time': avg.gpu_time, 'data_time': avg.data_time})


def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval()  # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50
        if args.modality == 'd':
            img_merge = None
        else:
            if args.modality == 'rgb':
                rgb = input
            elif args.modality == 'rgbd':
                rgb = input[:, :3, :, :]
                depth = input[:, 3:, :, :]

            if i == 0:
                if args.modality == 'rgbd':
                    img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    img_merge = utils.merge_into_row(rgb, target, pred)
            elif (i < 8 * skip) and (i % skip == 0):
                if args.modality == 'rgbd':
                    row = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    row = utils.merge_into_row(rgb, target, pred)
                img_merge = utils.add_row(img_merge, row)
            elif i == 8 * skip:
                filename = output_directory + '/comparison_' + str(epoch) + '.png'
                utils.save_image(img_merge, filename)

        if (i + 1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'
          'MAE={average.mae:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'REL={average.absrel:.3f}\n'
          'Lg10={average.lg10:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                             'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                             'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge


if __name__ == '__main__':
    main()
