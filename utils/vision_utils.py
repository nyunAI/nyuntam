import time
import logging
from datetime import datetime
import os
from typing import List

logger = logging.getLogger(__name__)


def validate(val_loader, model, criterion, args):
    from trailmet.utils import accuracy, AverageMeter
    import torch

    logger = logging.getLogger(__name__)
    name = "_".join(
        [
            "validate",
            datetime.now().strftime("%b-%d_%H-%M-%S"),
        ]
    )
    os.makedirs(f"{os.getcwd()}/logs/Validate", exist_ok=True)
    logger_file = f"{os.getcwd()}/logs/Validate/{name}.log"
    logging.basicConfig(
        filename=logger_file,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H-%M-%S",
        level=logging.INFO,
    )

    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to evaluate mode
    device = args.get("DEVICE", "cuda:0")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute output
            output = model(images.to(device))
            loss = criterion(output, target.to(device))

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.to("cpu"), target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            logger.info(
                " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(
                    top1=top1, top5=top5
                )
            )

        logger.info(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg, top5.avg


def modify_head_classification(model, model_name, num_classes):
    import torch
    import torch.nn as nn

    layer_names = [name for name, _ in model.named_children()]
    if "head" in layer_names:
        nc = [i for i in model.head.named_children()]
        if nc == []:
            setattr(model, "head", nn.Linear(model.head.in_features, num_classes))
        else:
            if "fc" in model.head.named_children():
                setattr(
                    model.head, "fc", nn.Linear(model.head.fc.in_features, num_classes)
                )
    elif "fc" in layer_names:
        setattr(model, "fc", nn.Linear(model.fc.in_features, num_classes))
    elif "classifier" in layer_names:
        setattr(
            model, "classifier", nn.Linear(model.classifier.in_features, num_classes)
        )
    elif "vanillanet" in model_name:
        model.switch_to_deploy()
        model.cls[2] = nn.Conv2d(
            model.cls[2].in_channels,
            num_classes,
            kernel_size=model.cls[2].kernel_size,
            stride=model.cls[2].stride,
        )
    else:
        raise ValueError(
            f"Not able to find the last fc layer from the layer list {layer_names}"
        )
    return model
