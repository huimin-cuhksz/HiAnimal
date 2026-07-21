import json
import os
import time

import torch
from torch.utils.data import DataLoader

from lib.data.TrainDataset_Refine import TrainDataset_Refine
from lib.model.HGPIFuNetwNML import HGPIFuNetwNML
from lib.options import BaseOptions
from lib.train_util import (
    adjust_learning_rate,
    reshape_multiview_tensors,
    reshape_sample_tensor,
)


def train(opt):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for HiAnimal training")

    device = torch.device(f"cuda:{opt.gpu_id}")
    train_dataset = TrainDataset_Refine(opt, phase="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=opt.num_threads,
        pin_memory=opt.pin_memory,
    )
    print(f"train data size: {len(train_loader)}", flush=True)

    net_g = HGPIFuNetwNML(opt, train_dataset.projection_mode).to(device)
    optimizer = torch.optim.RMSprop(
        net_g.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0
    )
    learning_rate = opt.learning_rate
    print(f"Using Network: {net_g.name}", flush=True)

    if opt.load_netG_checkpoint_path is not None:
        print(f"Loading net G: {opt.load_netG_checkpoint_path}", flush=True)
        state_dict = torch.load(opt.load_netG_checkpoint_path, map_location=device)
        net_g.load_state_dict(state_dict)

    if opt.continue_train:
        checkpoint_name = (
            "netG_latest"
            if opt.resume_epoch < 0
            else f"netG_epoch_{opt.resume_epoch}"
        )
        checkpoint = os.path.join(opt.checkpoints_path, opt.name, checkpoint_name)
        print(f"Resuming from: {checkpoint}", flush=True)
        net_g.load_state_dict(torch.load(checkpoint, map_location=device))

    checkpoint_dir = os.path.join(opt.checkpoints_path, opt.name)
    result_dir = os.path.join(opt.results_path, opt.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "opt.txt"), "w", encoding="utf-8") as file:
        json.dump(vars(opt), file, indent=2)

    start_epoch = 0 if not opt.continue_train else max(opt.resume_epoch, 0)
    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start = time.time()
        net_g.train()
        learning_rate = adjust_learning_rate(
            optimizer, epoch, learning_rate, opt.schedule, opt.gamma
        )

        for step, batch in enumerate(train_loader):
            iteration_start = time.time()
            images = batch["img"].to(device)
            calibrations = batch["calib"].to(device)
            samples = batch["samples"].to(device)
            labels = batch["labels"].to(device)

            images, calibrations = reshape_multiview_tensors(images, calibrations)
            if opt.num_views > 1:
                samples = reshape_sample_tensor(samples, opt.num_views)

            _, error = net_g(images, samples, calibrations, labels=labels)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()

            if step % opt.freq_plot == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / (step + 1) * (len(train_loader) - step - 1)
                print(
                    f"Name: {opt.name} | Epoch: {epoch} | "
                    f"{step}/{len(train_loader)} | Err: {error.item():.6f} | "
                    f"LR: {learning_rate:.6f} | Sigma: {opt.sigma:.2f} | "
                    f"netT: {time.time() - iteration_start:.5f} | "
                    f"ETA: {int(eta // 60):02d}:{int(eta % 60):02d}",
                    flush=True,
                )

        if epoch % opt.freq_save == 0 and len(train_loader) > 0:
            state_dict = net_g.state_dict()
            torch.save(state_dict, os.path.join(checkpoint_dir, "netG_latest"))
            torch.save(
                state_dict, os.path.join(checkpoint_dir, f"netG_epoch_{epoch}")
            )


if __name__ == "__main__":
    train(BaseOptions().parse())
