import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from IPython.display import clear_output
from tqdm import tqdm


def generate_attention_mask(size, device="cpu"):
    return torch.tril(torch.ones(size, size, device=device))


# useful utility class for computing averages
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, optimizer, tokenizer, loader, scheduler=None, device="cpu"):
    model.train()
    loss_m = AverageMeter()
    epoch_lrs = []
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for stories in tqdm(loader):
        tokens_story = [
            torch.tensor(tokenizer.encode(story, add_bos=True), dtype=torch.long)
            for story in stories
        ]
        padded_tokens_story = nn.utils.rnn.pad_sequence(
            tokens_story, batch_first=True, padding_value=0
        ).to(device)
        attention_story_mask = generate_attention_mask(
            padded_tokens_story.shape[1] - 1, device=device
        )
        outputs = model(padded_tokens_story[..., :-1], attention_story_mask)
        loss = criterion(outputs.transpose(1, 2), padded_tokens_story[..., 1:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update stats
        loss_m.update(loss.item(), padded_tokens_story.shape[0])
        epoch_lrs += [optimizer.param_groups[0]["lr"]]
        # we use step-wise scheduler
        if scheduler is not None:
            scheduler.step()
    return loss_m.avg, epoch_lrs


@torch.no_grad()
def val_epoch(model, tokenizer, loader, device="cpu"):
    model.eval()
    loss_m = AverageMeter()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for stories in tqdm(loader):
        tokens_story = [
            torch.tensor(tokenizer.encode(story, add_bos=True), dtype=torch.long)
            for story in stories
        ]
        padded_tokens_story = nn.utils.rnn.pad_sequence(
            tokens_story, batch_first=True, padding_value=0
        ).to(device)
        attention_story_mask = generate_attention_mask(
            padded_tokens_story.shape[1] - 1, device=device
        )
        outputs = model(padded_tokens_story[..., :-1], attention_story_mask)
        loss = criterion(outputs.transpose(1, 2), padded_tokens_story[..., 1:])
        # update stats
        loss_m.update(loss.item(), padded_tokens_story.shape[0])
    return loss_m.avg


def plot_history(train_losses, val_losses, lrs, figsize=(18, 6)):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    ax[0].plot(train_losses, label="train")
    ax[0].plot(val_losses, label="val")
    ax[0].set_xlabel("Epoch", fontsize=16)
    ax[0].set_ylabel("Loss", fontsize=16)
    ax[0].legend()

    ax[1].plot(lrs)
    ax[1].set_xlabel("Step", fontsize=16)
    ax[1].set_ylabel("Learning rate", fontsize=16)

    fig.tight_layout()
    plt.show()


def train(
    model,
    num_epochs,
    optimizer,
    tokenizer,
    scheduler,
    train_loader,
    val_loader,
    wandb_log=False,
    device="cpu",
):
    train_losses = []
    val_losses = []
    lrs = []
    for i in range(num_epochs):
        # run train epoch
        train_loss, epoch_lrs = train_epoch(
            model, optimizer, tokenizer, train_loader, scheduler, device
        )
        train_losses.append(train_loss)
        # run val epoch
        val_loss = val_epoch(model, tokenizer, val_loader, device)
        val_losses.append(val_loss)
        # update lr
        lrs += epoch_lrs

        if wandb_log:
            wandb.log(
                {"train_loss": train_loss, "val_loss": val_loss, "lr": epoch_lrs[0]}
            )

        clear_output()
        plot_history(train_losses, val_losses, lrs)
