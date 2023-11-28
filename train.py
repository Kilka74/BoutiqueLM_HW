import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
from IPython.display import clear_output
from tqdm import tqdm
import torch.nn.functional as F


def generate_attention_mask(size, device="cpu"):
    mask = torch.tril(torch.ones(size, size, device=device, dtype=torch.float))
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


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


def train_epoch(
    model,
    optimizer,
    loader,
    scheduler=None,
    device="cpu",
    scaler=torch.cuda.amp.GradScaler(),
    wandb_log=False,
):
    model.train()
    loss_m = AverageMeter()
    epoch_lrs = []
    accum_iter = 4
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    loss = 0
    for batch_idx, stories in tqdm(enumerate(loader)):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            stories = stories.long().to(device)

            attention_story_mask = generate_attention_mask(
                stories.shape[1] - 1, device=device
            )
            outputs = model(stories[..., :-1], attention_story_mask)
            loss = criterion(outputs.transpose(1, 2), stories[..., 1:])
        loss_m.update(loss.item(), stories.shape[0])

        loss = loss / accum_iter
        #             loss.bacward()
        scaler.scale(loss).backward()
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(loader)):
            scaler.step(optimizer)
            scaler.update()
            #                 optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

            epoch_lrs += [optimizer.param_groups[0]["lr"]]
            if wandb_log:
                wandb.log(
                    {"train_loss": loss_m.avg, "lr": optimizer.param_groups[0]["lr"]}
                )

            # update stats
            # we use step-wise scheduler
    return loss_m.avg, epoch_lrs


@torch.no_grad()
def val_epoch(model, loader, device="cpu", wandb_log=False):
    model.eval()
    loss_m = AverageMeter()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    for stories in tqdm(loader):
        stories = stories.long().to(device)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            attention_story_mask = generate_attention_mask(
                stories.shape[1] - 1, device=device
            )
            outputs = model(stories[..., :-1], attention_story_mask)
            loss = criterion(outputs.transpose(1, 2), stories[..., 1:].long())
            # update stats
            loss_m.update(loss.item(), stories.shape[0])
    if wandb_log:
        wandb.log({"val_loss": loss_m.avg})
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
    scheduler,
    train_loader,
    val_loader,
    wandb_log=False,
    device="cpu",
):
    scaler = torch.cuda.amp.GradScaler()
    train_losses = []
    val_losses = []
    lrs = []
    for i in range(num_epochs):
        # run train epoch
        train_loss, epoch_lrs = train_epoch(
            model, optimizer, train_loader, scheduler, device, scaler, wandb_log
        )
        train_losses.append(train_loss)
        # run val epoch
        val_loss = val_epoch(model, val_loader, device, wandb_log)
        val_losses.append(val_loss)
        # update lr
        lrs += epoch_lrs

        # if wandb_log:
        #     wandb.log(
        #         {"train_loss": train_loss, "val_loss": val_loss, "lr": epoch_lrs[0]}
        #     )

        clear_output()
        plot_history(train_losses, val_losses, lrs)


def generate_story_argmax(model, tokenizer, beginning, story_length):
    model.eval()
    eos_idx = tokenizer.eos_id
    input_tokens = torch.tensor(
        tokenizer.encode(beginning, add_bos=True), dtype=torch.long
    )
    for i in range(story_length):
        pred_token = torch.argmax(model(input_tokens.unsqueeze(0))[:, -1])
        if pred_token == eos_idx:
            break
        else:
            input_tokens = torch.tensor(
                input_tokens.tolist() + [pred_token], dtype=torch.long
            )
    return tokenizer.decode(input_tokens.squeeze(0).tolist())


def generate_story_temp(model, tokenizer, beginning, story_length, k, tau):
    model.eval()
    eos_idx = tokenizer.eos_id
    input_tokens = torch.tensor(
        tokenizer.encode(beginning, add_eos=True), dtype=torch.long
    )
    for i in range(story_length):
        pred_tokens = model(input_tokens.unsqueeze(0))[:, -1]
        tokens = torch.argsort(pred_tokens, descending=True)[0, :k]
        probs = F.softmax(pred_tokens[0, tokens] / tau, dim=-1)
        pred_token = tokens[torch.multinomial(probs, 1)]
        if pred_token == eos_idx:
            break
        else:
            input_tokens = torch.tensor(
                input_tokens.tolist() + [pred_token], dtype=torch.long
            )
    return tokenizer.decode(input_tokens.squeeze(0).tolist())


@torch.no_grad()
def generate_nucleus(
    model, tokenizer, batch_size: int, prefix, max_len=32, nucleus=0.9, tau=1
):
    """
    Samples output sequence from probability distribution obtained by model

    :params
        model: predict next token for the whole batch of sequences
        tokenizer: tokenizer for the model and [BOS] token
        batch_size: number of sequence
        prefix: Tensor of tokens with shape: [batch_size, seq_len]
        max_len: max length of predicted sequence
        nucleus: parameter of nucleus sampling

    :return
        the Tensor of tokens of shape: [batch_size, max_len]
    """
    prefix = torch.tensor(
        tokenizer.encode(prefix, add_eos=False), dtype=torch.long
    ).unsqueeze(0)
    print(prefix)
    for i in range(max_len):
        probs = F.softmax(model(prefix)[:, -1, :] / tau, dim=1)
        argsorted = torch.argsort(probs, dim=1)
        sorted_probs = probs[
            torch.arange(batch_size)[:, None].repeat(1, probs.shape[1]), argsorted
        ]
        sorted_probs[torch.cumsum(sorted_probs, dim=1) < 1 - nucleus] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=1)[:, None]
        next_tokens = argsorted[
            torch.arange(batch_size), torch.multinomial(sorted_probs, 1).squeeze(1)
        ]
        prefix = torch.cat((prefix, next_tokens[:, None]), dim=-1)
    return tokenizer.decode(prefix.squeeze(0).tolist())
