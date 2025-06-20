from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
import os
from data_loader import DataLoader
from model import GPTModel
from config import GPTConfig
from torch.optim.lr_scheduler import OneCycleLR
import time
import wandb

# set up distributed data parallel
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "Distributed data parallel requires CUDA"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"  # each process is assigned a GPU
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # identify master process (GPU 0)
else:
    # vanilla, non-distributed training
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

torch.manual_seed(1337)  # for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)  # for reproducibility on GPU

# gradient accumulation (add gradients up over multiple steps, then step optimizer)
total_batch_size = 524288
total_steps = 10_000_000_000 // total_batch_size 
B = 64  # micro-batch size (64 on 8 GPUs --> total batch size 512)
T = 1024  # sequence length
assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), "total batch size must be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Total desired batch size: {total_batch_size:,}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# data loaders for training and validation
train_loader = DataLoader(
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    master_process=master_process,
    split="train",
)

val_loader = DataLoader(
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    master_process=master_process,
    split="val",
)

# set float32 matmul precision for better performance
# reduces precision to fit more data into memory on GPUs
torch.set_float32_matmul_precision("high")

# create the model
# vocab size set to 50304 for GPU memory efficiency
model = GPTModel(GPTConfig(vocab_size=50304))
model.to(device)

# compile the model for better performance
model = torch.compile(model) 
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    
# always keep a reference to the raw model for saving (unwrapped model)
raw_model = model.module if ddp else model 

if master_process:
    total_params = sum(p.numel() for p in raw_model.parameters())
    print(f"Total parameters: {total_params:,}")
    
# optimizer
optimizer = raw_model.setup_optimizer(
    weight_decay=0.1, learning_rate=6e-4, device=device
)
    
# learning rate scheduler
lr_scheduler = OneCycleLR(
    optimizer,
    max_lr=6e-4,  # max learning rate
    total_steps=total_steps,
    pct_start=0.05, # percent of total steps for warmup
    anneal_strategy="cos",  # cosine annealing
    final_div_factor=0.1,  # final learning rate will be 10x smaller than max_lr
)

# log directory
log_dir = "model_checkpoints"
os.makedirs(log_dir, exist_ok=True)

# wandb logging
wandb.init(
    project="gpt2-pretraining",
    config={
        "model_size": "medium",
        "total_steps": total_steps,
        "batch_size": total_batch_size,
        "micro_batch_size": B,
        "sequence_length": T,
        "learning_rate": 6e-4,  # your max LR
        "grad_accum_steps": grad_accum_steps,
        "ddp_world_size": ddp_world_size,
    }
)

for step in range(total_steps):
    t0 = time.time()
    last_step = step == total_steps - 1
    
    # evaluate validation loss every 250 steps
    if step % 250 == 0 or last_step:
        model.eval()  # set model to eval mode
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss at step {step}: {val_loss_accum.item()}")
            wandb.log({"val/loss": val_loss_accum.item(), "train/step": step, "val/perplexity": torch.exp(val_loss_accum).item()})
            if step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "val_loss": val_loss_accum.item(),
                    "total_tokens": total_tokens,
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(),
                }
                torch.save(checkpoint, checkpoint_path)
                
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)  # move tensors to device
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)  # forward pass
        loss = loss / grad_accum_steps  # normalize loss
        loss_accum += loss.detach()
        # if using DDP, require backward grad sync only at the end of each grad accumulation
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        loss.backward()  # backpropagate and compute gradients
        
    # average loss across all processes
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
    # clip gradients to prevent extreme gradient magnitudes
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step() # update model parameters
    lr_scheduler.step()  # update learning rate

    torch.cuda.synchronize()  # wait for GPU to finish work
    
    t1 = time.time()

    tokens_processed = (
        train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    )
    dt = t1 - t0
    tokens_per_sec = tokens_processed / dt
    lr = lr_scheduler.get_last_lr()[0]  # current learning rate
    if master_process:
        total_tokens += tokens_processed
        wandb.log({
            "train/loss": loss_accum.item(),
            "train/learning_rate": lr,
            "train/step": step,
            "train/perplexity": torch.exp(loss_accum).item(),
            "perf/tokens_per_second": tokens_per_sec,
            "perf/grad_norm": norm,
            "perf/duration": dt,
            "perf/total_tokens": total_tokens
        })
        print(
            f"Step {step}: loss={loss_accum.item():.4f}, "
            f"lr={lr:.6f}, tokens/sec={tokens_per_sec:.2f}, "
            f"grad_norm={norm:.4f}, duration={dt:.2f}s"
        )

if master_process:
    wandb.finish()  # finish wandb logging

if ddp:
    destroy_process_group()  # clean up distributed process group

