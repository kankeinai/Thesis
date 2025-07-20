import torch
import heapq

def relative_error(prediction: torch.Tensor,
                    target: torch.Tensor,
                    dim: int = None,
                    eps: float = 1e-8) -> torch.Tensor:
    
    num = torch.norm(prediction - target, p=2, dim=dim)
    den = torch.norm(target, p=2, dim=dim).clamp(min=eps)
    return num / den

def calculate_true_error(x, u, t, optimal_x, optimal_u, device):
    # x, u: [1, m] tensors, t: [1, m] tensor
    t_flat = t[0].detach().cpu().numpy()
    true_u = torch.tensor(optimal_u, dtype=torch.float32, device=device).unsqueeze(0)
    true_x = torch.tensor(optimal_x, dtype=torch.float32, device=device).unsqueeze(0)
    
    rel_err_u = relative_error(u, true_u)
    rel_err_x = relative_error(x, true_x)

    return rel_err_u.item(), rel_err_x.item()


def calculate_test_errors(model: torch.nn.Module,
                          test_loader: torch.utils.data.DataLoader,
                          top_k: int = None,
                          device: torch.device = None, architecture='deeponet') -> dict:
    """
    Compute mean/std of your relative error over the test set;
    optionally also return the top‐k worst errors (with inputs/targets/preds).
    
    If top_k is None or <= 0, skips all top‐k bookkeeping.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    # Pre‐compute uniform time grid
    if architecture == 'deeponet':
        t_uniform = torch.linspace(0, 1, test_loader.dataset.m, device=device).unsqueeze(1)

    running_sum = 0.0
    running_sum_sq = 0.0
    total_samples = 0

    # only initialize heap if top_k requested
    if top_k and top_k > 0:
        top_k_heap = []  # will store (error, idx, u, traj, pred)
    sample_idx = 0
    all_errors = []

    with torch.no_grad():
        for batch in test_loader:
            if architecture == 'deeponet':
                u, t, t0, ut, trajectory, mask = batch
                t_uniform = t_uniform.to(device)
                pred = model(u, t_uniform)
            else:
                u, t,  trajectory, mask = batch
                pred = model(u, t)
            errors = relative_error(pred, trajectory,
                                    dim=list(range(1, trajectory.ndim)))

            batch_errors = errors.cpu()
            all_errors.append(batch_errors)
            running_sum += batch_errors.sum().item()
            running_sum_sq += (batch_errors ** 2).sum().item()
            total_samples += batch_errors.numel()

            # only do top‐k tracking if requested
            if top_k and top_k > 0:
                for i, err_val in enumerate(batch_errors):
                    err = err_val.item()
                    if len(top_k_heap) < top_k:
                        heapq.heappush(top_k_heap, (err, sample_idx + i,
                                                   u[i].cpu(), trajectory[i].cpu(), pred[i].cpu()))
                    elif err > top_k_heap[0][0]:
                        heapq.heapreplace(top_k_heap, (err, sample_idx + i,
                                                       u[i].cpu(), trajectory[i].cpu(), pred[i].cpu()))
            sample_idx += batch_errors.size(0)

    errors_tensor = torch.cat(all_errors)
    mean_error = errors_tensor.mean()
    std_error = errors_tensor.std()

    result = {
        'mean_error': mean_error,
        'std_error': std_error,
    }
    

    # unpack top‐k only if we did the work
    if top_k and top_k > 0:
        # reverse‐sort so worst first
        top_k_heap.sort(reverse=True)
        result.update({
            'topk_indices': torch.tensor([item[1] for item in top_k_heap]),
            'topk_errors':  torch.tensor([item[0] for item in top_k_heap]),
            'topk_u':       [item[2] for item in top_k_heap],
            'topk_trajectory':   [item[3] for item in top_k_heap],
            'topk_prediction':   [item[4] for item in top_k_heap],
        })

    return result