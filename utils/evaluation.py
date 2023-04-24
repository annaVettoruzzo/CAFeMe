from .common import DEVICE, func_call, accuracy

def local_evaluation(model, dataloader, loss_fn):
    model.eval()
    tot_loss = 0
    num_samples = 0
    acc = 0
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = func_call(model, None, x)
        tot_loss += loss_fn(y_pred, y)
        acc += accuracy(y_pred, y)
        num_samples += y.size(-1)

    model.train()
    return tot_loss, acc/num_samples