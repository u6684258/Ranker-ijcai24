import torch


def hgn_rank_train(model, device, train_loader, criterion, optimiser):
    model.train()
    train_loss = 0

    for data in train_loader:
        data = data.to(device)

        optimiser.zero_grad(set_to_none=True)
        h_pred, h_true = model.forward(data)
        h_true = [h_t.to(device) for h_t in h_true]

        loss = criterion.forward(h_pred, h_true)
        loss.backward()
        optimiser.step()
        train_loss += loss.detach().cpu().item()

    stats = {
        "loss": train_loss / len(train_loader),
    }
    return stats

def hgn_rank_evaluate(model, device, val_loader, criterion, return_true_preds=False):
    model.eval()
    val_loss = 0

    # if return_true_preds:
    #     y_true = torch.tensor([])
    #     y_pred = torch.tensor([])

    for data in val_loader:
        data = data.to(device)
        h_pred, h_true = model.forward(data)
        h_pred = [h_t.to(device) for h_t in h_pred]
        h_true = [h_t.to(device) for h_t in h_true]

        loss = criterion.forward(h_pred, h_true)
        val_loss += loss.detach().cpu().item()

        # if return_true_preds:
        #     y_pred = torch.cat((y_pred, h_pred.detach().cpu()))
        #     y_true = torch.cat((y_true, h_true.detach().cpu()))

    stats = {
        "loss": val_loss / len(val_loader),
    }
    # if return_true_preds:
    #     stats["y_true"] = y_true
    #     stats["y_pred"] = y_pred

    return stats
