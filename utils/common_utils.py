import torch    

def save_checkpoints(state:dict, filename = "checkpoint.pth.tar"):
    print("==> Saving model checkpoint")
    torch.save(state, filename)

def load_checkpoints(checkpoint,model,optimizer):
    print("==> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])