import torch
from torch import nn
from torch.functional import F

__simi_model = None


def siminet_similarity(prev_vec, cur_vec):
    assert __simi_model is not None, "Siminet Model cannot be none"
    siminet_input1 = torch.tensor(prev_vec).float()
    siminet_input2 = torch.tensor(cur_vec).float()
    inputs = torch.cat([siminet_input1, siminet_input2], dim=-1)
    outputs = __simi_model(inputs)
    outputs = torch.sigmoid(outputs)
    siminet_alpha = outputs.cpu().detach().numpy()[0]
    return siminet_alpha.item()


class SimiNet(nn.Module):
    def __init__(self, n_classes):
        super(SimiNet, self).__init__()
        self.output = 1
        self.input = n_classes * 2
        N_HIDDEN = 1024
        self.l1 = nn.Linear(self.input, N_HIDDEN, bias=True)
        self.l2 = nn.Linear(N_HIDDEN, N_HIDDEN, bias=True)
        self.l3 = nn.Linear(N_HIDDEN, self.output, bias=True)

    def forward(self, x):
        x = x.view(-1, self.input)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


def load_siminet_model(n_classes, resume_path):
    model = SimiNet(n_classes)
    model = nn.DataParallel(model, device_ids=None)

    print('loading checkpoint {}'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
    res = [val for key, val in checkpoint['state_dict'].items()
           if 'module' in key]
    if len(res) == 0:
        # Model wrapped around DataParallel but checkpoints are not
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    global __simi_model
    __simi_model = model
    return model
