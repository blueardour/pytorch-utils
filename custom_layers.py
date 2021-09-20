
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import pdb

def fast_sigmoid(x):
    x = x / (1.0 + abs(x))
    x = (x + 1.0) / 2.
    return x

def tanh(x, sigmoid_func=fast_sigmoid):
    #return sigmoid_func(2.*x) * 2. + (-1.)
    return sigmoid_func(x)

class CustomTanH(nn.Module):
    def __init__(self, sigmoid_func=torch.sigmoid):
        super(CustomTanH, self).__init__()
        self.sigmoid_func = sigmoid_func
        self.bn1 = nn.BatchNorm1d(60)
        self.bn2 = nn.BatchNorm1d(60)
        #with torch.no_grad():
        #    self.bn1.weight.fill_(2.)
        #    self.bn1.bias.fill_(0.)

    def forward(self, x):
        #x = self.bn1(x)
        x = self.sigmoid_func(x)
        #x = self.bn2(x)
        return x
        
class CustomLSTM(nn.LSTM):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False):
        # proj_size=0 is available from Pytorch 1.8
        super(CustomLSTM, self).__init__(input_size, hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

        sz = self.hidden_size
        self.weight_ihf_0 = nn.Parameter(self.weight_ih_l0[sz*0:sz*1,:])
        self.weight_ihf_1 = nn.Parameter(self.weight_ih_l0[sz*1:sz*2,:])
        self.weight_ihf_2 = nn.Parameter(self.weight_ih_l0[sz*2:sz*3,:])
        self.weight_ihf_3 = nn.Parameter(self.weight_ih_l0[sz*3:sz*4,:])
        self.weight_hhf_0 = nn.Parameter(self.weight_hh_l0[sz*0:sz*1,:])
        self.weight_hhf_1 = nn.Parameter(self.weight_hh_l0[sz*1:sz*2,:])
        self.weight_hhf_2 = nn.Parameter(self.weight_hh_l0[sz*2:sz*3,:])
        self.weight_hhf_3 = nn.Parameter(self.weight_hh_l0[sz*3:sz*4,:])
        self.bias_ihf_0 = nn.Parameter(self.bias_ih_l0[sz*0:sz*1])
        self.bias_ihf_1 = nn.Parameter(self.bias_ih_l0[sz*1:sz*2])
        self.bias_ihf_2 = nn.Parameter(self.bias_ih_l0[sz*2:sz*3])
        self.bias_ihf_3 = nn.Parameter(self.bias_ih_l0[sz*3:sz*4])
        self.bias_hhf_0 = nn.Parameter(self.bias_hh_l0[sz*0:sz*1])
        self.bias_hhf_1 = nn.Parameter(self.bias_hh_l0[sz*1:sz*2])
        self.bias_hhf_2 = nn.Parameter(self.bias_hh_l0[sz*2:sz*3])
        self.bias_hhf_3 = nn.Parameter(self.bias_hh_l0[sz*3:sz*4])

        self.weight_ihr_0 = nn.Parameter(self.weight_ih_l0_reverse[sz*0:sz*1,:].transpose(0, 1))
        self.weight_ihr_1 = nn.Parameter(self.weight_ih_l0_reverse[sz*1:sz*2,:].transpose(0, 1))
        self.weight_ihr_2 = nn.Parameter(self.weight_ih_l0_reverse[sz*2:sz*3,:].transpose(0, 1))
        self.weight_ihr_3 = nn.Parameter(self.weight_ih_l0_reverse[sz*3:sz*4,:].transpose(0, 1))
        self.weight_hhr_0 = nn.Parameter(self.weight_hh_l0_reverse[sz*0:sz*1,:].transpose(0, 1))
        self.weight_hhr_1 = nn.Parameter(self.weight_hh_l0_reverse[sz*1:sz*2,:].transpose(0, 1))
        self.weight_hhr_2 = nn.Parameter(self.weight_hh_l0_reverse[sz*2:sz*3,:].transpose(0, 1))
        self.weight_hhr_3 = nn.Parameter(self.weight_hh_l0_reverse[sz*3:sz*4,:].transpose(0, 1))
        self.bias_ihr_0 = nn.Parameter(self.bias_ih_l0_reverse[sz*0:sz*1])
        self.bias_ihr_1 = nn.Parameter(self.bias_ih_l0_reverse[sz*1:sz*2])
        self.bias_ihr_2 = nn.Parameter(self.bias_ih_l0_reverse[sz*2:sz*3])
        self.bias_ihr_3 = nn.Parameter(self.bias_ih_l0_reverse[sz*3:sz*4])
        self.bias_hhr_0 = nn.Parameter(self.bias_hh_l0_reverse[sz*0:sz*1])
        self.bias_hhr_1 = nn.Parameter(self.bias_hh_l0_reverse[sz*1:sz*2])
        self.bias_hhr_2 = nn.Parameter(self.bias_hh_l0_reverse[sz*2:sz*3])
        self.bias_hhr_3 = nn.Parameter(self.bias_hh_l0_reverse[sz*3:sz*4])

        self.sigmoid = torch.sigmoid
        #self.tanh = partial(tanh, sigmoid_func=self.sigmoid)
        self.tanh = CustomTanH()

        self.fcf1 = nn.Linear(input_size, hidden_size)
        self.fcf2 = nn.Linear(input_size, hidden_size)
        self.fcf3 = nn.Linear(input_size, hidden_size)
        self.fcf4 = nn.Linear(input_size, hidden_size)
        self.fcf5 = nn.Linear(hidden_size, hidden_size)
        self.fcf6 = nn.Linear(hidden_size, hidden_size)
        self.fcf7 = nn.Linear(hidden_size, hidden_size)
        self.fcf8 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, exporting_onnx=True):
        if exporting_onnx:
            assert self.num_layers == 1
            if isinstance(x , list):
                seq = len(x) - 2
                assert seq > 0, "seq is {}".format(seq)
                assert x[0].size(0) == 1 #and x[0].size(1) == 1
                bs = 1
                inputs = [x]
                h_t, c_t = x[-2], x[-1]
            else:
                if self.batch_first:
                    bs, seq, _ = x.size() 
                    inputs = x.split(seq, dim=1)
                else:
                    bs, seq, _ = (x.size(1), x.size(0), x.size(2))
                    inputs = x.split(seq, dim=0)
                h_t, c_t = (torch.zeros(bs, 1, self.hidden_size).to(x.device), torch.zeros(bs, 1, self.hidden_size).to(x.device))

            hidden_seq_forward = []
            for t in range(seq):
                x_t = inputs[0][t]
                x_t = x_t.view(1, -1)
                #h_t = h_t.view(1, -1)
                #c_t = c_t.view(1, -1)
                #print(t, inputs[0][t].shape, x_t.shape, h_t.shape, c_t.shape)
                #pdb.set_trace()

                #i_t = F.linear(x_t, self.weight_ihf_0) + self.bias_ihf_0 + F.linear(h_t, self.weight_hhf_0) + self.bias_hhf_0
                #f_t = F.linear(x_t, self.weight_ihf_1) + self.bias_ihf_1 + F.linear(h_t, self.weight_hhf_1) + self.bias_hhf_1
                #g_t = F.linear(x_t, self.weight_ihf_2) + self.bias_ihf_2 + F.linear(h_t, self.weight_hhf_2) + self.bias_hhf_2
                #o_t = F.linear(x_t, self.weight_ihf_3) + self.bias_ihf_3 + F.linear(h_t, self.weight_hhf_3) + self.bias_hhf_3
                i_t = self.fcf1(x_t) + self.fcf5(h_t)
                f_t = self.fcf2(x_t) + self.fcf6(h_t)
                g_t = self.fcf3(x_t) + self.fcf7(h_t)
                o_t = self.fcf4(x_t) + self.fcf8(h_t)
                i_t = self.sigmoid(i_t)
                f_t = self.sigmoid(f_t)
                g_t = self.tanh(g_t)
                o_t = self.sigmoid(o_t)
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * self.tanh(c_t)
                hidden_seq_forward.append(h_t)

            #pdb.set_trace()
            hidden_seq_forward = torch.cat(hidden_seq_forward, dim=0) # [seq, bs, self.hidden_size]
            #print(hidden_seq_forward.shape)
            return hidden_seq_forward

            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), torch.zeros(bs, self.hidden_size).to(x.device))
            hidden_seq_reverse = []
            for t in range(seq):
                x_t = inputs[0][seq-t-1]
                i_t = x_t @ self.weight_ihr_0 + self.bias_ihr_0 + h_t @ self.weight_hhr_0 + self.bias_hhr_0
                f_t = x_t @ self.weight_ihr_1 + self.bias_ihr_1 + h_t @ self.weight_hhr_1 + self.bias_hhr_1
                g_t = x_t @ self.weight_ihr_2 + self.bias_ihr_2 + h_t @ self.weight_hhr_2 + self.bias_hhr_2
                o_t = x_t @ self.weight_ihr_3 + self.bias_ihr_3 + h_t @ self.weight_hhr_3 + self.bias_hhr_3
                i_t = self.sigmoid(i_t)
                f_t = self.sigmoid(f_t)
                g_t = self.tanh(g_t)
                o_t = self.sigmoid(o_t)
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * self.tanh(c_t) # [bs * self.hidden_size]
                hidden_seq_reverse.append(h_t.unsqueeze(0))

            # stack hidden_seq_forward and hidden_seq_reverse to hidden_seq
            hidden_seq_forward = torch.cat(hidden_seq_forward, dim=0) # [seq, bs, self.hidden_size]
            return hidden_seq_forward

            hidden_seq_reverse = torch.cat(hidden_seq_reverse, dim=0) # [seq, bs, self.hidden_size]
            #print(hidden_seq_forward.shape, hidden_seq_reverse.shape)
            hidden_seq = torch.cat([hidden_seq_forward, hidden_seq_reverse], dim=2)
            #print(hidden_seq.shape)
            if self.batch_first:
                hidden_seq = hidden_seq.transpose(0, 1).contiguous()
            return hidden_seq

        else:
            return super().forward(x)


def lstm():
    model = CustomLSTM(100, 60, bidirectional=True)
    return model

class CustomLinear(nn.Module):
    def __init__(self, cin, cmid, cout):
        super(CustomLinear, self).__init__()
        self.fc1 = nn.Linear(cin, cmid)
        self.fc2 = nn.Linear(cmid, cout)
        self.weight = nn.Parameter(torch.ones(100, 200))
        print("weight", self.weight.shape)

    def forward(self, x, export_onnx=True):
        x = x.view(4, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.linear(x, self.weight)

        if not export_onnx:
            return x, (None, None)
        else:
            return x

def linear():
    model = CustomLinear(100, 100, 200)
    return model

def export(model):
    x = torch.rand(1, 1, 100)

    model.eval()
    y1, (hn, cn) = model(x, False)
    print(y1.shape)

    y2 = model(x, True)

    print("Export to onnx")
    dummy_input = x
    input_names = ["input_image"]
    output_names = []
    if isinstance(y2, list):
        for i, item in enumerate(y2):
            print(item.shape)
            output_names.append("embedding-{}".format(i))
    else:
        print(y2.shape)
        output_names = ["embedding"]

    torch.onnx.export(
        model,
        dummy_input,
        "output.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
    )
    print("Export to onnx done")

def expand(model):
    model.eval()
    x = torch.rand(2, 1, 100) # channel/ batch/ dim 

    print("Export to onnx")
    dummy_input = []
    input_names = []
    output_names = []

    for i in range(x.size(0)):
        input_names.append('input-{}'.format(i))
        dummy_input.append(x[i:i+1])
        #dummy_input.append(x[i])
        
    h_t, c_t = (torch.zeros(1, model.hidden_size).to(x.device), torch.zeros(1, model.hidden_size).to(x.device))
    dummy_input.append(h_t)
    dummy_input.append(c_t)
    input_names.append("h_t")
    input_names.append("c_t")

    y2 = model(dummy_input, True)

    if isinstance(y2, list):
        for i, item in enumerate(y2):
            print(item.shape)
            output_names.append("embedding-{}".format(i))
    else:
        print(y2.shape)
        output_names = ["embedding"]

    torch.onnx.export(
        model,
        dummy_input,
        "output.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
    )
    print("Export to onnx done")

if __name__ == "__main__":
    #export(linear())
    #export(lstm())
    expand(lstm())


