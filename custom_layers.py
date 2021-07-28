
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
    return sigmoid_func(2*x) * 2 - 1


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
        self.tanh = partial(tanh, sigmoid_func=self.sigmoid)

    def forward(self, x, exporting_onnx=True):
        if exporting_onnx:
            assert self.num_layers == 1
            if self.batch_first:
                bs, seq, _ = x.size() 
                inputs = x.split(seq, dim=1)
            else:
                bs, seq, _ = (x.size(1), x.size(0), x.size(2))
                inputs = x.split(seq, dim=0)
            #pdb.set_trace()

            h_t, c_t = (torch.zeros(bs, 1, self.hidden_size).to(x.device), torch.zeros(bs, 1, self.hidden_size).to(x.device))
            hidden_seq_forward = []
            for t in range(seq):
                x_t = inputs[0][t]
                print("f", t, x_t.shape, self.weight_ihf_0.shape, h_t.shape)
                i_t = F.linear(x_t, self.weight_ihf_0) + self.bias_ihf_0 + F.linear(h_t, self.weight_hhf_0) + self.bias_hhf_0
                f_t = F.linear(x_t, self.weight_ihf_1) + self.bias_ihf_1 + F.linear(h_t, self.weight_hhf_1) + self.bias_hhf_1
                g_t = F.linear(x_t, self.weight_ihf_2) + self.bias_ihf_2 + F.linear(h_t, self.weight_hhf_2) + self.bias_hhf_2
                o_t = F.linear(x_t, self.weight_ihf_3) + self.bias_ihf_3 + F.linear(h_t, self.weight_hhf_3) + self.bias_hhf_3
                i_t = self.sigmoid(i_t)
                f_t = self.sigmoid(f_t)
                g_t = self.tanh(g_t)
                o_t = self.sigmoid(o_t)
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * self.tanh(c_t)
                hidden_seq_forward.append(h_t.unsqueeze(0))

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

            return hidden_seq_forward
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
    x = torch.rand(4, 1, 100)

    model.eval()
    y1, (hn, cn) = model(x, False)
    print(y1.shape)

    y2 = model(x, True)
    #pdb.set_trace()

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
        keep_initializers_as_inputs=True,
    )
    print("Export to onnx done")
    #pdb.set_trace()

def linear():
    model = nn.Sequantial()

if __name__ == "__main__":
    lstm()

