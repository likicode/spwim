import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hashlib
import numpy as np
import itertools

def hard_pad2d(x, pad):
    def pad_side(idx):
        pad_len = max(pad - x.size(idx), 0)
        return [0, pad_len]
    padding = pad_side(3)
    padding.extend(pad_side(2))
    x = F.pad(x, padding)
    return x[:, :, :pad, :pad]

class VDPWIConvNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        def make_conv(n_in, n_out):
            conv = nn.Conv2d(n_in, n_out, 3, padding=1)
            conv.bias.data.zero_()
            nn.init.xavier_normal_(conv.weight)
            return conv
        self.conv1 = make_conv(12, 128)
        self.conv2 = make_conv(128, 164)
        self.conv3 = make_conv(164, 192)
        self.conv4 = make_conv(192, 192)
        self.conv5 = make_conv(192, 128)
        self.maxpool2 = nn.MaxPool2d(2, ceil_mode=True)
        self.dnn = nn.Linear(128, 128)
        self.output = nn.Linear(128, config['n_labels'])
        self.input_len = 32

    def forward(self, x):
        x = hard_pad2d(x, self.input_len)
        pool_final = nn.MaxPool2d(2, ceil_mode=True) if x.size(2) == 32 else nn.MaxPool2d(3, 1, ceil_mode=True)
        x = self.maxpool2(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.maxpool2(F.relu(self.conv3(x)))
        x = self.maxpool2(F.relu(self.conv4(x)))
        x = pool_final(F.relu(self.conv5(x)))
        x = F.relu(self.dnn(x.view(x.size(0), -1)))
        return F.log_softmax(self.output(x), 1)


class VDPWIResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        def make_conv(n_in, n_out):
            conv = nn.Conv2d(n_in, n_out, 3, padding=1)
            conv.bias.data.zero_()
            nn.init.xavier_normal_(conv.weight)
            return conv
        self.conv1 = make_conv(12, 128)
        self.conv2 = make_conv(128, 164)
        self.conv3 = make_conv(164, 192)
        self.conv4 = make_conv(192, 192)
        self.conv5 = make_conv(192, 128)
        self.maxpool2 = nn.MaxPool2d(2, ceil_mode=True)
        self.dnn = nn.Linear(128, 128)
        self.output = nn.Linear(128, config['n_labels'])
        self.input_len = 32
        self.resconv1 = nn.Conv2d(12, 164, kernel_size=3, stride=2, padding=1)
        self.resconv2 = nn.Conv2d(164, 192, kernel_size=3, stride=2, padding=1)


    def forward(self, x):
        x = hard_pad2d(x, self.input_len)
        pool_final = nn.MaxPool2d(2, ceil_mode=True) if x.size(2) == 32 else nn.MaxPool2d(3, 1, ceil_mode=True)
        # residual block 1
        old_x = x
        x = self.maxpool2(F.relu(self.conv1(x)))
        middle_x = self.conv2(x)
        new_x = self.resconv1(old_x) + middle_x
        x = self.maxpool2(F.relu(new_x))
        # residual block 2
        old_x = x
        x = self.maxpool2(F.relu(self.conv3(x)))
        middle_x = self.conv4(x)
        new_x = self.resconv2(old_x) + middle_x
        x = self.maxpool2(F.relu(new_x))
        x = pool_final(F.relu(self.conv5(x)))
        x = F.relu(self.dnn(x.view(x.size(0), -1)))
        return F.log_softmax(self.output(x), 1)

class ChildSumTreeLSTM(nn.Module):
    # This class is implemented by https://github.com/dasguptar/treelstm.pytorch/blob/master/treelstm/model.py
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.cache = {}

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs, raw_text):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs, raw_text)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: self.cache[id(x)], tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        tree_key = id(tree)
        try:
            self.cache[tree_key] = self.node_forward(inputs[tree.idx], child_c, child_h)
        except IndexError:
            self.cache[tree_key] = self.node_forward(inputs[len(inputs)-1], child_c, child_h)
            print("parent length uncompatible")

        return self.cache[tree_key][1]


class VDPWIModel(nn.Module):
    def __init__(self, dim, config, tree_file=None):
        super().__init__()
        self.arch = 'vdpwi'
        self.hidden_dim = config['rnn_hidden_dim']
        self.rnn = nn.LSTM(dim, self.hidden_dim, 1, batch_first=True)
        self.device = config['device']
        self.tree_file = tree_file
        self.treeLSTM = ChildSumTreeLSTM(300, config['rnn_hidden_dim'])
        self.wh = nn.Linear(config['rnn_hidden_dim'] * 2, config['rnn_hidden_dim'])
        self.wp = nn.Linear(config['rnn_hidden_dim'], config['n_labels'])

        self.rnn2 = nn.LSTM(dim + self.hidden_dim * 2, self.hidden_dim, 1, batch_first=True, bidirectional=True)
        self.rnn3 = nn.LSTM(dim + self.hidden_dim * 2 + self.hidden_dim * 2, self.hidden_dim ,1, batch_first=True, bidirectional=True)

        self.classifier_net = VDPWIResNet(config)


    def create_pad_cube(self, sent1, sent2):
        pad_cube = []
        sent1_lengths = [len(s.split()) for s in sent1]
        sent2_lengths = [len(s.split()) for s in sent2]
        max_len1 = max(sent1_lengths)
        max_len2 = max(sent2_lengths)

        for s1_length, s2_length in zip(sent1_lengths, sent2_lengths):
            pad_mask = np.ones((max_len1, max_len2))
            pad_mask[:s1_length, :s2_length] = 0
            pad_cube.append(pad_mask)

        pad_cube = np.array(pad_cube)
        return torch.from_numpy(pad_cube).float().to(self.device).unsqueeze(0)

    def compute_sim_cube(self, seq1, seq2):
        def compute_sim(prism1, prism2):
            prism1_len = prism1.norm(dim=3)
            prism2_len = prism2.norm(dim=3)

            dot_prod = torch.matmul(prism1.unsqueeze(3), prism2.unsqueeze(4))
            dot_prod = dot_prod.squeeze(3).squeeze(3)
            cos_dist = dot_prod / (prism1_len * prism2_len + 1E-8)
            l2_dist = ((prism1 - prism2).norm(dim=3))
            return torch.stack([dot_prod, cos_dist, l2_dist], 1)

        def compute_prism(seq1, seq2):
            prism1 = seq1.repeat(seq2.size(1), 1, 1, 1)
            prism2 = seq2.repeat(seq1.size(1), 1, 1, 1)
            prism1 = prism1.permute(1, 2, 0, 3).contiguous()
            prism2 = prism2.permute(1, 0, 2, 3).contiguous()
            return compute_sim(prism1, prism2)

        sim_cube = torch.Tensor(seq1.size(0), 12, seq1.size(1), seq2.size(1))
        sim_cube = sim_cube.to(self.device)
        seq1_f = seq1[:, :, :self.hidden_dim]
        seq1_b = seq1[:, :, self.hidden_dim:]
        seq2_f = seq2[:, :, :self.hidden_dim]
        seq2_b = seq2[:, :, self.hidden_dim:]
        sim_cube[:, 0:3] = compute_prism(seq1, seq2)
        sim_cube[:, 3:6] = compute_prism(seq1_f, seq2_f)
        sim_cube[:, 6:9] = compute_prism(seq1_b, seq2_b)
        sim_cube[:, 9:12] = compute_prism(seq1_f + seq1_b, seq2_f + seq2_b)
        return sim_cube

    def compute_tree_sim_cube(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)
        out = F.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out), dim=1)
        return out

    def compute_focus_cube(self, sim_cube, pad_cube):
        neg_magic = -10000
        pad_cube = pad_cube.repeat(12, 1, 1, 1)
        pad_cube = pad_cube.permute(1, 0, 2, 3).contiguous()
        sim_cube = neg_magic * pad_cube + sim_cube
        mask = torch.Tensor(*sim_cube.size()).to(self.device)
        mask[:, :, :, :] = 0.1

        def build_mask(index):
            max_mask = sim_cube[:, index].clone()
            for _ in range(min(sim_cube.size(2), sim_cube.size(3))):
                # values, indices: max for each one in a batch
                values, indices = torch.max(max_mask.view(sim_cube.size(0), -1), 1)
                # sim_cube.size(3): second sentence length
                row_indices = indices / sim_cube.size(3)
                col_indices = indices % sim_cube.size(3)
                row_indices = row_indices.unsqueeze(1)
                col_indices = col_indices.unsqueeze(1).unsqueeze(1)
                for i, (row_i, col_i, val) in enumerate(zip(row_indices, col_indices, values)):
                    if val < neg_magic / 2:
                        continue
                    mask[i, :, row_i, col_i] = 1
                    max_mask[i, row_i, :] = neg_magic
                    max_mask[i, :, col_i] = neg_magic
        build_mask(9)
        build_mask(10)
        focus_cube = mask * sim_cube * (1 - pad_cube)
        return focus_cube


    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        # The computation of word-to-word matrix part

        pad_cube = self.create_pad_cube(raw_sent1, raw_sent2)
        sent1 = sent1.permute(0, 2, 1).contiguous()
        sent2 = sent2.permute(0, 2, 1).contiguous()
        seq1f, _ = self.rnn(sent1)
        seq2f, _ = self.rnn(sent2)
        seq1b, _ = self.rnn(torch.cat(sent1.split(1, 1)[::-1], 1))
        seq2b, _ = self.rnn(torch.cat(sent2.split(1, 1)[::-1], 1))

        seq1_layer1 = torch.cat([seq1f, seq1b], 2)
        seq2_layer1 = torch.cat([seq2f, seq2b], 2)

        # For Layer 2
        sent1_layer2_in = torch.cat([sent1, seq1_layer1], dim=2)
        sent2_layer2_in = torch.cat([sent2, seq2_layer1], dim=2)
        seq1_layer2_out, _ = self.rnn2(sent1_layer2_in)
        seq2_layer2_out, _ = self.rnn2(sent2_layer2_in)

        # For Layer 3
        sent1_layer3_in = torch.cat([sent1, seq1_layer1, seq1_layer2_out], dim=2)
        sent2_layer3_in = torch.cat([sent2, seq2_layer1, seq2_layer2_out], dim=2)
        seq1_layer3_out, _ = self.rnn3(sent1_layer3_in)
        seq2_layer3_out, _ = self.rnn3(sent2_layer3_in)

        seq1 = seq1_layer3_out
        seq2 = seq2_layer3_out

        sim_cube = self.compute_sim_cube(seq1, seq2)

        truncate = self.classifier_net.input_len
        sim_cube = sim_cube[:, :, :pad_cube.size(2), :pad_cube.size(3)].contiguous()
        if truncate is not None:
            sim_cube = sim_cube[:, :, :truncate, :truncate].contiguous()
            pad_cube = pad_cube[:, :, :sim_cube.size(2), :sim_cube.size(3)].contiguous()
        focus_cube = self.compute_focus_cube(sim_cube, pad_cube)
        log_prob = self.classifier_net(focus_cube)

        tree_prob = []
        self.treeLSTM.cache = {}
        for i in range(len(raw_sent1)):
            _sent1 = raw_sent1[i]
            _sent2 = raw_sent2[i]
            sent1_len = len(_sent1.split())
            sent2_len = len(_sent2.split())
            # Load data from file
            ltree = self.tree_file[hashlib.sha224(_sent1.encode()).hexdigest()]
            rtree = self.tree_file[hashlib.sha224(_sent2.encode()).hexdigest()]

            linput = sent1[i][:sent1_len]
            rinput = sent2[i][:sent2_len]

            # re-organize hidden states in original sentence order
            ltree_state = self.treeLSTM(ltree, linput, _sent1.split())
            rtree_state = self.treeLSTM(rtree, rinput, _sent2.split())
            prob = self.compute_tree_sim_cube(ltree_state, rtree_state)
            tree_prob.append(prob)
        tree_prob = torch.stack(tree_prob).squeeze(1)

        return (log_prob, tree_prob)

