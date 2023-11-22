""" CIBN model"""
import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
from sim_Reason import Reason_i2t, Reason_t2i
from pytorch_pretrained_bert import BertModel

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class WordEmbeddings(nn.Module):
    def __init__(self):
        super(WordEmbeddings, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)[0]
        x = x[-4:]
        x = torch.stack(x, dim=1)
        x = torch.sum(x, dim=1)
        return x

class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderImagePrecomp, self).load_state_dict(new_state)

# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # bert embedding
        self.embedd = WordEmbeddings()

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=use_bi_gru)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # bert_embedding
        x = self.embedd(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, h_n = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] +
                       cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len


class EncoderText_positive(nn.Module):

    def __init__(self, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText_positive, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # bert embedding
        self.embedd = WordEmbeddings()

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=use_bi_gru)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # bert_embedding
        x = self.embedd(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] +
                       cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.Rank_Loss = opt.Rank_Loss

    def forward(self, scores):
        #self.opt.count_num = self.opt.count_num + 1
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.opt.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.opt.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the DynamciTopK, maximum or all violating negative for each query

        if self.Rank_Loss == 'DynamicTopK_Negative':
            topK = int((cost_s > 0.).sum() / (cost_s.size(0) + 0.00001) + 1)
            cost_s, index1 = torch.sort(cost_s, descending=True, dim=-1)
            cost_im, index2 = torch.sort(cost_im, descending=True, dim=0)

            return cost_s[:, 0:topK].sum() + cost_im[0:topK, :].sum()

        elif self.Rank_Loss == 'Hardest_Negative':
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

            return cost_s.sum() + cost_im.sum()

        else:
            return cost_s.sum() + cost_im.sum()

class ContrastiveSemanticLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0.1):
        super(ContrastiveSemanticLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.Rank_Loss = 'DynamicTopK_Negative'
        self.fc1 = nn.Linear(opt.embed_size, opt.embed_size // 2)
        self.fc2 = nn.Linear(opt.embed_size, opt.embed_size // 2)
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, init_features, target_features):
        init_features = torch.mean(init_features, dim=1).squeeze()
        target_features = torch.mean(target_features, dim=1).squeeze()
        init_features = self.fc1(init_features)
        init_features = init_features.unsqueeze(1)
        target_features = self.fc2(target_features)
        target_features = target_features.unsqueeze(0)
        scores = torch.cosine_similarity(init_features, target_features, dim=-1)
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column

        cost_s = (self.opt.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row

        cost_im = (self.opt.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the DynamciTopK, maximum or all violating negative for each query

        if self.Rank_Loss == 'DynamicTopK_Negative':
            topK = int((cost_s > 0.).sum() / (cost_s.size(0) + 0.00001) + 1)
            cost_s, index1 = torch.sort(cost_s, descending=True, dim=-1)
            cost_im, index2 = torch.sort(cost_im, descending=True, dim=0)

            return cost_s[:, 0:topK].sum() + cost_im[0:topK, :].sum()

        elif self.Rank_Loss == 'Hardest_Negative':
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

            return cost_s.sum() + cost_im.sum()

        else:
            return cost_s.sum() + cost_im.sum()

    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(ContrastiveSemanticLoss, self).load_state_dict(new_state)


class MultiLabelClassify(nn.Module):

    def __init__(self, opt):
        super(MultiLabelClassify, self).__init__()
        self.numLabel = opt.numLabel
        self.tanh = nn.Tanh()
        self.out_1 = nn.Linear(1024, 1000)
        self.out_2 = nn.Linear(1000, 1000)
        # self.loss = nn.MultiLabelSoftMarginLoss(reduction='mean')

        self.loss = nn.BCEWithLogitsLoss()
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, weighted_features, traget_labels):
        weighted_features = self.tanh(self.out_1(weighted_features))
        weighted_features = self.out_2(weighted_features)
        loss = self.loss(weighted_features, traget_labels)
        return loss

    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(MultiLabelClassify, self).load_state_dict(new_state)

class MultiLabelClassify1(nn.Module):

    def __init__(self, opt):
        super(MultiLabelClassify1, self).__init__()
        self.numLabel = opt.numLabel
        self.tanh = nn.Tanh()
        self.out_1 = nn.Linear(1024, 1000)
        self.out_2 = nn.Linear(1000, 1000)
        # self.loss = nn.MultiLabelSoftMarginLoss(reduction='mean')

        self.loss = nn.BCEWithLogitsLoss()
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, weighted_features, traget_labels):
        weighted_features = self.tanh(self.out_1(weighted_features))
        weighted_features = self.out_2(weighted_features)
        loss = self.loss(weighted_features, traget_labels)
        return loss

    def load_state_dict(self, state_dict):

        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(MultiLabelClassify1, self).load_state_dict(new_state)


# Multi-task Learning Network
class MTLN(object):

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip

        # Encoding visual and textual features
        self.img_enc = EncoderImagePrecomp(opt.img_dim, opt.embed_size, opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.word_dim, opt.embed_size, opt.num_layers, use_bi_gru=opt.bi_gru, no_txtnorm=opt.no_txtnorm)
        self.po_txt_enc = EncoderText_positive(opt.word_dim, opt.embed_size, opt.num_layers, use_bi_gru=opt.bi_gru, no_txtnorm=opt.no_txtnorm)

        # Matching
        self.i2t_match = Reason_i2t(opt.feat_dim, opt.hid_dim, opt.out_dim, opt)
        self.t2i_match = Reason_t2i(opt.feat_dim, opt.hid_dim, opt.out_dim, opt)
        self.label_criterion = MultiLabelClassify(opt)
        self.label_criterion1 = MultiLabelClassify1(opt)

        self.criterion = ContrastiveLoss(opt=opt, margin=opt.margin)
        self.semanticCrit = ContrastiveSemanticLoss(opt=opt, margin=0.1)

        if torch.cuda.is_available():
            self.semanticCrit.cuda()
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.po_txt_enc.cuda()
            self.i2t_match.cuda()
            self.t2i_match.cuda()
            self.label_criterion.cuda()
            self.label_criterion1.cuda()
            cudnn.benchmark = True

        params = list(self.img_enc.parameters())
        params += list(self.txt_enc.parameters())
        params += list(self.i2t_match.parameters())
        params += list(self.t2i_match.parameters())
        params += list(self.label_criterion.parameters())
        params += list(self.po_txt_enc.parameters())
        params += list(self.semanticCrit.parameters())
        params += list(self.label_criterion1.parameters())
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0
        self.opt = opt

    def state_dict(self):
        state_dict = [
                      self.img_enc.state_dict(),
                      self.txt_enc.state_dict(),
                      self.po_txt_enc.state_dict(),
                      self.i2t_match.state_dict(),
                      self.t2i_match.state_dict(),
                      self.label_criterion.state_dict(),
                      self.semanticCrit.state_dict(),
                      self.label_criterion1.state_dict()
                      ]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.po_txt_enc.load_state_dict(state_dict[2])
        self.i2t_match.load_state_dict(state_dict[3])
        self.t2i_match.load_state_dict(state_dict[4])
        self.label_criterion.load_state_dict(state_dict[5])
        self.semanticCrit.load_state_dict(state_dict[6])
        self.label_criterion1.load_state_dict(state_dict[7])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.po_txt_enc.train()
        self.i2t_match.train()
        self.t2i_match.train()
        self.label_criterion.train()
        self.semanticCrit.train()
        self.label_criterion1.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.po_txt_enc.eval()
        self.i2t_match.eval()
        self.t2i_match.eval()
        self.label_criterion.eval()
        self.semanticCrit.eval()
        self.label_criterion1.eval()

    def forward_emb(self, images, captions, lengths, po_captions, po_lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        po_captions = Variable(po_captions, volatile=volatile)

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            po_captions = po_captions.cuda()

        # Encoding:
        img_emb = self.img_enc(images)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        po_cap_emb, po_cap_lens = self.po_txt_enc(po_captions, po_lengths)

        return img_emb, cap_emb, cap_lens, po_cap_emb, po_cap_lens

    def forward_sim(self, img_emb, cap_emb, cap_lens):
        if self.opt.Matching_direction == 'i2t':
            i2t_scores, visual_features, textual_features = self.i2t_match(img_emb, cap_emb, cap_lens, self.opt)
            return i2t_scores, visual_features, textual_features
        elif self.opt.Matching_direction == 't2i':
            t2i_scores, visual_features, textual_features = self.t2i_match(img_emb, cap_emb, cap_lens, self.opt)
            return t2i_scores, visual_features, textual_features
        else:
            t2i_scores, visual_features, textual_features = self.t2i_match(img_emb, cap_emb, cap_lens, self.opt)
            i2t_scores, visual_features, textual_features = self.i2t_match(img_emb, cap_emb, cap_lens, self.opt)
            return t2i_scores + i2t_scores

    def forward_loss(self, scores, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(scores)
        # self.logger.update('retrieval_loss', loss.item())
        return loss

    def train_emb(self, images, captions, lengths, labels, ids, po_captions, po_lengths, *args):
        """One training step given images and captions.
        """

        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        img_emb, cap_emb, cap_lens, po_cap_emb, po_cap_lens = self.forward_emb(
            images, captions, lengths, po_captions, po_lengths)

        scores, visual_features, textual_features = self.forward_sim(img_emb, cap_emb, cap_lens)
        labels = labels.cuda()
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss1 = self.forward_loss(scores)
        loss2 = self.semanticCrit(po_cap_emb, cap_emb)
        loss3 = self.label_criterion(visual_features, labels)
        loss4 = self.label_criterion1(textual_features, labels)

        self.logger.update('reLoss', loss1.item())
        self.logger.update('seLoss', loss2.item())
        self.logger.update('imLLoss', loss3.item())
        self.logger.update('teLLoss', loss4.item())

        loss = loss1 + loss2+ loss3+ loss4
        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

