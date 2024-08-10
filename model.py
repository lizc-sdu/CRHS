import csv
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.utils import shuffle

from loss import *
import random
from tasks import predict_task


def pairwise_inner_product(mat_1, mat_2):
    n, m = mat_1.shape  # (180, 144)
    mat_expand = torch.unsqueeze(mat_2, 0)
    mat_expand = mat_expand.expand(n, n, m)
    mat_expand = mat_expand.permute(1, 0, 2)
    inner_prod = torch.mul(mat_expand, mat_1)
    inner_prod = torch.sum(inner_prod, axis=-1)
    return inner_prod


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=False):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class CrossLayer(nn.Module):
    def __init__(self, embedding_size):
        super(CrossLayer, self).__init__()
        self.alpha_poi = nn.Parameter(torch.tensor(0.5))  # 0.8
        self.alpha_d = nn.Parameter(torch.tensor(0.5))
        self.alpha_s = nn.Parameter(torch.tensor(0.5))
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_size, num_heads=4)

    def forward(self, poi_emb, s_emb, d_emb):
        stk_emb = torch.stack((poi_emb, d_emb, s_emb))
        fusion, _ = self.attn(stk_emb, stk_emb, stk_emb)
        poi_f = fusion[0] * self.alpha_poi + (1 - self.alpha_poi) * poi_emb
        d_f = fusion[1] * self.alpha_d + (1 - self.alpha_d) * d_emb
        s_f = fusion[2] * self.alpha_s + (1 - self.alpha_s) * s_emb
        return poi_f, s_f, d_f


class AttentionFusionLayer(nn.Module):
    def __init__(self, embedding_size):
        super(AttentionFusionLayer, self).__init__()
        self.q = nn.Parameter(torch.randn(embedding_size))
        self.fusion_lin = nn.Linear(embedding_size, embedding_size)

    def forward(self, poi_f, s_f, d_f):
        poi_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(poi_f)) * self.q, dim=1))
        s_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(s_f)) * self.q, dim=1))
        d_w = torch.mean(torch.sum(F.leaky_relu(
            self.fusion_lin(d_f)) * self.q, dim=1))

        w_stk = torch.stack((poi_w, s_w, d_w))
        w = torch.softmax(w_stk, dim=0)

        region_feature = w[0] * poi_f + w[1] * s_f + w[2] * d_f
        return region_feature


def device_as(t1, t2):
    return t1.to(t2.device)


class ADV(nn.Module):
    def __init__(self, config):
        super(ADV, self).__init__()
        self.poi_similarity = torch.tensor(np.load(config[config['city']]['poi_simi_path']), dtype=torch.float32)
        self.source_simi = torch.tensor(np.load(config[config['city']]['source_simi_path']), dtype=torch.float32)
        self.destina_simi = torch.tensor(np.load(config[config['city']]['destina_simi_path']), dtype=torch.float32)

        self.config = config
        self.tau = 0.5
        self.pos_eps = 0.6
        self.neg_eps = 0.6

        self.embedding_size = config['Autoencoder']['arch1'][-1]
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.ReLU()
        )

        self.regions_num = config[config['city']]['region_num']

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.poi_similarity = self.poi_similarity.to(device)
        self.source_simi = self.source_simi.to(device)
        self.destina_simi = self.destina_simi.to(device)
        self.projection.to(device)

    def forward(self, poi_f, s_f, d_f, poi_f_aug, s_f_aug, d_f_aug):
        hard_neg_s = self.generate_adv(s_f, self.source_simi, neg_eps=self.neg_eps)  # 生成tricker负样本
        hard_neg_d = self.generate_adv(d_f, self.destina_simi, neg_eps=self.neg_eps)
        hard_neg_poi = self.generate_adv(poi_f, self.poi_similarity, neg_eps=self.neg_eps)

        hard_pos_s = self.generate_cont_adv(s_f_aug,s_f,self.tau, self.pos_eps)  # 生成devcopy正样本
        hard_pos_d = self.generate_cont_adv(d_f_aug,d_f, self.tau, self.pos_eps)
        hard_pos_poi = self.generate_cont_adv(poi_f_aug,poi_f, self.tau, self.pos_eps)

        cont_s = cont_loss_(anchor=s_f, e_pos=s_f_aug, h_pos=hard_pos_s, h_neg=hard_neg_s)
        cont_d = cont_loss_(anchor=d_f, e_pos=d_f_aug, h_pos=hard_pos_d, h_neg=hard_neg_d)
        cont_poi = cont_loss_(anchor=poi_f, e_pos=poi_f_aug, h_pos=hard_pos_poi, h_neg=hard_neg_poi, )

        cont_loss = cont_s + cont_d + cont_poi
        loss = [cont_s.item(), cont_d.item(), cont_poi.item()]

        return cont_loss, poi_f, s_f, d_f, loss

    def generate_adv(self, Anchor_hiddens, lm_labels, neg_eps=0.5):
        Anchor_hiddens = Anchor_hiddens.detach()
        lm_labels = lm_labels.detach()
        Anchor_hiddens.requires_grad_(True)

        avg_Anchor_hiddens = self.projection(Anchor_hiddens)
        inner_prod = pairwise_inner_product(avg_Anchor_hiddens, avg_Anchor_hiddens)

        loss_adv = F.mse_loss(inner_prod, lm_labels).requires_grad_()
        loss_adv.backward()
        dec_grad = Anchor_hiddens.grad.detach()

        l2_norm = torch.norm(dec_grad, dim=-1)

        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)
        perturbed_Anc = Anchor_hiddens + neg_eps * dec_grad.detach()
        perturbed_Anc = perturbed_Anc  # [b,t,d]

        self.zero_grad()
        return perturbed_Anc

    def generate_cont_adv(self, STNPos_hiddens,
                          Anchor_hiddens,
                          tau, eps):  # positive
        STNPos_hiddens = STNPos_hiddens.detach()
        Anchor_hiddens = Anchor_hiddens.detach()
        STNPos_hiddens.requires_grad = True
        Anchor_hiddens.requires_grad = True

        avg_STNPos = self.projection(STNPos_hiddens)
        avg_Anchor = self.projection(Anchor_hiddens)

        cos = nn.CosineSimilarity(dim=-1)
        k, _ = avg_Anchor.size()
        mask = getmask(k)
        e_pos_logits = cos(avg_Anchor, avg_STNPos).reshape(k, 1)
        neg_logits = cos(avg_Anchor.unsqueeze(1),
                         avg_Anchor.unsqueeze(0))[mask].reshape(k, -1)
        logits = torch.cat((e_pos_logits, neg_logits), dim=1) / tau
        pos_label = torch.Tensor([1 for _ in range(e_pos_logits.size(1))]).to(e_pos_logits.device).long()
        neg_label = torch.Tensor([0 for _ in range(neg_logits.size(1))]).to(neg_logits.device).long()
        labels = torch.cat([pos_label, neg_label])
        cont_crit = nn.CrossEntropyLoss()
        loss_cont_adv = cont_crit(logits, labels)
        loss_cont_adv.backward()

        dec_grad = Anchor_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)
        perturb_Anchor_hidden = Anchor_hiddens + eps * dec_grad

        # step 2
        perturb_Anchor_hidden = perturb_Anchor_hidden.detach()
        perturb_Anchor_hidden.requires_grad = True
        Anchor_hiddens = Anchor_hiddens.detach()

        inner_prod = pairwise_inner_product(Anchor_hiddens, Anchor_hiddens)
        preturb_prod = pairwise_inner_product(perturb_Anchor_hidden, Anchor_hiddens)

        loss_adv = F.mse_loss(inner_prod, preturb_prod).requires_grad_()
        loss_adv.backward()
        dec_grad = perturb_Anchor_hidden.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)

        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)
        perturb_Anchor_hidden = perturb_Anchor_hidden - eps * dec_grad
        return perturb_Anchor_hidden


class CRHS():
    """ReCP module."""

    def __init__(self,
                 config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        self._config = config
        self.city = config['city']
        self.embedding_size = config['Autoencoder']['arch1'][-1]

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        # View-specific autoencoders
        config['Autoencoder']['arch1'][0] = config[config['city']]['category_num']
        config['Autoencoder']['arch2'][0] = config[config['city']]['region_num']
        self.autoencoder_a = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                         config['Autoencoder']['batchnorm'])
        self.autoencoder_s = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                         config['Autoencoder']['batchnorm'])
        self.autoencoder_d = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                         config['Autoencoder']['batchnorm'])

        self.cross_layer = CrossLayer(self.embedding_size)
        self.fusion_layer = AttentionFusionLayer(self.embedding_size)

        # adv Contrastive
        self.adv = ADV(config)

        self.poi_most_similar_dict = pickle.load(open(config[config['city']]['poi_most_sim_path'], 'rb'))
        self.s_most_similar_dict = pickle.load(open(config[config['city']]['source_most_sim_path'], 'rb'))
        self.d_most_similar_dict = pickle.load(open(config[config['city']]['destina_most_sim_path'], 'rb'))

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder_a.to(device)
        self.autoencoder_s.to(device)
        self.autoencoder_d.to(device)
        self.adv.to_device(device)
        self.cross_layer.to(device)
        self.fusion_layer.to(device)

    def mix_aug(self, emb, similar_dict, alpha, seed=42):
        random.seed(seed)

        similar_matrix = torch.zeros_like(emb)
        for key, value in similar_dict.items():
            aug = value
            similar_matrix[key, :] = emb[aug, :]

        new_poi_f = emb + (1 - alpha) * similar_matrix

        return new_poi_f

    def train(self, config, xs,
              optimizer, scheduler, device):
        """Training the model.

            Args:
              config: parameters which defined in configure.py.
              redata: data augmentation
              xs: data matrix A,S,D
              optimizer: adam is used in our experiments
              scheduler: learning rate decay
              device: to cuda if gpu is used

        """

        for epoch in range(config['training']['epoch']):
            poi_emb = self.autoencoder_a.encoder(xs[0])
            s_emb = self.autoencoder_s.encoder(xs[1])
            d_emb = self.autoencoder_d.encoder(xs[2])

            alpha_n = np.random.uniform(low=0.9, high=1., size=poi_emb.shape[0])
            alpha = torch.tensor(alpha_n, dtype=torch.float32).view(-1, 1).to(poi_emb.device)
            poi_emb_aug = self.mix_aug(poi_emb, similar_dict=self.poi_most_similar_dict, alpha=alpha,
                                       seed=epoch)
            s_emb_aug = self.mix_aug(s_emb, similar_dict=self.s_most_similar_dict, alpha=alpha,
                                     seed=epoch)
            d_emb_aug = self.mix_aug(d_emb, similar_dict=self.d_most_similar_dict, alpha=alpha,
                                     seed=epoch)

            # Contrastive Loss
            poi_f, s_f, d_f = self.cross_layer(poi_emb, s_emb, d_emb)
            poi_f_aug, s_f_aug, d_f_aug = self.cross_layer(poi_emb_aug, s_emb_aug, d_emb_aug)

            cont_loss, poi_f, s_f, d_f, loss = self.adv(poi_f=poi_f, s_f=s_f, d_f=d_f,
                                                                                poi_f_aug=poi_f_aug, s_f_aug=s_f_aug,
                                                                                d_f_aug=d_f_aug)

            region_emb = self.fusion_layer(poi_f, s_f, d_f)
            # Reconstruction Loss
            recon1 = F.mse_loss(self.autoencoder_a.decoder(region_emb), xs[0])  # 样本的平均MSE损失
            recon2 = F.mse_loss(self.autoencoder_s.decoder(region_emb), xs[1])
            recon3 = F.mse_loss(self.autoencoder_d.decoder(region_emb), xs[2])

            loss = cont_loss + (config['training']['lambda1'] * recon1 + recon2 + recon3) * config['training'][
                'lambda2']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)

            # evalution
            if (epoch + 1) % config['print_num'] == 0 or epoch == 0:
                print(
                    "Epoch : {:.0f}/{:.0f} ===>cont Loss = {:.4f}, ===>recon1 Loss = {}, ===>recon1 Loss = {},===>recon3 Loss = {},".format(
                        (epoch + 1), config['training']['epoch'], cont_loss, recon1, recon2, recon3, ))
                pre1, pre2, pre3, region_emb = self.test(config,xs[0], xs[1], xs[2], epoch + 1)
                print(optimizer.param_groups[0]['lr'])

        return pre1, pre2, pre3

    def test(self, config,attribute_m, source_matrix, destina_matrix, epoch):
        with torch.no_grad():
            self.autoencoder_a.eval(), self.autoencoder_s.eval(), self.autoencoder_d.eval()
            self.adv.eval(),

            latent_a = self.autoencoder_a.encoder(attribute_m)
            latent_s = self.autoencoder_s.encoder(source_matrix)
            latent_d = self.autoencoder_d.encoder(destina_matrix)
            poi, s, d = self.cross_layer(latent_a, latent_s, latent_d)
            region_emb = self.fusion_layer(poi, s, d)

            region_emb = region_emb.cpu().numpy()
            print('==========================================>')
            print("co2_sum Prediction")
            prediction1 = predict_task(emb=region_emb, label=np.load(self._config[self.city]['co2_sum_path']),
                                       name='co2')
            print('==========================================>')
            print("gdp_sum Prediction")
            prediction2 = predict_task(emb=region_emb, label=np.load(self._config[self.city]['gdp_sum_path']),
                                       name='gdp')
            print('==========================================>')
            print("population_sum Prediction")
            prediction3 = predict_task(emb=region_emb, label=np.load(self._config[self.city]['population_sum_path']),
                                       name='popu')

            self.autoencoder_a.train(), self.autoencoder_s.train(), self.autoencoder_d.train()
            self.adv.train(),

        return prediction1, prediction2, prediction3, region_emb