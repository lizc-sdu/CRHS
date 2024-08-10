import numpy as np
import random
import torch
import torch.nn as nn

import pickle

FType = torch.FloatTensor
LType = torch.LongTensor

class RegionData:

    def __init__(self, config):
        self.config = config
        self.city = config['city']

        poi_info_path = config[config['city']]['poi_info_path']
        poi_simi_path = config[config['city']]['poi_simi_path']
        a_path = config[config['city']]['poi_f_path']
        s_path = config[config['city']]['source_path']
        d_path = config[config['city']]['destina_path']

        self.a_m = np.load(a_path)
        self.s_m = np.load(s_path)
        self.d_m = np.load(d_path)
        self.poi_simi = np.load(poi_simi_path)
        self.poi_info = pickle.load(open(poi_info_path, "rb"))
        self.region_num = config[config['city']]['region_num']

        self.poi_most_similar_dict = pickle.load(open(config[config['city']]['poi_most_sim_path'], 'rb'))
        self.s_most_similar_dict = pickle.load(open(config[config['city']]['source_most_sim_path'], 'rb'))
        self.d_most_similar_dict = pickle.load(open(config[config['city']]['destina_most_sim_path'], 'rb'))
