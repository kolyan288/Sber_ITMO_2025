import pandas as pd
import numpy as np
import pickle as pi
import random
import argparse
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import warnings
warnings.filterwarnings('ignore')


class Classifier:

    def __init__(self):

        work_dir = 'ml_models/checkpoints/'
        
        self.gbc_unobstructed = pi.load(open(f'{work_dir}gbc_Unobstructed.pkl', 'rb'))
        self.gbc_orthogonal_planes = pi.load(open(f'{work_dir}gbc_Orthogonal planes.pkl', 'rb'))
        self.gbc_h_bond_bridging = pi.load(open(f'{work_dir}gbc_H-bonds bridging.pkl', 'rb'))

        self.features_unobstructed = open(f'{work_dir}result_features/features_Unobstructed.txt','r').read().split('\n')
        self.features_orthogonal_planes = open(f'{work_dir}result_features/features_Orthogonal planes.txt','r').read().split('\n')
        self.features_h_bond_bridging = open(f'{work_dir}result_features/features_H-bonds bridging.txt','r').read().split('\n')

        self.min_max_scaler = pi.load(open(f'{work_dir}/min_max_scaler.pkl', 'rb'))

        self.feature_num = 43

        self.desired_value = 1


    def get_drug_descriptors(self, drug):

        descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
        get_descriptors = rdMolDescriptors.Properties(descriptor_names)
        num_descriptors = len(descriptor_names)
        descriptors_set = np.empty((0, num_descriptors), float)

        drug_obj = Chem.MolFromSmiles(drug)
        descriptors = np.array(get_descriptors.ComputeProperties(drug_obj)).reshape((-1,num_descriptors))
        descriptors_set = np.append(descriptors_set, descriptors, axis=0)
        drug_table = pd.DataFrame(descriptors_set, columns=descriptor_names)

        return drug_table

    def get_coformer_descriptors(self, coformer):

        descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
        descriptor_coformer_names = [name + '.1' for name in descriptor_names]
        get_descriptors = rdMolDescriptors.Properties(descriptor_names)
        num_descriptors = len(descriptor_names)
        descriptors_set = np.empty((0, num_descriptors), float)

        gen_obj = Chem.MolFromSmiles(coformer)
        descriptors = np.array(get_descriptors.ComputeProperties(gen_obj)).reshape((-1,num_descriptors))
        descriptors_set = np.append(descriptors_set, descriptors, axis=0)
        gen_table = pd.DataFrame(descriptors_set, columns=descriptor_coformer_names)

        return gen_table

    def create_clf_dataframe(self, drug, generated_coformers):

        drug_table = self.get_drug_descriptors(drug)
        gen_table = self.get_coformer_descriptors(generated_coformers)

        clf_data = drug_table.merge(gen_table, how='cross')

        list_of_params = clf_data.columns.tolist()

        for feat_idx in range(self.feature_num):
            clf_data[list_of_params[feat_idx] + '_sum'] = \
                clf_data.iloc[:, feat_idx] + clf_data.iloc[:, feat_idx + self.feature_num]
            clf_data[list_of_params[feat_idx] + '_mean'] = \
                (clf_data.iloc[:, feat_idx] + clf_data.iloc[:, feat_idx + self.feature_num]) / 2

        clf_data_scaled = pd.DataFrame(self.min_max_scaler.transform(clf_data), columns=clf_data.columns)

        return clf_data_scaled

    def predict_properties(self, drug, coformer, properties):
        clf_data = self.create_clf_dataframe(drug, coformer)
        
        try:
            output = {}
            if 'unobstructed' in properties:
                clf_data_unobstructed = pd.DataFrame(clf_data[self.features_unobstructed])
                clf_prediction_unobstructed = self.gbc_unobstructed.predict(clf_data_unobstructed)
                output['unobstructed'] = clf_prediction_unobstructed[0]
            if 'orthogonal_planes' in properties:
                clf_data_orthogonal_planes = pd.DataFrame(clf_data[self.features_orthogonal_planes])
                clf_prediction_orthogonal_planes = self.gbc_orthogonal_planes.predict(clf_data_orthogonal_planes)
                output['orthogonal_planes'] = clf_prediction_orthogonal_planes[0]
            if 'h_bond_bridging' in properties:
                clf_data_h_bond_bridging = pd.DataFrame(clf_data[self.features_h_bond_bridging])
                clf_prediction_h_bond_bridging = self.gbc_h_bond_bridging.predict(clf_data_h_bond_bridging)
                output['h_bond_bridging'] = clf_prediction_h_bond_bridging[0]
            
            return output
        except Exception as e:
            print(e)
    
    def predict_properties_proba(self, drug, coformer, properties):
        clf_data = self.create_clf_dataframe(drug, coformer)
        
        try:
            output = {}
            if 'unobstructed' in properties:
                clf_data_unobstructed = pd.DataFrame(clf_data[self.features_unobstructed])
                clf_prediction_unobstructed = self.gbc_unobstructed.predict_proba(clf_data_unobstructed)
                output['unobstructed'] = clf_prediction_unobstructed[0][1]
            if 'orthogonal_planes' in properties:
                clf_data_orthogonal_planes = pd.DataFrame(clf_data[self.features_orthogonal_planes])
                clf_prediction_orthogonal_planes = self.gbc_orthogonal_planes.predict_proba(clf_data_orthogonal_planes)
                output['orthogonal_planes'] = clf_prediction_orthogonal_planes[0][1]
            if 'h_bond_bridging' in properties:
                clf_data_h_bond_bridging = pd.DataFrame(clf_data[self.features_h_bond_bridging])
                clf_prediction_h_bond_bridging = self.gbc_h_bond_bridging.predict_proba(clf_data_h_bond_bridging)
                output['h_bond_bridging'] = clf_prediction_h_bond_bridging[0][0]
            
            return output
        except Exception as e:
            print(e)
        