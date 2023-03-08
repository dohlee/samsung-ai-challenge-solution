import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import math
import itertools

from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdMolTransforms import CanonicalizeConformer
from rdkit.Chem.rdMolTransforms import GetAngleDeg, GetDihedralDeg
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import Descriptors3D
from rdkit.Chem.rdmolfiles import CanonicalRankAtoms
from openbabel import pybel as pb
RDLogger.DisableLog('rdApp.*') # Suppress warnings from rdkit

from sac2021.const import fp, ignored_uids, max_n_atoms, DUMMY_ATOM, a2i, hyb2int, bondtype2int

class SACData(Dataset):
    # self.version = '0.0.1'
    # self.version = '0.0.2'
    # self.version = '0.0.3' # normalize xyz features to have mu=0, std=1
    # version = '0.0.4' # divide input features by sqrt(d_model)
    # version = '0.0.5' # divide input features by sqrt(d_model)
    # version = '0.0.6' # pairwise distance
    # version = '0.0.7' # graph adjacency
    # version = '0.0.8' # use dev data for train
    # version = '0.0.9' # donor & acceptor
    # version = '0.0.10' # molecular feature

    version = '0.1.0' # Finalized.

    def __init__(self, idx=None, augs=[], meta=fp['train_meta'], data=fp['train_data'], pretrain=False):
        super(SACData, self).__init__()
        if pretrain:
            meta = fp['pretrain_meta']
            data = fp['pretrain_data']
        self.data_dir = data

        self.meta = pd.read_csv(meta)
        self.meta = self.meta[~self.meta.uid.isin(ignored_uids)].reset_index(drop=True)

        if idx is not None:
            self.meta = self.meta.loc[idx].reset_index(drop=True)
        self.meta = self.meta.to_records()

        # Needed for donor/acceptor features.
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self.factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

        self.augs = augs
        self.pretrain = pretrain

    def __getitem__(self, i):
        m = self.meta[i]

        mol = Chem.SDMolSupplier(f'{self.data_dir}/{m.uid}.sdf')[0]
        pb_mol = pb.readstring("smi", Chem.MolToSmiles(mol))
        conformer = mol.GetConformer()

        n_atoms = mol.GetNumAtoms()

        # Atom index features.
        atom_idx = torch.tensor(
            [self._get_atom_index(a.GetSymbol()) for a in mol.GetAtoms()] +
            [DUMMY_ATOM] + # note that a dummy atom is added after all atoms in the molecule.
            [-1] * (max_n_atoms - (n_atoms + 1)),
            dtype=torch.long
        )

        # Atom coordinate features. (max_n_atoms x 3)
        xyz = self._get_atom_xyz(mol, pad=max_n_atoms - n_atoms)

        # Mask for self-attention. 1 means masked. (1 x max_n_atoms x max_n_atoms)
        mask = torch.ones([max_n_atoms, max_n_atoms], dtype=bool)
        mask_1d = (atom_idx != -1)

        mask[mask_1d] = 0
        mask[:, mask_1d] = 0
        mask[~mask_1d] = 1
        mask[:, ~mask_1d] = 1

        mask = mask.unsqueeze(0)

        # Pairwise distance feature. (max_n_atoms x max_n_atoms)
        pdist = torch.cdist(xyz, xyz, p=2.0)
        
        # set distance to dummy node as 1e6
        pdist[n_atoms] = 1e6
        pdist[:, n_atoms] = 1e6

        # Adjacency matrix. (max_n_atoms x max_n_atoms)
        adj = torch.zeros([max_n_atoms, max_n_atoms], dtype=torch.float32)
        _adj = GetAdjacencyMatrix(mol)
        adj[:n_atoms, :n_atoms] = torch.tensor(_adj, dtype=torch.float32)

        # Angles and 2-adjacency. (max_n_atoms x max_n_atoms)
        angle = torch.zeros([max_n_atoms, max_n_atoms], dtype=torch.float32)
        adj_2 = _adj @ _adj

        for a in range(n_atoms):
            for b in range(a + 1, n_atoms):
                if adj_2[a, b] != 0:  # a, b are two-hop neighbors, so we find common neighbor and measure angles. 
                    common_neighbor = (_adj[a] * _adj[b]).argmax()
                    angle[a, b] = angle[b, a] = GetAngleDeg(conformer, a, int(common_neighbor), b) / 90.0 * math.pi

        # Torsion angles. (max_n_atoms x max_n_atoms)
        torsion_angle = torch.zeros([max_n_atoms, max_n_atoms], dtype=torch.float32)
        adj_3 = (adj_2 @ _adj > 0).astype(int) - _adj
        
        for a in range(n_atoms):
            for b in range(a + 1, n_atoms):
                if adj_3[a, b] != 0:  # a, b are three-hop neighbors.
                    a_neighbors = np.where(_adj[a] == 1)[0]
                    b_neighbors = np.where(_adj[b] == 1)[0]

                    for a_n, b_n in itertools.product(a_neighbors, b_neighbors):
                        if mol.GetBondBetweenAtoms(int(a_n), int(b_n)) is not None:
                            torsion_angle[a, b] = torsion_angle[b, a] = GetDihedralDeg(conformer, a, int(a_n), int(b_n), b) / 90.0 * math.pi
                            break
        
        # Aggregate angle features into single tensor. (max_n_atoms x max_n_atoms x 2)
        angle = torch.cat([angle.unsqueeze(2), torsion_angle.unsqueeze(2)], dim=2)

        # Bond types and conjugation.
        # Bond types (max_n_atoms x max_n_atoms x 5)
        # Conjugation (max_n_atoms x max_n_atoms)
        bond_types = torch.zeros([max_n_atoms, max_n_atoms, 5], dtype=torch.float32)
        conjugation = torch.zeros([max_n_atoms, max_n_atoms], dtype=torch.float32)
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt_idx = bondtype2int[bond.GetBondType()]
            bond_types[a, b, bt_idx] = bond_types[b, a, bt_idx] = 1.0
            conjugation[a, b] = conjugation[b, a] = float(bond.GetIsConjugated())

        # Symmetry class. (max_n_atoms x max_n_atoms)
        sym_groups = np.array(CanonicalRankAtoms(mol, breakTies=False))
        sym_mat = torch.zeros([max_n_atoms, max_n_atoms], dtype=torch.bool)
        for i, group in enumerate(sym_groups):
            sym_mat[i, [idx for idx in np.where(sym_groups == group)[0] if idx != i]] = 1

        # One-hot atom hybridization (7 types) (max_n_atoms x 7)
        hyb = self._get_atom_hybridization(mol)
        # Is an atom contained in aromatic ring? (max_n_atoms x 1)
        is_aromatic = self._get_atom_is_aromatic(mol)
        # Formal charge. (max_n_atoms x 1)
        formal_charge = self._get_formal_charge(mol)
        # Total_numh. (max_n_atoms x 1)
        total_numh = self._get_total_num_H(mol)
        # Total_valence. (max_n_atoms x 1)
        total_valence = self._get_total_valence(mol)
        # One-hot label for donor & acceptor status. (max_n_atoms x 2)
        donor_acceptor = self._get_donor_acceptor(mol)

        spin = torch.zeros([max_n_atoms, 2])
        spin[:, pb_mol.spin - 1] = 1 # Spin multiplicity

        # Molecule-level features.
        npr1 = torch.ones([max_n_atoms, 1]) * Descriptors3D.NPR1(mol) # NPR1
        npr2 = torch.ones([max_n_atoms, 1]) * Descriptors3D.NPR2(mol) # NPR2

        fp2 = pb_mol.calcfp(fptype='FP2')
        fp = torch.zeros([1021], dtype=torch.float)
        for b in fp2.bits:
            fp[b - 1] = 1.0

        feat = torch.cat([
            is_aromatic,    # dim=1
            formal_charge,  # dim=1
            total_numh,     # dim=1
            total_valence,  # dim=1
            npr1,           # dim=1
            npr2,           # dim=1
        ], dim=-1)          # total dim=6

        item = {
            'mask': mask.bool(),
            'out_mask': mask_1d.bool(),
            'n_atoms': mask_1d.sum(),
            'atom_idx': atom_idx.long(),
            'hyb': hyb,
            'donac': donor_acceptor,
            'spin': spin,
            'pdist': pdist,
            'sym': sym_mat,
            'angle': angle,
            'adj': adj,
            'feat': feat,
            'fp': fp,
        }

        if self.pretrain:
            item.update({
                'homo': m['homo'],
                'lumo': m['lumo'],
                'target': m['lumo'] - m['homo'],
            })
        else:
            item.update({
                's1': m['S1_energy(eV)'],
                't1': m['T1_energy(eV)'],
                'target': m['S1_energy(eV)'] - m['T1_energy(eV)'],
            })

        return item

    def _get_donor_acceptor(self, mol):
        v = torch.zeros([max_n_atoms, 2])
        feats = self.factory.GetFeaturesForMol(mol)
        for feat in feats:
            if feat.GetFamily() == 'Donor':
                for a_idx in feat.GetAtomIds():
                    v[a_idx, 0] = 1

            if feat.GetFamily() == 'Acceptor':
                for a_idx in feat.GetAtomIds():
                    v[a_idx, 1] = 1
        
        return v

    def _get_total_valence(self, mol):
        # valence = 0, 1, 2, 3, 4, 5, 6
        v = torch.zeros([max_n_atoms, 1])
        for i, atom in enumerate(mol.GetAtoms()):
            v[i] = atom.GetTotalValence() / 1.5 - 2.
        return v

    def _get_total_num_H(self, mol):
        # total_num_h = 0, 1, 2, 3, 4
        v = torch.zeros([max_n_atoms, 1])
        for i, atom in enumerate(mol.GetAtoms()):
            v[i] = atom.GetTotalNumHs(includeNeighbors=True) - 2.
        return v

    def _get_formal_charge(self, mol):
        # formal_charge = -3, -2, -1, 0, 1, 2, 3
        v = torch.zeros([max_n_atoms, 1])
        for i, atom in enumerate(mol.GetAtoms()):
            v[i] = atom.GetFormalCharge() / 1.5
        return v

    def _get_atom_is_aromatic(self, mol):
        onehot = torch.zeros([max_n_atoms, 1])
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetIsAromatic():
                onehot[i, 0] = 1.

        return onehot

    def _get_atom_hybridization(self, mol):
        hyb_onehot = torch.zeros([max_n_atoms, 7])
        for i, atom in enumerate(mol.GetAtoms()):
            hyb_onehot[i, hyb2int.get(atom.GetHybridization(), 6)] = 1.

        return hyb_onehot
    
    def _get_atom_xyz(self, mol, pad):
        conf = mol.GetConformer()
        CanonicalizeConformer(conf)
        position = torch.tensor(conf.GetPositions()).float()
        return torch.cat([position, torch.zeros([pad, 3])], dim=0)

    def _get_atom_index(self, a):
        return a2i.get(a, 2)
    
    def __len__(self):
        return len(self.meta)


class SACDataInfer(SACData):
    # self.version = '0.0.1'
    # self.version = '0.0.2'
    # self.version = '0.0.3' # normalize xyz features to have mu=0, std=1
    # version = '0.0.4' # divide input features by sqrt(d_model)
    # version = '0.0.5' # divide input features by sqrt(d_model)
    # version = '0.0.6' # pairwise distance
    # version = '0.0.7' # graph adjacency
    # version = '0.0.8' # use dev data for train
    # version = '0.0.9' # donor & acceptor
    # version = '0.0.10' # molecular feature

    version = '0.1.0' # Finalized.
    def __init__(self, idx=None, augs=[], meta=fp['test_meta'], data=fp['test_data'], pretrain=False):
        super(SACData, self).__init__()

        self.augs = augs

        self.meta = pd.read_csv(meta)
        self.meta = self.meta[~self.meta.uid.isin(ignored_uids)].reset_index(drop=True)
        if idx is not None:
            self.meta = self.meta.loc[idx].reset_index(drop=True)
        self.meta = self.meta.to_records()
        
        self.data_dir = data

        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self.factory = ChemicalFeatures.BuildFeatureFactory(fdefName) # Needed for donor/acceptor features.

        self.pretrain = pretrain

    def __getitem__(self, i):
        m = self.meta[i]
        mol = Chem.SDMolSupplier(f'{self.data_dir}/{m.uid}.sdf')[0]
        pb_mol = pb.readstring("smi", Chem.MolToSmiles(mol))

        n_atoms = mol.GetNumAtoms()

        # Atom index features.
        atom_idx = torch.tensor(
            [self._get_atom_index(a.GetSymbol()) for a in mol.GetAtoms()] +
            [DUMMY_ATOM] + # note that a dummy atom is added after all atoms in the molecule.
            [-1] * (max_n_atoms - n_atoms - 1),
            dtype=torch.long
        )

        # Atom coordinate features.
        xyz = self._get_atom_xyz(mol, pad=max_n_atoms - n_atoms)

        mask = torch.ones([max_n_atoms, max_n_atoms], dtype=bool)
        mask_1d = (atom_idx != -1)
        mask[mask_1d] = 0
        mask[:, mask_1d] = 0
        mask[~mask_1d] = 1
        mask[:, ~mask_1d] = 1
        mask = mask.unsqueeze(0)

        # pairwise distance
        pdist = torch.cdist(xyz, xyz, p=2.0)
        # set distance to dummy node as 1e6
        pdist[n_atoms] = 1e6
        pdist[:, n_atoms] = 1e6

        # adjacency
        adj = torch.zeros([max_n_atoms, max_n_atoms], dtype=bool)
        adj[:n_atoms, :n_atoms] = torch.tensor(GetAdjacencyMatrix(mol), dtype=bool)

        # atom hybridization (one-hot, 7 types) : max_n_atoms x 7
        hyb = self._get_atom_hybridization(mol)
        # is_aromatic : max_n_atoms x 1
        is_aromatic = self._get_atom_is_aromatic(mol)
        # formal_charge : max_n_atoms x 1
        formal_charge = self._get_formal_charge(mol)
        # total_numh : max_n_atoms x 1
        total_numh = self._get_total_num_H(mol)
        # total_valence : max_n_atoms x 1
        total_valence = self._get_total_valence(mol)
        # donor & acceptor
        donor_acceptor = self._get_donor_acceptor(mol)

        # Molecule-level features.
        spin = torch.zeros([max_n_atoms, 2])
        spin[:, pb_mol.spin - 1] = 1 # Spin multiplicity

        npr1 = torch.ones([max_n_atoms, 1]) * Descriptors3D.NPR1(mol) # NPR1
        npr2 = torch.ones([max_n_atoms, 1]) * Descriptors3D.NPR2(mol) # NPR2

        feat = torch.cat([
            hyb,
            is_aromatic,
            formal_charge,
            total_numh,
            total_valence,
            donor_acceptor,
            spin,
            npr1,
            npr2
        ], dim=-1)

        item = {
            'mask': mask.bool(),
            'out_mask': mask_1d.bool(),
            'n_atoms': mask_1d.sum(),
            'atom_idx': atom_idx.long(),
            'pdist': pdist,
            'adj': adj,
            'feat': feat,
        }
        return item

if __name__ == '__main__':
    from tqdm import tqdm

    dataset = SACData(pretrain=True)
    loader = DataLoader(dataset, batch_size=1)

    for data in tqdm(loader):
        # if torch.isnan(data['hyb']).any():
            # print(data)
        if torch.isnan(data['angle']).any():
            # print(data['angle'])
            print(data['uid'])

        # xyz, target, mask, atom_idx = data['xyz'], data['target'], data['mask'], data['atom_idx']
        # adj = data['adj']
        # print(adj.shape)
        # break
        # print(data['target'])

    print(f'len={len(dataset)}')
    print(SACData.version)
