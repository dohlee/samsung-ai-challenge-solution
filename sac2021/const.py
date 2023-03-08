from rdkit import Chem
from collections import defaultdict

max_n_atoms = 148
# max_n_atoms = 310
num_unique_atoms = DUMMY_ATOM = 12
atoms = ['H', 'B', 'C', 'N', 'O', 'F', 'S', 'Si', 'P', 'Cl', 'Br', 'I']
a2i = {s:i for i, s in enumerate(atoms)}

fp = {
    'train_meta': '../../data/traindev.csv',
    'test_meta': '../../data/test_no590.csv',

    'train_data': '../../data/pretrain_sdf',
    'test_data': '../../data/test_sdf',

    'pretrain_meta': '../../data/qm9+oe62.csv',
    'pretrain_data': '../../data/pretrain_sdf',

    'sample_submission': '../../data/sample_submission_no590.csv',
}

ignored_uids = [
    'train_1688',
    'train_14782',
    'train_28906',
    'train_29068',

    'train_29036',  # cannot apply CanonicalizeConformer
    'train_29828',  # cannot apply CanonicalizeConformer

    'gdb_1',
    'gdb_2',
    'gdb_3',
    
    # Errors in angle computation.
    'gdb_23',
    'gdb_24',
    'gdb_486',
    'PAMMAS',
    'QUFCID',
    'LOVWOI',
    'QEYQUH',
    'ENIGEO',
]

hyb2int = {
    Chem.rdchem.HybridizationType.S: 0,
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
    Chem.rdchem.HybridizationType.UNSPECIFIED: 6,
}

bondtype2int = defaultdict(lambda: 4)
bondtype2int[Chem.rdchem.BondType.SINGLE] = 0
bondtype2int[Chem.rdchem.BondType.AROMATIC] = 1
bondtype2int[Chem.rdchem.BondType.DOUBLE] = 2
bondtype2int[Chem.rdchem.BondType.TRIPLE] = 3
