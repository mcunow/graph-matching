from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit.Chem import AddHs
from rdkit.Chem import SanitizeMol
import torch
from torch_geometric.data import Data
import utils

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _transform_graph(data):
    
    atom_ls=torch.tensor([6,7,8,9],device=DEVICE) 
    atom_x=torch.argmax(data.x,dim=1)
    atom_x=atom_ls[atom_x]
    return Data(atom_x,data.edge_index,edge_attr=torch.argmax(data.edge_attr,dim=1),batch=data.batch)

def data_to_molecule(in_data):
    data=_transform_graph(in_data)
    if data.x.size(0)==0:
        raise Exception
    mol = RWMol()

    for i in range(data.num_nodes):
        atomic_number = int(data.x[i].item())  # Assuming node features are atomic numbers
        atom = Chem.Atom(atomic_number)
        mol.AddAtom(atom)

    ls=[Chem.BondType.SINGLE,Chem.BondType.AROMATIC,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE]

    for i in range(data.num_edges):
        start_idx = int(data.edge_index[0, i].item())
        end_idx = int(data.edge_index[1, i].item())
        if start_idx<end_idx:
            bond_type=ls[data.edge_attr[i]]
            if bond_type is not None:
                mol.AddBond(start_idx, end_idx,order=bond_type)
    mol = mol.GetMol()
    # mol.UpdatePropertyCache(strict=False)
    # mol=AddHs(mol)
    return mol
def is_valid_molecule(data):
    try:
        mol = data_to_molecule(data)
    except Exception:
        print("Zero-node graph")
        return None

    try:
        # Perform sanitization first
        #Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_RDKIT)
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol)
        
        # Generate SMILES
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=False, allHsExplicit=False, canonical=True)
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.CanonSmiles(smiles)
        return smiles
    except Chem.KekulizeException:
        print("kekulization error")
        return None
    except Exception as e:
        # Other exceptions can be handled here if needed
        return None


def get_smiles_ls(dataset):
    dataset_smiles=[]
    for data in dataset:
        mol=data_to_molecule(data)
        Chem.CanonicalRankAtoms(mol)
        dataset_smiles.append(Chem.MolToSmiles(mol, kekuleSmiles=True,allHsExplicit=False,canonical=True))
    return set(dataset_smiles)
    

def compute_validity(data_ls):
    length=len(data_ls)
    valid_ls=[]
    for data in data_ls:
        if data is None:
            pass
        elif data.x.size(0)!=0:
            try:
                smiles=is_valid_molecule(data)
                smiles=utils.extract_largest_smiles_string(smiles)
                if smiles is not None:
                    valid_ls.append(smiles)
            except:
                pass
    return valid_ls,len(valid_ls)/(length)

def compute_uniqueness(smiles_ls):
    if len(smiles_ls)==0:
        return [],0.
    ls=set(smiles_ls)
    ls=list(ls)
    
    return ls,len(ls)/(len(smiles_ls))

def compute_novelty(smiles_ls,dataset_smiles_set):
    sample_set=set(smiles_ls)
    inter_sec=sample_set.intersection(dataset_smiles_set)
    if len(sample_set)==0:
        return [],0.
    else:
        frac=1-(len(inter_sec)/(len(sample_set)))
        return sample_set.difference(inter_sec),frac

def compute_metrics(data_ls,dataset_smiles):

    valid_smiles,validity_frac=compute_validity(data_ls)
    unique_smiles,unique_frac=compute_uniqueness(valid_smiles)
    novel_smiles,novelty_frac=compute_novelty(unique_smiles,dataset_smiles)

    return validity_frac,unique_frac,novelty_frac,novel_smiles

def smiles_to_txt(path,smiles,i):
    with open(path+f"smiles_epoch{i}.txt", "w") as file:
    # Write each string on a new line
        if len(smiles)!=0:
            for item in smiles:
                file.write(item + "\n")

def evaluate_reconstruction(data,data_rec):
    data_ls=data.to_data_list()
    #data_rec=utils.oh_encode_reconstructed_data(data_rec)
    data_rec_ls=utils.batch_to_list(data_rec)
    length=len(data_ls)
    valid_num=0
    reconstructed_num=0
    for input,rec in zip(data_ls,data_rec_ls):
        if rec is not None:
            smiles_in=is_valid_molecule(input)
            smiles_rec=is_valid_molecule(rec)
            if smiles_rec is not None:
                valid_num+=1
                if smiles_in==smiles_rec:
                    reconstructed_num+=1
    
    return valid_num/length, reconstructed_num/length

def plot_mol(data):
    try:
        mol=data_to_molecule(data)
        return mol
    except:
        print("not valid")

