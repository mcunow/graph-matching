import os
import sys
import gzip
import pickle
import torch
import torch.nn.functional as F
import wandb
import argparse
import json



use_wandb = os.getenv("USE_WANDB", False)=="True"
wandb_key=os.getenv("WANDB_MODE","online")


with open('../src/wandb_key.txt', 'r') as file:
    key = file.read()


wandb_key=os.getenv("WANDB_KEY",key)
if use_wandb==True and wandb_key is None:
    raise Exception("No valid key specified")

# Import necessary modules
import gnn
import util_metrics
import vgae
import matching
from permute import permute_batch, permute_complete_dataset
from decoder import TransformerDecoder
from torch_geometric.loader import DataLoader


# Load datasets
def load_datasets():
    with open('../data/smiles_canonical_train.pkl', 'rb') as file:
        dataset_smiles = pickle.load(file)
    
    with gzip.open('../data/dataset_train_fc.pkl.gz', 'rb') as f:
        dataset = pickle.load(f)
    
    with gzip.open('../data/dataset_validation_fc.pkl.gz', 'rb') as f:
        dataset_val = pickle.load(f)
    
    return dataset_smiles, dataset, dataset_val

# Initialize Weights and Biases

def init_wandb(config):
    wandb.login(key=wandb_key)
    wandb.init(
        project="Graph Matching",
        config=config
    )
    return  wandb.run.name

def matching_loss(matching_type,matched_loss,unmatched_loss):
    if matching_type=="matched":
        return matched_loss
    elif matching_type=="unmatched":
        return unmatched_loss
    elif matching_type=="mixed":
        return 0.5*matched_loss+0.5*unmatched_loss
    

def generate_perm_inv_loss(loss_type):
    if loss_type=="Optimal":
        return matching.BruteForceMatcher()
    elif loss_type=="Top10":
        return matching.BruteForceSampleMatcher(10)
    elif loss_type=="Top50":
        return matching.BruteForceSampleMatcher(50)
    elif loss_type=="Top100":
        return matching.BruteForceSampleMatcher(100)
    elif loss_type=="Statistics":
        return matching.BruteForceMatcher()
    elif loss_type=="GNN":
        return matching.GNN_Loss(6,50)
    else:
        raise Exception("Permutation-invariant loss is not correct specified.")

def perm_inv_loss(matcher,data,data_rec,loss_type):
    if loss_type=="Optimal" or loss_type=="Top10" or loss_type=="Top50" or loss_type=="Top100":
        return matcher.match(data,data_rec)
    elif loss_type=="Statistics":
        return matcher.statistics_loss(data,data_rec)
    elif loss_type=="GNN":
        return matcher.gnn_loss(data,data_rec)
    else:
        raise Exception



# Training function
def train(model, train_loader, optimizer, matcher, device, config,epoch,permutations):
    model.train()
    alpha, beta = config["alpha"], config["beta"]
    loss_ls,num_ls,kl_ls,matched_ls,unmatched_ls=[],[],[],[],[]
    
    for data in train_loader:
        data = data.to(device)
        data=permute_batch(data,permutations,device)
        optimizer.zero_grad()
        
        data_rec, mu, logvar, _, pred_num = model(data)
        num_nodes = 0.5 * (torch.sqrt(8 * torch.bincount(data.batch) + 1) - 1)
        num_loss = alpha * F.mse_loss(pred_num, num_nodes)

        matched_loss, unmatched_loss = perm_inv_loss(matcher,data,data_rec,config["perm_inv_loss"])
        kl_loss = beta * model.kl_loss(mu, logvar)
        rec_loss = matched_loss
        loss = rec_loss + kl_loss + num_loss
        
        loss.backward()
        optimizer.step()
        
        kl_ls.append(kl_loss.item())
        num_ls.append(num_loss.item())
        loss_ls.append(loss.item())
        matched_ls.append(matched_loss.item())
        unmatched_ls.append(unmatched_loss.item())
    length = len(loss_ls)
    
    # Log training metrics
    if use_wandb:
        wandb.log({
            "train/matched_loss": sum(matched_ls)/length,
            "train/unmatched_loss": sum(unmatched_ls)/length,
            "train/kl_loss": sum(kl_ls)/length,
            "train/loss": sum(loss_ls)/length,
            "train/num_node_pred_loss": sum(num_ls)/length
        },step=epoch)

# Validation function
def validate(model, valid_loader, matcher, device, config,epoch):
    model.eval()
    alpha, beta = config["alpha"], config["beta"]
    
    num_ls, match_ls,unmatch_ls, kl_ls, loss_ls = [], [], [], [],[]
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device)
            data_rec, mu, logvar, _, pred_num = model(data)
            num_nodes = 0.5 * (torch.sqrt(8 * torch.bincount(data.batch) + 1) - 1)
            num_loss = alpha * F.mse_loss(pred_num, num_nodes)
            
            matched_loss, unmatched_loss = perm_inv_loss(matcher,data,data_rec,config["perm_inv_loss"])
            kl_loss = beta * model.kl_loss(mu, logvar)
            rec_loss = matched_loss
            loss = rec_loss + kl_loss + num_loss
            
            num_ls.append(num_loss.item())
            unmatch_ls.append(unmatched_loss.item())
            match_ls.append(matched_loss.item())
            kl_ls.append(kl_loss.item())
            loss_ls.append(loss.item())
    
    # Compute average validation metrics
    length = len(loss_ls)
    if use_wandb:
        wandb.log({
            "validation/matched_loss": sum(match_ls) / length,
            "validation/unmatched_loss": sum(unmatch_ls) / length,
            "validation/kl_loss": sum(kl_ls) / length,
            "validation/loss": sum(loss_ls) / length,
            "validation/num_node_pred_loss": sum(num_ls) / length
        },step=epoch)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Main script for graph matching project")

    # General settings
    parser.add_argument('--perm_inv_loss', type=str, default="Optimal", help="Optimal, Top10, Top50, Top100, Statistics, GNN")
    return parser.parse_args()

# Main function
def main(args):
    # Load datasets
    dataset_smiles, dataset, dataset_val = load_datasets()
    dataset=dataset[:10]
    dataset_val=dataset_val[:10]

    #Read hyperparameters
    with open("../src/config.json", "r") as f:
        config = json.load(f)
    config["perm_inv_loss"]=args.perm_inv_loss
    
    # Initialize WandB
    if use_wandb:
        run_name = init_wandb(config)
    else:
        run_name=""
    
    # Set device and initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = gnn.GIN(
        in_channels=6, hidden_channels=config["encoder_hidden_dims"], out_channels=config["encoder_output_dim"], 
        norm=config["encoder_norm"], num_layers=config["encoder_layers"], act="leakyrelu", dropout=config["dropout"]
    ).to(device)
    decoder = TransformerDecoder(config["latent_dim"], config["decoder_hidden_dim"],config["attention_heads"],config["decoder_layers"],
                                 config["decoder_norm"]).to(device)
    matcher = generate_perm_inv_loss(args.perm_inv_loss)
    vgae_model = vgae.VGAE(encoder=encoder, decoder=decoder, latent_dims=config["latent_dim"], 
                           embedding="graph", eval=False).to(device)
    
    # Set up data loaders
    batch_size = config["batch_size"]
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    
    #Precompute permutation structure. Not trivial because of the edges as node representation.
    permutations=[matching.permute_graphs_over_n(i).to(device) for i in range(1,10)]
    
    #Permute validation set
    dataset_val=permute_complete_dataset(dataset_val,permutations,device)
    valid_loader = DataLoader(dataset_val, batch_size, shuffle=True)
    
    # Define optimizer
    optimizer = torch.optim.Adam(vgae_model.parameters(), lr=config["lr"], weight_decay=1e-5)
    
    # Training loop
    for epoch in range(config["epochs"]):
        train(vgae_model, train_loader, optimizer, matcher, device, config,epoch,permutations)
        
        if epoch % 20 == 0:
            if epoch%50==0:
                # Save model checkpoint
                folder = "../models"
                path = f"{folder}/vgae_model_{run_name}_{epoch}.pth"
                os.makedirs(folder, exist_ok=True)
                torch.save(vgae_model.state_dict(), path)
            
            # Generate and log samples
            sample = vgae_model.sample(config["sample_size"])
            validity, unique, novelty, smiles = util_metrics.compute_metrics(sample, dataset_smiles)
            util_metrics.smiles_to_txt(f"../models/{run_name}_", smiles, epoch)
            
            if use_wandb:
                wandb.log({
                    "inference/Validity": validity,
                    "inference/Uniqueness": unique,
                    "inference/Novelty": novelty,
                    "inference/Overall_frac":1*validity*unique*novelty
                },step=epoch)
            
            # Validation
            validate(vgae_model, valid_loader, matcher, device, config,epoch)
        
    validate(vgae_model, valid_loader, matcher, device, config,epoch)
    # Save final model
    final_path = f"../models/vgae_model_{run_name}_final.pth"
    torch.save(vgae_model.state_dict(), final_path)
    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)