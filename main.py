from src.base.trainer import *
from src.ppi_data import *
from models.REFLEX import REFLEX
import argparse
import wandb

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes","True", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no","False", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def run_single_experiment(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"\n{'='*80}")
    print(f"Running experiment with ESM-2 encoding")
    print(f"{'='*80}\n")
    
    # Loading data & Preprocessing
    PPIData_ = PPIData(args)
    if args.torch_seed != 0:
        torch.manual_seed(args.torch_seed)
    data = PPIData_.data
    prot_token_ids = PPIData_.prot_token_ids  # List[Tensor]
    pad_idx = PPIData_.pad_value
    vocab_size = PPIData_.vocab_size
    data.to(device)
    
    # Initialize REFLEX model
    model = REFLEX(
        input_dim=data.embed1.shape[-1],
        vocab_size=vocab_size,
        pad_value=pad_idx,
        class_num=7,
        hidden_dim=376,
        ff_dim=1024,
        heads=8,
        layers=4,
        max_len=args.protein_max_length,
        device=device,
        layer_num=args.ln,
        args=args,
        disable_hierarchical_fusion=getattr(args, 'ablation_no_fusion', False),
        disable_generation_task=getattr(args, 'ablation_no_generation', False)
    ).to(device)

    # Initialize wandb
    is_ablation = (args.ablation_no_fusion or 
                   args.ablation_no_generation)
    wandb_project = 'PPI_ablation' if is_ablation else 'PPIGEN'
    
    wandb.init(
        project=wandb_project,
        name=args.run_name,
        config=vars(args)
    )
    
    # Train model
    trainer = BaseTrainer(model=model, data=data, args=args)
    trainer.train(PPIData_=PPIData_)

    if args.wandb:
        wandb.finish()
    
    print(f"\n{'='*80}")
    print(f"Experiment completed: {args.run_name}")
    print(f"Results saved to: {args.log_dir}")
    print(f"{'='*80}\n")
    
    return args.log_dir

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SHS27k', help='dataset')
    parser.add_argument('--split_type', type=str, default='random', help='random/bfs/dfs (default: random)')
    parser.add_argument('--split_seed', type=int, default=42, help='Random seed for data splitting (default: 42)')
    parser.add_argument('--torch_seed', type=int, default=42, help='Random seed for model parameter (default: 42)')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size (default: 256) -b')
    parser.add_argument('--model', type=str, default='REFLEX', help='used model')
    parser.add_argument('--task', type=str, default='regression', help='regression or classification')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)
    parser.add_argument('--max_epochs', type=int, default=500, help="e")
    parser.add_argument('--patience', type=int, default=25, help='Early stop epoch')
    parser.add_argument('--save_iter', type=int, default=0)
    parser.add_argument('--n_exp', type=int, default=1, help='Experiment Index')
    parser.add_argument('--wandb', type=str2bool, default=True, help='Whether to use wandb')
    parser.add_argument('--input_path', default=None, type=str, help='path for sequence and relation file')
    parser.add_argument('--structure', default=None, type=str, help='prefix for the pre generated structure')
    parser.add_argument('--ln', default=2, type=int, help='graph layer num')
    parser.add_argument('--protein_max_length', default=100, type=int, help='hidden layer')
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--lm_weight', default=5.0, type=float, help='weight for LM loss in HIPPI_Gen')
    parser.add_argument('--kl_weight', default=1, type=float, help='weight for KL loss in HIPPI_Gen')
    parser.add_argument('--ablation_no_fusion', type=str2bool, default=False, help='Ablation: disable Hierarchical Attribute Extractor')
    parser.add_argument('--ablation_no_generation', type=str2bool, default=False, help='Ablation: disable VAE/generation task')
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    args.steps = [100, 200, 300, 400]
    
    # Set dataset paths
    if args.dataset == 'SHS27k':
        args.input_path = 'data/27K.txt'
        args.structure = 'data/27K/27K'
    elif args.dataset == 'SHS148k':
        args.input_path = 'data/148K.txt'
        args.structure = 'data/148K/148K'
    
    # Read sequence and action files
    with open(args.input_path, 'r') as f:
        args.sequence = f.readline().strip()
        args.action = f.readline().strip()
    
    # Setup wandb
    if args.wandb:
        wandb.login(key="03455359de6e84d9bb06ffcf977c65a92e8fb8e8")
    
    # Run single experiment
    folder_path = os.path.dirname(os.path.abspath(__file__))
    param_str_list = [args.model, args.dataset, args.split_type, args.split_seed, 
                      args.torch_seed, args.batch_size, args.lm_weight, args.kl_weight, 
                      args.protein_max_length]
    if args.ablation_no_fusion:
        param_str_list.append('no_fusion')
    if args.ablation_no_generation:
        param_str_list.append('no_generation')
    param_str = '-'.join(str(value) for value in param_str_list)
    args.run_name = param_str
    args.log_dir = "{}/experiment/{}/{}".format(folder_path, args.dataset, param_str)
    
    print(args)
    print(f"Run name: {param_str}")
    
    run_single_experiment(args)
