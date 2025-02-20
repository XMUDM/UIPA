import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="input GPU id", default="0")
    parser.add_argument("-m", "--model", type=str, help="the backbones, select from: lightgcn, MF", default='lightgcn')
    parser.add_argument("-s", "--source_platform", type=str, help="the source platform, select from: DB, ML, MT, DB_Bo, BX_Bo", default="DB")
    parser.add_argument("-t", "--target_platform", type=str, help="the target platform, select from: DB, ML, MT, DB_Bo, BX_Bo", default="ML")
    parser.add_argument('--bpr_batch_size', type=int, help="bpr_batch_size", default=1024)
    parser.add_argument('-dim', '--latent_dim_rec', type=int, help="the dim of embeddings", default=64)
    parser.add_argument('--lightGCN_n_layers', type=int, help="lightGCN_n_layers", default=3)
    parser.add_argument('--lr', type=float, help="the learning rate", default=0.005)
    parser.add_argument('--seed', type=int, help="seed", default=2024)
    parser.add_argument('--epochs', type=int, help="the max epoch number", default=1000)
    parser.add_argument('--topks', type=list, help="topks", default=[10])
    parser.add_argument('--pretrained_model', type=int, help="whether to use pretrained_model", default=0)
    parser.add_argument('-pn','--prototype_num', type=int, help="the number of prototypes", default=100)
    parser.add_argument('-lw','--llm_weight', type=float, help="the weight of llm", default=0.3)
    parser.add_argument('-pw','--prototype_weight', type=float, help="the weight of prototype", default=0.1)
    parser.add_argument('-pl','--prototype_loss_weight', type=float, help="the weight of prototype_loss", default=1e-2)
    args = parser.parse_args()
    return args
