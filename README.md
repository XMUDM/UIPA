# UIPA
This repository is the official implementation of "Joint User and Item Prototype Alignment for Cross-Platform Recommendation". In this paper, we propose a cross-platform recommendation framework based on **U**ser and **I**tem **P**rototype **A**lignment (**UIPA**). UIPA is a plug-and-play framework that can be applied to single-platform collaborative filtering backbones and simultaneously enhance performance on both platform. 


## 1 Download
To use the UIPA code smoothly, you need to download the repository from GitHub and the datasets and models from GoogleDrive.

**1.1 Download the repository**
```
git clone https://github.com/XMUDM/UIPA.git
```
**1.2 Download the files in directories *Models* and *Datasets***

[Models](https://drive.google.com/file/d/1jDTF8H8L7i_b8E9SAhgnAVX8QpaPZ3HH/view?usp=drive_link)
[Datasets](https://drive.google.com/file/d/10VdpmroXeuMqvpIqLVzil7dAroVKLp66/view?usp=drive_link)


## 2 Setup
You can utilize the following commands to inital a virtual environment and install the required packages.

```
# Create and activate a new virtual environment
conda create -n UIPA python=3.8
conda activate UIPA

# Install the package
pip install -r requirements.txt
```

## 3 Quick Start
Please make sure the packages required in the `requirements.txt` are properly installed. Then you can utilize the following command to run the code and train the UIPA model:
```
python main.py
```
If you want to skip the training part and use the trained model, you can run:
```
python main.py --pretrained_model=1 
```

## 4 Parameters Setting
All the parameters along with their descriptions are in `parse.py`. You can also run UIPA with any combination of parameters you want.
```
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
```
