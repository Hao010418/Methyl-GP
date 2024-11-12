import argparse
from typing import List


def get_config():
    parse = argparse.ArgumentParser(description='common train config')

    parse.add_argument('-kmers', type=List[int], default=[3, 4, 5, 6])
    parse.add_argument('-kmer', type=int, default=3)

    # TODO train on GPU or CPU
    parse.add_argument('-cuda', type=bool, default=True)
    parse.add_argument('-device', type=int, default=0)

    # TODO change training set and testing set
    parse.add_argument('-num_class', type=int, default=2)
    parse.add_argument('-dataset_name', type=str, default='dataset')
    parse.add_argument('-methylation_type', type=str, default='4mC')
    # parse.add_argument('-methylation_type', type=str, default='5hmC')
    # parse.add_argument('-methylation_type', type=str, default='6mA')
    parse.add_argument('-species', type=str, default='Tolypocladium')
    # parse.add_argument('-test_species', type=str, default='A_thaliana')
    """
    A_thaliana, C_elegans, C_equisetifolia, D_melanogaster, E_coli, F_vesca, G_pickeringii, G_subterraneus,
    H_sapiens, M_musculus, R_chinensis, S_cerevisiae, T_thermophile, Tolypocladium, Xoc BLS256
    """

    # TODO change params of optimizer or bacth size
    parse.add_argument('-lr', type=float, default=0.00003)
    parse.add_argument('-wd', type=float, default=0.001)
    parse.add_argument('-batch_size', type=int, default=128)

    # TODO change training mode and model parameters
    parse.add_argument('-seed', type=int, default=42)
    parse.add_argument('-mode', type=str, default='independent test')
    # parse.add_argument('-mode', type=str, default='cross-species')
    parse.add_argument('-model', type=str, default='iDNA-M2CR')
    parse.add_argument('-feature_dims', type=List[int], default=[768, 768, 768, 768],
                       help='dimension of four pooler layer outputs')
    parse.add_argument('-emb_dim', type=int, default=512, help='embedding dimension of encoder layer')
    parse.add_argument('-fc_hidden', type=int, default=512, help='dimension of fully-connection layer')
    parse.add_argument('-beta', type=float, default=0.0125, help='regularization term of pretrain loss')
    parse.add_argument('-sigma', type=float, default=0.01)
    parse.add_argument('-lambda_1', type=float, default=8.0)
    parse.add_argument('-lambda_2', type=float, default=0.01)
    parse.add_argument('-froze', type=bool, default=False, help='to freeze bert model during training')
    parse.add_argument('-check_params', type=bool, default=False, help='to check modules of model')
    parse.add_argument('-count_params', type=bool, default=True, help='counting trainable params of model')

    # TODO change type of Loss function and train epochs
    parse.add_argument('-pre_epoch', type=int, default=20, help='epochs of pre-train part')
    parse.add_argument('-epoch', type=int, default=50, help='epochs of fine-tuning part')
    parse.add_argument('-b', type=float, default=0.06, help='flooding model')
    parse.add_argument('-loss_function', type=str, default='CE')
    parse.add_argument('-reduction', type=str, default='mean')

    # save path and other parameters
    parse.add_argument('-save_model', type=bool, default=False)
    parse.add_argument('-save_pred_data', type=bool, default=False)
    parse.add_argument('-model_save_path', type=str, default='results/', help='save the best model')
    parse.add_argument('-model_save_name', type=str, default='iDNA-M2CR')

    config = parse.parse_args()
    return config
