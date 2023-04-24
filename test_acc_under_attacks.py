import os
import math
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F

from utils.testing import txt2dict
from utils.mess_proj import message_to_vector, vector_to_message
from model.SI_SWE.model import Mapping, Generator, Predictor
from model.SI_SWE.utils import hiding_process, extracting_process
from attacks import (BaseAttack, JPEGCompressionAttacker, WebPCompressionAttacker,
                     GaussianNoiseAttacker, PepperAndSaltNoiseAttacker, SpeckleNoiseAttacker,
                     GaussianBlurAttacker, AverageBlurAttacker, MedianBlurAttacker)

if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SI-SWE')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=500000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--sigma', type=int, default=1)
    parser.add_argument('--t_max', type=int, default=10)
    parser.add_argument('--test_num', type=int, default=1000)

    parser.add_argument('--exp_root', type=str, default='./experiments')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--test_name', type=str, default='robustness_evaluation')

    args = parser.parse_args()

    result_dir = os.path.join(args.result_dir, args.test_name, args.model, f'sigma={args.sigma}')
    os.makedirs(result_dir, exist_ok=True)

    ''' Load model '''
    exp_root = os.path.join(args.exp_root, args.model, args.exp_name)
    config_path = os.path.join(exp_root, 'config.txt')
    ckpt_args = argparse.Namespace(**txt2dict(config_path))
    model_path = os.path.join(exp_root, 'model', f'{args.ckpt}.pt')
    print(f'Load model from {model_path}')
    ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)

    model = argparse.Namespace()
    model.vector_dim = ckpt_args.vector_dim
    model.factor_dim = ckpt_args.factor_dim

    model.mapping = Mapping(input_dim=model.factor_dim, latent_dim=model.factor_dim).to(device)
    model.generator = Generator(channel=32, vector_dim=model.vector_dim, factor_dim=model.factor_dim).to(device)
    model.predictor = Predictor(channel=32, vector_dim=model.vector_dim).to(device)

    model.mapping.eval()
    model.generator.eval()
    model.predictor.eval()

    model.mapping.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['mapping'].items()})
    model.generator.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['generator'].items()})
    model.predictor.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['predictor'].items()})

    print(f'Model Loaded')

    ''' Test config '''
    sigma = args.sigma
    t_max = args.t_max
    message_dim = int(model.vector_dim * sigma)

    attack_and_intensity_list = [
        (BaseAttack(), ['']),
        (GaussianNoiseAttacker(), [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]),
        (PepperAndSaltNoiseAttacker(), [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]),
        (SpeckleNoiseAttacker(), [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]),
        (JPEGCompressionAttacker(), [100, 90, 80, 70, 60, 50]),
        (WebPCompressionAttacker(), [100, 90, 80, 70, 60, 50]),
        (GaussianBlurAttacker(), [3, 5, 7, 9, 11, 13]),
        (AverageBlurAttacker(), [3, 5, 7, 9, 11, 13]),
        (MedianBlurAttacker(), [3, 5, 7, 9, 11, 13]),
    ]

    ''' Testing '''

    Acc_dict = defaultdict(lambda: defaultdict(list))
    messages = torch.randint(low=0, high=2, size=(args.test_num, message_dim))
    vectors = message_to_vector(messages, sigma=sigma).to(device)
    factors = torch.randint(low=0, high=2, size=(args.test_num, model.factor_dim), dtype=torch.float).to(device)

    for i in tqdm(range(math.ceil(args.test_num / args.batch_size)),desc='Testing'):
        message = messages[i * args.batch_size:(i + 1) * args.batch_size]
        vector = vectors[i * args.batch_size:(i + 1) * args.batch_size]
        factor = factors[i * args.batch_size:(i + 1) * args.batch_size]

        with torch.no_grad():
            container_image = hiding_process(model, vector, factor)
            for attack, intensities in attack_and_intensity_list:
                for intensity in intensities:
                    attacked_image = attack.attack(container_image.clone(), intensity)
                    recovered_vector = extracting_process(model, attacked_image, factor, t_max)
                    recovered_message = vector_to_message(recovered_vector, sigma=sigma)

                    Acc_dict[attack.get_attack_name()][attack.get_intensity_name(intensity)].append(
                        1 - F.l1_loss(recovered_message, message).mean().item()
                    )

    attack_name_list = []
    accuracy_list = []
    for attack, intensities in attack_and_intensity_list:
        for intensity in intensities:
            attack_name_list.append(f'{attack.get_attack_name()} {attack.get_intensity_name(intensity)}')
            temp_list = Acc_dict[attack.get_attack_name()][attack.get_intensity_name(intensity)]
            accuracy_list.append(sum(temp_list) / len(temp_list))

    dataframe = pd.DataFrame()
    dataframe['Att. Name'] = attack_name_list
    dataframe['Accuracy'] = accuracy_list

    dataframe.to_csv(os.path.join(result_dir, f'{args.exp_name}.csv'))
