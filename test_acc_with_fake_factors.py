import os
import math
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from utils.testing import txt2dict
from utils.mess_proj import message_to_vector, vector_to_message
from model.SI_SWE.model import Mapping, Generator, Predictor
from model.SI_SWE.utils import hiding_process, extracting_process

if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SI-SWE')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=500000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--t_max', type=int, default=10)
    parser.add_argument('--test_num', type=int, default=1000)

    parser.add_argument('--exp_root', type=str, default='./experiments')

    args = parser.parse_args()

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

    ''' Testing '''
    real_factors_ACCs = []
    fake_factors_ACCs = []
    messages = torch.randint(low=0, high=2, size=(args.test_num, message_dim))
    vectors = message_to_vector(messages, sigma=sigma).to(device)
    real_factors = torch.randint(low=0, high=2, size=(args.test_num, model.factor_dim), dtype=torch.float).to(device)
    fake_factors = torch.randint(low=0, high=2, size=(args.test_num, model.factor_dim), dtype=torch.float).to(device)

    for i in tqdm(range(math.ceil(args.test_num / args.batch_size)),desc='Testing'):
        message = messages[i * args.batch_size:(i + 1) * args.batch_size]
        vector = vectors[i * args.batch_size:(i + 1) * args.batch_size]
        real_factor = real_factors[i * args.batch_size:(i + 1) * args.batch_size]
        fake_factor = fake_factors[i * args.batch_size:(i + 1) * args.batch_size]

        with torch.no_grad():
            container_image = hiding_process(model, vector, real_factor)
            recovered_vector_using_real_factor = extracting_process(model, container_image, real_factor, t_max)
            recovered_message_using_real_factor = vector_to_message(recovered_vector_using_real_factor, sigma=sigma)
            recovered_vector_using_fake_factor = extracting_process(model, container_image, fake_factor, t_max)
            recovered_message_using_fake_factor = vector_to_message(recovered_vector_using_fake_factor, sigma=sigma)
            
        real_factors_ACCs.append(1 - F.l1_loss(recovered_message_using_real_factor, message).item())
        fake_factors_ACCs.append(1 - F.l1_loss(recovered_message_using_fake_factor, message).item())

    real_factors_acc = sum(real_factors_ACCs) / len(real_factors_ACCs)
    fake_factors_acc = sum(fake_factors_ACCs) / len(fake_factors_ACCs)
    print(f'{args.exp_name} Acc with real: {real_factors_acc}; Acc with fake: {fake_factors_acc}')
