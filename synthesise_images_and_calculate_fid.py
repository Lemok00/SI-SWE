import os
import math
import argparse
from tqdm import tqdm
from cleanfid import fid

import torch
from torchvision.utils import save_image

from utils.testing import txt2dict
from utils.mess_proj import message_to_vector
from model.SI_SWE.model import Mapping, Generator, Predictor
from model.SI_SWE.utils import hiding_process

if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='SI-SWE')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=500000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--sigma', type=float, default=1)
    parser.add_argument('--test_num', type=int, default=10000)

    parser.add_argument('--synthesise_images', action='store_true')
    parser.add_argument('--calculate_fid', action='store_true')

    parser.add_argument('--exp_root', type=str, default='./experiments')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--test_name', type=str, default='synthesised_images')

    args = parser.parse_args()

    assert args.synthesise_images or args.calculate_fid, \
        'Select at least one task (--synthesise_images or --calculate_fid).'

    result_dir = os.path.join(args.result_dir, args.test_name, args.model, args.exp_name)

    ''' Synthesise images '''
    if args.synthesise_images:

        assert not os.path.exists(result_dir) or not len(os.listdir(result_dir)) > 0, \
            'The target directory already exists and is not empty.'
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

        model.mapping.eval()
        model.generator.eval()

        model.mapping.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['mapping'].items()})
        model.generator.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['generator'].items()})

        print(f'Model Loaded')

        ''' Synthesis config '''
        sigma = args.sigma
        message_dim = int(model.vector_dim * sigma)
        
        messages = torch.randint(low=0, high=2, size=(args.test_num, message_dim))
        vectors = message_to_vector(messages, sigma=sigma).to(device)
        factors = torch.randint(low=0, high=2, size=(args.test_num, model.factor_dim), dtype=torch.float).to(device)

        for i in tqdm(range(math.ceil(args.test_num / args.batch_size)),desc='Synthesising'):
            vector = vectors[i * args.batch_size:(i + 1) * args.batch_size]
            factor = factors[i * args.batch_size:(i + 1) * args.batch_size]

            with torch.no_grad():
                container_image = hiding_process(model, vector, factor).cpu().data

            for j in range(container_image.shape[0]):
                save_image(container_image[j],
                           os.path.join(result_dir, f'{i * args.batch_size + j + 1:05d}.png'),
                           normalize=True,
                           value_range=(-1, 1))

    ''' Calculate fid scores '''
    if args.calculate_fid:

        if 'Bedroom' in args.exp_name or args.dataset_name == 'Bedroom':
            dataset_name = 'bedroom'
        elif 'Church' in args.exp_name or args.dataset_name == 'Church':
            dataset_name = 'church'
        elif 'FFHQ' in args.exp_name or args.dataset_name == 'FFHQ':
            dataset_name = 'ffhq'
        else:
            raise NotImplementedError

        fid_score = fid.compute_fid(result_dir, dataset_name=dataset_name, mode="clean", dataset_split="custom", verbose=True)

        print(args.model, args.exp_name, f'FID: {fid_score: .2f}')
