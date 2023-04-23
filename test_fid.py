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

    parser.add_argument('--resynthesise_images', action='store_true')

    parser.add_argument('--exp_root', type=str, default='./experiments')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--test_name', type=str, default='synthesis_fidelity_evaluation')

    args = parser.parse_args()

    result_dir = os.path.join(args.result_dir, args.test_name, args.model, args.exp_name, f'sigma={args.sigma}')
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
    message_dim = int(model.vector_dim * sigma)

    ''' Testing '''

    if not len(os.listdir(result_dir)) == args.test_num or args.resynthesise_images:
        
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
                           range=(-1, 1))

    fid_score = 0

    if 'Bedroom' in args.exp_name or args.dataset_name == 'Bedroom':
        fid_score = fid.compute_fid(result_dir, dataset_name='bedroom',
                                    mode="clean", dataset_split="custom", verbose=False)
    elif 'Church' in args.exp_name or args.dataset_name == 'Church':
        fid_score = fid.compute_fid(result_dir, dataset_name='church',
                                    mode="clean", dataset_split="custom", verbose=False)
    elif 'FFHQ' in args.exp_name or args.dataset_name == 'FFHQ':
        fid_score = fid.compute_fid(result_dir, dataset_name='ffhq',
                                    mode="clean", dataset_split="custom", verbose=False)

    print(args.model, args.exp_name, f'sigma={sigma}', f'FID: {fid_score: .4f}')
