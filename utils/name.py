def set_exp_name(args):
    exp_name = ''
    if 'Bedroom' in args.dataset_path:
        exp_name += 'Bedroom'
    elif 'Church' in args.dataset_path:
        exp_name += 'Church'
    elif 'FFHQ' in args.dataset_path:
        exp_name += 'FFHQ'

    if args.model == 'DualContrastGAN':
        exp_name += '_' + args.generator + '+' + args.discriminator
        exp_name += f'_inDim{args.input_dim}'
    elif args.model == 'DisentangledContrastGAN_shareD':
        exp_name += f'_inDim{args.input_dim}'
    elif args.model == 'DisentangledContrastGAN_Rec':
        exp_name += f'_struDim{args.structure_dim}'
        exp_name += f'_recIter{args.recover_iter_max}'
        exp_name += f'_num{args.dataset_size // 1000}K'
    return exp_name
