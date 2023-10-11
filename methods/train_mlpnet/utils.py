def get_model_name(args):
    mechanism_name = ''
    for m in args.mechanisms:
        if m == 'first':
            mechanism_name += 'f|'
        elif m == 'second':
            mechanism_name += 's|'
        elif m == 'first+entry':
            mechanism_name += 'fe|'
        elif m == 'second+entry':
            mechanism_name += 'se|'
        else:
            raise ValueError(f"Unknown mechanism name:{m}")
    mechanism_name = mechanism_name[:-1]

    distribution_name = ''
    for dist in args.distributions:
        if dist == 'uniform':
            distribution_name += 'u|'
        elif dist == 'gaussian':
            distribution_name += 'g|'
    distribution_name += 'z' if args.start_from_zero else 'nz'
    update_mode = 'update_max' if args.only_update_max else 'update_all'
    return f'MLPNet_{mechanism_name}_{distribution_name}_n={args.max_player}_v={args.valuation_range - 1}_batch={args.batch_size}_lr={args.lr}_{update_mode}_epoch={args.max_epoch}'
