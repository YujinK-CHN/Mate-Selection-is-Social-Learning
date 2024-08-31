def training(config, algo):
    x, y = algo.train()

    if algo.name == 'mtppo':
        algo.save(f"./models/{config['hidden_size']}_{config['batch_size']}_{config['epoch_opt']}_{config['total_episodes']}.pt")

    return x, y
