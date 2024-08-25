def training(config, algo_list):
    for algo in algo_list:
        algo.train()
        
        if algo.name == 'ippo' or algo.name == 'mappo':
            algo.save(f"./models/{config['env_name']}_{algo.name}_{config['n_agents']}_{config['max_cycles']}_{config['total_episodes']}.pt")
        if algo.name == 'ppo':
            algo.save(f"./models/{algo.name}_{config['n_agents']}_{config['max_cycles']}_{config['total_episodes']}.pt")
        if algo.name == 'mtppo':
            algo.save(f"./models/{config['pop_size']}_{config['max_cycles']}_{config['total_episodes']}_{config['lr']}.pt")
