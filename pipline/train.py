def training(config, algo_list):
    for algo in algo_list:
        algo.train()
        
        if algo.name == 'ippo' or algo.name == 'mappo':
            algo.save(f"./models/{config['env_name']}_{algo.name}_{config['n_agents']}_{config['max_cycles']}_{config['total_episodes']}.pt")
        else:
            algo.save(f"./models/{config['env_name']}_{algo.name}_{config['n_agents']}_{config['n_skills']}_{config['max_cycles']}_{config['total_episodes']}.pt")
