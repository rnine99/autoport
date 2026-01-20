
from utility_objective_functions import sinr_balancing_power_constraint
import numpy as np 
def select_ports(K, N_selected, N_Ports, Pt, n, H, noise):
    """   
   Select N_selected out of N_Ports ports to maximize the average objective value of n communications channels.

    Args:
        K: The number of users.
        N_selected: The number ports to be selected.  
        N_ports: Total number of ports available for each channel realization.        
        Pt: Total transmit power available.
        n: Total number of channel realizations.
        H: Numpy array of shape (n, N_Ports,K). It denotes n channel realizations.
        noise: Noise power, it is a NumPy scalar.

    Returns:
        port_sample: Numpy array of shape (n, N_selected), where n and N_selected are defined above. 
        For each row of it, all values should be integers from 0 to N_Ports-1 and cannot be repeated.
      

    For the n-th channel realization H(n, :,:), a valid port selection solution p must have the shape (1,N_selected)
    where N_selected is defined above and its elements are integers from 0 to N_Ports-1 and cannot be repeated. 
    The effective channel becomes h_n = H[n,p,:]. The objective value will be calculated using 
    the pre-defined function f_n=sinr_balancing_power_constraint(N_selected, K, h_n, Pt, noise).
    This function will maximize the average objective value of n channels. 
    """
    POP_SIZE = 30
    GENERATIONS = 50
    MUTATION_RATE = 0.1
    ELITE_SIZE = 6
    
    def fitness(individual):
        total = 0.0
        for i in range(n):
            h_eff = H[i, individual, :]
            total += sinr_balancing_power_constraint(N_selected, K, h_eff, Pt, noise)
        return total / n
    
    def create_individual():
        return np.random.choice(N_Ports, size=N_selected, replace=False)
    
    def crossover(parent1, parent2):
        mask = np.random.rand(N_selected) < 0.5
        child = np.full(N_selected, -1, dtype=int)
        child[mask] = parent1[mask]
        remaining = [g for g in parent2 if g not in child]
        idx = np.where(child == -1)[0]
        child[idx] = remaining[:len(idx)]
        if len(set(child)) < N_selected:
            return create_individual()
        return child
    
    def mutate(individual):
        if np.random.rand() < MUTATION_RATE:
            idx = np.random.choice(N_selected, 2, replace=False)
            available = [p for p in range(N_Ports) if p not in individual]
            if available:
                individual[idx[0]] = np.random.choice(available)
        return individual
    
    population = [create_individual() for _ in range(POP_SIZE)]
    for gen in range(GENERATIONS):
        scores = np.array([fitness(ind) for ind in population])
        elite_idx = np.argsort(scores)[-ELITE_SIZE:]
        new_pop = [population[i] for i in elite_idx]
        while len(new_pop) < POP_SIZE:
            parents = np.random.choice(elite_idx, 2, replace=False)
            child = crossover(population[parents[0]], population[parents[1]])
            new_pop.append(mutate(child))
        population = new_pop
    
    best = max(population, key=fitness)
    return np.tile(best, (n, 1))

