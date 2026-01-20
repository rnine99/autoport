
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
    POP_SIZE = 20
    GENERATIONS = 30
    MUTATION_RATE = 0.2
    
    def random_subset():
        return np.random.choice(N_Ports, N_selected, replace=False)
    
    def evaluate(subset, channel):
        h_sub = channel[subset, :]
        return sinr_balancing_power_constraint(N_selected, K, h_sub, Pt, noise)
    
    def mutate(subset):
        if np.random.rand() < MUTATION_RATE:
            new = subset.copy()
            idx = np.random.randint(N_selected)
            available = [p for p in range(N_Ports) if p not in new]
            if available:
                new[idx] = np.random.choice(available)
            return new
        return subset
    
    def crossover(parent1, parent2):
        common = np.intersect1d(parent1, parent2)
        remaining = np.setdiff1d(np.union1d(parent1, parent2), common)
        np.random.shuffle(remaining)
        child = np.concatenate([common, remaining[:N_selected - len(common)]])
        return child[:N_selected]
    
    port_sample = np.zeros((n, N_selected), dtype=int)
    for i in range(n):
        channel = H[i]
        population = [random_subset() for _ in range(POP_SIZE)]
        
        for _ in range(GENERATIONS):
            scores = np.array([evaluate(ind, channel) for ind in population])
            elite_idx = np.argsort(scores)[-POP_SIZE//4:]
            elites = [population[idx] for idx in elite_idx]
            
            new_pop = elites.copy()
            while len(new_pop) < POP_SIZE:
                p1, p2 = np.random.choice(elite_idx, 2, replace=False)
                child = crossover(population[p1], population[p2])
                child = mutate(child)
                new_pop.append(child)
            population = new_pop
        
        best = max(population, key=lambda ind: evaluate(ind, channel))
        port_sample[i] = np.sort(best)
    
    return port_sample

