
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
    def random_subset():
        return np.random.choice(N_Ports, N_selected, replace=False)
    
    def evaluate(subset, channel):
        h_sub = channel[subset, :]
        return sinr_balancing_power_constraint(N_selected, K, h_sub, Pt, noise)
    
    def neighbor(subset):
        new = subset.copy()
        idx = np.random.randint(N_selected)
        available = [p for p in range(N_Ports) if p not in new]
        if available:
            new[idx] = np.random.choice(available)
        return new
    
    port_sample = np.zeros((n, N_selected), dtype=int)
    for i in range(n):
        channel = H[i]
        current = random_subset()
        current_score = evaluate(current, channel)
        T_init = 10.0
        T_final = 0.01
        cooling_rate = 0.95
        steps = 100
        
        T = T_init
        for _ in range(steps):
            candidate = neighbor(current)
            candidate_score = evaluate(candidate, channel)
            delta = candidate_score - current_score
            
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current = candidate
                current_score = candidate_score
            T = max(T * cooling_rate, T_final)
        
        port_sample[i] = np.sort(current)
    
    return port_sample

