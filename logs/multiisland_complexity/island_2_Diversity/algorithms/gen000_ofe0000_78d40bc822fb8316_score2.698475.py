
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
    def objective(port_sets):
        total = 0.0
        for i in range(n):
            h_eff = H[i, port_sets[i], :]
            total += sinr_balancing_power_constraint(N_selected, K, h_eff, Pt, noise)
        return total / n
    
    current = [np.random.choice(N_Ports, size=N_selected, replace=False) for _ in range(n)]
    current_val = objective(current)
    best = [c.copy() for c in current]
    best_val = current_val
    
    T_start = 1.0
    T_end = 1e-4
    steps = 2000
    for step in range(steps):
        T = T_start * (T_end / T_start) ** (step / steps)
        i = np.random.randint(n)
        new_set = current[i].copy()
        a, b = np.random.choice(N_selected, 2, replace=False)
        candidate = np.random.choice([p for p in range(N_Ports) if p not in new_set])
        new_set[a] = candidate
        new_set = np.unique(new_set)
        while len(new_set) < N_selected:
            new_port = np.random.choice([p for p in range(N_Ports) if p not in new_set])
            new_set = np.append(new_set, new_port)
        new = current.copy()
        new[i] = new_set
        new_val = objective(new)
        if new_val > current_val or np.random.rand() < np.exp((new_val - current_val) / T):
            current = new
            current_val = new_val
            if current_val > best_val:
                best = [c.copy() for c in current]
                best_val = current_val
    return np.array(best)

