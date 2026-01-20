
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
    port_sample = np.zeros((n, N_selected), dtype=int)
    for i in range(n):
        channel_gains = np.linalg.norm(H[i], axis=1)
        ranked_ports = np.argsort(channel_gains)[::-1]
        selected = []
        candidate_pool = list(ranked_ports[:min(N_selected*4, N_Ports)])
        for _ in range(N_selected):
            best_port = -1
            best_value = -np.inf
            for port in candidate_pool:
                if port in selected:
                    continue
                candidate = selected + [port]
                h_n = H[i, candidate, :]
                f_val = sinr_balancing_power_constraint(len(candidate), K, h_n, Pt, noise)
                if f_val > best_value:
                    best_value = f_val
                    best_port = port
            selected.append(best_port)
            candidate_pool.remove(best_port)
            if len(selected) > 1:
                for idx in range(len(selected)):
                    temp = selected.copy()
                    removed = temp.pop(idx)
                    h_temp = H[i, temp, :]
                    f_temp = sinr_balancing_power_constraint(len(temp), K, h_temp, Pt, noise)
                    if f_temp > best_value:
                        selected = temp
                        candidate_pool.append(removed)
                        break
        port_sample[i] = selected
    return port_sample

