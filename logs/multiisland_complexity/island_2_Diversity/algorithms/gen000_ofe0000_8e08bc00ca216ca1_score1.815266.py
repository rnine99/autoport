
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
        channel = H[i]
        selected = []
        remaining = list(range(N_Ports))
        while len(selected) < N_selected:
            best_gain = -np.inf
            best_port = None
            for port in remaining:
                candidate_set = selected + [port]
                val = sinr_balancing_power_constraint(len(candidate_set), K, channel[candidate_set, :], Pt, noise)
                if val > best_gain:
                    best_gain = val
                    best_port = port
            selected.append(best_port)
            remaining.remove(best_port)
        improved = True
        while improved:
            improved = False
            for idx in range(N_selected):
                for cand in range(N_Ports):
                    if cand in selected:
                        continue
                    temp = selected.copy()
                    temp[idx] = cand
                    current_val = sinr_balancing_power_constraint(N_selected, K, channel[selected, :], Pt, noise)
                    new_val = sinr_balancing_power_constraint(N_selected, K, channel[temp, :], Pt, noise)
                    if new_val > current_val:
                        selected = temp
                        improved = True
        port_sample[i] = selected
    return port_sample

