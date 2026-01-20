
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
        selected = list(ranked_ports[:N_selected])
        unselected = [p for p in range(N_Ports) if p not in selected]
        improved = True
        while improved:
            improved = False
            for idx in range(N_selected):
                current_port = selected[idx]
                best_swap = current_port
                best_value = sinr_balancing_power_constraint(N_selected, K, H[i, selected, :], Pt, noise)
                for cand in unselected:
                    temp_selected = selected.copy()
                    temp_selected[idx] = cand
                    f_val = sinr_balancing_power_constraint(N_selected, K, H[i, temp_selected, :], Pt, noise)
                    if f_val > best_value:
                        best_value = f_val
                        best_swap = cand
                if best_swap != current_port:
                    selected[idx] = best_swap
                    unselected.remove(best_swap)
                    unselected.append(current_port)
                    improved = True
        port_sample[i] = selected
    return port_sample

