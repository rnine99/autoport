
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
    selected = []
    for _ in range(N_selected):
        best_gain = -np.inf
        best_port = None
        for port in range(N_Ports):
            if port in selected:
                continue
            candidate = selected + [port]
            total = 0.0
            for i in range(n):
                h_eff = H[i, candidate, :]
                total += sinr_balancing_power_constraint(len(candidate), K, h_eff, Pt, noise)
            avg_gain = total / n
            if avg_gain > best_gain:
                best_gain = avg_gain
                best_port = port
        selected.append(best_port)
    
    # Local refinement via pairwise swaps
    improved = True
    while improved:
        improved = False
        for idx in range(N_selected):
            for new_port in range(N_Ports):
                if new_port in selected:
                    continue
                swapped = selected.copy()
                swapped[idx] = new_port
                total_orig = 0.0
                total_swap = 0.0
                for i in range(n):
                    h_orig = H[i, selected, :]
                    h_swap = H[i, swapped, :]
                    total_orig += sinr_balancing_power_constraint(N_selected, K, h_orig, Pt, noise)
                    total_swap += sinr_balancing_power_constraint(N_selected, K, h_swap, Pt, noise)
                if total_swap > total_orig:
                    selected = swapped
                    improved = True
                    break
            if improved:
                break
    
    return np.tile(selected, (n, 1))

