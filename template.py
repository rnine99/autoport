template_program = '''
from utility_objective_functions import sinr_balancing_power_constraint
import numpy as np

def select_ports(K, N_selected, N_Ports, Pt, n, H, noise):
    """
    Select N_selected ports from N_Ports to maximize average SINR.

    Args:
        K: Number of users
        N_selected: Number of ports to select
        N_Ports: Total available ports
        Pt: Total transmit power
        n: Number of channel realizations
        H: Channel matrix, shape (n, N_Ports, K), dtype=complex128
        noise: Noise power

    Returns:
        port_sample: shape (n, N_selected), dtype=int

    CRITICAL:
        - H is COMPLEX: use np.abs() or np.linalg.norm()
        - Output shape: (n, N_selected)
        - Output dtype: int

    BASELINES:

    # 1. Random
    # port_sample = np.zeros((n, N_selected), dtype=int)
    # for j in range(n):
    #     p = np.random.choice(N_Ports, N_selected, replace=False)
    #     port_sample[j, :] = p
    # return port_sample

    # 2. Greedy
    # port_sample = np.zeros((n, N_selected), dtype=int)
    # for j in range(n):
    #     H_temp = H[j, :, :]
    #     gains = np.linalg.norm(H_temp, axis=1)
    #     p = np.argsort(gains)[-N_selected:]
    #     port_sample[j, :] = p
    # return port_sample

    # 3. Local Search
    # port_sample = np.zeros((n, N_selected), dtype=int)
    # for j in range(n):
    #     H_temp = H[j, :, :]
    #
    #     # Start with greedy
    #     gains = np.linalg.norm(H_temp, axis=1)
    #     p = np.argsort(gains)[-N_selected:]
    #     best_p = p.copy()
    #     h_n = H_temp[best_p, :]
    #     best_score = sinr_balancing_power_constraint(N_selected, K, h_n, Pt, noise)
    #
    #     # Local search: try swapping
    #     unselected = [i for i in range(N_Ports) if i not in best_p]
    #     for i in range(N_selected):
    #         for new_port in unselected:
    #             p_new = best_p.copy()
    #             p_new[i] = new_port
    #             h_n_new = H_temp[p_new, :]
    #             score_new = sinr_balancing_power_constraint(N_selected, K, h_n_new, Pt, noise)
    #             if score_new > best_score:
    #                 best_score = score_new
    #                 best_p = p_new.copy()
    #                 unselected = [i for i in range(N_Ports) if i not in best_p]
    #                 break
    #
    #     port_sample[j, :] = best_p
    # return port_sample

    YOUR ALGORITHM (goal: beat than baselines):
    """

    port_sample = np.zeros((n, N_selected), dtype=int)

    for j in range(n):
        H_temp = H[j, :, :]

        # TODO: Your algorithm
        p = np.random.choice(N_Ports, N_selected, replace=False)

        port_sample[j, :] = p

    return port_sample
'''

task_description = """
Select N_selected ports to maximize average SINR.

REQUIREMENTS:
- Output: (n, N_selected), dtype=int
- H is COMPLEX: use np.linalg.norm()

"""
