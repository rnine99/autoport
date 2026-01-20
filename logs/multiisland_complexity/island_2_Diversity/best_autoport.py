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