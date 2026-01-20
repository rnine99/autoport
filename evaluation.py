from __future__ import annotations

from template import template_program, task_description
from llm4ad.base import Evaluation
from utility_objective_functions import sinr_balancing_power_constraint
from typing import Any
import scipy.io
import os
import itertools
import numpy as np



__all__ = ['FasPortRateEvaluation']


class FasPortRateEvaluation(Evaluation):
    """Evaluator for circle packing problem in a unit square."""

    def __init__(self,
                 timeout_seconds=30,
                 **kwargs):
        """
        Args:
            timeout_seconds: Time limit for evaluation
            n_instance: Number of problem instances to evaluate
            max_circles: Maximum number of circles to pack (n)
        Raises:
            ValueError: If invalid parameters are provided
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        base_path = os.path.dirname(__file__)

        self.K = 8
        self.Selected_port = self.K

        Port_N1 = 8
        Port_N2 = Port_N1
        self.N_Ports = Port_N1 * Port_N2
        self.SINR_target = np.ones((self.K,1))

        self.noise = 1

        P = 30  # (dBm)

        self.Pt = 10 ** ((P - 30) / 10)


        # training datasets
        BATCH_SIZE = 1000
        length=2
        filename_train = f'./FA_Channel/train_channel_N_{Port_N1}_U_{self.K}_W_{length}_S_{BATCH_SIZE}_dBm.mat'

        # load training data
        data = scipy.io.loadmat(filename_train)
        Htemp =   np.transpose(data['Hmat'], (2, 1, 0))   # (BATCH_SIZE, Port_N, K)
        num_train = 50

        # dataset for training
        H_train = Htemp[0:num_train, :,:]
        self.H_current = H_train
        self.n,_,_ = self.H_current.shape


    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)


    def evaluate(self, eva: callable) -> float:
        """Evaluate the circle packing solution."""
        np.random.seed(2025)
        population  = eva(self.K,self.Selected_port,self.N_Ports,self.Pt,self.n,self.H_current,self.noise)

        rewards = np.array([sinr_balancing_power_constraint(self.Selected_port, self.K, self.H_current[j, population[j,:].astype(int),:], self.Pt, self.noise) for j in range(self.n)])
        score = np.sum(rewards)/self.n
        return score




if __name__ == '__main__':




    def select_ports(K,N_selected,N_Ports,Pt,n,H_current,noise):
        """
        Select Selected_port out of N_Ports ports to maximize sum rate.

        Args:
            K: Number of users
            N_ports: Total number of ports available for each channel realization
            Pt: Total transmit power
            n: Total number of channel realizations
            H: Numpy array of shape (n, N_Ports,K). It denotes n channel realizations
            noise: Noise power


        Returns:
            port_sample: Numpy array of shape (n, Selected_port). For each row or channel realizations, all values should be integers from 1 to N_Ports and cannot be repeated.
            sum_rate: the value of the sum rate
        """
        # rate=np.zeros(n,)
        # port_sample = np.zeros((n,Selected_port))
        # for j in range(n):
        #     H_temp = H_current[j,:,:]
        #     p = np.random.choice(N_Ports, Selected_port, replace=False)
        #     port_sample[j,:] =p#.reshape((1,Selected_port))
        #     # print(H_temp.shape)
        #     h_test = H_temp[p,:]
        #     rate[j] = sinr_balancing_power_constraint(Selected_port, K, h_test, Pt, noise)
        # sum_rate = np.sum(rate)/n

        # return port_sample

        rate=np.zeros(n,)
        port_sample = np.zeros((n,N_selected))
        for j in range(n):
            H_temp = H_current[j,:,:]
            p = np.random.choice(N_Ports, N_selected, replace=False)
            port_sample[j,:] =p.reshape((1,N_selected))
            h_test = H_temp[p,:]
            rate[j] = sinr_balancing_power_constraint(N_selected, K, h_test, Pt, noise)
        average_rate = np.sum(rate)/n

        return port_sample

    pack = FasPortRateEvaluation()
    pack.evaluate_program('_', select_ports)
