# -*- coding: utf-8 -*-

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##
# calc_ro - given proportion p, calculates the utility from FORUM based on Example1
#
# Parameters:
#   p - Proportion p_t
#
def calc_ro(p):
    return 1-p;

##
# calc_rg - Calculates the utility from GenAI based on Example1
#
# Parameters:
#   t - Number of rounds since the last training round, corresponds to gamma_t
#
def calc_rg(t):
    return pow(0.5,t) * 3


##
# calc_p - Calculate the next proportion p_t+1
#
# Parameters:
#   g - utility value from GenAI at round t
#   o - utility value from FORUM at round t
#   beta - Users' sensitivity
#
def calc_p(g, o, beta):
    return exp(beta * g) / (exp(beta * g) + exp(beta * o))

##
# run_scheme - Run a given training scheme, calculating the users' social 
#              welfare and GenAI's revenue.
#
# Parameters:
#   p - Initial proportion p_1
#   beta - Users' sensitivity
#   T - Horizon, number of rounds
#   training_scheme - Training scheme x
#
# Return:
#   u - List of instantaneous welfare for each round.
#   v - List of revenues for each round.
#   P - List of proportions in each round.
#
def run_scheme(p, beta, r, c_m, c_t, T, rg, ro, training_scheme):
    trn_idx = 0
    v = []
    u = []
    P = []
    
    # Iterate over every round in [T]
    for t in range(1, T+1):
        P.append(p)
        v_t = 0

        # Calculate gamma_t        
        while trn_idx + 1 < len(training_scheme) and t >= training_scheme[trn_idx + 1]:
            trn_idx = trn_idx + 1
        dis_to_trn = t - training_scheme[trn_idx]
        
        # Calculate the revenue v_t
        if dis_to_trn == 0:
            v_t = v_t - c_t
        v_t = v_t + p*r - c_m
        v.append(v_t)
        
        # Calculate the instantaneous welfare u_t
        u.append(p*rg(dis_to_trn) + (1-p)*ro(p))
        
        # Calculate the next proportion p_t+1
        p = calc_p(rg(dis_to_trn), ro(p), beta)
        
    return u, v, P


##
# plot_example - Plot the graph of Example1 in Section2
#
def plot_example():
    p = [1]
    p_cur = p[0]
    beta = 1
    R = 1
    c_t = 0.5
    c_m = 0.6
    T = 20
    
    # Calculate the instantaneous welfare for each training scheme
    u_no_train, v, P_no_train = run_scheme(p[0], beta, 0, 0, 0, T, calc_rg, calc_ro, [1])
    u_rev_opt, v, P_rev_opt = run_scheme(p[0], beta, 0, 0, 0, T, calc_rg, calc_ro, [1, 4, 7, 9, 12, 14, 17])
    u_usr_opt, v, P_usr_opt = run_scheme(p[0], beta, 0, 0, 0, T, calc_rg, calc_ro, range(1, T+1))
    
    
    U_nt = []
    U_rev = []
    U_uo = []
    U_s = []
    U_cur_nt = 0
    U_cur_rev = 0
    U_cur_uo = 0
    U_cur_s = 0
    
    # Sum the instantaneous welfare sum_i=1^t u_t
    for t in range(T):
        U_cur_nt = U_cur_nt + u_no_train[t]
        U_cur_rev = U_cur_rev + u_rev_opt[t]
        U_cur_uo = U_cur_uo + u_usr_opt[t]
        
        U_nt.append(U_cur_nt)
        U_rev.append(U_cur_rev)
        U_uo.append(U_cur_uo)
        
        # Repeat for the counterfactual welfare
        U_cur_s = U_cur_s + calc_ro(0)
        U_s.append(U_cur_s)
        
    # Plot the graph
    rounds = range(1, T+1)
    
    plt.figure(0)
    plt.plot(rounds, U_nt, label='x0')
    plt.plot(rounds, U_rev, label='Revenue maximizing')
    plt.plot(rounds, U_uo, label='welfare maximizing')
    plt.plot(rounds, U_s, label='counterfactual')
    
    plt.xlabel("round t")
    plt.ylabel("welfare")
    plt.legend()
    plt.show
        
    # Save the data to an excel document
    exc_labels = ['steps', 'U_nt', 'U_rev', 'U_uo', 'U_s']
    exc_data = (np.c_[rounds, U_nt, U_rev, U_uo, U_s])
    df = pd.DataFrame(exc_data, columns=exc_labels)
    df.to_excel('example.xlsx', sheet_name='sheet1')





##
# generate_k_cyc_scheme - Generate a cyclic training scheme which trains every 
#                         k rounds
#
# Parameters:
#   k - Number of rounds between each round GenAI trains
#   T - Horizon, number of rounds
#
# Return:
#   x - list of training rounds of the form 1+kn for n in N.
#
def generate_k_cyc_scheme(k, T):
    i = 1
    x = []
    
    while i <= T:
        x.append(i)
        i = i + k
        
    return x

##
# generate_mixed_cyc_scheme - Generate a training scheme which alternate its  
#                             training every k1 and k2 rounds
#
# Parameters:
#   k1 - First number of rounds
#   k2 - Second number of rounds
#   T - Horizon
#
# Return:
#   x - list of training rounds of the form [1, 1+k1, 1+k1+k2, 1+2k1+k2 ...]
#
def generate_mixed_cyc_scheme(k1, k2, T):
    i = 1
    x = []
    
    # While the current training round does not exceed the horizon
    while i <= T:
        # Train after k1 rounds
        x.append(i)
        i = i + k1
        
        if i>T:
            break
        
        # Train after k2 rounds
        x.append(i)
        i = i + k2
        
    return x

##
# do_theorem_2 - Calculate the proportions and evaluations for Theorem2 in Sec3
#
def do_theorem_2():
    beta = 1
    p = 1
    r = 1
    c_m = 0.6
    c_t = 0.504
    T = 100
    
    # Calculate the revenue for the training scheme that does not train.
    # The goal here is to find the round in which sum_i=1^t v_t is negative
    x0 = generate_k_cyc_scheme(T, T)
    u, v0, P0 = run_scheme(p, beta, r, c_m, c_t, T, calc_rg, calc_ro, x0)
    # sum_i=1^t v_t < 0 for t = 8.
    
    # Calculate the revenue and proportions of every cyclic training scheme k in [8]
    for i in range(1, 8+1):
        # Generate the scheme and calculate Proportions + revenue
        x = generate_k_cyc_scheme(i, T)
        u, vi, pi = run_scheme(p, beta, r, c_m, c_t, T, calc_rg, calc_ro, x)
        
        # Print the information for the evaluation
        print('-----------------------------------------------')
        print('------------------ k = ' + str(i) + ' ----------------------')
        print('a5 - a4 = ' + str(pi[1 + 5*i - 1] - pi[1 + 4*i + 1]))
        print('sum_t=0^k p_t = ' + str(sum(pi[4*i:5*i])))
        print('V' + str(i) + ': ' + str(sum(vi[i*10:i*11])/i))
        print('V' + str(i) + ' + c_m: ' + str(sum(vi[i*4:i*5])/i + c_m))
        print('sum_t=1^8 v_t = ' + str(sum(vi[0:8])))
    
    # Repeat the same process for the scheme x23 for alpha1=2 and alpha2 = 3.
    print('-----------------------------------------------')
    print('----------------- k = 2,3 ---------------------')
    x23 = generate_mixed_cyc_scheme(2, 3, T)
    u, v23, P23 = run_scheme(p, beta, r, c_m, c_t, T, calc_rg, calc_ro, x23)
    print('a5 - a4 = ' + str(P23[1 + 5*i - 1] - P23[1 + 4*i + 1]))
    print('sum_t=0^k p_t = ' + str(sum(P23[4*i:5*i])))
    print('V' + str(i) + ': ' + str(sum(v23[i*10:i*11])/i))
    print('V' + str(i) + ' + c_m: ' + str(sum(v23[i*4:i*5])/i + c_m))
    print('sum_t=1^8 v_t = ' + str(sum(v23[0:8])))
    

##
# plot_figure_2a - Plot figure 2a in Section5
#
def plot_figure_2a():
    beta = 1
    T = 10
    Pp = []
    
    plt.figure(2)
    
    t = range(1,T+1)
    
    # Run the instance in the example with the following initial proportions.
    for p in [0, 0.2, 0.4, 0.6, 0.8, 1]:
        u, v, P = run_scheme(p, beta, 0, 0, 0, T, calc_rg, calc_ro, [1])
        Pp.append(P)
        
        # Plot the induced proportions
        plt.plot(t, P, label='p_1 = ' + str(p))
    
    plt.xlabel("round t")
    plt.ylabel("p_t")
    plt.legend()
    plt.show()
    
    # Save the proportions into an excel document
    exc_labels = ['round', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    exc_data = (np.c_[t, np.transpose(Pp)])
    df = pd.DataFrame(exc_data, columns=exc_labels)
    df.to_excel('pvst.xlsx', sheet_name='sheet1')
    

##
# calc_ro_fig2b - given proportion p, calculates the utility from FORUM for
#                 figure 2b.
#
# Parameters:
#   p - Proportion p_t
#
def calc_ro_fig2b(p):
    b = 100
    return exp(-b*p)/(exp(-b*p) + exp(-b * 0.8))

##
# calc_rg_fig2b - Calculates the utility from GenAI based for figure 2b
#
# Parameters:
#   t - Number of rounds since the last training round, corresponds to gamma_t
#
def calc_rg_fig2b(t):
    return pow(0.8,t) * 1.1

##
# plot_figure_2b - Plot figure 2b in Section5
#
def plot_figure_2b():
    beta = 10
    T = 10
    Pp = []
    
    plt.figure(3)
    
    t = range(1,T+1)
    
    # Run the instance of figure 2b for the following initial proportions
    for p in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        u, v, P = run_scheme(p, beta, 0, 0, 0, T, calc_rg_fig2b, calc_ro_fig2b, [1])
        Pp.append(P)
        
        # Plot the induced proportions
        plt.plot(t, P, label='p_1 = ' + str(p))
    
    plt.xlabel("round t")
    plt.ylabel("p_t")
    plt.legend()
    plt.show()

    # Save the proportions into an excel document
    exc_labels = ['round', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    exc_data = (np.c_[t, np.transpose(Pp)])
    df = pd.DataFrame(exc_data, columns=exc_labels)
    df.to_excel('pvst_nocont.xlsx', sheet_name='sheet1')
    

##
# do_calc_combinations - recoursive function to calculate all the combinations
#                        of training schemes with horizon T that train n times
#
# parameters:
#   T - horizon
#   n - number of times the training schemes train
#   start - starting index for the iteration. represents a training round.
#   ret - list of training schemes, which of horizon T and trains n times
#   to_append - temporary list representing a specific training scheme   
#
def do_calc_combinations(T, n, start, ret, to_append):
    # Recoursion ending condition
    if n == 0:
        ret.append(to_append)
        return
    
    # Start another permutation where start+i is a training round
    for i in range(1, T + 1):        
        if start + i > T:
            break
            
        append_clone = to_append.copy()
        append_clone.append(start+i)
        do_calc_combinations(T, n-1, start+i, ret, append_clone)
         

##
# do_calc_combinations - wraping recoursive function to calculate all the combinations
#                        of training schemes with horizon T that train n times
#
# parameters:
#   T - horizon
#   n - number of times the training schemes train
#
# return:
#   ret - list of training schemes, which of horizon T and trains n times
#
def calc_combinations(T, n):
    ret = []
    do_calc_combinations(T, n-1, 1, ret, [1])
    return ret

##
# calc_optimal_training_scheme - Brute force method to calculate the optimal 
#                                training scheme
#
# parameters:
#   beta - Users' sensitivity
#   p1 - initial proportion
#   r - GenAI reward from users choosing it
#   c_m - maintenance cost
#   c_train - training cost
#   T - horizon
#
def calc_optimal_training_scheme(beta, p1, r, c_m, c_train, T):
    training_times_result = []

    V_results = []
    training_times = []
    
    # Iterate over the number of times a training scheme can train
    for n in range(1, T + 1): 
        nV_results = []
        
        # Get all the training schemes that train n times
        n_training_times = calc_combinations(T, n) 
        
        # Run each training scheme and save the induced revenue
        for i in range(len(n_training_times)):
            u, v, p = run_scheme(p1, beta, r, c_m, c_train, T, calc_rg, calc_ro, n_training_times[i]) 
            nV_results.append(sum(v))
        
        # Save the n-training times corresponding revenue
        V_results.append(max(nV_results))
        training_times.append(n_training_times[np.argmax(nV_results)])
    
    # Get the training scheme which maximizes the revenue over every n
    max_idx = np.argmax(V_results)
    T_opt_training_scheme_sparse = training_times[max_idx]
    T_opt_training_scheme = [0]*(T+1)
    
    # Translate the format of the training scheme
    for i in range(len(T_opt_training_scheme_sparse)):
        T_opt_training_scheme[T_opt_training_scheme_sparse[i]] = 1
    training_times_result.append(T_opt_training_scheme)
    
    # Print the information of the optimal training scheme
    print("T = " + str(T) + ": "  + ''.join(map(str,training_times[max_idx])))
    print("revenue = " + str(V_results[max_idx]))