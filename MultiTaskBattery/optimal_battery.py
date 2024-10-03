# Module for functions used for optimal battery construction
# Author: Bassel Arafat
# Date: Oct 1st 2024

import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
import PcmPy as pcm


def align_conditions(Ya, Yb, info_a, info_b):
    """
    Align two datasets based on shared conditions, align all conditions to the mean of shared conditions,
    then average shared conditions and append unique conditions.

    Args:
    Ya (numpy array): Dataset A (subjects x conditions x voxels) or (conditions x voxels)
    Yb (numpy array): Dataset B (subjects x conditions x voxels) or (conditions x voxels)
    info_a (pandas.DataFrame): Info file for Dataset A
    info_b (pandas.DataFrame): Info file for Dataset B

    Returns:
    combined_data (numpy array): Combined dataset
    combined_info (pandas.DataFrame): Combined info
    """

    shared_conditions = np.intersect1d(info_a['cond_code'], info_b['cond_code'])
    if len(shared_conditions) == 0:
        raise ValueError("No shared conditions between datasets.")

    # Standardize input dimensions to 3D if needed
    if len(Ya.shape) == 2:
        Ya = Ya[None, :, :]
    if len(Yb.shape) == 2:
        Yb = Yb[None, :, :]

    # Sort shared conditions and get indices
    shared_sorted = np.sort(shared_conditions)
    order_a = []  # Indices for shared conditions in Dataset A
    order_b = []  # Indices for shared conditions in Dataset B

    # Loop through each condition in the sorted shared conditions array
    for cond in shared_sorted:
        # Find the index of the condition in info_a that matches the current condition code
        idx_a = info_a[info_a['cond_code'] == cond].index[0]
        # Find the index of the condition in info_b that matches the current condition code
        idx_b = info_b[info_b['cond_code'] == cond].index[0]

        # Append the indices to the respective lists
        order_a.append(idx_a)
        order_b.append(idx_b)

    # Align shared conditions
    Ya_shared = Ya[:, order_a, :]
    Yb_shared = Yb[:, order_b, :]
    Ya_mean, Yb_mean = Ya_shared.mean(1, keepdims=True), Yb_shared.mean(1, keepdims=True)
    Ya_aligned, Yb_aligned = Ya - Ya_mean, Yb - Yb_mean

    # Average aligned shared conditions
    shared_avg = (Ya_aligned[:, order_a, :] + Yb_aligned[:, order_b, :]) / 2.0

    # Combine shared and unique conditions
    unique_a = np.setdiff1d(info_a['cond_code'], shared_sorted)
    unique_a_indices = info_a['cond_code'].isin(unique_a)
    unique_b = np.setdiff1d(info_b['cond_code'], shared_sorted)
    unique_b_indices = info_b['cond_code'].isin(unique_b)

    Ya_aligned_unique = Ya_aligned[:, unique_a_indices, :]
    Yb_aligned_unique = Yb_aligned[:, unique_b_indices, :]
    combined_data = np.concatenate([shared_avg, Ya_aligned_unique,
                                    Yb_aligned_unique], axis=1)

    # Create combined info file
    shared_info = info_a.loc[order_a, ['cond_name', 'cond_code']].copy()
    shared_info['source'] = 'averaged'
    unique_info = pd.concat([info_a[unique_a_indices], info_b[unique_b_indices]])
    unique_info['source'] = 'Novel'
    combined_info = pd.concat([shared_info, unique_info[['cond_name', 'cond_code', 'source']]], ignore_index=True)

    if Ya.shape[0] == 1:
        combined_data = combined_data[0]
    else:
        combined_data = combined_data


    return combined_data, combined_info

def find_optimal_battery(task_matrix, task_names, num_tasks=4, function='trace', top_n=1, sample_size=1000, average_across_subjects=True):

    """
    Finds the top N combinations of tasks based on a specified function, either on the group-averaged second moment matrix or by averaging matrices across subjects.
    
    Args:
        task_matrix (torch.Tensor): The task data matrix of shape (num_subjects, num_tasks, num_voxels)
        task_names (list): List of task names corresponding to the tasks in the task_matrix.
        num_tasks (int): The number of tasks required for the battery
        function (str): The function to optimize for. Can be 'trace' (for total variance) or 'inverse_trace' (for total precision).
        top_n (int): The number of top combinations to return. Default is 1.
        sample_size (int or None): If specified, randomly sample this many combinations from all possible combinations. If None, use all combinations.
        average_across_subjects (bool): If True, perform the search on the group second moment matrix based on group-averaged data. 
                                        If False, perform the search on a group second moment matrix based on averaging individual second moment matrices.
    
    Returns:
        list of tuples: A list of the top N combinations. Each tuple contains:
            - function_result (float): The result of the specified function for this combination.
            - combination (numpy.array): The indices of the tasks in the combination.
    """

    # if task matrix has nan values, replace them with zeros
    task_matrix[np.isnan(task_matrix)] = 0

    total_tasks = len(np.unique(task_names))
    num_runs = task_matrix.shape[1] // total_tasks
    
    # Generate all task indices
    task_indices = np.arange(total_tasks)

    if sample_size is not None:
        sampled_combinations = np.random.randint(0, len(task_indices), (sample_size, num_tasks))

    else:
        # Generate all possible combinations if sample_size is None
        all_combinations = list(combinations_with_replacement(task_indices, num_tasks))
        sampled_combinations = all_combinations

    # create condition_v and partition vector
    cond_vec = np.tile(np.arange(1, total_tasks+1), num_runs)

    # make a vector of 1 repated 16 times then 2 repeated 16 times and so on
    part_vec = np.repeat(np.arange(1, num_runs+1), total_tasks)
    
    # If we are averaging across subjects, average task_matrix across subjects (dim=0)
    if average_across_subjects:
        avg_task_matrix = np.nanmean(task_matrix, axis=0)  # Averaged across subjects
        G_group,E = pcm.util.est_G_crossval(avg_task_matrix,cond_vec,part_vec)

    else:
        # Compute the covariance matrix for each subject individually
        G_matrices = []
        for subj in range(task_matrix.shape[0]): 
            G_s,E_s = pcm.util.est_G_crossval(task_matrix[subj], cond_vec, part_vec)
            G_matrices.append(G_s)
        G_matrices_stacked = np.stack(G_matrices, 0)
        G_group = np.nanmean(G_matrices_stacked, axis=0)  # Averaged across subjects

        
    eye_matrix = 0.0001 * np.eye(num_tasks)
    ones_vector = np.ones((num_tasks, num_tasks))
    centering_matrix = np.eye(num_tasks) - ones_vector / num_tasks

    # Initialize top results based on function
    if function in ['trace', 'determinant']:
        top_results = [(-float('inf'), None)] * top_n
    elif function == 'inverse_trace':
        top_results = [(float('inf'), None)] * top_n


    for i, comb in enumerate(sampled_combinations):
        if i % 100000 == 0:
            print(f"Processing sample {i+1}/{len(sampled_combinations)}")

        # Extract subset covariance for the averaged data
        subset_varcov = G_group[comb, :][:, comb]
        centered_varcov = centering_matrix @ subset_varcov @ centering_matrix.T
        centered_varcov = centered_varcov + eye_matrix

        eigenvalues, _ = np.linalg.eigh(centered_varcov)

        # Compute trace or inverse trace
        if function == 'trace':
            function_result = np.sum(eigenvalues)
        elif function == 'inverse_trace':
            inverse_eigenvalues = 1.0 / eigenvalues
            function_result = np.sum(inverse_eigenvalues)
        elif function == 'determinant':
            function_result = np.prod(eigenvalues)
        else:
            raise ValueError("Invalid function argument")

        function_result_value = function_result.item()


        # After initialization, update only if the new result is better
        if function == 'inverse_trace':
            if function_result_value < top_results[-1][0]:  # We want the smallest values for 'inverse_trace'
                top_results[-1] = (function_result_value,comb)
        else:  # 'trace'
            if function_result_value > top_results[-1][0]:  # We want the largest values for 'trace'
                top_results[-1] = (function_result_value,comb)

        # Sort only after an update
        top_results.sort(reverse=(function in ['trace', 'determinant']))

    return top_results



# def genetic_algorithm(task_matrix, task_names, num_tasks=4, function='trace', population_size=100, generations=50, mutation_rate=0.1, top_n=1):
#         """evolution based algorithm to find the best combination of tasks."""
    
#     task_indices = pt.arange(len(task_names)).to(device)
#     full_varcov = task_matrix @ task_matrix.T
#     eye_matrix = 0.000004 * pt.eye(num_tasks).to(device)
#     ones_vector = pt.ones((num_tasks, num_tasks), device=device)
#     centering_matrix = pt.eye(num_tasks, device=device) - ones_vector / num_tasks
    
#     # Step 1: Initialize the population randomly
#     population = [random.sample(list(task_indices.cpu().numpy()), num_tasks) for _ in range(population_size)]

#     def fitness_function(combination):
#         """Evaluate the fitness of a combination (higher fitness is better)."""
#         subset_varcov = full_varcov[combination, :][:, combination]
#         centered_varcov = centering_matrix @ subset_varcov @ centering_matrix.T
#         centered_varcov = centered_varcov + eye_matrix
#         if function == 'inverse_trace':
#             fitness = pt.trace(pt.linalg.inv(centered_varcov)).item()
#         elif function == 'trace':
#             fitness = pt.trace(centered_varcov).item()
#         return fitness

#     def select_parents(population, fitnesses):
#         """Select parents based on fitness (using roulette wheel selection)."""
#         total_fitness = sum(fitnesses)
#         probabilities = [f / total_fitness for f in fitnesses]
#         parents = random.choices(population, weights=probabilities, k=2)
#         return parents

#     def crossover(parent1, parent2):
#         """Create a new combination by crossing over two parents."""
#         crossover_point = random.randint(1, num_tasks - 1)
#         child = parent1[:crossover_point] + parent2[crossover_point:]
#         return child

#     def mutate(combination, mutation_rate):
#         """Randomly mutate a combination by replacing one task with a new task."""
#         if random.random() < mutation_rate:
#             # Replace one task with a task not already in the combination
#             new_task = random.choice([task for task in task_indices.cpu().numpy() if task not in combination])
#             replace_index = random.randint(0, len(combination) - 1)
#             combination[replace_index] = new_task
#         return combination

#     # Step 2: Iterate over generations
#     for generation in range(generations):
#         print(generation)
#         # Step 3: Calculate fitness for each combination
#         fitnesses = [fitness_function(comb) for comb in population]
        
#         # Step 4: Select the top combinations
#         top_combinations = sorted(zip(fitnesses, population), reverse=(function == 'trace'))[:top_n]
        
#         # Print top combination for this generation
#         top_fitness, top_comb = top_combinations[0]
#         # print(f"Generation {generation+1} | Best fitness: {top_fitness} | Combination: {[task_names[i] for i in top_comb]}")
        
#         # Step 5: Create the next generation
#         new_population = []
#         while len(new_population) < population_size:
#             # Step 6: Select parents and perform crossover
#             parent1, parent2 = select_parents(population, fitnesses)
#             child = crossover(parent1, parent2)
            
#             # Step 7: Mutate the child (not everytime)
#             child = mutate(child, mutation_rate)
            
#             # aappend the child to the new population
#             new_population.append(child)
        
#         # Update population
#         population = new_population
    
#     return top_combinations