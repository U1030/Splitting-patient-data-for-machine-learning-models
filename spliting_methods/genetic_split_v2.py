import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
from tqdm import tqdm


class Individual:
    def __init__(self,patients,classes) :
        """
        patients : list of patient id
        classes : dict  key = patient id  , value = dict key : class id ; value : class count
        """
        self.patients = patients
        self.classes = classes  
        self.total_patients = len(patients)
        self.set_repartition = self.initialize_repartition()    
        self.train_patients , self.test_patients = self.compute_sets()
        class_ids = set(class_id for patient_classes in classes.values() for class_id in patient_classes.keys())
        self.classes_id = list(class_ids)
        self.train_lesions , self.test_lesions = self.determine_split_count()
        self.total_lesions = self.calculate_total_lesions()
        self.train_target, self.test_target = self.determine_target_counts()
        self.fitness = self.calculate_fitness()  
      

    def initialize_repartition(self):
        #print('--> initializing split')
        repartition = np.zeros(self.total_patients, dtype=int)
        test_indices = np.random.choice(self.total_patients, size=self.total_patients // 3, replace=False)
        repartition[test_indices] = 1
        return repartition 

    def compute_sets(self):
        #print('--> computing sets of patients')
        train_patients = [patient for patient in self.patients if self.set_repartition[self.patients.index(patient)] == 0]
        test_patients = [patient for patient in self.patients if self.set_repartition[self.patients.index(patient)] == 1]
        return train_patients,test_patients

    
    def calculate_total_lesions(self):
        #print('--> calculating total number of lesions of each class')
        total_lesions = {0: 0, 1: 0, 2: 0}
        for lesions in self.classes.values():
            for cls, count in lesions.items():
                total_lesions[cls] += count
        return total_lesions
    
    def determine_split_count(self):
        #print('--> determining split counts')
        train_count = {cls: 0 for cls in self.classes_id}
        for patient in self.train_patients:
            for cls, count in self.classes[patient].items():
                train_count[cls] += count
        test_count = {cls: 0 for cls in self.classes_id}
        for patient in self.test_patients:
            for cls, count in self.classes[patient].items():
                test_count[cls] += count
        return train_count, test_count

    def determine_target_counts(self, train_ratio=2/3):
        #print('--> determining target count')
        train_target = {cls: int(count * train_ratio) for cls, count in self.total_lesions.items()}
        test_target = {cls: self.total_lesions[cls] - train_target[cls] for cls in self.total_lesions}
        return train_target, test_target

    def calculate_fitness(self):
        #print('--> calculating fitness')
        score = 0
        for cls in self.classes_id:
            score += abs(self.train_lesions[cls] - self.train_target[cls])
            score += abs(self.test_lesions[cls] - self.test_target[cls])
        return score
    
    def update(self):        
        self.train_patients, self.test_patients = self.compute_sets()
        self.train_lesions, self.test_lesions = self.determine_split_count()
        self.train_target, self.test_target = self.determine_target_counts()
        self.fitness = self.calculate_fitness()
                                                                                                                                                                                                

def mutate(individual):
    new_repartition = individual.set_repartition.copy()
    total_patients = individual.total_patients    
    index_train = np.where(new_repartition == 0)[0]
    index_test = np.where(new_repartition == 1)[0]   
    if np.random.rand() < 0.5:
        # Mutate a train index to test
        if len(index_train) > 0 :
            mutation_indices_train = np.random.choice(index_train, size=1, replace=False)
            new_repartition[mutation_indices_train] = 1
        else :
            print("train set empty")
            mutation_indices_test = np.random.choice(index_test, size=1, replace=False)
            new_repartition[mutation_indices_test] = 0
    else:
        if len(index_test) > 0 :
            # Mutate a test index to train
            mutation_indices_test = np.random.choice(index_test, size=1, replace=False)
            new_repartition[mutation_indices_test] = 0
        else :
            print(" test set empty ")
            mutation_indices_train = np.random.choice(index_train, size=1, replace=False)
            new_repartition[mutation_indices_train] = 1
    individual.set_repartition = new_repartition
    individual.update()
    return


def linear_decay(current_gen, total_gens,initial_rate=0.7, final_rate=0.1):
    return initial_rate - ((initial_rate - final_rate) * (current_gen / total_gens))

def exponential_decay(current_gen,initial_rate=0.7,decay_rate =0.05):
    return initial_rate * np.exp(-decay_rate * current_gen)

def check_fitness_pop(population):
    fitnesses = [ind.fitness for ind in population]
    unique_fitness = set(fitnesses)
    if len(unique_fitness) == 1 :
        print("WARNING all indiduals have same fitness")
    return


def create_new_generation(population, elite_size_ratio, mutation_rate):
    # Evaluate fitness of all individuals
    pop_fitness = []
    for individual in population:       
        pop_fitness.append(individual.fitness)
    counter = Counter(pop_fitness)
    num_diff_fitness = len(counter)
    if num_diff_fitness <= 2:
        print(" WARNING : number of different fitness :",len(counter))
    # Sort population based on fitness
    population = sorted(population, key=lambda ind: ind.fitness) 
    check_fitness_pop(population)     
    # Elitism: carry the best individuals to the next generation 
    elite_size = int(len(population) * elite_size_ratio)
    new_generation = population[:elite_size]      
    # Mutation: all individuals exept elite are mutated     
    remaining_individuals = population[elite_size:]
    num_mutated_individuals = int(len(remaining_individuals) * mutation_rate)
    mutated_individuals = random.sample(remaining_individuals, num_mutated_individuals)    
    # Mutate the selected individuals
    for individual in mutated_individuals:
        mutate(individual)           
    # Add mutated individuals to the new generation
    new_generation.extend(mutated_individuals)  
    # Add the remaining individuals who are not elite and not mutated
    non_mutated_individuals = [ind for ind in remaining_individuals if ind not in mutated_individuals]
    new_generation.extend(non_mutated_individuals)        
    return new_generation


def find_split_genetic(patients,classes, population_size,nb_generations,elite_size_ratio,plot=False):   
    # Initialize population
    population = [Individual(patients,classes) for _ in range(population_size)]
    # Track best fitness over generations
    best_fitness_over_generations = []
    # Evolve over generations   
    best_individuals = []
    for gen in tqdm(range(nb_generations), desc='Processing generations'):
        mutation_rate = linear_decay(gen,nb_generations)       
        population = create_new_generation(population, elite_size_ratio, mutation_rate)
        best_fitness = min(individual.fitness for individual in population)        
        best_individual = min(population, key=lambda ind: ind.fitness)        
        best_fitness_over_generations.append(best_fitness)
        best_individuals.append(best_individual)        
        #print(f'Generation {generation + 1}, Best Fitness: {best_fitness}')        
    best_index = np.argmin(best_fitness_over_generations)    
    best_indiv = best_individuals[best_index] 
    #print("best fitness :", best_fitness_over_generations[best_index])  
    # Plot best fitness over generations
    if plot :
        plt.plot(best_fitness_over_generations)
        plt.xlabel('Generations')
        plt.ylabel('Best Fitness')
        plt.title('Best Fitness Over Generations')
        plt.show()       
    return best_indiv,best_fitness_over_generations[best_index]

def print_indiv(indiv):
    print('fitness :',indiv.fitness)  
    print('train lesions :',indiv.train_lesions)
    print('test lesions :',indiv.test_lesions)    
    return


def split_genetic(patients,classes, population_size,nb_generations,elite_size_ratio,plot):
    best_indiv,best_fitness = find_split_genetic(patients,classes, population_size,nb_generations,elite_size_ratio,plot) 
    split = best_indiv.set_repartition
    train_patient_ids = best_indiv.train_patients
    test_patient_ids = best_indiv.test_patients
    return train_patient_ids,test_patient_ids, best_fitness, split


"""

import sys
sys.path.append("/home/utilisateur/Bureau/CD8_RS_Pipeline")
import utils.organize_patient_data as org
import utils.read_data as rd

X_path = "/home/utilisateur/Bureau/CD8_RS_Pipeline/final results/final_data_acses_2_labels.csv"


grouped_data,df = rd.read_data(X_path,num_labels=2)
df["label"] = df["tumor_evolution"].apply(lambda x: 0 if x <= -30 else (1 if x >= 20 else 2))
patients,lesions,classes, classes_ratio = org.organize_data(grouped_data)
population_size = 1000
nb_generations = 500
elite_size_ratio = 0.4
train_patient_ids,test_patient_ids, best_fitness, split = split_genetic(patients,classes, population_size,nb_generations,elite_size_ratio,plot=True)

print(" best fitness :", best_fitness)
print("train :", train_patient_ids)
print("test :", test_patient_ids)

train_patients = [patient for patient in patients if split[patients.index(patient)] == 0]
test_patients = [patient for patient in patients if split[patients.index(patient)] == 1]

nb_lesions_train = sum(value for patient, value in lesions.items() if patient in train_patients)
print(f'Total number of lesions for patients in train: {nb_lesions_train}')
nb_lesions_test = sum(value for patient, value in lesions.items() if patient in test_patients)
print(f'Total number of lesions for patients in test: {nb_lesions_test}')

"""