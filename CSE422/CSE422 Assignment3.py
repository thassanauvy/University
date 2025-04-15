import numpy as np


def fitness(population, n):
    
    population_size = population.shape[0]
    fitness_scores = np.zeros(population_size)
    
    max_attacking_pairs = n * (n - 1) // 2
    
    for i in range(population_size):
        individual = population[i]
        attacking_pairs = 0
        
        for j in range(n):
            for k in range(j + 1, n):
              
                if abs(j - k) == abs(individual[j] - individual[k]):
                    attacking_pairs += 1
        
        fitness_scores[i] = max_attacking_pairs - attacking_pairs
    
    return fitness_scores


def select(population, fit):
    
    total_fitness = np.sum(fit)
    if total_fitness == 0: 
        probabilities = np.ones(len(fit)) / len(fit)
    else:
        probabilities = fit / total_fitness
    
    selected_index = np.random.choice(len(population), size=1, p=probabilities)[0]
    return population[selected_index].copy()


def crossover(x, y):
    
    n = len(x)
    c = np.random.randint(1, n)
    
    child = np.concatenate((x[:c], y[c:]))
    return child


def mutate(child):
    
    n = len(child)
    position = np.random.randint(0, n)
    child[position] = np.random.randint(0, n)
    return child


def GA(population, n, mutation_threshold=0.3, max_generations=1000):
    
    max_possible_fitness = n * (n - 1) // 2
    
    best_fitness = 0
    best_solution = None
    
    generation = 0
    
    while best_fitness < max_possible_fitness and generation < max_generations:
        fit = fitness(population, n)
        
        current_best_idx = np.argmax(fit)
        current_best_fitness = fit[current_best_idx]
        
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_idx].copy()
            print(f"Generation {generation}: New best fitness = {best_fitness}/{max_possible_fitness}")
            print(f"Best solution so far: {best_solution}")
        
        new_population = np.zeros_like(population)
        
        for i in range(len(population)):
            parent_x = select(population, fit)
            parent_y = select(population, fit)
            
            child = crossover(parent_x, parent_y)
            
            if np.random.random() < mutation_threshold:
                child = mutate(child)
            
            new_population[i] = child
        
        population = new_population
        
        generation += 1
    
    
    print("\n=== Final Result ===")
    print(f"Best fitness: {best_fitness}/{max_possible_fitness}")
    print(f"Solution found: {best_solution}")
    
    print("\nChessboard visualization:")
    visualize_board(best_solution, n)
    
    return best_solution, best_fitness


def visualize_board(solution, n):
    
    for i in range(n):
        row = ""
        for j in range(n):
            if solution[j] == i:
                row += "Q "
            else:
                row += ". "
        print(row)


'''for 8 queen problem, n = 8'''
n = 8

'''start_population denotes how many individuals/chromosomes are there
  in the initial population n = 8'''
start_population = 10

'''if you want you can set mutation_threshold to a higher value,
   to increase the chances of mutation'''
mutation_threshold = 0.3

'''creating the population with random integers between 0 to 7 inclusive
   for n = 8 queen problem'''
population = np.random.randint(0, n, (start_population, n))

'''calling the GA function'''
GA(population, n, mutation_threshold)


import random
import math

def strength(x):
    return math.log2(x + 1) + x / 10

def calculate_utility(maxV, minV):
  
    i = random.randint(0, 1)
    random_component = ((-1) ** i) * random.randint(1, 10) / 10
    return strength(maxV) - strength(minV) + random_component

def generate_game_tree(depth, maxV, minV):
    if depth == 0:
        return calculate_utility(maxV, minV)
    return [generate_game_tree(depth - 1, maxV, minV) for _ in range(2)]

def minimax_ab(node, depth, alpha, beta, is_maximizing, maxV, minV):
    if depth == 0 or not isinstance(node, list):
        return node

    if is_maximizing:
        value = float('-inf')
        for child in node:
            value = max(value, minimax_ab(child, depth - 1, alpha, beta, False, maxV, minV))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  
        return value
    else:
        value = float('inf')
        for child in node:
            value = min(value, minimax_ab(child, depth - 1, alpha, beta, True, maxV, minV))
            beta = min(beta, value)
            if alpha >= beta:
                break  
        return value

def minimax_with_mind_control(node, depth, alpha, beta, is_maximizing, maxV, minV):
  
    if depth == 0 or not isinstance(node, list):
        return node

    if is_maximizing:
        value = float('-inf')
        for child in node:
            value = max(value, minimax_with_mind_control(child, depth - 1, alpha, beta, True, maxV, minV))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float('inf')
        for child in node:
            value = min(value, minimax_with_mind_control(child, depth - 1, alpha, beta, False, maxV, minV))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

def determine_winner(utility_value, max_player, min_player):
    """Determine the winner based on utility value"""
    if utility_value > 0:
        return f"{max_player} (Max) (Utility value: {utility_value:.2f})"
    elif utility_value < 0:
        return f"{min_player} (Min) (Utility value: {utility_value:.2f})"
    else:
        return f"Draw (Utility value: {utility_value:.2f})"

# Main function for Problem 1
def problem1():
    print("==== Problem 1: Chess Masters ====")
    
    # Get inputs
    starting_player = int(input("Enter starting player for game 1 (0 for Carlsen, 1 for Caruana): "))
    carlsen_strength = float(input("Enter base strength for Carlsen: "))
    caruana_strength = float(input("Enter base strength for Caruana: "))
    
    players = ["Magnus Carlsen", "Fabiano Caruana"]
    wins = {players[0]: 0, players[1]: 0, "Draw": 0}
    
    # Play 4 games
    for game in range(1, 5):
      
        max_player_index = (starting_player + game - 1) % 2
        min_player_index = 1 - max_player_index
        max_player = players[max_player_index]
        min_player = players[min_player_index]
        
        if max_player_index == 0:  # Carlsen is Max
            maxV = carlsen_strength
            minV = caruana_strength
        else:  # Caruana is Max
            maxV = caruana_strength
            minV = carlsen_strength
        
        # Generate game tree with depth 5
        game_tree = generate_game_tree(5, maxV, minV)
        
        
        utility_value = minimax_ab(game_tree, 5, float('-inf'), float('inf'), True, maxV, minV)
        
        # Determine winner
        result = determine_winner(utility_value, max_player, min_player)
        print(f"Game {game} Winner: {result}")
        
        # Update win counts
        if utility_value > 0:
            wins[max_player] += 1
        elif utility_value < 0:
            wins[min_player] += 1
        else:
            wins["Draw"] += 1
    
    # Print overall results
    print("Overall Results:")
    print(f"Magnus Carlsen Wins: {wins['Magnus Carlsen']}")
    print(f"Fabiano Caruana Wins: {wins['Fabiano Caruana']}")
    print(f"Draws: {wins['Draw']}")
    
    # Determine overall winner
    if wins['Magnus Carlsen'] > wins['Fabiano Caruana']:
        print("Overall Winner: Magnus Carlsen")
    elif wins['Magnus Carlsen'] < wins['Fabiano Caruana']:
        print("Overall Winner: Fabiano Caruana")
    else:
        print("Overall Winner: Draw")

# Main function for Problem 2
def problem2():
    print("\n==== Problem 2: Chess Noobs with Magic ====")
    
    # Get inputs
    first_player = int(input("Enter who goes first (0 for Light, 1 for L): "))
    mind_control_cost = float(input("Enter the cost of using Mind Control: "))
    light_strength = float(input("Enter base strength for Light: "))
    l_strength = float(input("Enter base strength for L: "))
    
    # Set players
    players_p2 = ["Light", "L"]
    max_player = players_p2[first_player]
    min_player = players_p2[1 - first_player]
    
    # Set strength values
    if first_player == 0:  # Light is Max
        maxV = light_strength
        minV = l_strength
    else:  # L is Max
        maxV = l_strength
        minV = light_strength
    
    # Generate game tree
    game_tree = generate_game_tree(5, maxV, minV)
    
    # Apply regular minimax with alpha-beta pruning
    regular_value = minimax_ab(game_tree, 5, float('-inf'), float('inf'), True, maxV, minV)
    
    # Apply mind control minimax
    mind_control_value = minimax_with_mind_control(game_tree, 5, float('-inf'), float('inf'), True, maxV, minV)
    
    # Calculate final value after cost
    final_mind_control_value = mind_control_value - mind_control_cost
    
    # Print results
    print(f"Minimax value without Mind Control: {regular_value:.2f}")
    print(f"Minimax value with Mind Control: {mind_control_value:.2f}")
    print(f"Minimax value with Mind Control after incurring the cost: {final_mind_control_value:.2f}")
    
    # Recommendation
    if regular_value > 0 and final_mind_control_value > 0:
        print(f"{max_player} should NOT use Mind Control as the position is already winning.")
    elif regular_value < 0 and final_mind_control_value > 0:
        print(f"{max_player} should use Mind Control.")
    elif regular_value < 0 and final_mind_control_value < 0:
        print(f"{max_player} should NOT use Mind Control as the position is losing either way.")
    elif regular_value > 0 and final_mind_control_value < 0:
        print(f"{max_player} should NOT use Mind Control as it backfires.")
    else:
        print(f"{max_player} should NOT use Mind Control.")

# Main function
def main():
    problem1()
    problem2()

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()