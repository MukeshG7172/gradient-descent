import numpy as np
import matplotlib.pyplot as plt

def make_data(size=100, slope=2.0, intercept=1.0, seed=42):
    # Make data with y = slope*x + intercept + noise
    np.random.seed(seed)
    x = np.linspace(0, 10, size)
    noise = np.random.normal(0, 1, size)
    y = slope * x + intercept + noise
    return x, y

def calc_error(x, y, slope, intercept):
    # Get mean squared error
    pred = slope * x + intercept
    return np.mean((y - pred) ** 2)

def plot_search(x, y, points=100):
    # Search through slope values and plot MSE
    slopes = np.linspace(0, 5, points)
    
    # Get best intercept for each slope
    intercepts = [np.mean(y - m * x) for m in slopes]
    errors = [calc_error(x, y, m, b) for m, b in zip(slopes, intercepts)]
    
    # Find best values
    best_idx = np.argmin(errors)
    best_slope = slopes[best_idx]
    best_intercept = intercepts[best_idx]
    best_error = errors[best_idx]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(slopes, errors)
    plt.scatter(best_slope, best_error, color='red', s=100)
    
    plt.text(best_slope + 0.5, best_error + 0.1, 
             f'Best: {best_slope:.2f}, {best_intercept:.2f}, MSE: {best_error:.2f}')
    
    plt.xlabel('Slope')
    plt.ylabel('Error')
    plt.title('Linear Search')
    plt.savefig('linear_search_graph.png')
    plt.show()
    
    return best_slope, best_intercept, best_error

def best_intercept(x, y, slope):
    # Get best intercept for given slope
    return np.mean(y - slope * x)

def plot_gradient_descent(x, y, learn_rate=0.01, max_iter=None):
    # Random starting point
    slope = np.random.uniform(0, 3)
    intercept = np.random.uniform(0, 3)
    
    if max_iter is None:
        max_iter = 1000
    
    slope_history = [slope]
    intercept_history = [intercept]
    error_history = [calc_error(x, y, slope, intercept)]
    
    # Gradient descent
    iter_count = 0
    prev_error = float('inf')
    
    while iter_count < max_iter:
        iter_count += 1
        
        # Get prediction and error
        pred = slope * x + intercept
        error = y - pred
        
        # Calculate gradients
        grad_slope = -2 * np.mean(x * error)
        grad_intercept = -2 * np.mean(error)
        
        # Update parameters
        slope = slope - learn_rate * grad_slope
        intercept = intercept - learn_rate * grad_intercept
        
        # Calculate current error
        current_error = calc_error(x, y, slope, intercept)
        
        slope_history.append(slope)
        intercept_history.append(intercept)
        error_history.append(current_error)
        
            
        prev_error = current_error
    
    plt.figure(figsize=(10, 6))
    
    # Set plot range
    slopes = np.linspace(0, 5, 100)
    errors = []
    
    # Calculate error curve
    for m in slopes:
        b = best_intercept(x, y, m)
        err = calc_error(x, y, m, b)
        errors.append(err)
    
    plt.plot(slopes, errors)
    
    # Plot GD path
    path_errors = []
    for m in slope_history:
        b = best_intercept(x, y, m)
        path_errors.append(calc_error(x, y, m, b))
    
    plt.scatter(slope_history, path_errors, color='red')
    plt.scatter(slope_history[0], path_errors[0], color='green', s=100)
    plt.scatter(slope_history[-1], path_errors[-1], color='purple', s=100)
    
    plt.text(slope_history[0] + 0.3, path_errors[0] + 0.2, 
             f'Start: {slope_history[0]:.2f}, {intercept_history[0]:.2f}')
    
    plt.text(slope_history[-1] + 0.3, path_errors[-1] - 0.2, 
             f'End: {slope_history[-1]:.2f}, {intercept_history[-1]:.2f}, Epochs: {iter_count}')
    
    plt.xlabel('Slope')
    plt.ylabel('Error')
    plt.title('Gradient Descent')
    plt.savefig('gradient_descent_graph.png')
    plt.show()
    
    return slope_history[-1], intercept_history[-1], error_history[-1], iter_count

def main(seed=42):
    np.random.seed(seed)
    
    x, y = make_data(size=50, slope=2.0, intercept=1.0, seed=seed)
    
    # Run linear search
    ls_slope, ls_intercept, ls_error = plot_search(x, y)
    print(f"Linear Search: slope = {ls_slope:.4f}, intercept = {ls_intercept:.4f}, MSE = {ls_error:.4f}")
    
    # Run gradient descent with dynamic epochs
    gd_slope, gd_intercept, gd_error, epochs = plot_gradient_descent(
        x, y, 
        learn_rate=0.01
    )
    
    print(f"Gradient Descent: slope = {gd_slope:.4f}, intercept = {gd_intercept:.4f}, MSE = {gd_error:.4f}")
    print(f"Epochs needed: {epochs}")

if __name__ == "__main__":
    main()