import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import imageio
import os
## set random seed for reproducibility
np.random.seed(42)

class EchoBackprop:
    """
    Improved Echo Backpropagation optimizer inspired by bat echolocation.
    """
    
    def __init__(self, 
                 learning_rate=0.01,
                 num_echoes=5,       # Number of echo directions
                 num_distances=8,    # Number of distances per direction
                 alpha_max=0.5,      # Maximum echo distance - INCREASED
                 adaptive_echoes=True,
                 momentum=0.9):      # Added momentum
        
        self.learning_rate = learning_rate
        self.num_echoes = num_echoes
        self.num_distances = num_distances
        self.alpha_max = alpha_max
        self.adaptive_echoes = adaptive_echoes
        self.momentum = momentum
        self.velocity = None  # Will be initialized on first call
        
    def _generate_orthogonal_directions(self, gradient, num_directions):
        """Generate orthogonal basis vectors to the gradient."""
        n = len(gradient)
        basis = np.zeros((num_directions, n))
        
        # Normalize gradient
        grad_norm = np.linalg.norm(gradient)
        if grad_norm < 1e-8:
            # If gradient is too small, use random directions
            basis = np.random.randn(num_directions, n)
            for i in range(num_directions):
                basis[i] = basis[i] / max(1e-8, np.linalg.norm(basis[i]))
            return basis
            
        norm_gradient = gradient / grad_norm
        
        # First direction is negative gradient
        basis[0] = -norm_gradient
        
        # Generate random vectors and use Gram-Schmidt to make them orthogonal
        remaining = num_directions - 1
        if remaining > 0:
            # Create random vectors
            rand_vecs = np.random.randn(remaining, n)
            
            # Apply Gram-Schmidt process
            for i in range(remaining):
                v = rand_vecs[i]
                # Make orthogonal to gradient
                v = v - np.dot(v, norm_gradient) * norm_gradient
                
                # Make orthogonal to previous basis vectors
                for j in range(1, i + 1):
                    v = v - np.dot(v, basis[j]) * basis[j]
                
                # Normalize
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-8:  # Avoid numerical issues
                    v = v / v_norm
                    basis[i+1] = v
                else:
                    # If we get a zero vector, just use a random unit vector
                    new_vec = np.random.randn(n)
                    new_vec = new_vec / np.linalg.norm(new_vec)
                    basis[i+1] = new_vec
        
        return basis
    
    def _quadratic_fit(self, alphas, losses):
        """Fit a quadratic function to (alpha, loss) pairs with robust error handling."""
        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c
        
        try:
            # Add regularization to avoid ill-conditioning
            popt, _ = curve_fit(quadratic, alphas, losses, p0=[1.0, -0.1, losses[0]], 
                                maxfev=5000, method='trf')
            a, b, c = popt
            
            # Sanity checks on quadratic fit
            if a <= 0:  # Non-convex case
                # Find the best empirical point
                best_idx = np.argmin(losses)
                best_alpha = alphas[best_idx]
                return 0.0, -1.0, losses[0]  # Will lead to using alpha_max
            
            return a, b, c
        except Exception as e:
            # Fallback for curve fitting failures
            print(f"Curve fitting failed: {e}. Using fallback.")
            min_idx = np.argmin(losses)
            best_alpha = alphas[min_idx]
            
            # Use empirical best alpha
            if min_idx == 0:  # If minimum is at the smallest alpha
                return 1.0, -0.1, losses[0]  # Small conservative step
            else:
                # Move towards the empirically best point
                return 0.0, -1.0, losses[0]  # Will lead to using best_alpha or alpha_max
    
    def step(self, params, loss_fn, grad_fn):
        """
        Perform one optimization step using Echo Backpropagation.
        
        Args:
            params: Current parameters
            loss_fn: Function that computes loss given parameters
            grad_fn: Function that computes gradient given parameters
            
        Returns:
            new_params: Updated parameters
        """
        # Initialize velocity if needed
        if self.velocity is None:
            self.velocity = np.zeros_like(params)
        
        # Current loss and gradient
        current_loss = loss_fn(params)
        gradient = grad_fn(params)
        
        # Debug info
        print(f"Current loss: {current_loss:.6f}, Gradient norm: {np.linalg.norm(gradient):.6f}")
        
        # Generate echo directions
        echo_directions = self._generate_orthogonal_directions(gradient, self.num_echoes)
        
        # Define echo distances (logarithmically spaced)
        # Use a wider range of step sizes
        echo_distances = np.logspace(-3, np.log10(self.alpha_max), self.num_distances)
        
        # Echo emission and reception matrix: [directions × distances]
        echo_responses = np.zeros((self.num_echoes, self.num_distances))
        
        # Send echoes and collect responses
        for i in range(self.num_echoes):
            direction = echo_directions[i]
            for j in range(self.num_distances):
                distance = echo_distances[j]
                # Emit echo
                echo_params = params + distance * direction
                # Receive echo (evaluate loss)
                echo_responses[i, j] = loss_fn(echo_params)
        
        # Echo analysis and parameter update
        updates = np.zeros_like(params)
        total_weight = 0
        
        for i in range(self.num_echoes):
            # Find best echo in this direction
            best_j = np.argmin(echo_responses[i])
            best_distance = echo_distances[best_j]
            best_loss = echo_responses[i, best_j]
            
            # Compute potential loss reduction
            loss_reduction = current_loss - best_loss
            
            # Fit quadratic to echo profile
            a, b, c = self._quadratic_fit(echo_distances, echo_responses[i])
            
            # Estimate optimal step size
            if a > 0:  # Positive curvature
                optimal_step = -b / (2 * a)
                # Clamp to reasonable range
                optimal_step = min(max(optimal_step, 0), self.alpha_max)
            else:  # Negative or zero curvature
                # Use the best observed step
                optimal_step = best_distance if loss_reduction > 0 else 0
            
            # Only use directions that reduce loss
            if loss_reduction > 0:
                weight = loss_reduction
                updates += weight * optimal_step * echo_directions[i]
                total_weight += weight
                print(f"Direction {i}: optimal_step={optimal_step:.4f}, loss_reduction={loss_reduction:.6f}")
        
        # If no direction reduces loss, use negative gradient with learning rate
        if total_weight <= 0:
            print("No direction reduces loss. Using gradient descent fallback.")
            updates = -self.learning_rate * gradient
        else:
            # Normalize updates by total weight
            updates = updates / total_weight
        
        # Apply momentum
        self.velocity = self.momentum * self.velocity + updates
        
        # Apply update to parameters
        new_params = params + self.velocity
        
        # Validate the update improved the situation
        new_loss = loss_fn(new_params)
        if new_loss >= current_loss:
            print("Warning: Loss increased. Reducing step size.")
            # Reduce step size and try again
            new_params = params + 0.5 * self.velocity
            new_loss = loss_fn(new_params)
            
            # If still no improvement, just do a small gradient step
            if new_loss >= current_loss:
                print("Still no improvement. Using minimal gradient step.")
                new_params = params - 0.01 * gradient / (np.linalg.norm(gradient) + 1e-8)
        
        # Adapt echo strategy for next iteration
        if self.adaptive_echoes:
            improvement_ratio = (current_loss - new_loss) / (current_loss + 1e-10)
            
            # Adjust alpha_max based on improvement
            if improvement_ratio > 0.1:  # Good improvement
                self.alpha_max = min(2.0, self.alpha_max * 1.2)  # Increase exploration range
                print(f"Increasing alpha_max to {self.alpha_max:.4f}")
            elif improvement_ratio < 0.001:  # Poor improvement
                self.alpha_max = max(0.01, self.alpha_max * 0.8)  # Decrease exploration range
                print(f"Decreasing alpha_max to {self.alpha_max:.4f}")
        
        print(f"Loss change: {current_loss:.6f} -> {new_loss:.6f} (delta: {current_loss - new_loss:.6f})")
        return new_params
    
    def _visualize_echoes(self, distances, responses):
        """Visualization disabled."""
        pass

def rosenbrock(params):
    """
    Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    Global minimum at (1,1)
    """
    x, y = params
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(params):
    """Gradient of the Rosenbrock function."""
    x, y = params
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

if __name__ == "__main__":
    # Configure the optimizer with better parameters
    optimizer = EchoBackprop(
        learning_rate=0.01,
        num_echoes=4,
        num_distances=10,
        alpha_max=0.5,  # Larger initial step size
        adaptive_echoes=True,
        momentum=0.9  # Added momentum
    )
    
    # Initial parameters
    params = np.array([-1.0, 1.0])
    
    # Keep track of optimization path
    path_x = [params[0]]
    path_y = [params[1]]
    losses = [rosenbrock(params)]
    
    # Create figure for animation
    fig = plt.figure(figsize=(10, 8))
    frames = []
    
    # Create contour plot of Rosenbrock function (only once)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rosenbrock(np.array([X[i, j], Y[i, j]]))
    
    # Optimization loop
    max_iterations = 100
    for i in range(max_iterations):
        print(f"\nIteration {i}")
        params = optimizer.step(params, rosenbrock, rosenbrock_grad)
        
        # Record path
        path_x.append(params[0])
        path_y.append(params[1])
        losses.append(rosenbrock(params))
        
        # Create frame for animation
        plt.clf()  # Clear the current figure
        
        # Plot contours
        plt.contourf(X, Y, np.log10(Z + 1), levels=50, cmap='viridis', alpha=0.7)
        plt.colorbar(label='log10(f(x,y) + 1)')
        
        # Plot optimization path up to current iteration
        plt.plot(path_x, path_y, 'r-', linewidth=2, label='Optimization Path')
        plt.scatter(path_x[0], path_y[0], color='green', s=100, label='Start')
        plt.scatter(path_x[-1], path_y[-1], color='red', s=100, label='Current')
        plt.scatter(1, 1, color='white', s=100, marker='*', label='Global Minimum')
        
        plt.title(f'Optimization Path (Iteration {i})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        
        # Save frame
        plt.savefig(f'frame_{i:03d}.png')
        frames.append(f'frame_{i:03d}.png')
        
        # Print progress
        if i % 5 == 0 or i == max_iterations - 1:
            print(f"Iteration {i}, Loss: {rosenbrock(params):.6f}, Params: {params}")
        
        # Check for convergence
        if np.linalg.norm(rosenbrock_grad(params)) < 1e-5:
            print(f"Converged after {i+1} iterations!")
            break
    
    print(f"Final parameters: {params}")
    print(f"Final loss: {rosenbrock(params):.6f}")
    print(f"Distance to optimum: {np.linalg.norm(params - np.array([1.0, 1.0])):.6f}")
    
    # Create GIF from frames
    with imageio.get_writer('optimization_path.gif', mode='I', duration=0.2) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)
            # Clean up frame file
            os.remove(frame)
    
    # Plot final loss curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(len(losses)), losses)
    plt.title('Loss vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()
    
    print("Optimization visualization saved as GIF and loss curve saved.")