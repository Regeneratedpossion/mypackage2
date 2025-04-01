def plot_gmm_2d(X, y, means, covariances):
    """
     Visualize 2D GMM data and theoretical distributions
    
    Parameters:
        X (np.ndarray): Data points, shape (n_samples, 2)
        y (np.ndarray): True class labels, shape (n_samples,)
        means (list): List of true mean vectors
        covariances (list):  List of true covariance matrices
    """
    
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    K = len(means) # Number of components
    
    # === Plot data points ===
    for k in range(K):
        plt.scatter(X[y == k, 0], X[y == k, 1], # x and y coordinates
                    c=colors[k], alpha=0.6,   # Color and transparenc
                    label=f'Component {k}', s=30,   # Color and transparenc
                    edgecolor='white'  # Edge color
                   )
    
     # === Plot covariance ellipses ===
    for k in range(K):
        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eigh(covariances[k])
        # Compute ellipse rotation angle (radians to degrees)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 4 * np.sqrt(eigvals)
        # Compute axis lengths (scaled by 4 for visibility)

        # Create ellipse patch
        ellipse = plt.matplotlib.patches.Ellipse(
            xy=means[k],  # Center coordinates
            width=width,
            height=height,
            angle=angle,
            fill=False,
            linestyle='--',
            linewidth=2,
            edgecolor=colors[k]
        )
        plt.gca().add_patch(ellipse)# Add ellipse to the plot

        # Mark the mean with an 'X'
        plt.scatter(means[k][0], means[k][1], 
                    c='black', marker='X', s=200, edgecolor='white')# Large black 'X' marker


    # === Plot styling ===
    plt.title(f'2D GMM Generated Data (K={K})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(alpha=0.3)  # Semi-transparent grid
    plt.show()