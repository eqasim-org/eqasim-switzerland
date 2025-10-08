import numpy as np
from geopy.distance import distance
from geopy.point import Point

def geo_indistinguishability_noise(epsilon, sensitivity=1.0):
    # Sample radius from exponential distribution
    r = np.random.exponential(scale=sensitivity / epsilon)

    # Sample angle uniformly
    theta = np.random.uniform(0, 2 * np.pi)

    # Convert polar to Cartesian
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return dx, dy

def add_geo_noise(lat, lon, epsilon):
    dx, dy = geo_indistinguishability_noise(epsilon)
    
    # Convert dx/dy in kilometers
    noisy_point = distance(kilometers=dy).destination(Point(lat, lon), bearing=0)
    noisy_point = distance(kilometers=dx).destination(noisy_point, bearing=90)
    
    return noisy_point.latitude, noisy_point.longitude

def analyze_noise_impact(n_samples=1000, epsilon=1.0):
    # Initialize arrays to store distances
    distances_km = []
    
    # Reference point (somewhere in Switzerland)
    base_lat, base_lon = 47.3769, 8.5417  # Zurich coordinates
    
    for _ in range(n_samples):
        # Get noisy coordinates
        noisy_lat, noisy_lon = add_geo_noise(base_lat, base_lon, epsilon)
        
        # Calculate distance between original and noisy points
        dist = distance((base_lat, base_lon), (noisy_lat, noisy_lon)).kilometers
        distances_km.append(dist)
    
    distances = np.array(distances_km)
    
    print(f"Statistics for ε = {epsilon}:")
    print(f"Average distance: {np.mean(distances):.2f} km")
    print(f"Median distance: {np.median(distances):.2f} km")
    print(f"Standard deviation: {np.std(distances):.2f} km")
    print(f"95th percentile: {np.percentile(distances, 95):.2f} km")
    print(f"Maximum distance: {np.max(distances):.2f} km")

if __name__ == "__main__":
    analyze_noise_impact(n_samples=10000, epsilon=1.0)
