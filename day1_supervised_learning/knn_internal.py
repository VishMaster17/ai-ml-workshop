from collections import Counter
import math


data = [
    ((160, 8), "apple"),
    ((170, 6), "banana"),
    ((180, 5), "banana"),
    ((190, 7), "apple"),
    ((200, 8), "banana"),
    ((210, 9), "apple"),
    ((220, 10), "banana"),
    ((230, 11), "apple"),
]

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def knn(new_point, k):
    """Find the k nearest neighbors of a new point."""
    distances = []
    for features, point in data:
        distance = euclidean_distance(features, new_point)
        distances.append((distance, point))
        
    distances.sort(key=lambda x: x[0])
    
    neighbors = [label for _, label in distances[:k]]
    vote = Counter(neighbors)
    most_common = vote.most_common(1)[0][0]
    print(f"The most common label among the {k} nearest neighbors is: {most_common}")
    
    return most_common

new_point = (190, 8)
k = 4
neighbors = knn(new_point, k)