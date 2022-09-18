import numpy as np


def clustering(img, tolerance=255, background=0):
    """Two-pass clustering algorithm.
    img: 2D numpy array
    tolerance: maximum difference between two pixels to be considered
    """
    labels = np.zeros(img.shape, dtype=np.int8)
    current_label = 1
    equivalence = {}

    # First pass
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row, col] == background:
                continue
            # Check if it's connected to a previous label
            neighbors = []
            if row > 0 and img[row - 1, col] != background and abs(img[row, col] - img[row - 1, col]) <= tolerance:
                neighbors.append(labels[row - 1, col])
            if col > 0 and img[row, col - 1] != background and abs(img[row, col] - img[row, col - 1]) <= tolerance:
                neighbors.append(labels[row, col - 1])
            if len(neighbors) == 0:
                # No neighbors, new label
                labels[row, col] = current_label
                current_label += 1
            else:
                # Connected to a previous label
                labels[row, col] = min(neighbors)
                for n in neighbors:
                    if n != labels[row, col]:
                        equivalence[n] = labels[row, col]
    
    # Second pass
    for child, parent in equivalence.items():
        labels[labels == child] = parent

    return labels

if __name__ == '__main__':
    test_img = np.array([
        [0, 1, 0, 0, 0, 9, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 9, 9, 0, 0],
        [0, 1, 1, 9, 0, 0, 0, 9, 9, 9],
        [1, 0, 5, 9, 0, 1, 0, 0, 0, 9],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ])
    print('--- Input image ---')
    print(test_img)
    print('--- standard 2-pass clustering ---')
    print(clustering(test_img))
    print('--- 2-pass clustering with tolerance 2 ---')
    print(clustering(test_img, tolerance=3))
    print('--- 2-pass clustering with tolerance 5 ---')
    print(clustering(test_img, tolerance=5))
