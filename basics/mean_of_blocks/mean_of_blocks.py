from typing import List

from basics.equipartition_vectors.equipartition_vectors import equipartition_vectors


def mean_of_blocks(blocks: List[List[List[float]]]) -> List[List[float]]:
    block_means = []
    dimension = None  # Initialize dimension as None to indicate it's unset

    for block in blocks:
        if block:
            num_vectors = len(block)
            if block[0]:
                dimension = len(block[0])  # Update dimension based on the first vector of the current block
            sums = [0] * dimension
            for vector in block:
                for i in range(dimension):
                    sums[i] += vector[i]
            means = [total / num_vectors for total in sums]
            block_means.append(means)
        else:
            # If dimension is still None, it means no previous blocks were non-empty to set a dimension
            if dimension is None:
                block_means.append([])  # Append an empty list if no dimension is known
            else:
                block_means.append([0] * dimension)  # Use last known dimension

    return block_means


if __name__ == "__main__":
    # Example usage:
    vectors = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]]
    blocks_ = equipartition_vectors(vectors, 3)
    means_ = mean_of_blocks(blocks_)

    print("Mean of each block:", means_)
