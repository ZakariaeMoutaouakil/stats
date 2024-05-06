from typing import List, Any


def equipartition_vectors(vectors: List[List[Any]], k: int):
    n = len(vectors)  # Total number of vectors
    base_size = n // k  # Minimum number of vectors per block
    remainder = n % k  # Extra vectors to distribute

    blocks: List[List[List[Any]]] = []
    start_index = 0

    # Create each block
    for i in range(k):
        # Calculate the number of vectors in the current block
        if i < remainder:
            block_size = base_size + 1
        else:
            block_size = base_size

        # Slice the portion of the list that corresponds to the current block
        block = vectors[start_index:start_index + block_size]
        blocks.append(block)

        # Update the start index for the next block
        start_index += block_size

    return blocks


if __name__ == "__main__":
    # Example usage:
    # Define some vectors (as lists for simplicity)
    vectors_ = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18]]

    # Partition into 3 blocks
    blocks_ = equipartition_vectors(vectors_, 3)

    # Print the blocks
    for j, block_ in enumerate(blocks_):
        print(f"Block {j + 1}: {block_}")
