# Python code to implement the above approach
import argparse

def calc_fitness(W, items, n):
    # Making the dp array
    dp = [0 for i in range(W + 1)]

    # Taking first i elements
    for i in range(1, n + 1):

        # Starting from back,
        # so that we also have data of
        # previous computation when taking i-1 items
        for w in range(W, 0, -1):
            if items[i - 1][1] <= w:
                # Finding the maximum value
                dp[w] = max(dp[w], dp[w - items[i - 1][1]] + items[i - 1][0])

            # Returning the maximum value of knapsack
    return dp[W]

def load_knapsack_data(filename):
    # Load the knapsack data from the file
    # Return the number of items, the maximum weight of the knapsack, and the list of items (price, weight)
    with open(filename, "r") as f:
        n, W = map(int, f.readline().split())
        items = [tuple(map(int, line.split())) for line in f]
    return n, W, items

def print_fitness(individual, items, W):
    print('\t' + f"Best Price = {calc_fitness(W, items, n)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolutionary Algorithm for Knapsack Problem")
    parser.add_argument("filename", type=str, help="Input file containing knapsack data")
    args = parser.parse_args()

    n, W, items = load_knapsack_data(args.filename)
    print('\t' + f"Best Price = {calc_fitness(W, items, n)}")

# This code is contributed by Suyash Saxena
