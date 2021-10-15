import numpy as np

# Uses least squares regression to find the solution
# when there is one unknown variable.
def print_lstsq_solution(A, B):
    A_inv = np.linalg.pinv(A)
    x = np.matmul(A_inv, B)
    print("least squares solution:", x[0][0])

# Uses the pseudoinverse matrix to find the solution
# when there are two unknown variables.
def print_pinv_solution(A, mv, B):
    new_A = np.concatenate((A, mv), axis=1)
    new_A_inv = np.linalg.pinv(new_A)
    new_x = np.matmul(new_A_inv, B)
    print("pinv solution:", new_x[0][0], new_x[1][0])

# Traverses the data and prints out one value for
# each update type.
def print_solutions(file_path):
    data = np.genfromtxt(file_path, delimiter="\t")

    prev_update = 0
    split_list_indices = list()
    for i, val in enumerate(data):
        if prev_update != val[3]:
            split_list_indices.append(i)
            prev_update = val[3]

    split = np.split(data, split_list_indices)

    for array in split:
        A, mv, B, update = np.hsplit(array, 4)
        print("update type:", update[0][0])
        print_lstsq_solution(A, B)
        print_pinv_solution(A, mv, B)
        print()

if __name__ == "__main__":
    print_solutions("data/lowres_64f_target150_data.txt")
