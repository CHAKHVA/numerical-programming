import numpy as np

while True:
    try:
        vector_components = input("Enter a vector: ").split()
        if not vector_components:
            raise ValueError
        vector = [float(x) for x in vector_components]
        break
    except:
        print("Invalid input. Please enter real numbers only.")


first_norm = np.sum(np.abs(vector))

second_form = np.sqrt(np.sum(np.abs(vector)**2))

infinite_norm = np.max(np.abs(vector))

p = 3
p_norm = np.sum(np.abs(vector)**p)**(1/p)

print(f"First norm: {first_norm}")
print(f"Second norm: {second_form}")
print(f"P norm: {p_norm}")
print(f"Infinite norm: {infinite_norm}")


