import random
import math

def generar_uniforme(a, b, n):
    numbers = []
    for i in range(n):
        random.random()
        rnd = (math.trunc(random.random() * 10000) / 10000)
        x = a + (rnd * (b - a))
        numbers.append(x)
    print(len(numbers))
    return numbers

