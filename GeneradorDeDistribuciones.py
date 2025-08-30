import random
import math

def generar_uniforme(a, b, n):
    numbers = []
    for i in range(n):
        rnd = random.random()
        x = a + (rnd * (b - a))
        x_truncated = math.trunc(x * 10000) / 10000
        numbers.append(x_truncated)
    return numbers

def generar_normal(media, desv_estandar, n):
    numbers = []
    for i in range(n):
        r1 = random.random()
        r2 = random.random()
        n1 = (math.sqrt(-2 * math.log(r1, math.e)) * math.cos(2 * math.pi * r2)) * desv_estandar + media
        n1_truncated = math.trunc(n1 * 10000) / 10000
        numbers.append(n1_truncated)
        n2 = (math.sqrt(-2 * math.log(r1, math.e)) * math.sin(2 * math.pi * r2)) * desv_estandar + media
        n2_truncated = math.trunc(n2 * 10000) / 10000
        numbers.append(n2_truncated)
    return numbers
