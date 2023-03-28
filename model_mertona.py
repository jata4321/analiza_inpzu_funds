from scipy.stats import norm
import numpy as np

# Dane wejściowe
V = 100  # Wartość rynkowa aktywów
D = 50  # Wartość nominalna zadłużenia
r = 0.05  # Stopa wolna od ryzyka
T = 1  # Czas do zapadalności zadłużenia
sigma = 0.3  # Zmienność aktywów

# Obliczenia
d1 = (np.log(V / D) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
B = D * np.exp(-r * T) * norm.cdf(d2)  # Wartość długu
V = V * norm.cdf(d1)  # Wartość aktywów
E = V - B  # Wartość kapitału własnego
p = 1 - norm.cdf(d2)  # Prawdopodobieństwo niewypłacalności

# Wyniki:
print("Wartość długu:", B)
print("Wartość kapitału własnego:", E)
print("Prawdopodobieństwo niewypłacalności:", p)
