import math

# z pdfa przepisane
w1 = 0.2;  w2 = -0.3;  b1 = 0.4
w3 = -0.5; w4 = 0.1;   b2 = -0.2
w5 = 0.3;  w6 = -0.4;  b3 = 0.2

x1 = 0.6
x2 = 0.1
y = 0.8

eta = 0.1

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_pochodna(h):
    return h * (1 - h)

# propagacja w przod
def forward(x1, x2):
    z1 = w1*x1 + w2*x2 + b1
    h1 = sigmoid(z1)

    z2 = w3*x1 + w4*x2 + b2
    h2 = sigmoid(z2)

    y_hat = w5*h1 + w6*h2 + b3

    return h1, h2, y_hat

# a) propagacja w przod
h1, h2, y_hat = forward(x1, x2)
print("h1 =", round(h1, 4))
print("h2 =", round(h2, 4))
print("y_hat =", round(y_hat, 4), "  (oczekiwane ok. 0.234)")

# c) obliczenie straty
L = 0.5 * (y_hat - y) ** 2
print("strata L =", round(L, 4))

# krok 1 blad na wyjsciu
delta3 = y_hat - y
print("delta3 =", round(delta3, 4))

# krok 2 gradienty wag wyjsciowych
grad_w5 = delta3 * h1
grad_w6 = delta3 * h2
grad_b3 = delta3

# krok 3propagacja bledu do warstwy ukrytej
delta1 = delta3 * w5 * sigmoid_pochodna(h1)
delta2 = delta3 * w6 * sigmoid_pochodna(h2)

# krok 4 gradienty wag ukrytych
grad_w1 = delta1 * x1
grad_w2 = delta1 * x2
grad_b1 = delta1

grad_w3 = delta2 * x1
grad_w4 = delta2 * x2
grad_b2 = delta2

# d) inne wagi sprawdzamy
w5_new = w5 - eta * grad_w5
w6_new = w6 - eta * grad_w6
b3_new = b3 - eta * grad_b3

w1_new = w1 - eta * grad_w1
w2_new = w2 - eta * grad_w2
b1_new = b1 - eta * grad_b1

w3_new = w3 - eta * grad_w3
w4_new = w4 - eta * grad_w4
b2_new = b2 - eta * grad_b2

print("\nnowe wagi warstwy ukrytej:")
print("w1_new =", round(w1_new, 4), "  w2_new =", round(w2_new, 4), "  b1_new =", round(b1_new, 4))
print("w3_new =", round(w3_new, 4), "  w4_new =", round(w4_new, 4), "  b2_new =", round(b2_new, 4))
print("\nnowe wagi warstwy wyjsciowej:")
print("w5_new =", round(w5_new, 4), "  w6_new =", round(w6_new, 4), "  b3_new =", round(b3_new, 4))