import numpy as np
import matplotlib.pyplot as plt

# Valori di lambda
lambda_values = np.array([0, 0.01, 0.05, 0.1, 1, 2, 5])

# Dati della tabella
training_source_mag = np.array([5.12, 6.878, 7.503, 7.879, 9.075, 9.473, 9.619])
training_source_env = np.array([0.883, 1.304, 1.435, 1.506, 1.205, 1.247, 1.255])
training_source_rt60 = np.array([4.434, 4.356, 4.236, 4.293, 4.337, 4.399, 4.502])
unexplored_source_mag = np.array([5.053, 6.712, 7.391, 7.670, 9.409, 9.468, 9.882])
unexplored_source_env = np.array([1.138, 1.412, 1.682, 1.614, 1.520, 1.503, 1.606])
unexplored_source_rt60 = np.array([5.724, 5.008, 5.002, 4.942, 4.99, 4.958, 5.025])

# Dizionario per i dati
data = {
    "Mag": (training_source_mag, unexplored_source_mag),
    "ENV": (training_source_env, unexplored_source_env),
    "RT60 error [ms]": (training_source_rt60, unexplored_source_rt60)
}

# Creazione dei grafici
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Titoli per le due sezioni
axes[0, 1].set_title("Training Source Position", fontsize=16)
axes[1, 1].set_title("Unexplored Source Position", fontsize=16)

for i, (label, (train_values, unexplored_values)) in enumerate(data.items()):
    axes[0, i].plot(lambda_values, train_values, marker='o', linestyle='-')
    axes[1, i].set_xlabel("λ", fontsize=12)
    axes[0, i].set_ylabel(label, fontsize=12)
    axes[0, i].grid(True)
    
    axes[1, i].plot(lambda_values, unexplored_values, marker='o', linestyle='-')
    axes[1, i].set_xlabel("λ", fontsize=12)
    axes[1, i].set_ylabel(label, fontsize=12)
    axes[1, i].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
