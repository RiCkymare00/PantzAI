import matplotlib.pyplot as plt
import numpy as np

def plot_columns_from_file(file_path):
    data = []

    with open(file_path, 'r') as f:
        for line in f:
            values = [float(val) for val in line.split()]
            while len(values) < 11:
                values.append(np.nan)
            data.append(values)

    data = np.array(data)
    x = data[:, 0]
    y1, y2, y3, y4 = data[:, 1], data[:, 2], data[:, 3], data[:, 4]

    # Creazione della figura con due grafici affiancati
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Primo grafico: prima e terza colonna di Y
    ax1.scatter(x, y1, label='ansatz: [x], mean', marker='o', color='blue', s =4)
    ax1.scatter(x, y3, label='ansatz [], mean', marker='o', color='red', s = 4)
    ax1.set_xlabel('time_steps')
    ax1.set_ylabel('reward')
    ax1.legend()

    # Secondo grafico: seconda e quarta colonna di Y
    ax2.scatter(x, y2, label='ansatz [x], total', marker='o', color='blue', s = 4)
    ax2.scatter(x, y4, label='ansatz [], total', marker='o', color='red', s = 4)
    ax2.set_xlabel('time_steps')
    ax2.set_ylabel('reward')
    ax2.legend()

    # Salva il grafico come un'unica immagine
    plt.tight_layout()
    plt.savefig("grafico_output.png")

# Esempio di utilizzo
plot_columns_from_file("results.txt")

