import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Teil 1: wandb-Daten exportieren ---
api = wandb.Api()
runs = api.runs("svenja-dreyer-tu-dortmund/Messungen-noTu-Tu")

data = []
for run in runs:
    row = run.config.copy()
    row.update(run.summary)
    row["name"] = run.name
    row["id"] = run.id
    data.append(row)

df = pd.DataFrame(data)
csv_path = "wandb_export.csv"
df.to_csv(csv_path, index=False)

# --- Teil 2: CSV laden ---
df = pd.read_csv(csv_path)

# Ordner für Plots erstellen, falls nicht vorhanden
os.makedirs("plots", exist_ok=True)

# --- Funktion zum Plotten ---
def plot_metric(metric_name, ylabel, filename):
    grouped = df.groupby("train_samples_used").agg({
        metric_name: ["mean", "std"]
    }).reset_index()
    grouped.columns = ["train_samples_used", f"{metric_name}_mean", f"{metric_name}_std"]

    plt.figure(figsize=(10, 6))
    plt.errorbar(grouped["train_samples_used"], grouped[f"{metric_name}_mean"],
                 yerr=grouped[f"{metric_name}_std"], fmt='none', ecolor='cornflowerblue',
                 capsize=5, label="Fehlerbalken")

    plt.scatter(grouped["train_samples_used"], grouped[f"{metric_name}_mean"],
                color='red', marker='x', label="Mittelwert")

    plt.xlabel("Trainingssamples")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} in Abhängigkeit der Trainingssamples")
    plt.xticks(grouped["train_samples_used"])
    plt.grid(True)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{filename}")
    plt.show()

# --- Teil 3: Drei Plots erzeugen ---
plot_metric("val_accuracy", "Validation Accuracy / %", "Accuracy_mean.pdf")
plot_metric("val_sensitivity", "Validation Sensitivity ", "Sensitivity_mean.pdf")
plot_metric("val_specificity", "Validation Specificity ", "Specificity_mean.pdf")
