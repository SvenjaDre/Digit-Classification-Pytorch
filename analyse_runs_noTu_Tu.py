import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os

# Falls du noch nicht eingeloggt bist, öffnet sich ein Login-Link
wandb.login()

# Deine Projektinfos
entity = "svenja-dreyer-tu-dortmund"
project = "2-Messungen-noTu-Tu"

# W&B API initialisieren
api = wandb.Api()

# Runs aus dem Projekt abrufen
runs = api.runs(f"{entity}/{project}")

# Liste zum Speichern der Run-Daten
data = []

# Alle Runs durchgehen
for run in runs:
    summary = run.summary
    config = run.config
    name = run.name
    run_id = run.id

    # Relevante Infos aus summary & config extrahieren
    row = {
        "run_id": run_id,
        "name": name,
        "test_accuracy": summary.get("test_accuracy"),
        "test_sensitivity": summary.get("test_sensitivity"),
        "test_specificity": summary.get("test_specificity"),
        "val_accuracy": summary.get("val_accuracy"),
        "train_samples": config.get("train_samples"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "epochs": config.get("epochs"),
    }

    data.append(row)

# In DataFrame umwandeln
df = pd.DataFrame(data)

# Ordner für CSV-Dateien und Plots erstellen
os.makedirs("csv", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Als CSV speichern (rohe Daten)
csv_filename = "csv/wandb_runs_2-Messungen-noTu-Tu.csv"
df.to_csv(csv_filename, index=False)

print(f"✅ {len(df)} Runs erfolgreich exportiert in: {csv_filename}")
#print(df.head())

# --- Aggregierte Mittelwerte und Std berechnen ---
def get_agg_df(metric_name):
    grouped = df.groupby("train_samples").agg({
        metric_name: ["mean", "std"]
    }).reset_index()
    grouped.columns = ["train_samples", f"{metric_name}_mean", f"{metric_name}_std"]
    return grouped

# Aggregierte DataFrames erzeugen
acc_df = get_agg_df("test_accuracy")
sens_df = get_agg_df("test_sensitivity")
spec_df = get_agg_df("test_specificity")

# Alle in einem DataFrame zusammenführen
agg_df = acc_df.merge(sens_df, on="train_samples").merge(spec_df, on="train_samples")

# Aggregierte Daten als CSV speichern
agg_csv_filename = "noTu_Tu_aggregated_metrics.csv"
agg_df.to_csv(agg_csv_filename, index=False)

print(f"Aggregierte Mittelwert- und Std-Daten gespeichert in: {agg_csv_filename}")

# Ordner für Plots erstellen, falls nicht vorhanden
os.makedirs("plots", exist_ok=True)

# --- Funktion zum Plotten ---
def plot_metric(metric_name, ylabel, filename):
    grouped = agg_df[["train_samples", f"{metric_name}_mean", f"{metric_name}_std"]]

    plt.figure(figsize=(10, 6))
    plt.errorbar(grouped["train_samples"], grouped[f"{metric_name}_mean"],
                 yerr=grouped[f"{metric_name}_std"], fmt='none', ecolor='cornflowerblue',
                 capsize=5, label="Fehlerbalken")

    plt.scatter(grouped["train_samples"], grouped[f"{metric_name}_mean"],
                color='red', marker='x', label="Mittelwert")

    plt.xlabel("Trainingssamples")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} in Abhängigkeit der Trainingssamples bei der Unterscheidung no Tumor & Tumor")
    plt.xticks(grouped["train_samples"])
    plt.grid(True)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{filename}")
    plt.show()

# --- Drei Plots erzeugen ---
plot_metric("test_accuracy", "Test Accuracy / %", "Accuracy_noTu_Tu_mean.pdf")
plot_metric("test_sensitivity", "Test Sensitivity", "Sensitivity_noTu_Tu_mean.pdf")
plot_metric("test_specificity", "Test Specificity", "Specificity_noTu_Tu_mean.pdf")
