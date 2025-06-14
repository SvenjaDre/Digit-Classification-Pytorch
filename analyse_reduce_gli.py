import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os

# W&B Login
wandb.login()

# Projektinfos
entity = "svenja-dreyer-tu-dortmund"

project = "Reduzierung-Gli + Balnce"

# Basisverzeichnisse setzen
base_dir = "tudothesis-main"
csv_dir = os.path.join(base_dir, "csv")
plots_dir = os.path.join(base_dir, "plots")
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# W&B API initialisieren & Runs laden
api = wandb.Api()
runs = api.runs(f"{entity}/{project}")

# Daten sammeln
data = []
for run in runs:
    summary = run.summary
    config = run.config
    row = {
        "run_id": run.id,
        "name": run.name,
        "test_accuracy": summary.get("test_accuracy"),
        "test_sensitivity": summary.get("test_sensitivity"),
        "test_specificity": summary.get("test_specificity"),
        "val_accuracy": summary.get("val_accuracy"),
        "glioma_samples": summary.get("glioma_train"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "epochs": config.get("epochs"),
    }
    data.append(row)

# DataFrame erzeugen & CSV speichern
df = pd.DataFrame(data)
csv_filename = os.path.join(csv_dir, f"{project}_raw_runs.csv")
df.to_csv(csv_filename, index=False)
print(f"✅ {len(df)} Runs exportiert nach: {csv_filename}")

# Aggregierte Statistik berechnen
def get_agg_df(metric):
    grouped = df.groupby("glioma_samples").agg({metric: ["mean", "std"]}).reset_index()
    grouped.columns = ["glioma_samples", f"{metric}_mean", f"{metric}_std"]
    return grouped

acc_df = get_agg_df("test_accuracy")
sens_df = get_agg_df("test_sensitivity")
spec_df = get_agg_df("test_specificity")
agg_df = acc_df.merge(sens_df, on="glioma_samples").merge(spec_df, on="glioma_samples")

# Aggregierte CSV speichern
agg_csv_filename = os.path.join(csv_dir, f"{project}_aggregated_metrics.csv")
agg_df.to_csv(agg_csv_filename, index=False)
print(f"✅ Aggregierte Daten gespeichert nach: {agg_csv_filename}")

# Plotfunktion definieren
def plot_metric(metric, ylabel, suffix):
    grouped = agg_df[["glioma_samples", f"{metric}_mean", f"{metric}_std"]]

    plt.figure(figsize=(10, 6))
    plt.errorbar(grouped["glioma_samples"], grouped[f"{metric}_mean"],
                 yerr=grouped[f"{metric}_std"], fmt='none', ecolor='cornflowerblue',
                 capsize=5, label="Fehlerbalken")

    plt.scatter(grouped["glioma_samples"], grouped[f"{metric}_mean"],
                color='red', marker='x', label="Mittelwert")

    plt.xlabel("Glioma samples")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} in Abhängigkeit der Trainingssamples")
    plt.xticks(grouped["glioma_samples"])
    plt.grid(True)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(plots_dir, f"{project}_{suffix}.pdf")
    plt.savefig(plot_path)
    plt.show()
    print(f"📈 Plot gespeichert: {plot_path}")

# Plots erstellen
plot_metric("test_accuracy", "Test Accuracy / %", "Accuracy_mean")
plot_metric("test_sensitivity", "Test Sensitivity", "Sensitivity_mean")
plot_metric("test_specificity", "Test Specificity", "Specificity_mean")
