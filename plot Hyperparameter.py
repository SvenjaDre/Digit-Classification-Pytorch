import pandas as pd
import matplotlib.pyplot as plt

#------- Klassifikation notumor und tumor
# CSV laden
df1 = pd.read_csv("/nfs/homes/sdreyer/Digit-Classification-Pytorch/tudothesis-main/csv/Hyperparameter-noTU-TU_valloss.csv")

# Liste der val_loss-Spalten, die du plotten möchtest
val_loss_spalten_1 = [
    "dulcet-sweep-88 - val_loss",
    "lunar-sweep-81 - val_loss",
    "young-sweep-1 - val_loss",
    "proud-sweep-92 - val_loss",
    "splendid-sweep-83 - val_loss"
]

#custom_labels = [
#    "Dropout= 0.55, Batch= 128, LR= 0.005",
#    "Dropout= 0.5 , Batch= 128, LR= 0.0005",
#    "Dropout= 0.5, Batch= 16 , LR= 0.0001",
#    "Dropout= 0.4, Batch= 128, LR= 0.0005",
#    "Dropout= 0.5, Batch= 128, LR= 0.0005"
#]
custom_labels = [
    "Run 1", "Run 2", "Run 3", "Run 4", "Run 5"
]

# Plot
plt.figure(figsize=(10, 6))
for spalte, label in zip(val_loss_spalten_1, custom_labels):
    plt.plot(df1["epoch"], df1[spalte], label=label)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss bei der Verwendung verschiedener Hyperparameter")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Val_loss_noTu_Tu.pdf")
plt.show()

#---- Klassifikation Glioma und Meningioma

# CSV laden
df2 = pd.read_csv("/nfs/homes/sdreyer/Digit-Classification-Pytorch/tudothesis-main/csv/Hyperparameter-Gli-Men.csv")

# Liste der val_loss-Spalten, die du plotten möchtest
val_loss_spalten_2 = [
    "apricot-sweep-83 - val_loss",
    "swift-sweep-75 - val_loss",
    "sage-sweep-9 - val_loss",
    "eager-sweep-5 - val_loss",
    "mild-sweep-4 - val_loss"
]

#custom_labels = [
#    "Dropout= 0.55, Batch= 128, LR= 0.005",
#    "Dropout= 0.5 , Batch= 128, LR= 0.0005",
#    "Dropout= 0.5, Batch= 16 , LR= 0.0001",
#    "Dropout= 0.4, Batch= 128, LR= 0.0005",
#    "Dropout= 0.5, Batch= 128, LR= 0.0005"
#]


# Plot
plt.figure(figsize=(10, 6))
for spalte, label in zip(val_loss_spalten_2, custom_labels):
    plt.plot(df2["epoch"], df2[spalte], label=label)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss bei der Verwendung verschiedener Hyperparameter")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Val_loss_Gli_Men.pdf")
plt.show()