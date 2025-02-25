import os

# Récupérer les fichiers du dossier courant
files = [f for f in os.listdir() if f.endswith(".json") or f.endswith(".wav")]

# Séparer les fichiers json et wav
json_files = {f[:-5] for f in files if f.endswith(".json")}
wav_files = {f[:-4] for f in files if f.endswith(".wav")}

# Vérifier les correspondances
missing_json = wav_files - json_files
missing_wav = json_files - wav_files

# Afficher les fichiers classés
print("📂 Fichiers présents (JSON et WAV) par ordre alphabétique :")
for file in sorted(files):
    print(file)

# Afficher les anomalies
if missing_json:
    print("\n⚠️  Fichiers .wav sans .json correspondant :", sorted(missing_json))
if missing_wav:
    print("\n⚠️  Fichiers .json sans .wav correspondant :", sorted(missing_wav))
