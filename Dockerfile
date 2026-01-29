# 1. Image de base Python
FROM python:3.10-slim

# 2. Dossier de travail dans le container
WORKDIR /app

# 3. Copier les fichiers
COPY requirements.txt .
COPY main.py .

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Commande de lancement
CMD ["python", "main.py"]
