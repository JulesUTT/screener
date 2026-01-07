# 🚀 Crypto Screener

Un screener crypto complet qui analyse le Top 50 des cryptomonnaies sur Binance.

## 📊 Fonctionnalités

- **Top 50 Cryptos** : Analyse automatique des 50 cryptos les plus tradées sur Binance
- **Timeframe 4H** : Tous les calculs sont basés sur des bougies de 4 heures
- **Indicateurs Techniques** :
  - RSI Stochastique (14 périodes)
  - EMA 13, 25, 32 et 200
- **Signaux de Trading** :
  - **LONG** : Prix > EMA 200 et prix entre EMA 13-32
  - **SHORT** : Prix < EMA 200 et prix entre EMA 13-32
- **Rafraîchissement automatique** toutes les 5 minutes
- **Filtres** pour afficher uniquement les signaux actifs

## 🛠️ Installation

### 1. Prérequis

- Python 3.9 ou supérieur
- pip (gestionnaire de paquets Python)

### 2. Installation des dépendances

```bash
cd crypto_screener
pip install -r requirements.txt
```

### 3. Lancement du serveur

```bash
python app.py
```

Le serveur démarre sur `http://localhost:5000`

## 📁 Structure du projet

```
crypto_screener/
├── app.py              # Serveur Flask (backend)
├── screener.py         # Module de calcul des indicateurs
├── requirements.txt    # Dépendances Python
├── README.md           # Documentation
├── templates/
│   └── index.html      # Interface web
└── static/
    └── style.css       # Styles CSS
```

## 🔌 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Page principale du screener |
| `GET /api/data` | Toutes les données du screener |
| `GET /api/signals` | Uniquement les cryptos avec signal actif |
| `GET /api/refresh` | Force le rafraîchissement des données |
| `GET /api/status` | Statut du serveur et du cache |

## 📈 Logique des signaux

### Conditions pour un signal LONG :
1. Le prix actuel doit être **au-dessus** de l'EMA 200
2. Le prix actuel doit être **entre** l'EMA 13 et l'EMA 32

### Conditions pour un signal SHORT :
1. Le prix actuel doit être **en-dessous** de l'EMA 200
2. Le prix actuel doit être **entre** l'EMA 13 et l'EMA 32

### Pas de signal (AUCUN) :
- Si le prix n'est pas dans la zone EMA 13-32

## ⚙️ Configuration

Le cache des données est configuré pour une durée de 5 minutes (300 secondes).
Vous pouvez modifier cette valeur dans `app.py` :

```python
CACHE_DURATION = 300  # secondes
```

## 🎨 Interface

L'interface est moderne et responsive avec :
- Design sombre pour le confort visuel
- Tableau triable par colonne
- Filtres par type de signal
- Indicateur de chargement
- Rafraîchissement automatique

## 📝 Notes

- Les données proviennent de l'API publique de Binance (pas de clé API requise)
- Le screener analyse uniquement les paires en USDT
- Les calculs sont effectués côté serveur pour optimiser les performances

## 🐛 Dépannage

### Le serveur ne démarre pas
- Vérifiez que Python est installé : `python --version`
- Vérifiez que les dépendances sont installées : `pip list`

### Les données ne se chargent pas
- Vérifiez votre connexion internet
- L'API Binance peut être momentanément indisponible

### Erreur de module
- Réinstallez les dépendances : `pip install -r requirements.txt --force-reinstall`

## 📜 Licence

Ce projet est fourni à des fins éducatives uniquement. 
**Les signaux ne constituent pas des conseils financiers.**

---

Développé avec ❤️ en Python + Flask
