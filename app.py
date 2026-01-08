"""
Screener Crypto - Application Flask
====================================
Serveur web pour afficher les résultats du screener crypto.
Fournit une API REST et une interface web.
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
from screener import CryptoScreener
from datetime import datetime
import threading
import time

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)  # Autoriser les requêtes cross-origin

# Instance du screener
screener = CryptoScreener()

# Cache pour les données (pour éviter de surcharger l'API Binance)
cache = {
    'data': [],
    'last_update': None,
    'is_updating': False
}

# Durée de validité du cache en secondes (15 minutes)
CACHE_DURATION = 900


def update_cache():
    """
    Met à jour le cache avec les dernières données du screener.
    Cette fonction est thread-safe.
    """
    global cache
    
    if cache['is_updating']:
        return
    
    cache['is_updating'] = True
    
    try:
        print(f"[{datetime.now()}] Mise à jour du cache...")
        data = screener.scan_all()
        cache['data'] = data
        cache['last_update'] = datetime.now()
        print(f"[{datetime.now()}] Cache mis à jour avec {len(data)} cryptos")
    except Exception as e:
        print(f"Erreur lors de la mise à jour du cache: {e}")
    finally:
        cache['is_updating'] = False


def get_cached_data():
    """
    Récupère les données du cache, en les rafraîchissant si nécessaire.
    
    Returns:
        Liste des données du screener
    """
    # Vérifier si le cache est valide
    if (cache['last_update'] is None or 
        (datetime.now() - cache['last_update']).total_seconds() > CACHE_DURATION):
        update_cache()
    
    return cache['data']


# ========== Routes Web ==========

@app.route('/')
def index():
    """
    Page principale du screener.
    """
    return render_template('index.html')


# ========== Routes API ==========

@app.route('/api/data')
def get_data():
    """
    API pour récupérer les cryptos avec un setup potentiel (LONG ou SHORT uniquement).
    
    Returns:
        JSON avec les données des cryptos ayant un signal actif
    """
    try:
        data = get_cached_data()
        
        # Filtrer pour ne garder que les signaux LONG ou SHORT
        filtered_data = [item for item in data if item['signal'] in ['LONG', 'SHORT']]
        
        return jsonify({
            'success': True,
            'count': len(filtered_data),
            'total_scanned': len(data),
            'last_update': cache['last_update'].isoformat() if cache['last_update'] else None,
            'data': filtered_data
        })
    except Exception as e:
        print(f"ERREUR API /api/data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'data': []
        })


@app.route('/api/signals')
def get_signals():
    """
    API pour récupérer uniquement les cryptos avec un signal actif (LONG ou SHORT).
    
    Returns:
        JSON avec les données des cryptos ayant un signal
    """
    data = get_cached_data()
    signals = [item for item in data if item['signal'] != 'AUCUN']
    
    return jsonify({
        'success': True,
        'count': len(signals),
        'last_update': cache['last_update'].isoformat() if cache['last_update'] else None,
        'data': signals
    })


@app.route('/api/refresh')
def refresh_data():
    """
    Force le rafraîchissement des données.
    
    Returns:
        JSON avec le statut du rafraîchissement
    """
    if cache['is_updating']:
        return jsonify({
            'success': False,
            'message': 'Une mise à jour est déjà en cours'
        })
    
    # Lancer la mise à jour dans un thread séparé
    thread = threading.Thread(target=update_cache)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Rafraîchissement lancé'
    })


@app.route('/api/status')
def get_status():
    """
    API pour vérifier le statut du serveur et du cache.
    
    Returns:
        JSON avec les informations de statut
    """
    return jsonify({
        'success': True,
        'is_updating': cache['is_updating'],
        'last_update': cache['last_update'].isoformat() if cache['last_update'] else None,
        'cache_size': len(cache['data']),
        'cache_duration': CACHE_DURATION
    })


# ========== Lancement de l'application ==========

if __name__ == '__main__':
    import os
    
    print("=" * 50)
    print("Screener Crypto - Démarrage du serveur")
    print("=" * 50)
    print(f"URL: http://localhost:5000")
    print(f"Cache duration: {CACHE_DURATION} secondes")
    print("=" * 50)
    
    # Premier chargement des données au démarrage
    # Ne charger que dans le processus principal (pas dans le reloader)
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        print("Chargement initial des données...")
        update_cache()
    
    # Lancer le serveur Flask
    app.run(debug=True, host='0.0.0.0', port=5000)
