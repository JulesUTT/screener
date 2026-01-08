"""
Screener Crypto - Module de calcul des indicateurs
===================================================
Ce module récupère les données de l'API Binance et calcule les indicateurs techniques :
- RSI Stochastique (14 périodes)
- EMA 13, 25, 32 et 200
- Supports et Résistances (basés sur les pivots)

Timeframe : 4 heures
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class CryptoScreener:
    """
    Classe principale pour le screener crypto.
    Récupère les données de Binance et calcule les indicateurs techniques.
    """
    
    # URL de base de l'API Binance
    BASE_URL = "https://api.binance.com/api/v3"
    
    # Timeframe 4 heures
    TIMEFRAME = "4h"
    
    # Nombre de bougies à récupérer (pour calculer EMA 200 avec précision, on prend 500 bougies)
    LIMIT = 500
    
    # URL de l'API CoinGecko (gratuite)
    COINGECKO_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self):
        """Initialise le screener."""
        self.top_50_symbols: List[str] = []
        self.market_caps: Dict[str, float] = {}  # Pour stocker les market caps
        self.fdvs: Dict[str, float] = {}  # Pour stocker les FDV (Fully Diluted Valuation)
        self.images: Dict[str, str] = {}  # Pour stocker les URLs des images
        
        # Liste des stablecoins à exclure
        self.stablecoins = {
            'USDCUSDT', 'USDTUSDT', 'USDEUSDT', 'DAIUSDT', 'TUSDUSDT', 
            'BUSDUSDT', 'USDPUSDT', 'GUSDUSDT', 'FRAXUSDT', 'LUSDUSDT',
            'USTUSDT', 'FDUSDUSDT', 'PYUSDUSDT', 'EURUSDT', 'GBPUSDT',
            'AABORAUSDT', 'USD1USDT', 'BFUSDUSDT'
        }
        
    def get_top_100_symbols(self) -> List[str]:
        """
        Récupère le Top 100 des cryptomonnaies par MARKET CAP via CoinGecko.
        Exclut les stablecoins.
        Filtre uniquement les paires disponibles sur Binance en USDT.
        
        Returns:
            Liste des symboles (ex: ['BTCUSDT', 'ETHUSDT', ...])
        """
        try:
            # 1. Récupérer le top 150 par market cap depuis CoinGecko (marge pour les exclusions)
            print("Récupération du Top 150 par Market Cap depuis CoinGecko...")
            cg_url = f"{self.COINGECKO_URL}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 150,  # On prend 150 pour avoir assez de marge après exclusions
                'page': 1,
                'sparkline': 'false'
            }
            
            response = requests.get(cg_url, params=params, timeout=15)
            response.raise_for_status()
            coingecko_data = response.json()
            
            # 2. Récupérer les symboles disponibles sur Binance
            print("Vérification des paires disponibles sur Binance...")
            binance_url = f"{self.BASE_URL}/exchangeInfo"
            response = requests.get(binance_url, timeout=10)
            response.raise_for_status()
            binance_data = response.json()
            
            # Créer un set des symboles USDT disponibles sur Binance
            binance_usdt_symbols = set(
                s['symbol'] for s in binance_data['symbols']
                if s['symbol'].endswith('USDT') and s['status'] == 'TRADING'
            )
            
            # 3. Mapper les symboles CoinGecko vers Binance (exclure stablecoins)
            top_100 = []
            seen_symbols = set()  # Pour éviter les doublons
            
            for coin in coingecko_data:
                symbol = coin['symbol'].upper() + 'USDT'
                crypto_symbol = coin['symbol'].upper()
                
                # Exclure les stablecoins
                if symbol in self.stablecoins:
                    continue
                
                # Éviter les doublons (même symbole apparaissant plusieurs fois)
                if symbol in seen_symbols:
                    continue
                
                # Vérifier si la paire existe sur Binance
                if symbol in binance_usdt_symbols:
                    top_100.append(symbol)
                    seen_symbols.add(symbol)
                    # Stocker la market cap pour affichage
                    self.market_caps[symbol] = coin.get('market_cap', 0)
                    # Stocker la FDV (Fully Diluted Valuation)
                    self.fdvs[symbol] = coin.get('fully_diluted_valuation', 0) or 0
                    # Stocker l'URL de l'image - Utiliser CryptoCompare comme source alternative
                    # CryptoCompare est plus permissif pour les serveurs cloud
                    self.images[symbol] = f"https://assets.coincap.io/assets/icons/{crypto_symbol.lower()}@2x.png"
                    
                    if len(top_100) >= 100:
                        break
            
            self.top_50_symbols = top_100
            print(f"Top {len(top_100)} cryptos par Market Cap récupérées (stablecoins exclus)")
            
            return self.top_50_symbols
            
        except Exception as e:
            print(f"Erreur lors de la récupération du Top 50 par Market Cap: {e}")
            print("Fallback: utilisation du volume de trading...")
            return self._get_top_50_by_volume()
    
    def _get_top_50_by_volume(self) -> List[str]:
        """
        Méthode de fallback: récupère le Top 50 par volume de trading.
        Utilisée si CoinGecko est indisponible.
        
        Returns:
            Liste des symboles
        """
        try:
            url = f"{self.BASE_URL}/ticker/24hr"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            usdt_pairs = [
                item for item in data 
                if item['symbol'].endswith('USDT') 
                and not item['symbol'].startswith('USDT')
                and float(item['quoteVolume']) > 0
            ]
            
            usdt_pairs.sort(key=lambda x: float(x['quoteVolume']), reverse=True)

            # Filtrer les stablecoins connus
            filtered_pairs = []
            for item in usdt_pairs:
                if item['symbol'] not in self.stablecoins:
                    filtered_pairs.append(item['symbol'])
            
            self.top_50_symbols = filtered_pairs[:100]
            print(f"Fallback: {len(self.top_50_symbols)} cryptos par volume récupérées")
            
            return self.top_50_symbols
            
        except Exception as e:
            print(f"Erreur fallback volume: {e}")
            return []
    
    def get_klines(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Récupère les bougies (klines) pour un symbole donné.
        
        Args:
            symbol: Le symbole de la paire (ex: 'BTCUSDT')
            
        Returns:
            DataFrame avec les données OHLCV ou None en cas d'erreur
        """
        try:
            url = f"{self.BASE_URL}/klines"
            params = {
                'symbol': symbol,
                'interval': self.TIMEFRAME,
                'limit': self.LIMIT
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Créer le DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convertir les colonnes numériques
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convertir le timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            print(f"Erreur lors de la récupération des klines pour {symbol}: {e}")
            return None
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calcule l'EMA (Exponential Moving Average) pour une période donnée.
        Utilise la même formule que TradingView/Binance.
        
        Args:
            data: Série de prix (généralement les prix de clôture)
            period: Période de l'EMA
            
        Returns:
            Série avec les valeurs de l'EMA
        """
        return data.ewm(span=period, adjust=True).mean()
    
    def calculate_stochastic_rsi(self, data: pd.Series, rsi_length: int = 9, 
                                  stoch_length: int = 14,
                                  smooth_k: int = 3, smooth_d: int = 3) -> Dict[str, float]:
        """
        Calcule le RSI Stochastique.
        
        Le RSI Stochastique applique la formule du Stochastique au RSI.
        
        Args:
            data: Série de prix de clôture
            rsi_length: Longueur du RSI (défaut: 9)
            stoch_length: Longueur du Stochastique (défaut: 14)
            smooth_k: Lissage du %K (défaut: 3)
            smooth_d: Lissage du %D (défaut: 3)
            
        Returns:
            Dictionnaire avec les valeurs K et D du Stochastic RSI
        """
        # Calcul du RSI avec rsi_length
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_length).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calcul du Stochastic RSI avec stoch_length
        rsi_min = rsi.rolling(window=stoch_length).min()
        rsi_max = rsi.rolling(window=stoch_length).max()
        
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
        
        # Lissage
        stoch_rsi_k = stoch_rsi.rolling(window=smooth_k).mean()
        stoch_rsi_d = stoch_rsi_k.rolling(window=smooth_d).mean()
        
        # Retourner les dernières valeurs
        return {
            'k': round(stoch_rsi_k.iloc[-1], 2) if not pd.isna(stoch_rsi_k.iloc[-1]) else 0,
            'd': round(stoch_rsi_d.iloc[-1], 2) if not pd.isna(stoch_rsi_d.iloc[-1]) else 0
        }
    
    def calculate_support_resistance(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Calcule les niveaux de support et résistance basés sur le script Pine Script
        'Support Resistance Channels' de LonesomeTheBlue.
        
        Paramètres du script (adaptés pour H4):
        - prd = 10 : Période pour les pivots (10 barres à gauche et à droite)
        - ChannelW = 5 : Largeur max du canal en % du range sur 300 barres
        - minstrength = 1 : Force minimum (au moins 1 pivot point = strength >= 20)
        - maxnumsr = 6 : Nombre max de S/R
        - loopback = 290 : Période de lookback pour les pivots
        
        Args:
            df: DataFrame avec les données OHLCV
            current_price: Prix actuel
            
        Returns:
            Dict avec nearest_support, nearest_resistance, at_support, at_resistance
        """
        # Paramètres du script Pine
        prd = 10  # Pivot Period
        channel_width_pct = 5  # Maximum Channel Width %
        min_strength = 1  # Minimum Strength
        max_num_sr = 6  # Maximum Number of S/R
        loopback = 290  # Loopback Period
        
        # S'assurer qu'on a assez de données
        if len(df) < loopback:
            loopback = len(df) - prd - 1
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # --- Étape 1: Trouver tous les pivots (High/Low) ---
        pivot_vals = []  # Valeur du pivot
        pivot_locs = []  # Index du pivot
        
        for i in range(prd, len(df) - prd):
            # Pivot High: le plus haut sur prd barres à gauche ET à droite
            is_pivot_high = True
            for j in range(1, prd + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_pivot_high = False
                    break
            
            # Pivot Low: le plus bas sur prd barres à gauche ET à droite
            is_pivot_low = True
            for j in range(1, prd + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_pivot_low = False
                    break
            
            if is_pivot_high:
                pivot_vals.append(highs[i])
                pivot_locs.append(i)
            if is_pivot_low:
                pivot_vals.append(lows[i])
                pivot_locs.append(i)
        
        # Filtrer les pivots dans la période de loopback
        current_bar = len(df) - 1
        filtered_pivots = []
        for val, loc in zip(pivot_vals, pivot_locs):
            if current_bar - loc <= loopback:
                filtered_pivots.append({'val': val, 'loc': loc})
        
        if not filtered_pivots:
            return {
                'nearest_support': None,
                'nearest_resistance': None,
                'at_support': False,
                'at_resistance': False,
                'support_distance_pct': None,
                'resistance_distance_pct': None
            }
        
        # --- Étape 2: Calculer la largeur du canal ---
        # Basé sur le range des 300 dernières barres
        range_bars = min(300, len(df))
        prdhighest = df['high'].iloc[-range_bars:].max()
        prdlowest = df['low'].iloc[-range_bars:].min()
        cwidth = (prdhighest - prdlowest) * channel_width_pct / 100
        
        # --- Étape 3: Créer les canaux S/R pour chaque pivot ---
        def get_sr_vals(pivot_idx):
            """Crée un canal S/R autour d'un pivot et calcule sa force."""
            lo = filtered_pivots[pivot_idx]['val']
            hi = lo
            num_pp = 0
            
            for p in filtered_pivots:
                cpp = p['val']
                # Vérifie si le pivot rentre dans le canal
                if cpp <= hi:
                    wdth = hi - cpp
                else:
                    wdth = cpp - lo
                
                if wdth <= cwidth:  # Le pivot rentre dans le canal
                    if cpp <= hi:
                        lo = min(lo, cpp)
                    else:
                        hi = max(hi, cpp)
                    num_pp += 20  # Chaque pivot ajoute 20 à la force
            
            return hi, lo, num_pp
        
        # --- Étape 4: Calculer force et niveaux pour chaque pivot ---
        supres = []  # [strength, hi, lo] pour chaque pivot
        for i in range(len(filtered_pivots)):
            hi, lo, strength = get_sr_vals(i)
            supres.append({'strength': strength, 'hi': hi, 'lo': lo})
        
        # --- Étape 5: Ajouter la force basée sur les touches de prix ---
        for sr in supres:
            h = sr['hi']
            l = sr['lo']
            touches = 0
            for y in range(min(loopback, len(df))):
                idx = len(df) - 1 - y
                if idx >= 0:
                    if (highs[idx] <= h and highs[idx] >= l) or (lows[idx] <= h and lows[idx] >= l):
                        touches += 1
            sr['strength'] += touches
        
        # --- Étape 6: Sélectionner les meilleurs S/R ---
        sr_channels = []
        used_ranges = []
        
        # Trier par force décroissante
        sorted_supres = sorted(supres, key=lambda x: x['strength'], reverse=True)
        
        for sr in sorted_supres:
            if sr['strength'] < min_strength * 20:
                continue
            
            # Vérifier que ce canal ne chevauche pas un canal déjà sélectionné
            overlap = False
            for used in used_ranges:
                if (sr['hi'] <= used['hi'] and sr['hi'] >= used['lo']) or \
                   (sr['lo'] <= used['hi'] and sr['lo'] >= used['lo']):
                    overlap = True
                    break
            
            if not overlap:
                sr_channels.append({'hi': sr['hi'], 'lo': sr['lo'], 'strength': sr['strength']})
                used_ranges.append({'hi': sr['hi'], 'lo': sr['lo']})
                
                if len(sr_channels) >= max_num_sr:
                    break
        
        # --- Étape 7: Trouver support et résistance les plus proches ---
        nearest_support = None
        nearest_resistance = None
        support_strength = 0
        resistance_strength = 0
        at_support = False
        at_resistance = False
        
        for channel in sr_channels:
            channel_mid = (channel['hi'] + channel['lo']) / 2
            
            # Le prix est DANS le canal
            if current_price <= channel['hi'] and current_price >= channel['lo']:
                # C'est à la fois support et résistance (on est dedans)
                at_support = True
                at_resistance = True
                nearest_support = channel['lo']
                nearest_resistance = channel['hi']
                support_strength = channel['strength']
                resistance_strength = channel['strength']
                break
            
            # Canal en dessous du prix = Support
            if channel['hi'] < current_price:
                if nearest_support is None or channel['hi'] > nearest_support:
                    nearest_support = channel['hi']
                    support_strength = channel['strength']
            
            # Canal au dessus du prix = Résistance
            if channel['lo'] > current_price:
                if nearest_resistance is None or channel['lo'] < nearest_resistance:
                    nearest_resistance = channel['lo']
                    resistance_strength = channel['strength']
        
        # Vérifier la proximité (dans les 2%)
        proximity_pct = 0.02
        if not at_support and nearest_support is not None:
            at_support = (current_price - nearest_support) / current_price <= proximity_pct
        if not at_resistance and nearest_resistance is not None:
            at_resistance = (nearest_resistance - current_price) / current_price <= proximity_pct
        
        # Calculer les distances en pourcentage
        support_distance_pct = None
        resistance_distance_pct = None
        
        if nearest_support:
            support_distance_pct = round((current_price - nearest_support) / current_price * 100, 2)
        if nearest_resistance:
            resistance_distance_pct = round((nearest_resistance - current_price) / current_price * 100, 2)
        
        return {
            'nearest_support': round(nearest_support, 8) if nearest_support else None,
            'nearest_resistance': round(nearest_resistance, 8) if nearest_resistance else None,
            'support_strength': support_strength,
            'resistance_strength': resistance_strength,
            'at_support': at_support,
            'at_resistance': at_resistance,
            'support_distance_pct': support_distance_pct,
            'resistance_distance_pct': resistance_distance_pct
        }
    
    def calculate_rating(self, signal: str, stoch_rsi_k: float, sr_data: Dict) -> Dict:
        """
        Calcule une note pour le setup basée sur les critères.
        
        Système de notation (sur 5 étoiles):
        - EMA alignées seules: 1-2 étoiles
        - EMA + RSI stoch optimal: 3 étoiles  
        - EMA + RSI stoch + S/R: 4-5 étoiles
        
        Args:
            signal: 'LONG', 'SHORT' ou 'AUCUN'
            stoch_rsi_k: Valeur du RSI stochastique K
            sr_data: Données des supports/résistances
            
        Returns:
            Dict avec rating (1-5), rating_text, et les détails
        """
        if signal == 'AUCUN':
            return {
                'rating': 0,
                'rating_text': '-',
                'has_ema': False,
                'has_rsi': False,
                'has_sr': False
            }
        
        rating = 1  # Base: EMA alignées (sinon pas de signal)
        has_ema = True
        has_rsi = False
        has_sr = False
        
        if signal == 'LONG':
            # RSI stoch bas = bon pour LONG
            if stoch_rsi_k <= 20:
                rating += 2  # RSI excellent
                has_rsi = True
            elif stoch_rsi_k <= 35:
                rating += 1  # RSI bon
                has_rsi = True
            
            # Proche d'un support = excellent pour LONG
            if sr_data.get('at_support'):
                rating += 2
                has_sr = True
            elif sr_data.get('support_distance_pct') and sr_data['support_distance_pct'] <= 3:
                rating += 1
                has_sr = True
                
        elif signal == 'SHORT':
            # RSI stoch haut = bon pour SHORT
            if stoch_rsi_k >= 80:
                rating += 2  # RSI excellent
                has_rsi = True
            elif stoch_rsi_k >= 65:
                rating += 1  # RSI bon
                has_rsi = True
            
            # Proche d'une résistance = excellent pour SHORT
            if sr_data.get('at_resistance'):
                rating += 2
                has_sr = True
            elif sr_data.get('resistance_distance_pct') and sr_data['resistance_distance_pct'] <= 3:
                rating += 1
                has_sr = True
        
        # Limiter à 5 étoiles max
        rating = min(5, rating)
        
        # Texte de la note
        stars = '★' * rating + '☆' * (5 - rating)
        
        return {
            'rating': rating,
            'rating_text': stars,
            'has_ema': has_ema,
            'has_rsi': has_rsi,
            'has_sr': has_sr
        }
    
    def determine_signal(self, price: float, ema_13: float, ema_25: float, 
                         ema_32: float, ema_200: float) -> str:
        """
        Détermine le signal de trading basé sur les conditions.
        
        Règles:
        - LONG: Prix > EMA200 + Prix entre EMA13 et EMA32 + EMA13 > EMA25 > EMA32
        - SHORT: Prix < EMA200 + Prix entre EMA13 et EMA32 + EMA13 < EMA25 < EMA32
        
        Args:
            price: Prix actuel
            ema_13: Valeur de l'EMA 13
            ema_25: Valeur de l'EMA 25
            ema_32: Valeur de l'EMA 32
            ema_200: Valeur de l'EMA 200
            
        Returns:
            'LONG', 'SHORT' ou 'AUCUN'
        """
        # Vérifier si le prix est entre EMA 13 et EMA 32
        ema_min = min(ema_13, ema_32)
        ema_max = max(ema_13, ema_32)
        
        price_in_zone = ema_min <= price <= ema_max
        
        if not price_in_zone:
            return "AUCUN"
        
        # Conditions d'alignement des EMAs
        emas_bullish = ema_13 > ema_25 > ema_32  # Alignement haussier
        emas_bearish = ema_13 < ema_25 < ema_32  # Alignement baissier
        
        # Déterminer la direction basée sur EMA 200 ET alignement des EMAs
        if price > ema_200 and emas_bullish:
            return "LONG"
        elif price < ema_200 and emas_bearish:
            return "SHORT"
        else:
            return "AUCUN"
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Analyse complète d'un symbole.
        
        Args:
            symbol: Le symbole à analyser (ex: 'BTCUSDT')
            
        Returns:
            Dictionnaire avec toutes les données d'analyse ou None en cas d'erreur
        """
        # Récupérer les données
        df = self.get_klines(symbol)
        if df is None or len(df) < self.LIMIT:
            return None
        
        close_prices = df['close']
        current_price = close_prices.iloc[-1]
        
        # Calculer les EMAs
        ema_13 = self.calculate_ema(close_prices, 13).iloc[-1]
        ema_25 = self.calculate_ema(close_prices, 25).iloc[-1]
        ema_32 = self.calculate_ema(close_prices, 32).iloc[-1]
        ema_200 = self.calculate_ema(close_prices, 200).iloc[-1]
        
        # Calculer le Stochastic RSI
        stoch_rsi = self.calculate_stochastic_rsi(close_prices)
        
        # Calculer les supports et résistances
        sr_data = self.calculate_support_resistance(df, current_price)
        
        # Déterminer le signal
        signal = self.determine_signal(current_price, ema_13, ema_25, ema_32, ema_200)
        
        # Calculer la note du setup
        rating_data = self.calculate_rating(signal, stoch_rsi['k'], sr_data)
        
        # Extraire le nom de la crypto (sans USDT)
        crypto_name = symbol.replace('USDT', '')
        
        # Récupérer la market cap, FDV et l'image si disponibles
        market_cap = self.market_caps.get(symbol, 0)
        fdv = self.fdvs.get(symbol, 0)
        image_url = self.images.get(symbol, '')
        
        # Convertir les valeurs numpy en float Python natif pour la sérialisation JSON
        def to_python_float(val):
            if val is None:
                return None
            return float(val) if hasattr(val, 'item') else val
        
        return {
            'symbol': symbol,
            'name': crypto_name,
            'image': image_url,
            'price': float(round(current_price, 8)),
            'market_cap': float(market_cap) if market_cap else 0,
            'fdv': float(fdv) if fdv else 0,
            'ema_13': float(round(ema_13, 8)),
            'ema_25': float(round(ema_25, 8)),
            'ema_32': float(round(ema_32, 8)),
            'ema_200': float(round(ema_200, 8)),
            'stoch_rsi_k': float(stoch_rsi['k']),
            'stoch_rsi_d': float(stoch_rsi['d']),
            'signal': signal,
            # Supports et Résistances
            'nearest_support': to_python_float(sr_data['nearest_support']),
            'nearest_resistance': to_python_float(sr_data['nearest_resistance']),
            'at_support': bool(sr_data['at_support']),
            'at_resistance': bool(sr_data['at_resistance']),
            'support_distance_pct': to_python_float(sr_data['support_distance_pct']),
            'resistance_distance_pct': to_python_float(sr_data['resistance_distance_pct']),
            # Note du setup
            'rating': int(rating_data['rating']),
            'rating_text': rating_data['rating_text'],
            'has_ema': bool(rating_data['has_ema']),
            'has_rsi': bool(rating_data['has_rsi']),
            'has_sr': bool(rating_data['has_sr']),
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_signal_score(self, signal: str, stoch_rsi_k: float, rating: int = 0) -> float:
        """
        Calcule un score de probabilité pour le setup.
        Plus le score est bas, meilleur est le setup.
        
        Le score combine:
        - La note (rating): plus elle est haute, meilleur est le setup
        - Le RSI stochastique: RSI bas pour LONG, RSI haut pour SHORT
        
        Args:
            signal: 'LONG', 'SHORT' ou 'AUCUN'
            stoch_rsi_k: Valeur du RSI stochastique K
            rating: Note du setup (1-5)
            
        Returns:
            Score (plus bas = meilleur)
        """
        if signal == 'AUCUN':
            return 1000  # Pas de signal = score très haut
        
        # Score de base inversé par rapport à la note (rating 5 = score 0, rating 1 = score 40)
        base_score = (5 - rating) * 10
        
        if signal == 'LONG':
            # Pour LONG, RSI bas = meilleur setup
            rsi_score = stoch_rsi_k
        else:  # SHORT
            # Pour SHORT, RSI haut = meilleur setup  
            rsi_score = 100 - stoch_rsi_k
        
        # Score final: combinaison note + RSI (note prioritaire)
        return base_score + rsi_score
    
    def scan_all(self) -> List[Dict]:
        """
        Scanne toutes les cryptos du Top 100.
        
        Returns:
            Liste des analyses pour chaque crypto
        """
        # Récupérer le Top 100 si pas encore fait
        if not self.top_50_symbols:
            self.get_top_100_symbols()
        
        results = []
        
        for symbol in self.top_50_symbols:
            print(f"Analyse de {symbol}...")
            analysis = self.analyze_symbol(symbol)
            if analysis:
                # Ajouter le score de probabilité (basé sur la note + RSI)
                analysis['signal_score'] = self.calculate_signal_score(
                    analysis['signal'], 
                    analysis['stoch_rsi_k'],
                    analysis['rating']
                )
                results.append(analysis)
        
        # Trier par score de probabilité (meilleurs setups en premier)
        # Les signaux avec meilleure note + bon RSI apparaissent en premier
        results.sort(key=lambda x: x['signal_score'])
        
        return results


# Pour tester le module directement
if __name__ == "__main__":
    screener = CryptoScreener()
    
    print("Récupération du Top 50...")
    symbols = screener.get_top_50_symbols()
    print(f"Top 50 récupéré: {len(symbols)} symboles")
    
    print("\nAnalyse de BTC en exemple...")
    analysis = screener.analyze_symbol('BTCUSDT')
    if analysis:
        print(f"Prix: {analysis['price']}")
        print(f"EMA 13: {analysis['ema_13']}")
        print(f"EMA 25: {analysis['ema_25']}")
        print(f"EMA 32: {analysis['ema_32']}")
        print(f"EMA 200: {analysis['ema_200']}")
        print(f"Stoch RSI K: {analysis['stoch_rsi_k']}")
        print(f"Stoch RSI D: {analysis['stoch_rsi_d']}")
        print(f"Signal: {analysis['signal']}")
        print(f"Note: {analysis['rating_text']} ({analysis['rating']}/5)")
        print(f"Support: {analysis['nearest_support']} ({analysis['support_distance_pct']}%)")
        print(f"Résistance: {analysis['nearest_resistance']} ({analysis['resistance_distance_pct']}%)")
        print(f"Signal: {analysis['signal']}")
