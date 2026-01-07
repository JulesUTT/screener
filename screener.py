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
            for coin in coingecko_data:
                symbol = coin['symbol'].upper() + 'USDT'
                
                # Exclure les stablecoins
                if symbol in self.stablecoins:
                    continue
                
                # Vérifier si la paire existe sur Binance
                if symbol in binance_usdt_symbols:
                    top_100.append(symbol)
                    # Stocker la market cap pour affichage
                    self.market_caps[symbol] = coin.get('market_cap', 0)
                    # Stocker l'URL de l'image
                    self.images[symbol] = coin.get('image', '')
                    
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
            self.top_50_symbols = [item['symbol'] for item in usdt_pairs[:50]]
            
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
    
    def calculate_pivot_points(self, df: pd.DataFrame, prd: int = 10) -> Tuple[List[float], List[float]]:
        """
        Calcule les points pivots hauts et bas.
        
        Args:
            df: DataFrame avec les colonnes high, low, close
            prd: Période pour le calcul des pivots (regarde prd barres à gauche et à droite)
            
        Returns:
            Tuple (pivot_highs, pivot_lows) - listes des niveaux de pivots
        """
        pivot_highs = []
        pivot_lows = []
        
        highs = df['high'].values
        lows = df['low'].values
        
        for i in range(prd, len(df) - prd):
            # Pivot High: le plus haut des prd barres avant et après
            is_pivot_high = True
            for j in range(1, prd + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                pivot_highs.append(highs[i])
            
            # Pivot Low: le plus bas des prd barres avant et après
            is_pivot_low = True
            for j in range(1, prd + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                pivot_lows.append(lows[i])
        
        return pivot_highs, pivot_lows
    
    def calculate_support_resistance(self, df: pd.DataFrame, current_price: float, 
                                      channel_width_pct: float = 3.0) -> Dict:
        """
        Calcule les niveaux de support et résistance basés sur les pivots.
        
        Args:
            df: DataFrame avec les données OHLCV
            current_price: Prix actuel
            channel_width_pct: Largeur maximale du canal en pourcentage
            
        Returns:
            Dict avec nearest_support, nearest_resistance, at_support, at_resistance
        """
        pivot_highs, pivot_lows = self.calculate_pivot_points(df)
        
        # Combiner tous les pivots
        all_pivots = pivot_highs + pivot_lows
        
        if not all_pivots:
            return {
                'nearest_support': None,
                'nearest_resistance': None,
                'at_support': False,
                'at_resistance': False,
                'support_distance_pct': None,
                'resistance_distance_pct': None
            }
        
        # Calculer la largeur du canal basée sur le range
        price_range = df['high'].max() - df['low'].min()
        channel_width = price_range * channel_width_pct / 100
        
        # Grouper les pivots en clusters (niveaux S/R)
        all_pivots.sort()
        sr_levels = []
        
        i = 0
        while i < len(all_pivots):
            cluster = [all_pivots[i]]
            j = i + 1
            while j < len(all_pivots) and all_pivots[j] - cluster[0] <= channel_width:
                cluster.append(all_pivots[j])
                j += 1
            
            # Le niveau S/R est la moyenne du cluster, pondéré par le nombre de touches
            level = sum(cluster) / len(cluster)
            strength = len(cluster)  # Plus il y a de pivots, plus le niveau est fort
            sr_levels.append({'level': level, 'strength': strength})
            i = j
        
        # Trouver le support et la résistance les plus proches
        nearest_support = None
        nearest_resistance = None
        support_strength = 0
        resistance_strength = 0
        
        for sr in sr_levels:
            level = sr['level']
            if level < current_price:
                if nearest_support is None or level > nearest_support:
                    nearest_support = level
                    support_strength = sr['strength']
            elif level > current_price:
                if nearest_resistance is None or level < nearest_resistance:
                    nearest_resistance = level
                    resistance_strength = sr['strength']
        
        # Vérifier si le prix est proche d'un S/R (dans les 2%)
        proximity_threshold = current_price * 0.02
        
        at_support = nearest_support is not None and (current_price - nearest_support) <= proximity_threshold
        at_resistance = nearest_resistance is not None and (nearest_resistance - current_price) <= proximity_threshold
        
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
        
        # Récupérer la market cap et l'image si disponibles
        market_cap = self.market_caps.get(symbol, 0)
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
