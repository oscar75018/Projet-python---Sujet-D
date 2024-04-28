import numpy as np
from scipy.stats import norm
import numpy.random as npr
import pandas as pd
import yfinance as yf


# PARTIE POUR 2 PERSONNES

class OptionPricer:

  def __init__(self, S, K, T, sigma, r=0, q=0):
    """
    Initialise la classe pour le calcul du prix des options 
    Voici les différents paramètres :
    - S : Prix du sous-jacent
    - K : Strike Price
    - T : Date d’échéance
    - sigma : Volatilité implicite de l'actif sous-jacent
    - r : Taux d'intérêt sans risque (on dit que c’est 0 ici)
    - q : Rendement des dividendes du sous-jacent (par défaut, c’est 0)
    """  
    self.S = S
    self.K = K
    self.T = T
    self.sigma = sigma
    self.r = r
    self.q = q

  def d1(self, S, K, T, r, sigma, q):
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

  def d2(self, d1, sigma, T):
    return d1 - sigma * np.sqrt(T)

  def black_scholes(self, option_type='C'):
    """
    Permet la tarification d’une option en utilisant la méthode black-sholes ('C' pour Call, 'P' pour Put)
	  Cela renvoie le prix de l’option
    """
    # Calcul des variables intermédiaires d1 et d2
    d1_val = self.d1(self.S, self.K, self.T, self.r, self.sigma, self.q)
    d2_val = self.d2(d1_val, self.sigma, self.T)
    if option_type == 'C':
      prix = (self.S * np.exp(-self.q * self.T) * norm.cdf(d1_val) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2_val))
    elif option_type == 'P':
      prix = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2_val) - self.S * np.exp(-self.q * self.T) * norm.cdf(-d1_val))
    return prix

  def monte_carlo_simulation(self, option_type='C', num_simulations=100000):
    """
    On effectue des simulations de Monte Carlo pour estimer les prix des call et put pour différents scénarios de marché
    ST est le prix du sous-jacent
    WT est le mouvement brownien
    """   
    # Simuler les prix finaux en utilisant le mouvement brownien géométrique
    WT = np.random.normal(0, np.sqrt(self.T), num_simulations)
    ST = self.S * np.exp((self.r - self.q - 0.5 * self.sigma**2) * self.T + self.sigma * WT)
    
    if option_type == 'C':
      # Calculer les payoffs pour les calls
      payoffs = np.maximum(ST - self.K, 0)
    elif option_type == 'P':
      # Calculer les payoffs pour les puts
      payoffs = np.maximum(self.K - ST, 0)
    
    # Calculer le payoff moyen et l'actualiser à la valeur présente
    moyenne_payoff = np.mean(payoffs)
    actualise_payoff = np.exp(-self.r * self.T) * moyenne_payoff
        
    return actualise_payoff

# Exemple des utilisations possibles
prix = OptionPricer(S=100, K=100, T=1, sigma=0.2, r=0.05, q=0.02)
print("Prix méthode Black-Scholes Call:", prix.black_scholes('C'))
actualise_payoff = prix.monte_carlo_simulation(option_type='C', num_simulations=10000)
print("Résultats simulation Monte Carlo:", actualise_payoff)





# PARTIE POUR 3 PERSONNES

# Télécharge les données historiques pour une crypto-monnaie, par exemple Bitcoin (BTC-USD)
data = yf.download('BTC-USD', start='2023-01-01', end='2024-02-01')

# Calcule les rendements quotidiens
data['returns'] = data['Close'].pct_change()

# Affiche les premières lignes pour vérifier
print(data.head())

# Calcul de la volatilité historique et volatilité glissante
data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
data['volatilite']  = data['log_returns'].rolling(window=30).std() * np.sqrt(365)  # Fenêtre de 30 jours et annualisation
data['vol_glissante'] = data['log_returns'].abs().rolling(window=30).mean() * np.sqrt(365)

# Affiche les dernière lignes
print(data.tail())

notional_value = 10000  # Valeur notionnelle fixe pour chaque option vendue

def simulate_option_selling(data, strike_percent=1.10, option_type='C', num_days=365, notional_value=10000):
  pnl = []
  for i in range(1, num_days + 1):  # Commencer à l'indice 1 pour éviter NaN dans les rendements du premier jour

    # On recupere le prix à la date actuelle et la maturité + calcule strike price
    prix_SC_0 = data['Close'].iloc[i]
    strike_price = prix_SC_0 * strike_percent
    sigma = data['volatilite'].iloc[i] if not np.isnan(data['volatilite'].iloc[i]) else 0.2  # Utilisation de la volatilité historique ou valeur par défaut   # Utilisation de la volatilité 
    prix_SC_1 = data['Close'].iloc[i+30]

    # Calcul du prix initial de l'option et du nombre acheté
    prix_option = OptionPricer(S=prix_SC_0, K=strike_price, T=1/12, sigma=sigma).black_scholes(option_type)
    nombre_option = notional_value / prix_option

    # Calcul du daily pnl selon le type d'option (on regarde si l'acheteur exerce ou non son option à maturité)
    if option_type == 'C':
      daily_pnl = (prix_option - max(prix_SC_1 - strike_price, 0)) * nombre_option
    elif option_type == 'P':
      daily_pnl = (prix_option - max(strike_price - prix_SC_1, 0)) * nombre_option

    # Pour voir les daily_pnl
    print(daily_pnl, nombre_option)

    pnl.append(daily_pnl)

  return np.sum(pnl)

# Exemple d'utilisation (modifiez l'option type et le num days selon l'exemple voulu)
pnl_result = simulate_option_selling(data, strike_percent=1.05, option_type='P', num_days=90)
print("PnL total pour 90 jours:", pnl_result)
