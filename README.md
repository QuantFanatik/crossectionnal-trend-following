#### ATTENTION SURVIVORSHIP BIAS

# Crossectionnal trend following

This project is about building and testing a simple trend following strategy on all the stocks of the S&P500.
The goal is to study and experiment through the project, not ending with a wonderful trading strategy that will make us filthy rich. Hence it is not a financial advice.

## Trading strategy

It will be a long only strategy, the idea is the next:    we Long the stock if the 5 period EMA is above the 200 period EMA 
                                                          else we do nothing.
                                                          
## Asset allocation
Since this will be applied to every single stock of the S&P500 we will then construct a equally weighted portfolio with all the strategy returns of each individual stock.
Doing a portfolio will help to reduce the volatility, and also the drawdown by divesification. We could apply some mean variance annalysis/optimisation, but due to the computationnal intensity, this could be done in a future commit.

## Filtering for market state
Finally we used an Hidden Markov Model (HMM) to filter market states. *What is the utility of filtering bad market state when we are diversified ?* 

  - By CAPM a well divesifed portfolio express only systematik risk, since we have eliminated the idiosincratic risk by diversification; in other words we need to control market risk in order to control the portfolio risk. The HMM will help us to filter the time where market are too risky and 
  
  - The covariance matrix is evolving through time and have specific dynamics through market regimes. When markets are in a bearish regime, covariance between asset will increase and the majority of the asset will have higher correlation with one an other. Using the HMM to filter bullish and bearish state will help to minimize the portfolio exposition to bad market states.
