#ATTENTION SURVIVORSHIP BIAS

# crossectionnal trend following

This project is about building and test a simple trend following strategy on all the stocks of the S&P500.

# Trading strategy

It will be a long only strategy, the idea is the next: Long if the 5 period EMA is above the 200 period EMA.
Since this will be applied to every single stock of the S&P500 we will then construct a equally weighted portfolio with all the strategy returns of each individual stock.

# Asset allocation
Doing a portfolio will help to reduce the volatility, and also the drawdown by divesification. We could apply some mean variance annalysis/optimisation, but due to the computationnal intensity, this could be done in a future commit.

# Filtering for market state
Finally we used an Hidden Markov Model (HMM)
