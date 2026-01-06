# monte-carlo-portfolio
Advanced Portfolio Optimization Framework
A Quantitative Finance Project | Summer 2025 (8-Week Intensive)
Project Overview
This project is a comprehensive Python-based tool that uses Monte Carlo Simulations to determine optimal asset allocation for stock portfolios. By simulating 10,000 different scenarios, the framework identifies the "Efficient Frontier"—the point where an investor gets the highest return for the lowest possible risk.

Development Timeline: Summer 2025
I developed this framework over an 8-week period to bridge the gap between theoretical finance and practical data science.

Weeks 1-2: Foundations & Data Pipelines

Studied Modern Portfolio Theory (MPT).

Built the integration with the Yahoo Finance API (yfinance) to pull real-time historical data.

Weeks 3-4: The Simulation Engine

Developed the Monte Carlo engine to generate 10,000+ random portfolio weightings.

Implemented vectorization with NumPy to ensure high-performance calculations.

Weeks 5-6: Advanced Risk Modeling

Went beyond basic volatility to code sophisticated metrics: Value at Risk (VaR), Conditional VaR, and the Sortino Ratio.

Weeks 7-8: Backtesting & Visualization

Created a "Stress Test" environment to see how portfolios performed during the 2022 Bear Market vs. 2023 Bull Market.

Designed a multi-panel Matplotlib dashboard for professional data storytelling.

Key Features
Monte Carlo Simulation: 10,000 scenarios analyzing AAPL, MSFT, GOOGL, AMZN, and TSLA.

The Efficient Frontier: Visualizes the optimal risk-return tradeoff.

Strategy Comparison: Compares "Max Sharpe Ratio" (high growth) vs. "Minimum Volatility" (stability) against the S&P 500.

Risk Metrics: Calculates Max Drawdown and 95% Confidence VaR.

Technical Stack
Language: Python

Libraries: Pandas, NumPy, Scipy (Math/Data), Matplotlib, Seaborn (Visualization), yFinance (Financial Data).
