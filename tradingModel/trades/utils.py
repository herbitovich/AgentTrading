import pandas as pd
import math
def company_stats(trades):
    if not trades:
        return {}
    data = []
    for trade in trades:
        data.append({
            "date": trade.date.date(),
            "price": trade.current_price,
            "action": trade.action,
            "value": trade.value,
            "amount": trade.amount,
            "agent": trade.agent,
        })
    df = pd.DataFrame(data)
    results = {}
    agents = df["agent"].unique()
    for agent in agents:
        df_agent = df[df["agent"] == agent]
        daily_values = df_agent.groupby("date")["value"].mean()

        daily_profit = df_agent["value"].diff().dropna()
        avg_daily_profit = daily_profit.mean()
        avg_monthly_profit = avg_daily_profit * 21
        avg_yearly_profit = avg_daily_profit * 250

        daily_profit_per = daily_values.pct_change().dropna()
        avg_daily_profit_per = daily_profit_per.mean()
        avg_monthly_profit_per = avg_daily_profit_per * 21
        avg_yearly_profit_per = avg_daily_profit_per * 250
        sharpe_ratio = 0
        if len(daily_profit_per) > 0:
            sharpe_ratio = (daily_profit_per.mean() - 0.03) / daily_profit_per.std()

        sells = df_agent[df_agent["action"] == "sell"]

        avg_sell_profit = 0
        if not sells.empty:
           avg_sell_profit = sells["value"].diff().mean()

        price_growth = 0.0
        if not df_agent.empty:
            try:
                price_growth = ((df_agent["price"].iloc[-1] - df_agent["price"].iloc[0]) / df_agent["price"].iloc[0]) * 100
            except ZeroDivisionError:
                price_growth = 0.0

        def safe_round(x, digits=2):
            try:
                if pd.isna(x) or math.isinf(x):
                    return 0.0
                return float(round(x, digits))
            except Exception:
                return 0.0

        results[agent] = {
            "avg_daily_profit": safe_round(avg_daily_profit * 100, 2),
            "avg_monthly_profit": safe_round(avg_monthly_profit * 100, 2),
            "avg_yearly_profit": safe_round(avg_yearly_profit * 100, 2),
            "avg_daily_profit_per": safe_round(avg_daily_profit_per, 3),
            "avg_monthly_profit_per": safe_round(avg_monthly_profit_per, 3),
            "avg_yearly_profit_per": safe_round(avg_yearly_profit_per, 3),
            "sharpe_ratio": safe_round(sharpe_ratio, 15),
            "avg_sell_profit": safe_round(avg_sell_profit, 2),
            "price_growth": safe_round(price_growth, 2),
            "last_value": safe_round(df_agent["value"].iloc[-1]),
        }
    return results





