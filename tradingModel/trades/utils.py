import pandas as pd
import math

def calc_avg_sell_profit(df_agent):
    df_agent = df_agent.sort_values("date").reset_index(drop=True)
    buy_stack = []
    sell_profits = []
    for el, row in df_agent.iterrows():
        action = row["action"]
        amount = row["amount"]
        price = row["price"]

        if action == "buy":
            buy_stack.append([amount, price])
        elif action == "sell":
            sell_amount = amount
            profit = 0.0

            while sell_amount > 0 and buy_stack:
                buy_amount, buy_price = buy_stack[0]
                used_amount = min(sell_amount, buy_amount)

                profit += (price - buy_price) * used_amount

                buy_amount -= used_amount
                sell_amount -= used_amount

                if buy_amount == 0:
                    buy_stack.pop(0)
                else:
                    buy_stack[0][0] = buy_amount

            sell_profits.append(profit)

    if sell_profits:
        return sum(sell_profits) / len(sell_profits)
    else:
        return 0.0

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
        avg_daily_profit = (df_agent["value"]-df_agent["value"].shift(1)).mean()
        avg_monthly_profit = avg_daily_profit * 21
        avg_yearly_profit = avg_daily_profit * 250

        avg_daily_profit_per = avg_daily_profit/100
        avg_daily_profit_per = float(avg_daily_profit_per)
        avg_monthly_profit_per = avg_daily_profit_per * 21
        avg_yearly_profit_per = avg_daily_profit_per * 250

        df_agent["daily_return"] = df_agent["value"].pct_change()
        rf_daily = 0.03 / 250
        excess_returns = df_agent["daily_return"] - rf_daily
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * math.sqrt(250)

        avg_sell_profit = calc_avg_sell_profit(df_agent)
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
            "avg_daily_profit": safe_round(avg_daily_profit, 1),
            "avg_monthly_profit": safe_round(avg_monthly_profit, 1),
            "avg_yearly_profit": safe_round(avg_yearly_profit, 1),
            "avg_daily_profit_per": safe_round(avg_daily_profit_per, 2),
            "avg_monthly_profit_per": safe_round(avg_monthly_profit_per, 2),
            "avg_yearly_profit_per": safe_round(avg_yearly_profit_per, 2),
            "sharpe_ratio": safe_round(sharpe_ratio, 20),
            "avg_sell_profit": safe_round(avg_sell_profit, 1),
            "price_growth": safe_round(price_growth, 2),
            "last_value": safe_round(df_agent["value"].iloc[-1]),
        }
    return results





