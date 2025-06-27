from django.shortcuts import render
from trades.models import Trade
from trades.utils import company_stats
def available_companies(request):
    company_names = Trade.objects.values_list('company', flat=True).distinct()
    companies_data = []

    for name in company_names:
        trades = Trade.objects.filter(company=name).order_by('date')
        stats = company_stats(trades)

        if not stats:
            continue

        agents_data = []
        for agent_name, s in stats.items():
            agents_data.append({
                "agent": agent_name,
                "avg_daily_profit": s["avg_daily_profit"],
                "avg_monthly_profit": s["avg_monthly_profit"],
                "avg_yearly_profit": s["avg_yearly_profit"],
                "sharpe_ratio": s["sharpe_ratio"],
                "total_profit": s["last_value"] - 10000
            })

        companies_data.append({
            "company": name,
            "agents": agents_data
        })
    return render(request, 'main/available_companies.html', {'companies': companies_data})

