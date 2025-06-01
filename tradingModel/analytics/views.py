from django.shortcuts import render

def trades_stats_page(request, company):
    return render(request, 'analytics/trades_stats.html', {'company' : company})
