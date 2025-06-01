from django.shortcuts import render
from trades.models import Trade

def available_companies(request):
    companies = Trade.objects.values_list('company', flat=True).distinct()
    return render(request, 'main/available_companies.html', {'companies': companies})
