from django.urls import path
from . import views

urlpatterns= [
    path('trades/<str:company>', views.trades_stats_page, name='trades-stats')
]