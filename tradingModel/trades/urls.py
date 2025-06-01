from django.urls import path
from . import views

urlpatterns = [
    path('api/post-trades/', views.post_trades, name='post-trades'),
    path('api/get-trades/<str:company>', views.get_trades, name='get-trades'),
]

