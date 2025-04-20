from django.contrib import admin
from django.urls import path, include
urlpatterns = [
    path('admin/', admin.site.urls),
    path('trades/', include('trades.urls')),
    path('', include('analytics.urls')),
]
