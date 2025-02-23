from django.urls import path
from .views import HomePred

urlpatterns = [
    path('api/home/', HomePred.as_view(), name='home_Pred'),
]
