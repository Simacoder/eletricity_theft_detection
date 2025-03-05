from django.urls import path
from .views import GenerateReport

urlpatterns = [
    path('generate/', GenerateReport.as_view(), name='generate_report'),
]
