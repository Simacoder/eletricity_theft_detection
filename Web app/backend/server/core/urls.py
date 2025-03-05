from django.urls import path
from .views import MeterList, AnomalyList, AlertList, MeterDetail, UserSettingsView, DocumentationView, UserRegisterView, UserLoginView

urlpatterns = [
    path('register/', UserRegisterView.as_view(), name='register'),
    path('login/', UserLoginView.as_view(), name='login'),
    path('meters/', MeterList.as_view(), name='meter_list'),
    path('anomalies/', AnomalyList.as_view(), name='anomaly_list'),
    path('meters/<str:meter_id>/', MeterDetail.as_view(), name='meter_detail'),
    path('alerts/', AlertList.as_view(), name='alert_list'),
    path('settings/', UserSettingsView.as_view(), name='user_settings'),
    path('documentation/', DocumentationView.as_view(), name='documentation'),
]
