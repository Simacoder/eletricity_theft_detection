from django.urls import path
from .views import DashboardDataView, AlertList, UserSettingsView, DocumentationView, UserRegisterView, UserLoginView, MeterDataDetailView

urlpatterns = [
    path('register/', UserRegisterView.as_view(), name='register'),
    path('login/', UserLoginView.as_view(), name='login'),
    path('dashboard/', DashboardDataView.as_view(), name='dashboard_data'),
    path('alerts/', AlertList.as_view(), name='alert_list'),
    path('settings/', UserSettingsView.as_view(), name='user_settings'),
    path('documentation/', DocumentationView.as_view(), name='documentation'),
    path('meter-data/<str:meter_id>/', MeterDataDetailView.as_view(), name='meter_data_detail'),
]
