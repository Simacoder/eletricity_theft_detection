from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import Alert, UserSettings, Documentation, MeterData, Alert, Notification, Anomaly
from .serializers import MeterDataSerializer, AnomalySerializer, AlertSerializer, NotificationSerializer, UserSettingsSerializer, DocumentationSerializer, UserRegisterSerializer, UserLoginSerializer

class UserRegisterView(APIView):
    def post(self, request):
        serializer = UserRegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({"message": "User created successfully"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class UserLoginView(APIView):
    def post(self, request):
        serializer = UserLoginSerializer(data=request.data)
        if serializer.is_valid():
            return Response(serializer.validated_data, status=status.HTTP_200_OK)
        return Response({"error": "Invalid credentials"}, status=status.HTTP_400_BAD_REQUEST)

class DashboardDataView(APIView):
    def get(self, request):
        user = request.user

        # Fetching meter data (latest)
        meter_data = MeterData.objects.all().order_by('-timestamp')[:5]
        meter_data_serializer = MeterDataSerializer(meter_data, many=True)

        # Fetching recent alerts
        alerts = Alert.objects.all().order_by('-timestamp')[:5]
        alerts_serializer = AlertSerializer(alerts, many=True)

        # Fetching user notifications
        notifications = Notification.objects.filter(user=user).order_by('-timestamp')[:5]
        notifications_serializer = NotificationSerializer(notifications, many=True)

        return Response({
            'meter_data': meter_data_serializer.data,
            'alerts': alerts_serializer.data,
            'notifications': notifications_serializer.data,
        }, status=status.HTTP_200_OK)
        
class MeterDataDetailView(APIView):
    def get(self, request, meter_id):
        try:
            # Fetch the meter data for a specific meter ID
            meter_data = MeterData.objects.get(meter_id=meter_id)
            meter_data_serializer = MeterDataSerializer(meter_data)

            # Fetch anomalies related to this meter ID
            anomalies = Anomaly.objects.filter(meter=meter_data)
            anomalies_serializer = AnomalySerializer(anomalies, many=True)

            return Response({
                'meter_data': meter_data_serializer.data,
                'anomalies': anomalies_serializer.data
            }, status=status.HTTP_200_OK)

        except MeterData.DoesNotExist:
            return Response({'error': 'Meter data not found.'}, status=status.HTTP_404_NOT_FOUND)

class AlertList(APIView):
    def get(self, request):
        alerts = Alert.objects.all()
        serializer = AlertSerializer(alerts, many=True)
        return Response(serializer.data)

class UserSettingsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            settings = UserSettings.objects.get(user=request.user)
            serializer = UserSettingsSerializer(settings)
            return Response(serializer.data)
        except UserSettings.DoesNotExist:
            return Response({"error": "Settings not found."}, status=404)

    def put(self, request):
        try:
            settings = UserSettings.objects.get(user=request.user)
            serializer = UserSettingsSerializer(settings, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data)
            return Response(serializer.errors, status=400)
        except UserSettings.DoesNotExist:
            return Response({"error": "Settings not found."}, status=404)

class DocumentationView(APIView):
    def get(self, request):
        documentation = Documentation.objects.all()
        serializer = DocumentationSerializer(documentation, many=True)
        return Response(serializer.data)
    