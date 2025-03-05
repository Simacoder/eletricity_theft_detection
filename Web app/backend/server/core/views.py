from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import Meter, Anomaly, Alert, UserSettings, Documentation
from .serializers import MeterSerializer, AnomalySerializer, AlertSerializer, UserSettingsSerializer, DocumentationSerializer, UserRegisterSerializer, UserLoginSerializer

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

class MeterList(APIView):
    def get(self, request):
        meters = Meter.objects.all()
        serializer = MeterSerializer(meters, many=True)
        return Response(serializer.data)

class AnomalyList(APIView):
    def get(self, request):
        anomalies = Anomaly.objects.all()
        serializer = AnomalySerializer(anomalies, many=True)
        return Response(serializer.data)

class MeterDetail(APIView):
    def get(self, request, meter_id):
        try:
            meter = Meter.objects.get(meter_id=meter_id)
            anomalies = Anomaly.objects.filter(meter=meter)

            # Calculate consumption patterns: average, peak, off-peak
            readings = meter.readings
            total_readings = len(readings)
            total_consumption = sum([reading['value'] for reading in readings])
            average_consumption = total_consumption / total_readings if total_readings > 0 else 0
            peak_consumption = max([reading['value'] for reading in readings]) if readings else 0
            off_peak_consumption = min([reading['value'] for reading in readings]) if readings else 0

            # Prepare consumption patterns data
            consumption_patterns = {
                'average': average_consumption,
                'peak': peak_consumption,
                'off_peak': off_peak_consumption
            }

            meter_serializer = MeterSerializer(meter)
            anomalies_serializer = AnomalySerializer(anomalies, many=True)

            return Response({
                'meter': meter_serializer.data,
                'consumption_patterns': consumption_patterns,
                'anomalies': anomalies_serializer.data
            })
        except Meter.DoesNotExist:
            return Response({'error': 'Meter not found'}, status=404)

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
    
