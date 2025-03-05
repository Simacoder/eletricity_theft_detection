from django.contrib.auth.models import User
from rest_framework import serializers
from .models import Meter, Anomaly, UserSettings, Alert, Documentation
from rest_framework_simplejwt.tokens import RefreshToken

class UserRegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['username', 'password', 'email']

    def create(self, validated_data):
        user = User.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password'],
            email=validated_data['email']
        )
        return user

class UserLoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def validate(self, data):
        user = User.objects.filter(username=data['username']).first()
        if user and user.check_password(data['password']):
            refresh = RefreshToken.for_user(user)
            return {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
        raise serializers.ValidationError("Invalid credentials")

class MeterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Meter
        fields = ['meter_id', 'readings']

class AnomalySerializer(serializers.ModelSerializer):
    class Meta:
        model = Anomaly
        fields = ['meter', 'anomaly_type', 'timestamp', 'severity']

class AlertSerializer(serializers.ModelSerializer):
    anomaly = AnomalySerializer()

    class Meta:
        model = Alert
        fields = ['id', 'anomaly', 'alert_message', 'alert_severity', 'alert_timestamp']

class UserSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserSettings
        fields = ['email_notifications', 'push_notifications']

class DocumentationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Documentation
        fields = ['title', 'content']

