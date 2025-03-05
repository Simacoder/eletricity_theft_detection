from django.db import models
from django.contrib.auth.models import User

class MeterData(models.Model):
    meter_id = models.CharField(max_length=100, unique=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    value = models.FloatField()
    consumption_pattern = models.CharField(max_length=50)

    def __str__(self):
        return f'Meter {self.meter_id} - {self.timestamp}'
    
class Anomaly(models.Model):
    meter = models.ForeignKey(MeterData, on_delete=models.CASCADE)
    anomaly_type = models.CharField(max_length=100)
    severity = models.CharField(max_length=20)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Anomaly {self.anomaly_type} - {self.severity}'
        
class Alert(models.Model):
    meter = models.ForeignKey(MeterData, on_delete=models.CASCADE)
    anomaly_type = models.CharField(max_length=100)
    severity = models.CharField(max_length=20)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Alert {self.anomaly_type} - {self.severity}'

class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    def __str__(self):
        return f'Notification for {self.user.username}'
    
class UserSettings(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    email_notifications = models.BooleanField(default=True)
    push_notifications = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username}'s Settings"

class Documentation(models.Model):
    title = models.CharField(max_length=100)
    content = models.TextField()

    def __str__(self):
        return self.title
    