from django.db import models
from django.contrib.auth.models import User

class Meter(models.Model):
    meter_id = models.CharField(max_length=50, unique=True)
    readings = models.JSONField()

    def __str__(self):
        return self.meter_id

class Anomaly(models.Model):
    meter = models.ForeignKey(Meter, on_delete=models.CASCADE)
    anomaly_type = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)
    severity = models.CharField(max_length=20)

    def __str__(self):
        return f"{self.anomaly_type} - {self.severity}"

class Alert(models.Model):
    anomaly = models.ForeignKey(Anomaly, on_delete=models.CASCADE)
    alert_message = models.TextField()
    alert_severity = models.CharField(max_length=20)
    alert_timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Alert for {self.anomaly.anomaly_type} - {self.alert_severity}"
    
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

