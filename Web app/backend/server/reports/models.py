from django.db import models

class Report(models.Model):
    report_type = models.CharField(max_length=255)
    content = models.TextField()

    def __str__(self):
        return f"Report {self.report_type}"
