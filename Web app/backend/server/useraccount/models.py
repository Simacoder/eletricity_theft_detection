from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    ROLE_CHOICES = [
        ('admin', 'Administrator'),
        ('user', 'User'),
    ]
    role = models.CharField(max_length=5, choices=ROLE_CHOICES, default='user')