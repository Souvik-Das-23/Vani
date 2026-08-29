"""
=============================================================================
Project Vani: API URL Routing Configuration
=============================================================================
Defines endpoint routes for the Vani API application.
"""

from django.urls import path
from api import views

urlpatterns = [
    path("translate/", views.translate_gesture, name="translate_gesture"),
    path("health/", views.health_check, name="health_check"),
    path("actions/", views.list_actions, name="list_actions"),
]
