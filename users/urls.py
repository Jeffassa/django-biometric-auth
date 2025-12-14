# users/urls.py

from django.urls import path
from .views import register_face, login_face, dashboard_view, logout_view 

urlpatterns = [
    path("register/", register_face, name="register_face"),
    path("login/", login_face, name="login_face"),
    path("dashboard/", dashboard_view, name="dashboard"),
    path("logout/", logout_view, name="logout"), 
]