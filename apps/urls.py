from django.urls import path
from . import views

urlpatterns = [
    path('generate_code',views.generate_code,name='generate_code'),
    path('generate_image',views.generate_image,name='generate_image'),
    path('evaluate_assignment/', views.evaluate_assignment, name='evaluate_assignment'),        
] 