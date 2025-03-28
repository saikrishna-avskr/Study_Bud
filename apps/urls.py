from django.urls import path
from . import views

urlpatterns = [
    path('generate_code',views.generate_code,name='generate_code'),
    path('generate_image',views.generate_image,name='generate_image'),
    path('evaluate_assignment/', views.evaluate_assignment, name='evaluate_assignment'),    
    path('study_plan/', views.study_plan_view, name='study_plan'),    
    path('annotate/', views.annotate_image_view, name='annotate_image'),
    path('pomodoro/', views.pomodoro, name='pomodoro'),
] 