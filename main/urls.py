from django.urls import path
from . import views

urlpatterns = [

    # ---------- Core Pages ----------
    path('', views.home, name='home'),
    path('explore/', views.explore, name='explore'),

    # ---------- Bird Info ----------
    path('bird/<str:bird_name>/', views.bird_detail, name='bird_detail'),

    # ---------- Classification ----------
    path('classify/image/', views.classify_image, name='classify_image'),
    path('classify/audio/', views.classify_audio, name='classify_audio'),

    # ---------- Live YOLO Webcam ----------
    path('live-detect/', views.live_detect, name='live_detect'),
    
    # ---------- Admin Pages ----------
    path('admin-login/', views.admin_login, name='admin_login'),
    path('admin-confirm/', views.admin_confirm, name='admin_confirm'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),

    # ---------- Admin Add Bird Multi-Step ----------
    path('admin-add-bird/step1/', views.add_bird_step1, name='add_bird_step1'),
    path('admin-add-bird/step2/', views.add_bird_step2, name='add_bird_step2'),
    path('admin-add-bird/step3/', views.add_bird_step3, name='add_bird_step3'),
    path('admin-add-bird/submit/', views.add_bird_step3, name='add_bird_submit'),
    
    path('bird/admin/<int:id>/', views.bird_detail_admin, name='bird_detail_admin'),


]
