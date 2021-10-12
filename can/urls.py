from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from can import views

app_name = 'can'

urlpatterns = [
    path('img_upload/', views.img_upload, name='img_upload'),
    path('predict/', views.pred_img, name='pred_img'),
    path('detect_text/<str:file_name>/', views.detect_text, name='detect_text'),
    path('delete/<str:file_name>', views.delete, name='delete')
]
