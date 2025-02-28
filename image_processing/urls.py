from django.urls import path

from .views import ImageProcessingView, FaceRegistrationView

urlpatterns = [
    path("process-image/", ImageProcessingView.as_view(), name="process_image"),
    path("face-registration/", FaceRegistrationView.as_view(),
         name="face_registration")
]
