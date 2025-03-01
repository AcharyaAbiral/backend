import os

from django.shortcuts import render


# Create your views here.
# from asgiref.sync import async_to_sync

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictionImageUploadSerializer, FaceRegistrationImageUploadSerializer

from .utils import perform_object_detection, perform_face_registration


class ImageProcessingView(APIView):

    def post(self, request):
        serializer = PredictionImageUploadSerializer(data=request.data)

        if serializer.is_valid():
            image = serializer.validated_data['image']
            temp_image_path = f'media/uploads/{image.name}'
            os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
            with open(temp_image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            try:
                caption, results = perform_object_detection(temp_image_path)
                os.remove(temp_image_path)
                return Response({
                    'status': 'success',
                    'caption': caption,
                    'results': results


                }, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({
                    'status': 'error',
                    'message': str(e)
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class FaceRegistrationView(APIView):
    def post(self, request):
        serializer = FaceRegistrationImageUploadSerializer(data=request.data)

        if serializer.is_valid():
            image = serializer.validated_data['image']
            name = serializer.validated_data['name']
            temp_image_path = f'media/uploads/{image.name}'
            os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)

            with open(temp_image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            try:
                perform_face_registration(temp_image_path, name)
                print("face_registration successful")
                os.remove(temp_image_path)
                return Response({
                    'status': 'success',
                    # 'results': "j pai tei"


                }, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({
                    'status': 'error',
                    'message': str(e)
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
