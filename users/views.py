# users/views.py

import base64
import cv2
import numpy as np
import json
from django.shortcuts import render, redirect 
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout 
from mtcnn import MTCNN
from deepface import DeepFace
from scipy.spatial.distance import cosine
from .models import FaceProfile


# --- CONSTANTES GLOBALES ---

# MODÈLE : ArcFace (taille 512D)
MODEL_NAME = "ArcFace" 
# Le seuil de distance cosinus. Strict.
FACIAL_DISTANCE_THRESHOLD = 0.35 
# Initialisation du détecteur MTCNN
detector = MTCNN()


# --- VUES D'AFFICHAGE (HTML) ---

def register_page(request):
    """Rend la page HTML du formulaire d'inscription."""
    return render(request, 'register.html')

@login_required 
def dashboard_view(request):
    """Page accessible uniquement après connexion."""
    context = {
        'username': request.user.username,
    }
    return render(request, 'dashboard.html', context)

def logout_view(request):
    """Déconnecte l'utilisateur et affiche une page de confirmation."""
    logout(request) 
    return render(request, 'logged_out.html')


# --- VUES API (TRAITEMENT DE FOND) ---

@csrf_exempt
def register_face(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST requis"}, status=400)

    user = None 
    
    try:
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")
        image_data = request.POST.get("image")

        if not all([username, email, password, image_data]):
            return JsonResponse({"error": "Champs manquants"}, status=400)

        if User.objects.filter(username=username).exists():
            return JsonResponse({"error": "Utilisateur existe déjà"}, status=400)

        # 1. Décodage image et Détection visage (MTCNN)
        image_data = image_data.split(",")[1]
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        faces = detector.detect_faces(img)
        if len(faces) != 1:
            return JsonResponse({"error": "Un seul visage requis"}, status=400)

        x, y, w, h = faces[0]["box"]
        face_img = img[y:y+h, x:x+w]

        # 2. Extraction Embedding (ArcFace)
        embedding = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME, 
            # CORRECTION DE L'ERREUR 2: False pour éviter la double détection sur l'image déjà recadrée
            enforce_detection=False 
        )[0]["embedding"]
        new_embedding = np.array(embedding, dtype=np.float32)


        # 3. VÉRIFICATION D'UNICITÉ DU VISAGE (SÉCURITÉ)
        
        for profile in FaceProfile.objects.all():
            # Nécessite un FLUSH pour résoudre l'ERREUR 1 (512 vs 128)
            stored_embedding = np.frombuffer(profile.embedding, dtype=np.float32)
            distance = cosine(new_embedding, stored_embedding)
            
            if distance < FACIAL_DISTANCE_THRESHOLD:
                return JsonResponse({"error": "Ce visage est déjà enregistré pour un autre utilisateur."}, status=400)
        
        
        # 4. Création de l'utilisateur Django et Sauvegarde
        
        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )

        FaceProfile.objects.create(
            user=user,
            embedding=new_embedding.tobytes() 
        )

        return JsonResponse({"success": "Visage enregistré avec succès"})

    except Exception as e:
        if user and not FaceProfile.objects.filter(user=user).exists():
            user.delete()
        return JsonResponse({"error": f"Erreur serveur: {str(e)}"}, status=500)


@csrf_exempt
def login_face(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST requis"}, status=400)

    try:
        data = json.loads(request.body.decode('utf-8'))
        image_data = data.get("image")

        if not image_data:
            return JsonResponse({"error": "Image manquante"}, status=400)

        # Décodage image
        image_data = image_data.split(",")[1]
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # 1. Détection Visage (MTCNN)
        faces = detector.detect_faces(img)
        # Gestion de l'ERREUR 3: Un seul visage requis
        if len(faces) != 1:
            return JsonResponse({"error": "Un seul visage requis pour le login."}, status=400)

        x, y, w, h = faces[0]["box"]
        face_img = img[y:y+h, x:x+w]

        # 2. Extraction Embedding (ArcFace)
        current_embedding = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME, 
            enforce_detection=False 
        )[0]["embedding"]
        current_embedding = np.array(current_embedding, dtype=np.float32) 

        # 3. Comparaison
        best_match = None
        min_distance = float('inf')
        
        for profile in FaceProfile.objects.all():
            stored_embedding = np.frombuffer(profile.embedding, dtype=np.float32)
            distance = cosine(current_embedding, stored_embedding)

            if distance < min_distance:
                min_distance = distance
                best_match = profile.user

        # 4. Authentification et Connexion
        if best_match and min_distance < FACIAL_DISTANCE_THRESHOLD: 
            user_to_login = best_match
            
            user_to_login.backend = 'django.contrib.auth.backends.ModelBackend'
            login(request, user_to_login) 

            return JsonResponse({
                "success": f"Bienvenue, {user_to_login.username}!",
                "username": user_to_login.username
            })
        else:
            return JsonResponse({
                "error": "Visage inconnu ou niveau de confiance insuffisant.",
                "distance": float(min_distance) 
            }, status=401)

    except Exception as e:
        return JsonResponse({"error": f"Erreur serveur: {str(e)}"}, status=500)