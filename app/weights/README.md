Ce dossier reçoit les poids d'EmoVision (non versionnés) :

- `emovision_weights.pt` (257 Mo) : ResNet-50, ViT-Base et DBFN entraînés sur FER2013,
  à télécharger depuis la Release v1.0 du dépôt ou à reconstruire avec `app/tools/build_weights.py` ;
- `face_detection_yunet_2023mar.onnx` : détecteur de visages YuNet (OpenCV Zoo).

L'image Docker les télécharge automatiquement s'ils sont absents.
