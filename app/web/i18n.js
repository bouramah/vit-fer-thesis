// Textes de l'interface (français / anglais)
export const EMOTIONS = {
  angry:    { emoji: "😠", color: "#ef4444", fr: "Colère",    en: "Anger" },
  disgust:  { emoji: "🤢", color: "#65a30d", fr: "Dégoût",    en: "Disgust" },
  fear:     { emoji: "😨", color: "#a855f7", fr: "Peur",      en: "Fear" },
  happy:    { emoji: "😄", color: "#eab308", fr: "Joie",      en: "Happiness" },
  neutral:  { emoji: "😐", color: "#94a3b8", fr: "Neutre",    en: "Neutral" },
  sad:      { emoji: "😢", color: "#3b82f6", fr: "Tristesse", en: "Sadness" },
  surprise: { emoji: "😲", color: "#f97316", fr: "Surprise",  en: "Surprise" },
};

export const MODELS = {
  resnet:  { short: "R", name: "ResNet-50" },
  vit:     { short: "V", name: "ViT-Base" },
  dbfn:    { short: "D", name: "DBFN" },
};

const T = {
  fr: {
    "nav.home": "Accueil", "nav.live": "Direct", "nav.photo": "Photo", "nav.challenge": "Défi", "nav.about": "À propos",
    "model.label": "Modèle", "model.compare": "Comparer les 3",
    "hero.title": "Lisez les émotions, en direct.",
    "hero.sub": "EmoVision reconnaît 7 expressions faciales grâce au DBFN, une fusion d'un réseau convolutif (ResNet-50) et d'un Vision Transformer (ViT-Base), développée dans le cadre d'un mémoire de Master à l'Université Kofi Annan de Guinée.",
    "card.live.t": "Webcam en direct", "card.live.d": "Plusieurs visages suivis en temps réel, avec la confiance de chaque modèle.",
    "card.photo.t": "Analyser une photo", "card.photo.d": "Photos de groupe, explication visuelle de la décision (Grad-CAM).",
    "card.challenge.t": "Défi des 7 émotions", "card.challenge.d": "Mimez les 7 émotions et voyez si l'IA vous comprend. Environ 2 minutes.",
    "stats.contrib": "contributions au mémoire", "stats.feedback": "retours de visiteurs", "stats.correct": "jugées justes",
    "privacy.short": "Vos images sont analysées puis oubliées : rien n'est enregistré sans votre accord explicite.",
    "live.start": "Activer la caméra", "live.stop": "Arrêter", "live.capture": "Capturer",
    "live.hint": "Placez-vous face à la caméra, bien éclairé. Plusieurs personnes peuvent apparaître à l'écran.",
    "live.noface": "Aucun visage détecté", "live.track": "suivi", "live.faces": "visage(s)", "live.detect": "détection", "live.infer": "analyse",
    "live.panel": "Visage sélectionné", "live.panel.hint": "Touchez un visage pour le sélectionner.",
    "cam.error": "Impossible d'accéder à la caméra. Vérifiez l'autorisation du navigateur.",
   
    "photo.drop": "Glissez une photo ici ou", "photo.choose": "choisissez un fichier", "photo.snap": "Prendre une photo",
    "photo.analyzing": "Analyse en cours…", "photo.none": "Aucun visage détecté sur cette photo.",
    "photo.face": "Visage", "photo.explain": "Expliquer", "photo.explain.t": "Zones qui ont le plus pesé dans la décision du DBFN (Grad-CAM, branche ResNet-50).",
    "fb.q": "La prédiction est-elle juste ?", "fb.yes": "Oui", "fb.no": "Non", "fb.which": "Quelle émotion était-ce ?", "fb.thanks": "Merci pour votre retour !",
    "contrib.btn": "Contribuer au mémoire",
    "ch.title": "Défi des 7 émotions",
    "ch.intro": "L'application vous demande successivement les 7 émotions. Mimez chacune d'elles face à la caméra : dès que l'IA la reconnaît, l'image est capturée. Vous pouvez passer une émotion à tout moment.",
    "ch.start": "Commencer le défi", "ch.show": "Montrez-moi :", "ch.skip": "Passer", "ch.got": "Reconnu !",
    "ch.done": "Défi terminé", "ch.score": "émotions reconnues par le DBFN", "ch.again": "Rejouer",
    "ch.recap.hint": "Pour chaque émotion demandée : ce que chaque modèle a vu.",
    "ch.tip.angry": "Froncez les sourcils, serrez les lèvres.", "ch.tip.disgust": "Plissez le nez, relevez la lèvre supérieure.",
    "ch.tip.fear": "Ouvrez grand les yeux, sourcils relevés, bouche tendue.", "ch.tip.happy": "Un grand sourire !",
    "ch.tip.neutral": "Visage détendu, sans expression.", "ch.tip.sad": "Coins de la bouche vers le bas, regard abattu.",
    "ch.tip.surprise": "Sourcils hauts, yeux et bouche grands ouverts.",
    "c.title": "Contribuer au mémoire",
    "c.intro": "Avec votre accord, les images ci-dessous et les prédictions associées pourront illustrer et évaluer le modèle dans le mémoire de Master d'Ibrahima Douty OULARE (Université Kofi Annan de Guinée, sous la direction du Dr Adel Kermi).",
    "c.what": "Données conservées : les images de visage affichées, les prédictions des modèles, l'émotion demandée ou indiquée, la date. Aucun nom n'est exigé.",
    "c.where": "Stockage privé, accessible uniquement à l'auteur et à son encadrant. Conservation au plus 12 mois, sauf pour les images retenues dans le mémoire déposé.",
    "c.adult": "J'ai 18 ans ou plus.",
    "c.self": "Je suis la personne visible sur ces images, ou chaque personne visible a donné son accord.",
    "c.memoire": "J'accepte que ces images soient utilisées dans le mémoire (obligatoire pour contribuer).",
    "c.pub": "J'accepte aussi leur utilisation dans des présentations et publications scientifiques liées (facultatif).",
    "c.pseudo": "Prénom ou pseudonyme (facultatif)", "c.contact": "Contact pour un éventuel retrait (facultatif)",
    "c.submit": "Envoyer ma contribution", "c.cancel": "Annuler",
    "c.ok.t": "Merci pour votre contribution !",
    "c.ok.d": "Notez ce code de retrait : il ne s'affiche qu'une seule fois et n'est envoyé nulle part. Il vous permet de faire supprimer vos images à tout moment, depuis la page « À propos ».", "c.save": "Télécharger le reçu", "c.receipt": "reçu de contribution", "c.receipt.code": "Code de retrait", "c.receipt.date": "Date", "c.receipt.n": "Nombre d'images",
    "c.copy": "Copier", "c.copied": "Copié", "c.err": "L'envoi a échoué. Réessayez.",
    "w.title": "Retirer une contribution", "w.d": "Vous avez contribué au mémoire et souhaitez faire supprimer vos images ? Saisissez le code de retrait affiché à l'écran à la fin de votre contribution (il figure aussi dans le reçu téléchargeable).", "w.btn": "Retirer mes images",
    "w.ok": "Vos images ont été supprimées.", "w.no": "Code introuvable.",
    "about.title": "À propos d'EmoVision",
    "about.p1": "EmoVision est le prototype du mémoire de Master « Apprentissage profond basé sur les Vision Transformers pour la reconnaissance automatique des expressions faciales » (Master Intelligence Artificielle et Science des Données, Université Kofi Annan de Guinée, 2025-2026), associé à l'article accepté à la conférence CARI 2026.",
    "about.models": "Les trois modèles",
    "about.m.resnet": "ResNet-50 : réseau convolutif, sensible aux indices locaux (bouche, yeux).",
    "about.m.vit": "ViT-Base : Vision Transformer, qui relie les régions du visage par auto-attention.",
    "about.m.dbfn": "DBFN : les deux réseaux précédents, gelés, combinés par une petite tête de fusion de 1,58 million de paramètres.",
    "about.results": "Exactitude sur les jeux de test officiels",
    "about.limits": "Limites",
    "about.l1": "Les modèles ont été entraînés sur des images en niveaux de gris de 48×48 pixels (FER2013) : un bon éclairage et un visage de face améliorent nettement les résultats.",
    "about.l2": "Une expression faciale n'est pas une émotion ressentie : l'application reconnaît des expressions, elle ne lit pas dans les pensées.",
    "about.l3": "Les performances peuvent varier selon l'âge, l'origine ou le genre, ce qui n'a pas pu être évalué faute de données annotées.",
    "about.privacy": "Vie privée",
    "about.pv": "En direct et en mode photo, les images sont transmises au serveur, analysées puis immédiatement oubliées. Seules les contributions explicitement consenties sont conservées.",
    "about.code": "Code source", "about.contact": "Contact",
    "lat": "ms", "fps": "i/s",
  },
  en: {
    "nav.home": "Home", "nav.live": "Live", "nav.photo": "Photo", "nav.challenge": "Challenge", "nav.about": "About",
    "model.label": "Model", "model.compare": "Compare all 3",
    "hero.title": "Read emotions, live.",
    "hero.sub": "EmoVision recognises 7 facial expressions with the DBFN, a fusion of a convolutional network (ResNet-50) and a Vision Transformer (ViT-Base), developed for a Master's thesis at Kofi Annan University of Guinea.",
    "card.live.t": "Live webcam", "card.live.d": "Several faces tracked in real time, with each model's confidence.",
    "card.photo.t": "Analyse a photo", "card.photo.d": "Group photos, visual explanation of the decision (Grad-CAM).",
    "card.challenge.t": "7-emotion challenge", "card.challenge.d": "Act out the 7 emotions and see whether the AI gets you. About 2 minutes.",
    "stats.contrib": "contributions to the thesis", "stats.feedback": "visitor feedbacks", "stats.correct": "judged correct",
    "privacy.short": "Your images are analysed then forgotten: nothing is stored without your explicit consent.",
    "live.start": "Turn camera on", "live.stop": "Stop", "live.capture": "Capture",
    "live.hint": "Face the camera in good light. Several people can be on screen.",
    "live.noface": "No face detected", "live.track": "tracking", "live.faces": "face(s)", "live.detect": "detection", "live.infer": "analysis",
    "live.panel": "Selected face", "live.panel.hint": "Tap a face to select it.",
    "cam.error": "Cannot access the camera. Please check your browser permission.",
   
    "photo.drop": "Drop a photo here or", "photo.choose": "choose a file", "photo.snap": "Take a photo",
    "photo.analyzing": "Analysing…", "photo.none": "No face detected in this photo.",
    "photo.face": "Face", "photo.explain": "Explain", "photo.explain.t": "Regions that weighed most in the DBFN decision (Grad-CAM, ResNet-50 branch).",
    "fb.q": "Is the prediction right?", "fb.yes": "Yes", "fb.no": "No", "fb.which": "Which emotion was it?", "fb.thanks": "Thanks for your feedback!",
    "contrib.btn": "Contribute to the thesis",
    "ch.title": "7-emotion challenge",
    "ch.intro": "The app asks you for the 7 emotions one after another. Act each one out in front of the camera: as soon as the AI recognises it, the image is captured. You can skip an emotion at any time.",
    "ch.start": "Start the challenge", "ch.show": "Show me:", "ch.skip": "Skip", "ch.got": "Recognised!",
    "ch.done": "Challenge complete", "ch.score": "emotions recognised by the DBFN", "ch.again": "Play again",
    "ch.recap.hint": "For each requested emotion: what each model saw.",
    "ch.tip.angry": "Frown and press your lips together.", "ch.tip.disgust": "Wrinkle your nose, raise your upper lip.",
    "ch.tip.fear": "Eyes wide open, eyebrows raised, tense mouth.", "ch.tip.happy": "A big smile!",
    "ch.tip.neutral": "Relaxed face, no expression.", "ch.tip.sad": "Mouth corners down, downcast eyes.",
    "ch.tip.surprise": "Eyebrows up, eyes and mouth wide open.",
    "c.title": "Contribute to the thesis",
    "c.intro": "With your consent, the images below and their predictions may be used to illustrate and evaluate the model in the Master's thesis of Ibrahima Douty OULARE (Kofi Annan University of Guinea, supervised by Dr Adel Kermi).",
    "c.what": "Data kept: the face images shown, the models' predictions, the requested or reported emotion, the date. No name is required.",
    "c.where": "Private storage, only accessible to the author and the thesis supervisor. Kept for at most 12 months, except for images included in the submitted thesis.",
    "c.adult": "I am 18 or older.",
    "c.self": "I am the person shown in these images, or every person shown has agreed.",
    "c.memoire": "I agree that these images may be used in the thesis (required to contribute).",
    "c.pub": "I also agree to their use in related scientific presentations and publications (optional).",
    "c.pseudo": "First name or nickname (optional)", "c.contact": "Contact for a possible withdrawal (optional)",
    "c.submit": "Send my contribution", "c.cancel": "Cancel",
    "c.ok.t": "Thank you for your contribution!",
    "c.ok.d": "Write down this withdrawal code: it is shown only once and is not sent anywhere. It lets you have your images deleted at any time from the About page.", "c.save": "Download receipt", "c.receipt": "contribution receipt", "c.receipt.code": "Withdrawal code", "c.receipt.date": "Date", "c.receipt.n": "Number of images",
    "c.copy": "Copy", "c.copied": "Copied", "c.err": "Sending failed. Please try again.",
    "w.title": "Withdraw a contribution", "w.d": "Contributed to the thesis and want your images deleted? Enter the withdrawal code shown on screen at the end of your contribution (it is also in the downloadable receipt).", "w.btn": "Withdraw my images",
    "w.ok": "Your images have been deleted.", "w.no": "Code not found.",
    "about.title": "About EmoVision",
    "about.p1": "EmoVision is the prototype of the Master's thesis “Vision Transformer-based deep learning for automatic facial expression recognition” (Master in Artificial Intelligence and Data Science, Kofi Annan University of Guinea, 2025-2026), linked to the paper accepted at the CARI 2026 conference.",
    "about.models": "The three models",
    "about.m.resnet": "ResNet-50: a convolutional network, sensitive to local cues (mouth, eyes).",
    "about.m.vit": "ViT-Base: a Vision Transformer, linking face regions through self-attention.",
    "about.m.dbfn": "DBFN: both networks above, frozen, combined by a small 1.58-million-parameter fusion head.",
    "about.results": "Accuracy on the official test sets",
    "about.limits": "Limitations",
    "about.l1": "The models were trained on 48×48 greyscale images (FER2013): good lighting and a frontal face markedly improve results.",
    "about.l2": "A facial expression is not a felt emotion: the app recognises expressions, it does not read minds.",
    "about.l3": "Performance may vary with age, ethnicity or gender, which could not be assessed for lack of annotated data.",
    "about.privacy": "Privacy",
    "about.pv": "In live and photo modes, images are sent to the server, analysed and immediately forgotten. Only explicitly consented contributions are kept.",
    "about.code": "Source code", "about.contact": "Contact",
    "lat": "ms", "fps": "fps",
  },
};

let lang = "fr";
try { lang = localStorage.getItem("emovision.lang") || (navigator.language || "fr").slice(0, 2); } catch {}
if (!T[lang]) lang = "fr";

export const getLang = () => lang;
export function setLang(l) {
  lang = T[l] ? l : "fr";
  try { localStorage.setItem("emovision.lang", lang); } catch {}
  apply();
}
export const t = (k) => T[lang][k] ?? T.fr[k] ?? k;
export const emoName = (c) => EMOTIONS[c][lang];

export function apply(root = document) {
  document.documentElement.lang = lang;
  root.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  root.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
}
