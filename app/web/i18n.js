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
   
    "photo.drop": "Glissez une photo ici ou", "photo.choose": "choisissez une photo", "photo.snap": "Prendre une photo",
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
    "nav.guide": "Guide", "guide.link": "Comment bien mimer chaque émotion ?", "guide.title": "Guide des 7 émotions", "guide.intro": "Chaque émotion se reconnaît à quelques mouvements précis du visage (les « unités d'action » du système FACS). Voici comment les reproduire pour que l'IA vous comprenne.", "guide.try": "Essayer le défi", "guide.tip1": "💡 Éclairez votre visage de face : évitez le contre-jour (fenêtre derrière vous).", "guide.tip2": "🧍 Visage entier dans le cadre, de face, à 40-80 cm du téléphone.", "guide.tip3": "⏱️ Tenez l'expression 1 à 2 secondes : l'analyse se fait en continu.", "guide.tip4": "🎭 N'hésitez pas à exagérer : l'IA a appris sur des expressions bien marquées.", "guide.brows": "Sourcils :", "guide.eyes": "Yeux :", "guide.mouth": "Bouche :", "g.angry.brows": "froncés, abaissés et rapprochés vers le centre.", "g.angry.eyes": "regard fixe et dur, paupières tendues.", "g.angry.mouth": "lèvres serrées, ou bouche ouverte montrant les dents.", "g.angry.tip": "Fixez la caméra, mâchoire crispée, comme face à une injustice.", "g.angry.avoid": "Un regard baissé ou des yeux mi-clos font pencher vers la tristesse.", "g.disgust.brows": "légèrement abaissés.", "g.disgust.eyes": "plissés, joues remontées.", "g.disgust.mouth": "lèvre supérieure relevée, nez froncé (« beurk »).", "g.disgust.tip": "Imaginez une odeur très désagréable ou un aliment avarié.", "g.disgust.avoid": "Sans le nez plissé, l'expression est souvent lue comme de la colère.", "g.fear.brows": "relevés ET rapprochés (le front se plisse au centre).", "g.fear.eyes": "grands ouverts, on voit le blanc au-dessus de l'iris.", "g.fear.mouth": "lèvres étirées vers les oreilles, bouche légèrement ouverte et tendue.", "g.fear.tip": "Pensez à un danger soudain juste devant vous.", "g.fear.avoid": "Bouche ronde et détendue = surprise : pour la peur, la bouche est tendue et étirée.", "g.happy.brows": "détendus.", "g.happy.eyes": "plissés, avec de petites rides au coin (« pattes d'oie »).", "g.happy.mouth": "grand sourire, coins de la bouche relevés, dents visibles.", "g.happy.tip": "Un vrai sourire engage aussi les yeux et les pommettes.", "g.happy.avoid": "Un sourire trop léger peut être lu comme neutre.", "g.neutral.brows": "au repos.", "g.neutral.eyes": "ouverts normalement, regard droit vers la caméra.", "g.neutral.mouth": "fermée et détendue.", "g.neutral.tip": "Relâchez tous les muscles du visage, comme sur une photo d'identité.", "g.neutral.avoid": "Un regard vers le bas ou la bouche pincée peuvent être lus comme tristesse ou colère.", "g.sad.brows": "intérieur des sourcils relevé (forme en « accent circonflexe »).", "g.sad.eyes": "paupières tombantes, regard abattu.", "g.sad.mouth": "coins de la bouche nettement abaissés, menton qui remonte.", "g.sad.tip": "Pensez à une mauvaise nouvelle et laissez tomber les coins de la bouche.", "g.sad.avoid": "Trop discrète, la tristesse est confondue avec l'expression neutre : marquez bien la bouche.", "g.surprise.brows": "très hauts et arqués, front plissé horizontalement.", "g.surprise.eyes": "grands ouverts.", "g.surprise.mouth": "ouverte en « O », mâchoire relâchée.", "g.surprise.tip": "Réagissez comme à une nouvelle totalement inattendue : « Oh ! ».", "g.surprise.avoid": "Sourcils rapprochés et lèvres tendues font pencher vers la peur.",
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
   
    "photo.drop": "Drop a photo here or", "photo.choose": "choose a photo", "photo.snap": "Take a photo",
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
    "nav.guide": "Guide", "guide.link": "How to act out each emotion?", "guide.title": "Guide to the 7 emotions", "guide.intro": "Each emotion is recognised from a few precise facial movements (the “action units” of the FACS system). Here is how to reproduce them so that the AI understands you.", "guide.try": "Try the challenge", "guide.tip1": "💡 Light your face from the front: avoid backlight (a window behind you).", "guide.tip2": "🧍 Whole face in the frame, facing the camera, 40-80 cm from the phone.", "guide.tip3": "⏱️ Hold the expression for 1 to 2 seconds: analysis runs continuously.", "guide.tip4": "🎭 Feel free to exaggerate: the AI learned from clearly marked expressions.", "guide.brows": "Eyebrows:", "guide.eyes": "Eyes:", "guide.mouth": "Mouth:", "g.angry.brows": "frowning, lowered and drawn together.", "g.angry.eyes": "hard, fixed stare, tense eyelids.", "g.angry.mouth": "lips pressed together, or open mouth showing teeth.", "g.angry.tip": "Stare at the camera with a clenched jaw, as if facing an injustice.", "g.angry.avoid": "Looking down or half-closed eyes tip it towards sadness.", "g.disgust.brows": "slightly lowered.", "g.disgust.eyes": "narrowed, cheeks raised.", "g.disgust.mouth": "upper lip raised, nose wrinkled (“yuck”).", "g.disgust.tip": "Imagine a very bad smell or spoiled food.", "g.disgust.avoid": "Without the wrinkled nose, it is often read as anger.", "g.fear.brows": "raised AND drawn together (forehead wrinkles in the middle).", "g.fear.eyes": "wide open, white visible above the iris.", "g.fear.mouth": "lips stretched towards the ears, mouth slightly open and tense.", "g.fear.tip": "Think of a sudden danger right in front of you.", "g.fear.avoid": "A round, relaxed mouth means surprise: for fear, the mouth is tense and stretched.", "g.happy.brows": "relaxed.", "g.happy.eyes": "narrowed, with small wrinkles at the corners (crow's feet).", "g.happy.mouth": "big smile, mouth corners up, teeth visible.", "g.happy.tip": "A genuine smile also involves the eyes and cheeks.", "g.happy.avoid": "A faint smile may be read as neutral.", "g.neutral.brows": "at rest.", "g.neutral.eyes": "normally open, looking straight at the camera.", "g.neutral.mouth": "closed and relaxed.", "g.neutral.tip": "Relax every facial muscle, as for an ID photo.", "g.neutral.avoid": "Looking down or pursed lips may be read as sadness or anger.", "g.sad.brows": "inner corners raised (a “tent” shape).", "g.sad.eyes": "drooping eyelids, downcast look.", "g.sad.mouth": "mouth corners clearly pulled down, chin pushed up.", "g.sad.tip": "Think of bad news and let the corners of your mouth drop.", "g.sad.avoid": "Too subtle, sadness is mistaken for neutral: mark the mouth clearly.", "g.surprise.brows": "very high and arched, horizontal forehead wrinkles.", "g.surprise.eyes": "wide open.", "g.surprise.mouth": "open in an “O”, relaxed jaw.", "g.surprise.tip": "React as to completely unexpected news: “Oh!”.", "g.surprise.avoid": "Brows drawn together and tense lips tip it towards fear.",
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
