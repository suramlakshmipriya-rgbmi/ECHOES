
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

# ============================================================
# ECHOES
# Sound Classification & Acoustic Analysis
# ============================================================

st.set_page_config(
    page_title="ECHOES | Sound Classification",
    page_icon="🔊",
    layout="wide"
)

# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #FFF8F0;
}

.hero {
    background-color: #F8C9B4;
    padding: 35px;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    color: #3D2924;
    font-size: 48px;
    margin-bottom: 5px;
}

.hero p {
    color: #5B4037;
    font-size: 18px;
}

.section {
    background-color: #FFF0E8;
    padding: 22px;
    border-radius: 18px;
    margin: 15px 0;
}

.prediction {
    background-color: #DCEBD8;
    padding: 25px;
    border-radius: 18px;
}

.similar {
    background-color: #DCEAF5;
    padding: 22px;
    border-radius: 18px;
}

.acoustic {
    background-color: #FFF0B8;
    padding: 22px;
    border-radius: 18px;
}

.learn {
    background-color: #F8D7DE;
    padding: 25px;
    border-radius: 18px;
}

h2, h3, h4 {
    color: #3D2924 !important;
}

p, li, label {
    color: #3D2924 !important;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# PAGE HEADER
# ------------------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🔊 ECHOES</h1>
    <p>Sound Classification & Acoustic Analysis</p>
    <p><b>Listen. Identify. Understand.</b></p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# CLASS NAMES
# ------------------------------------------------------------

class_names = [
    "airplane", "breathing", "brushing_teeth", "can_opening",
    "car_horn", "cat", "chainsaw", "chirping_birds",
    "church_bells", "clapping", "clock_alarm", "clock_tick",
    "coughing", "cow", "crackling_fire", "crickets",
    "crow", "crying_baby", "dog", "door_wood_creaks",
    "door_wood_knock", "drinking_sipping", "engine",
    "fireworks", "footsteps", "frog", "glass_breaking",
    "hand_saw", "helicopter", "hen", "insects",
    "keyboard_typing", "laughing", "mouse_click", "pig",
    "pouring_water", "rain", "rooster", "sea_waves",
    "sheep", "siren", "sneezing", "snoring",
    "squeaking", "street_music", "thunderstorm",
    "toilet_flush", "train", "vacuum_cleaner", "washing_machine",
    "water_drops", "wind"
]

# ESC-50 has 50 classes
# Use first 50 unique names if needed.
class_names = class_names[:50]


# ------------------------------------------------------------
# EDUCATIONAL INFORMATION
# ------------------------------------------------------------

sound_education = {

    "airplane": {
        "what": "An airplane is an aircraft powered by engines that produces strong mechanical and aerodynamic sounds.",
        "how": "The engines, airflow around the aircraft, and vibration of mechanical parts create the sound.",
        "where": "Airports, skies near airports, and areas under flight paths.",
        "fun": "Large aircraft can be heard from several kilometres away."
    },

    "cat": {
        "what": "Cats produce sounds such as meows, purrs, hisses and growls.",
        "how": "Sound is produced using vibrations of structures in the cat's vocal system.",
        "where": "Homes, streets, farms and other places where cats live.",
        "fun": "Cats use different vocal sounds to communicate with humans and other cats."
    },

    "dog": {
        "what": "Dogs commonly produce barking, whining and growling sounds.",
        "how": "Air passing through the vocal system creates vibrations that produce sound.",
        "where": "Homes, streets, parks and other places where dogs are present.",
        "fun": "Dogs can change their bark depending on their situation."
    },

    "crying_baby": {
        "what": "A baby's cry is a vocal signal used to communicate needs or discomfort.",
        "how": "Air from the lungs vibrates the vocal folds and produces the cry.",
        "where": "Homes, hospitals, childcare centres and public places.",
        "fun": "Babies can produce different crying patterns for different needs."
    },

    "rain": {
        "what": "Rain produces a characteristic sound when water droplets hit surfaces.",
        "how": "Individual droplets create tiny impacts and vibrations when they hit objects.",
        "where": "Outdoors during rainfall, especially near roofs, windows and vegetation.",
        "fun": "Rain sounds can change dramatically depending on the surface being hit."
    },

    "thunderstorm": {
        "what": "A thunderstorm can produce thunder, rain and strong atmospheric sounds.",
        "how": "Thunder is created by the rapid expansion of air heated by lightning.",
        "where": "Outdoors during thunderstorms.",
        "fun": "Lightning heats surrounding air to extremely high temperatures."
    },

    "train": {
        "what": "Trains produce sounds from engines, wheels, brakes and tracks.",
        "how": "Mechanical vibrations and wheel-track interactions create the sound.",
        "where": "Railway stations, tracks and areas near railways.",
        "fun": "A train horn is designed to be heard from a long distance."
    },

    "church_bells": {
        "what": "Church bells are large metal bells used to produce loud ringing sounds.",
        "how": "A clapper strikes the metal bell, causing it to vibrate.",
        "where": "Churches, towers and religious buildings.",
        "fun": "Large bells can remain audible over considerable distances."
    },

    "clock_alarm": {
        "what": "An alarm clock produces a repetitive sound to attract attention.",
        "how": "An electronic or mechanical mechanism creates repeated vibrations.",
        "where": "Homes, bedrooms, offices and other indoor environments.",
        "fun": "Alarm sounds are designed to be attention-grabbing."
    },

    "clock_tick": {
        "what": "A clock tick is the small repetitive sound produced by a clock mechanism.",
        "how": "Mechanical parts move and interact at regular intervals.",
        "where": "Homes, offices and other quiet indoor environments.",
        "fun": "The regular rhythm of ticking makes it easy for humans to recognize."
    },

    "crickets": {
        "what": "Crickets are insects known for their characteristic chirping sounds.",
        "how": "Many crickets produce sound by rubbing their wings together.",
        "where": "Grasslands, gardens, forests and other outdoor environments.",
        "fun": "Cricket chirping is often associated with warm evenings."
    },

    "rooster": {
        "what": "A rooster produces a loud crowing sound.",
        "how": "Air passing through the vocal system creates the characteristic crow.",
        "where": "Farms, villages and places where chickens are raised.",
        "fun": "Roosters often crow around dawn but can also crow at other times."
    },

    "hen": {
        "what": "Hens produce clucks, calls and other vocal sounds.",
        "how": "Airflow through the vocal system creates their characteristic calls.",
        "where": "Farms, backyards and poultry environments.",
        "fun": "Hens use different calls to communicate with chicks and other birds."
    },

    "cow": {
        "what": "Cows produce low-frequency vocalizations commonly called mooing.",
        "how": "Air passing through the vocal system creates vibrations.",
        "where": "Farms, fields and cattle shelters.",
        "fun": "Cows use vocalizations to communicate with other cattle."
    },

    "sheep": {
        "what": "Sheep produce a characteristic bleating sound.",
        "how": "Air passing through their vocal system creates vibrations.",
        "where": "Farms, fields and grazing areas.",
        "fun": "Sheep can use vocal sounds to communicate with their flock."
    },

    "pig": {
        "what": "Pigs produce grunts, squeals and other vocal sounds.",
        "how": "Airflow through their vocal system creates vibrations.",
        "where": "Farms and pig shelters.",
        "fun": "Pigs use several different vocal sounds to communicate."
    },

    "frog": {
        "what": "Frogs produce croaking and calling sounds.",
        "how": "Air is pushed through the vocal system while vocal sacs help amplify the sound.",
        "where": "Ponds, wetlands, forests and gardens.",
        "fun": "Frog calls are especially noticeable during breeding seasons."
    },

    "crow": {
        "what": "Crows produce distinctive loud calls used for communication.",
        "how": "Air passing through the bird's vocal system creates vibrations.",
        "where": "Cities, forests, farms and open areas.",
        "fun": "Crows are highly intelligent birds and have complex communication."
    },

    "chirping_birds": {
        "what": "Birds produce chirps and calls to communicate.",
        "how": "Birds use a specialized vocal organ called the syrinx.",
        "where": "Gardens, forests, parks and urban areas.",
        "fun": "Different bird species can have remarkably different calls."
    },

    "sea_waves": {
        "what": "Sea waves create sound as moving water interacts with the shore.",
        "how": "Water movement, bubbles and impacts against rocks or sand create sound.",
        "where": "Beaches, coastlines and oceans.",
        "fun": "Wave sounds vary with wind, water depth and shoreline shape."
    },

    "wind": {
        "what": "Wind can create sounds as moving air interacts with objects.",
        "how": "Air turbulence and vibration of objects produce different sounds.",
        "where": "Outdoor environments, forests, buildings and open areas.",
        "fun": "Trees, wires and buildings can all produce different wind sounds."
    },

    "fireworks": {
        "what": "Fireworks produce loud explosive and crackling sounds.",
        "how": "Rapid chemical reactions create expanding gases and pressure waves.",
        "where": "Outdoor celebrations and festivals.",
        "fun": "Different chemical compositions can produce different visual effects."
    },

    "glass_breaking": {
        "what": "Glass breaking produces a sharp burst of high-frequency sound.",
        "how": "When glass fractures, many pieces vibrate and collide.",
        "where": "Homes, buildings and other environments containing glass objects.",
        "fun": "The sound contains many frequencies produced by rapidly moving fragments."
    },

    "door_wood_knock": {
        "what": "A knock is produced when an object strikes a door or another solid surface.",
        "how": "The impact causes the door material to vibrate and produce sound.",
        "where": "Homes, offices and buildings.",
        "fun": "Different doors produce different sounds depending on their material."
    },

    "door_wood_creaks": {
        "what": "A creaking door produces sound when its parts move against each other.",
        "how": "Friction and vibration between hinges and wooden surfaces create the sound.",
        "where": "Homes, buildings and older structures.",
        "fun": "Humidity can affect how wooden doors behave and sound."
    },

    "footsteps": {
        "what": "Footsteps are sounds created when a person walks or runs.",
        "how": "Shoes or feet strike and interact with the ground.",
        "where": "Homes, streets, corridors and public places.",
        "fun": "Different surfaces such as wood, concrete and grass produce different sounds."
    },

    "clapping": {
        "what": "Clapping is produced by striking the palms or hands together.",
        "how": "The impact creates a short pressure wave and vibration.",
        "where": "Events, classrooms, performances and social gatherings.",
        "fun": "Changing the shape of the hands changes the sound of a clap."
    },

    "laughing": {
        "what": "Laughter is a human vocal sound associated with amusement and emotion.",
        "how": "Repeated bursts of airflow through the vocal system create the sound.",
        "where": "Social environments, homes, schools and public places.",
        "fun": "Laughter can spread through groups and influence other people's emotions."
    },

    "coughing": {
        "what": "A cough is a sudden forceful release of air from the lungs.",
        "how": "The body rapidly pushes air outward through the respiratory system.",
        "where": "Homes, hospitals, classrooms and public places.",
        "fun": "Coughing is one of the body's protective reflexes."
    },

    "sneezing": {
        "what": "Sneezing is a sudden expulsion of air through the nose and mouth.",
        "how": "The body rapidly forces air outward to clear irritants.",
        "where": "Indoor and outdoor environments.",
        "fun": "Sneezing can be triggered by dust, pollen and other irritants."
    },

    "snoring": {
        "what": "Snoring is a sound produced during sleep when airflow causes tissues to vibrate.",
        "how": "Relaxed tissues in the airway can vibrate as air passes through.",
        "where": "Bedrooms and sleeping environments.",
        "fun": "The loudness of snoring can vary greatly between people."
    },

    "breathing": {
        "what": "Breathing is the regular movement of air into and out of the lungs.",
        "how": "The respiratory muscles move air through the airways.",
        "where": "Almost everywhere people are present.",
        "fun": "Normal breathing is usually quiet but can become louder during exercise."
    },

    "squeaking": {
        "what": "A squeak is a short, high-pitched sound.",
        "how": "Friction, movement or vibration can create a squeaking sound.",
        "where": "Many indoor and outdoor environments.",
        "fun": "Small changes in friction can dramatically change squeak sounds."
    },

    "water_drops": {
        "what": "Water drops create small impact sounds when they hit surfaces.",
        "how": "The impact causes vibrations in the water and the surface.",
        "where": "Sinks, roofs, puddles and wet environments.",
        "fun": "The sound changes depending on the surface and drop size."
    },

    "pouring_water": {
        "what": "Pouring water produces a continuous flowing sound.",
        "how": "Moving water creates turbulence and vibrations as it hits another surface.",
        "where": "Kitchens, bathrooms and other places where water is used.",
        "fun": "The sound changes with the height and speed of pouring."
    },

    "toilet_flush": {
        "what": "A toilet flush produces a strong rushing-water sound.",
        "how": "Water rapidly moves through the toilet and drainage system.",
        "where": "Bathrooms and washrooms.",
        "fun": "The sound varies depending on the design of the toilet."
    },

    "vacuum_cleaner": {
        "what": "A vacuum cleaner produces a continuous mechanical motor sound.",
        "how": "An electric motor drives a fan that moves air through the machine.",
        "where": "Homes, offices and other indoor environments.",
        "fun": "Different vacuum designs can have noticeably different acoustic signatures."
    },

    "washing_machine": {
        "what": "A washing machine produces mechanical sounds during washing and spinning.",
        "how": "The motor, drum movement and water flow create the sound.",
        "where": "Homes, laundries and other indoor environments.",
        "fun": "The spinning cycle can produce much louder sounds than the washing cycle."
    },

    "keyboard_typing": {
        "what": "Keyboard typing produces repeated clicking or tapping sounds.",
        "how": "Keys create small impacts when they are pressed and released.",
        "where": "Offices, classrooms, homes and computer labs.",
        "fun": "Mechanical, membrane and laptop keyboards can sound very different."
    },

    "mouse_click": {
        "what": "A mouse click is the short sound produced when a computer mouse button is pressed.",
        "how": "A small mechanical switch changes state and produces vibration.",
        "where": "Computer labs, offices and homes.",
        "fun": "Different mouse switches produce noticeably different clicks."
    },

    "brushing_teeth": {
        "what": "Brushing teeth creates repetitive brushing and friction sounds.",
        "how": "The toothbrush bristles interact with teeth and other surfaces.",
        "where": "Bathrooms and dental environments.",
        "fun": "Electric toothbrushes create a different acoustic pattern from manual brushes."
    },

    "drinking_sipping": {
        "what": "Drinking or sipping produces small sounds caused by moving liquid and airflow.",
        "how": "Liquid movement and air passing through the mouth create sound.",
        "where": "Homes, restaurants, offices and many everyday environments.",
        "fun": "Different drinks and containers can create different sounds."
    },

    "can_opening": {
        "what": "Opening a metal can produces a short mechanical popping sound.",
        "how": "The metal seal bends and releases pressure as the can opens.",
        "where": "Kitchens, homes and restaurants.",
        "fun": "The distinctive pop comes from the sudden movement of the metal seal."
    },

    "car_horn": {
        "what": "A car horn produces a loud warning sound.",
        "how": "An electrical system causes a vibrating diaphragm to generate sound.",
        "where": "Roads, parking areas and traffic environments.",
        "fun": "Car horns are designed to be easily noticed in noisy environments."
    },

    "engine": {
        "what": "An engine produces mechanical and combustion-related sounds.",
        "how": "Moving mechanical parts and combustion processes create vibrations.",
        "where": "Vehicles, machines and industrial environments.",
        "fun": "Engine sound can reveal information about speed and mechanical condition."
    },

    "helicopter": {
        "what": "A helicopter produces a characteristic rotor and engine sound.",
        "how": "Rapidly rotating blades create pressure changes in the air.",
        "where": "Airports, cities, emergency sites and skies.",
        "fun": "The rhythmic blade sound is one of the easiest features for people to recognize."
    },

    "chainsaw": {
        "what": "A chainsaw produces a loud mechanical cutting sound.",
        "how": "A motor drives a rapidly moving chain around a guide bar.",
        "where": "Forests, farms and construction environments.",
        "fun": "Chainsaws produce strong low- and high-frequency components."
    },

    "hand_saw": {
        "what": "A hand saw creates a repetitive cutting sound.",
        "how": "The saw teeth repeatedly interact with the material being cut.",
        "where": "Workshops, construction sites and woodworking areas.",
        "fun": "The sound changes depending on the material being cut."
    },

    "street_music": {
        "what": "Street music is musical performance occurring in public spaces.",
        "how": "Musical instruments and human voices generate sound.",
        "where": "Roads, markets, parks and public spaces.",
        "fun": "Street musicians can use the surrounding environment as natural acoustic amplification."
    },

    "crackling_fire": {
        "what": "A crackling fire produces irregular popping and crackling sounds.",
        "how": "Moisture and gases trapped inside burning wood rapidly expand and escape.",
        "where": "Campfires, fireplaces and outdoor fire environments.",
        "fun": "Each piece of wood can create a different crackling pattern."
    },

    "insects": {
        "what": "Insects produce many different buzzing, clicking and chirping sounds.",
        "how": "Different insects use wings, body parts or specialized structures to create sound.",
        "where": "Gardens, forests, fields and other outdoor environments.",
        "fun": "Some insects can produce sounds far louder than their body size suggests."
    }
}


# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------

@st.cache_resource
def load_models():

    model_path = os.path.join(
    os.path.dirname(__file__),
    "best_yamnet_classifier.keras"
)

    classifier = tf.keras.models.load_model(model_path)

    yamnet_model = hub.load(
        "https://tfhub.dev/google/yamnet/1"
    )

    return yamnet_model, classifier


# ------------------------------------------------------------
# AUDIO PROCESSING
# ------------------------------------------------------------

def prepare_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=16000,
        mono=True,
        duration=5
    )

    required_length = 16000 * 5

    if len(audio) < required_length:
        audio = np.pad(
            audio,
            (0, required_length - len(audio))
        )
    else:
        audio = audio[:required_length]

    return audio, sr


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

def predict_audio(file_path, yamnet_model, classifier):

    audio, sr = prepare_audio(file_path)

    scores, embeddings, spectrogram = yamnet_model(audio)

    embedding = tf.reduce_mean(
        embeddings,
        axis=0
    ).numpy()

    probabilities = classifier.predict(
        np.expand_dims(embedding, axis=0),
        verbose=0
    )[0]

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    top_results = []

    for index in top_indices:

        top_results.append(
            (
                class_names[index],
                float(probabilities[index] * 100)
            )
        )

    predicted_class = top_results[0][0]
    confidence = top_results[0][1]

    if confidence >= 50:
        status = "Confident Prediction"
    else:
        status = "Uncertain Prediction"

    rms = float(
        librosa.feature.rms(y=audio).mean()
    )

    zcr = float(
        librosa.feature.zero_crossing_rate(audio).mean()
    )

    spectral_centroid = float(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        ).mean()
    )

    return {
        "audio": audio,
        "sr": sr,
        "prediction": predicted_class,
        "confidence": confidence,
        "status": status,
        "top_results": top_results,
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": spectral_centroid
    }


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

try:

    yamnet_model, classifier = load_models()

except Exception as e:

    st.error(
        "Model files are not available yet. "
        "Make sure best_yamnet_classifier.keras is inside the ECHOES folder."
    )

    st.stop()


# ------------------------------------------------------------
# INTRO
# ------------------------------------------------------------

st.markdown("""
<div class="section">

### Welcome to ECHOES

ECHOES is an AI-powered acoustic analysis system that identifies
environmental sounds using deep learning.

Upload or record a short sound to discover what ECHOES hears.

</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# INPUT
# ------------------------------------------------------------

st.subheader("🎙️ Analyze a Sound")

uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"]
)

audio_input = st.audio_input(
    "Or record a sound"
)


# ------------------------------------------------------------
# SELECT INPUT
# ------------------------------------------------------------

selected_file = uploaded_file

if audio_input is not None:
    selected_file = audio_input


# ------------------------------------------------------------
# ANALYZE
# ------------------------------------------------------------

if selected_file is not None:

    st.audio(
        selected_file
    )

    if st.button(
        "🔍 Analyze Sound",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "ECHOES is listening..."
        ):

            suffix = ".wav"

            if hasattr(
                selected_file,
                "name"
            ):

                extension = os.path.splitext(
                    selected_file.name
                )[1]

                if extension:
                    suffix = extension

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    selected_file.getvalue()
                )

                temp_path = temp_file.name

            try:

                result = predict_audio(
                    temp_path,
                    yamnet_model,
                    classifier
                )

            finally:

                if os.path.exists(temp_path):
                    os.remove(temp_path)


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        st.markdown(
            '<div class="prediction">',
            unsafe_allow_html=True
        )

        st.subheader("🎯 Prediction")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"### {result['prediction'].replace('_', ' ').title()}"
            )

            st.write(
                f"Confidence: **{result['confidence']:.2f}%**"
            )

        with col2:

            if result["status"] == "Confident Prediction":

                st.success(
                    f"✓ {result['status']}"
                )

            else:

                st.warning(
                    f"⚠ {result['status']}"
                )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        st.markdown(
            '<div class="similar">',
            unsafe_allow_html=True
        )

        st.subheader("🔎 Similar Possibilities")

        for i, (name, confidence) in enumerate(
            result["top_results"],
            start=1
        ):

            st.write(
                f"**{i}. {name.replace('_', ' ').title()}** — "
                f"{confidence:.2f}%"
            )

            st.progress(
                min(confidence / 100, 1.0)
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # ACOUSTIC PROFILE
        # ----------------------------------------------------

        st.markdown(
            '<div class="acoustic">',
            unsafe_allow_html=True
        )

        st.subheader("📊 Acoustic Profile")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Duration",
                "5.00 sec"
            )

        with c2:
            st.metric(
                "RMS Energy",
                f"{result['rms']:.4f}"
            )

        with c3:
            st.metric(
                "Zero Crossing Rate",
                f"{result['zcr']:.4f}"
            )

        st.metric(
            "Spectral Centroid",
            f"{result['spectral_centroid']:.2f} Hz"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # WAVEFORM + MEL
        # ----------------------------------------------------

        st.subheader("📈 Sound Visualization")

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots(
                figsize=(7, 3)
            )

            time = np.linspace(
                0,
                len(result["audio"]) / result["sr"],
                len(result["audio"])
            )

            ax.plot(
                time,
                result["audio"]
            )

            ax.set_title(
                "Waveform"
            )

            ax.set_xlabel(
                "Time (seconds)"
            )

            ax.set_ylabel(
                "Amplitude"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        with col2:

            mel = librosa.feature.melspectrogram(
                y=result["audio"],
                sr=result["sr"],
                n_mels=128,
                fmax=8000
            )

            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )

            fig, ax = plt.subplots(
                figsize=(7, 3)
            )

            img = librosa.display.specshow(
                mel_db,
                sr=result["sr"],
                x_axis="time",
                y_axis="mel",
                ax=ax
            )

            ax.set_title(
                "Mel Spectrogram"
            )

            fig.colorbar(
                img,
                ax=ax,
                format="%+2.0f dB"
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


        # ----------------------------------------------------
        # LEARN
        # ----------------------------------------------------

        sound_name = result["prediction"]

        info = sound_education.get(
            sound_name
        )

        if info:

            st.markdown(
                '<div class="learn">',
                unsafe_allow_html=True
            )

            st.subheader(
                "📚 Learn About This Sound"
            )

            st.markdown(
                f"**What is it?**  \n{info['what']}"
            )

            st.markdown(
                f"**How is the sound produced?**  \n{info['how']}"
            )

            st.markdown(
                f"**Where might you hear it?**  \n{info['where']}"
            )

            st.markdown(
                f"**Fun fact:**  \n{info['fun']}"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# ------------------------------------------------------------
# NAVIGATION
# ------------------------------------------------------------

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "🔊 Sounds",
    "📊 Performance",
    "ℹ️ About"
])


with tab1:

    st.markdown("""
    ### ECHOES

    An intelligent acoustic analysis system designed to classify
    environmental sounds using deep learning.

    **Pipeline**

    Audio → YAMNet → Embeddings → Neural Network → 50-Class Prediction

    """)


with tab2:

    st.subheader("🔊 Supported Sound Classes")

    cols = st.columns(3)

    for i, sound in enumerate(class_names):

        with cols[i % 3]:

            st.write(
                f"• {sound.replace('_', ' ').title()}"
            )


with tab3:

    st.subheader("📊 Model Performance")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Test Accuracy",
            "79.75%"
        )

    with c2:
        st.metric(
            "Macro F1",
            "79.5%"
        )

    with c3:
        st.metric(
            "Classes",
            "50"
        )

    st.info(
        "The model was evaluated using the official ESC-50 test fold."
    )


with tab4:

    st.subheader("ℹ️ About ECHOES")

    st.write("""
    ECHOES — Sound Classification & Acoustic Analysis —
    uses YAMNet audio embeddings combined with a neural-network
    classifier to identify environmental sounds.

    The system processes a short audio recording, extracts
    acoustic information, generates a prediction and presents
    the result together with confidence and acoustic analysis.

    The system uses a confidence threshold to flag predictions
    as uncertain when the model is not sufficiently confident.
    """)

    st.markdown(
        "**Listen. Identify. Understand.**"
    )
