import logging

import numpy as np
import streamlit as st
import face_recognition
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client
import cv2
logger = logging.getLogger(__name__)


# Function to get ICE servers from Twilio
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.
    """
    try:
        account_sid = (
            "AC33632aa93a8aef5cdd18973197a2be57"  # Correct SID, should start with "AC"
        )
        auth_token = "204b198c1a053a12fb4fee7f2158d4e6"
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    try:
        token = client.tokens.create()
    except TwilioRestException as e:
        st.warning(
            f"Error occurred while accessing Twilio API. Fallback to a free STUN server from Google. ({e})"
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    return token.ice_servers


# Initialize the Streamlit app
st.title("Real-Time Face Recognition App")


# Function to load and encode a single image
def load_and_encode_image(upload):
    img = face_recognition.load_image_file(upload)
    img_encoding = face_recognition.face_encodings(img)[0]
    return img, img_encoding


# Upload images and encode faces
uploaded_files = st.file_uploader(
    "Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    known_face_encodings = []
    known_face_names = []

    for uploaded_file in uploaded_files:
        # Extract the name from the filename
        name = uploaded_file.name.split(".")[0]
        image, encoding = load_and_encode_image(uploaded_file)
        known_face_encodings.append(encoding)
        known_face_names.append(name)

    class FaceRecognitionTransformer(VideoTransformerBase):
        def __init__(self):
            self.known_face_encodings = known_face_encodings
            self.known_face_names = known_face_names
            self.frame_count = 0

        def transform(self, frame):
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")

            # Process every nth frame to reduce load
            if self.frame_count % 5 == 0:
                # Resize frame to 1/4 size for faster processing
                small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]  # BGR to RGB

                # Find all faces and face encodings in the current frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations
                )

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding
                    )
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                    face_names.append(name)

                # Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(
                        img,
                        (left, bottom - 35),
                        (right, bottom),
                        (0, 0, 255),
                        cv2.FILLED,
                    )
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(
                        img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
                    )

            return img

    ice_servers = get_ice_servers()

    try:
        webrtc_streamer(
            key="face-recognition",
            video_processor_factory=FaceRecognitionTransformer,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": ice_servers},
        )
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        st.error(f"An error occurred: {e}")

else:
    st.text("Please upload images to start the face recognition.")
