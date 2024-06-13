import logging
import os
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client
import face_recognition

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.
    """
    try:
        account_sid = os.getenv(
            "TWILIO_ACCOUNT_SID", "AC64d7c145744ff843129050829f7fce07"
        )
        auth_token = os.getenv("TWILIO_AUTH_TOKEN", "7681c93fce2db5de0dd1902cd6d0fedd")
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


class FaceRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_known_faces(self, uploaded_files):
        for uploaded_file in uploaded_files:
            try:
                # Load and encode each uploaded image
                image = face_recognition.load_image_file(uploaded_file)
                if face_recognition.face_encodings(image):
                    encoding = face_recognition.face_encodings(image)[0]
                    self.known_face_encodings.append(encoding)
                    # Use the file name (without extension) as the name
                    self.known_face_names.append(uploaded_file.name.split(".")[0])
                    logger.debug(f"Encoded face for {uploaded_file.name}")
                else:
                    logger.warning(f"No face found in {uploaded_file.name}")
            except Exception as e:
                logger.error(f"Error processing {uploaded_file.name}: {e}")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )
        face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding
            )
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(
                img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
            )

        return frame.from_ndarray(img, format="bgr24")


st.title("Face Recognition App")

# Upload images for known faces
st.sidebar.title("Upload Known Faces")
uploaded_files = st.sidebar.file_uploader(
    "Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    processor = FaceRecognitionProcessor()
    processor.load_known_faces(uploaded_files)
    st.sidebar.success("Images uploaded and faces encoded successfully!")

    st.title("Real-Time Face Recognition")
    ice_servers = get_ice_servers()
    logger.debug(f"ICE servers: {ice_servers}")

    webrtc_streamer(
        key="face-recognition",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: processor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": ice_servers},
    )
else:
    st.sidebar.info("Please upload images of known faces.")

st.write("Debug Information")
st.write(f"Streamlit version: {st.__version__}")
