# =========================
# Python 3.10.11
# =========================

import os
import cv2
import numpy as np
import streamlit as st
import time
from tensorflow.keras.models import load_model

# Constants
frame_width = 1280
frame_height = 720
font = cv2.FONT_HERSHEY_SIMPLEX

# Streamlit UI
st.title("Application for Detecting Littering Actions using CNN")

st.sidebar.title("Dashboard")

# Sidebar options for input source
option = st.sidebar.selectbox(
    "Choose an option",
    [
        "Detect from Image File",
        "Detect from Video File",
        "Open Webcam",
    ],
)

# Option to choose with or without preprocessing
preproc_option = st.sidebar.selectbox(
    "Choose preprocessing option",
    [
        "With Preprocessing",
        "Without Preprocessing",
    ],
)

# Load the appropriate model based on preprocessing option
if preproc_option == "With Preprocessing":
    # model = load_model("garbage_with_preprocessing_model.h5")
    model = load_model("people_garbage_with_preprocessing_model.h5")
else:
    # model = load_model("garbage_without_preprocessing_model.h5")
    model = load_model("people_garbage_without_preprocessing_model.h5")


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img, preproc):
    if preproc == "With Preprocessing":
        img = grayscale(img)
        img = equalize(img)
        img = img.reshape(72, 128, 1)  # Reshape to include one channel
    img = img / 255.0
    return img


def getClassName(class_no):
    classes = [
        "Not Littering",
        "Littering",
    ]
    return classes[class_no]


def processFrame(frame, preproc):
    # img = cv2.resize(frame, (128, 128))  # Garbage
    img = cv2.resize(frame, (72, 128))  # People Garbage
    img = preprocessing(img, preproc)
    if preproc == "With Preprocessing":
        # img = img.reshape(1, 128, 128, 1)  # Garbage With preprocessing
        img = img.reshape(1, 72, 128, 1)  # People Garbage With preprocessing
    else:
        # img = img.reshape(1, 128, 128, 3)  # Garbage Without preprocessing
        img = img.reshape(1, 72, 128, 3)  # People Garbage Without preprocessing

    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    probability_value = np.amax(predictions)

    font_class_coordinate = (
        (5, 115) if frame_width >= 1920 and frame_height >= 1080 else (5, 30)
    )
    font_score_coordinate = (
        (5, 190) if frame_width >= 1920 and frame_height >= 1080 else (5, 65)
    )
    text_size = 2 if frame_width >= 1920 and frame_height >= 1080 else 0.75
    font_thickness = 5 if frame_width >= 1920 and frame_height >= 1080 else 2

    cv2.putText(
        frame,
        "Class: ",
        font_class_coordinate,
        font,
        text_size,
        (0, 0, 255),
        font_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Score: ",
        font_score_coordinate,
        font,
        text_size,
        (0, 0, 255),
        font_thickness,
        cv2.LINE_AA,
    )

    label = getClassName(class_index)
    score = str(round(probability_value, 2))

    cv2.putText(
        frame,
        f"Class: {label}",
        font_class_coordinate,
        font,
        text_size,
        (0, 0, 255),
        font_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Score: {score}",
        font_score_coordinate,
        font,
        text_size,
        (0, 0, 255),
        font_thickness,
        cv2.LINE_AA,
    )

    return frame


if option == "Detect from Image File":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Save uploaded image
        uploads_dir = os.path.join("uploads")
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        image_path = os.path.join(uploads_dir, uploaded_image.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Display uploaded image
        st.image(image_path, caption="Uploaded Image", use_column_width=True)

        if st.button("Start Detection"):
            # Process image
            frame = cv2.imread(image_path)
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

            st.warning("Processing detection...")
            processed_frame = processFrame(frame, preproc_option)

            # Save and display processed image
            output_image_path = os.path.join("output", f"{uploaded_image.name}")
            cv2.imwrite(output_image_path, processed_frame)

            st.success("Image processing completed")

            # Convert BGR to RGB
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st.image(
                processed_frame_rgb, caption="Processed Image", use_column_width=True
            )

            with open(output_image_path, "rb") as f:
                cnn_data = f.read()

            # Optionally, remove the temporary uploaded file
            os.remove(image_path)

            @st.experimental_fragment
            def downloadButton():
                st.download_button(
                    label="Download Image",
                    data=cnn_data,
                    file_name=f"{uploaded_image.name}",
                    mime="image/jpeg",
                    help="Click to download the processed image",
                )

                with st.spinner("Waiting for 5 seconds!"):
                    time.sleep(5)

            downloadButton()

elif option == "Detect from Video File":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

    if uploaded_video is not None:
        uploads_dir = os.path.join("uploads")
        video_path = os.path.join(uploads_dir, uploaded_video.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.video(video_path)

        if st.button("Start Detection"):
            st.warning("Processing detection...")

            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Determine preprocessing status
            preproc_suffix = (
                "_preproc" if preproc_option == "With Preprocessing" else "_nopreproc"
            )
            output_filename = (
                os.path.splitext(uploaded_video.name)[0] + preproc_suffix + ".mp4"
            )
            out_path = os.path.join("output", output_filename)

            out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                20,
                (frame_width, frame_height),
            )

            frame_count = 0
            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    break

                frame_count += 1
                print(f"Processing frame {frame_count}/{total_frames}")
                st.text(f"Processing frame {frame_count}/{total_frames}")
                frame = processFrame(frame, preproc_option)
                out.write(frame)

            cap.release()
            out.release()
            st.success("Video processing completed")

            with open(out_path, "rb") as f:
                cnn_data = f.read()

            # Optionally, remove the temporary uploaded file
            os.remove(video_path)

            @st.experimental_fragment
            def downloadButton():
                st.download_button(
                    label="Download Video",
                    data=cnn_data,
                    file_name=output_filename,
                    mime="video/mp4",
                    help="Click to download the video",
                )

                with st.spinner("Waiting for 5 seconds!"):
                    time.sleep(5)

            downloadButton()


elif option == "Open Webcam":
    st.text("Press 'Start' to open the webcam")
    if st.button("Start"):
        cap = cv2.VideoCapture(0)
        cap.set(3, frame_width)
        cap.set(4, frame_height)

        while True:
            success, frame = cap.read()

            if not success:
                break

            frame = processFrame(frame, preproc_option)

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
