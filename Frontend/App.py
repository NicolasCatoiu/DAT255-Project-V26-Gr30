import streamlit as st
import random
import time

st.set_page_config(page_title="Voice Memory Game", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .sequence {
        font-size: 60px;
        font-weight: bold;
        text-align: center;
        letter-spacing: 10px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

if "sequence" not in st.session_state:
    st.session_state.sequence = []
if "game_started" not in st.session_state:
    st.session_state.game_started = False
if "show_numbers" not in st.session_state:
    st.session_state.show_numbers = False
if "countdown" not in st.session_state:
    st.session_state.countdown = 3
if "round" not in st.session_state:
    st.session_state.round = 0
if "result_shown" not in st.session_state:
    st.session_state.result_shown = False

st.title("Voice Memory Game")
st.markdown("Remember the number sequence, then say it out loud.")
st.markdown("---")

level = st.selectbox("Select difficulty", ["Easy", "Medium", "Hard"])
sequence_length = {"Easy": 4, "Medium": 6, "Hard": 9}[level]

if st.button("Start Game", use_container_width=True):
    st.session_state.sequence = [str(random.randint(0, 9)) for _ in range(sequence_length)]
    st.session_state.game_started = True
    st.session_state.show_numbers = True
    st.session_state.result_shown = False
    st.session_state.countdown = 3
    st.session_state.round += 1
    st.rerun()

st.markdown("---")

if st.session_state.game_started:

    if st.session_state.show_numbers:
        st.subheader("Remember this sequence!")
  
        st.markdown(
            f'<div class="sequence">{" ".join(st.session_state.sequence)}</div>',
            unsafe_allow_html=True
        )
        st.caption(f"Hiding in {st.session_state.countdown}...")
        time.sleep(1)

        if st.session_state.countdown > 1:
            st.session_state.countdown -= 1
        else:
            st.session_state.show_numbers = False
            st.session_state.countdown = 3

        st.rerun()

    elif not st.session_state.result_shown:
        st.subheader("Your turn! Say the numbers out loud:")
        audio = st.audio_input("Click to record", key=f"audio_{st.session_state.round}")

        if audio is not None:
            st.audio(audio, format="audio/wav")

            def predict_numbers(audio_data):
                # TODO: replace with real CNN model
                return st.session_state.sequence if random.random() > 0.5 else ["1", "2", "3"]

            predicted_sequence = predict_numbers(audio)
            st.session_state.result_shown = True
            st.markdown("---")

            if predicted_sequence == st.session_state.sequence:
                st.balloons()
                st.success(f"Correct! You said: {' '.join(predicted_sequence)}")
            else:
                st.error(f"Wrong! Correct answer: {' '.join(st.session_state.sequence)}")
                st.warning(f"Model heard: {' '.join(predicted_sequence)}")

            if st.button("Try Again"):
                st.session_state.game_started = False
                st.session_state.result_shown = False
                st.rerun()

else:
    st.info("Select a difficulty and press Start Game to begin.")