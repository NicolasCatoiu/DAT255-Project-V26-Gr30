import streamlit as st
import random
import time
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.predictor import load_model, predict_digit

MAX_ATTEMPTS = 3
DIFFICULTY   = {"Easy": 4, "Medium": 6, "Hard": 9}


@st.cache_resource
def get_model():
    m = load_model()
    m.predict(np.zeros((1, 64, 101, 1)), verbose=0)
    return m


st.set_page_config(page_title="Voice Memory Game", page_icon="🧠", layout="centered")
st.markdown("""
<style>
.sequence {
    font-size: 64px;
    font-weight: bold;
    text-align: center;
    letter-spacing: 14px;
    margin: 30px 0;
}
</style>
""", unsafe_allow_html=True)

for key, val in {
    "sequence":       [],
    "phase":          "idle",
    "countdown":      3,
    "round":          0,
    "streak":         0,
    "seq_length":     4,
    "current_digit":  0,
    "collected":      [],
    "digit_attempts": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.title("🧠 Voice Memory Game")
st.markdown(f"🔥 **Streak: {st.session_state.streak}**")
st.divider()


if st.session_state.phase == "idle":
    st.markdown("Remember the number sequence shown on screen, then say each digit one by one.")
    st.divider()

    level   = st.selectbox("Difficulty", list(DIFFICULTY.keys()), help="Easy = 4 digits, Medium = 6, Hard = 9")
    seq_len = DIFFICULTY[level]
    st.divider()

    if st.button("🎮  Start Game", use_container_width=True, type="primary"):
        st.session_state.sequence       = [str(random.randint(0, 9)) for _ in range(seq_len)]
        st.session_state.phase          = "showing"
        st.session_state.countdown      = 3
        st.session_state.round         += 1
        st.session_state.seq_length     = seq_len
        st.session_state.current_digit  = 0
        st.session_state.collected      = []
        st.session_state.digit_attempts = 0
        st.rerun()


elif st.session_state.phase == "showing":
    st.subheader("Memorize this sequence!")
    st.caption("It will disappear — then say each digit one by one.")
    st.markdown(
        f'<div class="sequence">{" ".join(st.session_state.sequence)}</div>',
        unsafe_allow_html=True,
    )
    st.progress(
        (3 - st.session_state.countdown) / 3,
        text=f"Hiding in {st.session_state.countdown}s..."
    )
    time.sleep(1)
    if st.session_state.countdown > 1:
        st.session_state.countdown -= 1
    else:
        st.session_state.phase = "recording"
    st.rerun()


elif st.session_state.phase == "recording":
    seq_len   = st.session_state.seq_length
    cur       = st.session_state.current_digit
    attempt   = st.session_state.digit_attempts
    collected = st.session_state.collected

    slots = []
    for i in range(seq_len):
        if i < len(collected):
            d = collected[i]["digit"]
            slots.append(d if d is not None else "?")
        elif i == cur:
            slots.append("▢")
        else:
            slots.append("·")
    st.markdown(
        f'<div class="sequence">{" ".join(slots)}</div>',
        unsafe_allow_html=True,
    )

    st.progress(cur / seq_len, text=f"Digit {cur + 1} of {seq_len}")
    st.divider()

    if attempt > 0:
        st.warning(f"Couldn't hear that clearly — attempt {attempt + 1} of {MAX_ATTEMPTS}. Try again.")
    else:
        st.subheader(f"Say digit {cur + 1} 🎤")

    audio = st.audio_input(
        f"Record digit {cur + 1}",
        key=f"rec_{st.session_state.round}_{cur}_{attempt}",
    )

    if audio is not None:
        with st.spinner("Recognizing..."):
            result = predict_digit(audio.getvalue(), model=get_model())

        if result["heard"]:
            st.session_state.collected.append(result)
            st.session_state.current_digit  += 1
            st.session_state.digit_attempts  = 0
            if st.session_state.current_digit >= seq_len:
                st.session_state.phase = "result"
            st.rerun()
        elif attempt + 1 >= MAX_ATTEMPTS:
            st.session_state.collected.append({"digit": None, "confidence": result["confidence"]})
            st.session_state.current_digit  += 1
            st.session_state.digit_attempts  = 0
            if st.session_state.current_digit >= seq_len:
                st.session_state.phase = "result"
            st.rerun()
        else:
            st.session_state.digit_attempts += 1
            st.rerun()


elif st.session_state.phase == "result":
    collected    = st.session_state.collected
    predicted    = [c["digit"] for c in collected]
    expected_seq = st.session_state.sequence
    seq_len      = st.session_state.seq_length

    st.subheader("Result")

    col_exp, col_heard = st.columns(2)
    with col_exp:
        st.metric("Expected", " ".join(expected_seq))
    with col_heard:
        heard_str = " ".join(d if d is not None else "?" for d in predicted)
        st.metric("Heard", heard_str if heard_str.strip() else "(nothing)")

    if collected:
        st.markdown("---")
        cols = st.columns(min(len(collected), seq_len))
        for i, c in enumerate(collected):
            pct   = int(c["confidence"] * 100)
            exp_d = expected_seq[i] if i < len(expected_seq) else "?"
            got_d = c["digit"]
            with cols[i] if i < len(cols) else st.container():
                if got_d is None:
                    st.metric(label=f"Pos {i+1} (exp: {exp_d})", value="?",   delta="❌ missed",  delta_color="inverse")
                elif got_d == exp_d:
                    st.metric(label=f"Pos {i+1}",                value=got_d, delta=f"✅ {pct}%")
                else:
                    st.metric(label=f"Pos {i+1} (exp: {exp_d})", value=got_d, delta=f"❌ {pct}%", delta_color="inverse")

    st.divider()

    if predicted == expected_seq:
        st.session_state.streak += 1
        st.balloons()
        st.success(f"🎉 Correct! Streak: {st.session_state.streak}")
    else:
        st.session_state.streak = 0
        st.error(f"Wrong! The answer was: {' '.join(expected_seq)}")
        st.caption("Streak reset to 0")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚙️  Change Settings", use_container_width=True):
            st.session_state.phase = "idle"
            st.rerun()
    with col2:
        if st.button("▶️  Play Again", use_container_width=True, type="primary"):
            st.session_state.sequence       = [str(random.randint(0, 9)) for _ in range(st.session_state.seq_length)]
            st.session_state.phase          = "showing"
            st.session_state.countdown      = 3
            st.session_state.round         += 1
            st.session_state.current_digit  = 0
            st.session_state.collected      = []
            st.session_state.digit_attempts = 0
            st.rerun()
