import streamlit as st
import random
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.predictor import load_model, predict_sequence

MAX_ATTEMPTS = 3
DIFFICULTY = {"Easy": 4, "Medium": 6, "Hard": 9}

@st.cache_resource
def get_model():
    return load_model()

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
    "sequence": [],
    "phase": "idle",
    "countdown": 3,
    "round": 0,
    "streak": 0,
    "attempts": 0,
    "last_result": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.title("🧠 Voice Memory Game")
st.markdown(f"🔥 **Streak: {st.session_state.streak}**")
st.divider()

if st.session_state.phase == "idle":
    st.markdown("Remember the number sequence shown on screen, then say it out loud.")
    st.divider()

    level = st.selectbox("Difficulty", list(DIFFICULTY.keys()), help="Easy = 4 digits, Medium = 6, Hard = 9")
    seq_len = DIFFICULTY[level]
    st.divider()

    if st.button("🎮  Start Game", use_container_width=True, type="primary"):
        st.session_state.sequence = [
            str(random.randint(0, 9)) for _ in range(seq_len)
        ]
        st.session_state.phase = "showing"
        st.session_state.countdown = 3
        st.session_state.round += 1
        st.session_state.attempts = 0
        st.session_state.last_result = None
        st.session_state.seq_length = seq_len
        st.rerun()

elif st.session_state.phase == "showing":
    st.subheader("Memorize this sequence!")
    st.caption("It will disappear — then say it back out loud.")
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
        st.session_state.phase = "listening"
    st.rerun()

elif st.session_state.phase == "listening":
    expected = st.session_state.seq_length
    attempt = st.session_state.attempts + 1

    if st.session_state.attempts > 0 and st.session_state.last_result:
        prev = st.session_state.last_result
        heard = prev["sequence"]
        n_heard = len(heard)
        st.warning(
            f"Only caught **{n_heard}** of **{expected}** digits — try again "
            f"(attempt {attempt} of {MAX_ATTEMPTS})."
        )
        if heard:
            st.caption(f"Partially heard: {' '.join(heard)}")
        st.divider()
    else:
        st.subheader("Your turn! 🎤")

    st.info(
        "**How to record:**\n"
        "1. Wait 1 second before speaking\n"
        "2. Say each digit clearly with a short pause between them\n"
        "3. Click stop when done"
    )

    audio = st.audio_input(
        "Record your answer",
        key=f"audio_{st.session_state.round}_{st.session_state.attempts}",
    )

    if audio is not None:
        st.audio(audio, format="audio/wav")

        with st.spinner(f"Recognizing {expected} digits..."):
            t0 = time.time()
            result = predict_sequence(
                audio.getvalue(),
                expected_length=expected,
                model=get_model(),
            )
            elapsed = time.time() - t0

        st.session_state.attempts += 1

        if result["heard_enough"]:
            st.session_state.result = result
            st.session_state.elapsed = elapsed
            st.session_state.phase = "result"
            st.rerun()
        elif st.session_state.attempts >= MAX_ATTEMPTS:
            st.session_state.result = result
            st.session_state.elapsed = elapsed
            st.session_state.phase = "result"
            st.rerun()
        else:
            st.session_state.last_result = result
            st.rerun()

elif st.session_state.phase == "result":
    result = st.session_state.result
    elapsed = st.session_state.elapsed
    predicted = result["sequence"]
    expected_seq = st.session_state.sequence
    expected_len = st.session_state.seq_length

    st.subheader("Result")

    col_exp, col_heard = st.columns(2)
    with col_exp:
        st.metric("Expected", " ".join(expected_seq))
    with col_heard:
        st.metric("Heard", " ".join(predicted) if predicted else "(nothing)")

    st.caption(f"⏱ {elapsed:.2f}s · {st.session_state.attempts} attempt(s)")

    if result["predictions"]:
        st.markdown("---")
        cols = st.columns(min(len(result["predictions"]), expected_len))
        for i, p in enumerate(result["predictions"]):
            pct = int(p["confidence"] * 100)
            with cols[i] if i < len(cols) else st.container():
                if i < len(expected_seq) and p["digit"] == expected_seq[i]:
                    st.metric(label=f"Pos {i+1}", value=p["digit"], delta=f"✅ {pct}%")
                else:
                    exp_d = expected_seq[i] if i < len(expected_seq) else "?"
                    st.metric(
                        label=f"Pos {i+1} (exp: {exp_d})",
                        value=p["digit"],
                        delta=f"❌ {pct}%",
                        delta_color="inverse",
                    )

    if not result["heard_enough"]:
        st.warning(
            f"Only detected {len(predicted)} of {expected_len} digits after {MAX_ATTEMPTS} attempts. "
            "Try speaking slower with a clear pause between each number."
        )

    st.divider()

    if predicted == expected_seq:
        st.session_state.streak += 1
        st.balloons()
        st.success(f"🎉 Correct!  Streak: {st.session_state.streak}")
    else:
        st.session_state.streak = 0
        st.error(f"Wrong!  The answer was: {' '.join(expected_seq)}")
        st.caption("Streak reset to 0")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚙️  Change Settings", use_container_width=True):
            st.session_state.phase = "idle"
            st.rerun()
    with col2:
        if st.button("▶️  Play Again", use_container_width=True, type="primary"):
            st.session_state.sequence = [
                str(random.randint(0, 9))
                for _ in range(st.session_state.seq_length)
            ]
            st.session_state.phase = "showing"
            st.session_state.countdown = 3
            st.session_state.round += 1
            st.session_state.attempts = 0
            st.session_state.last_result = None
            st.rerun()
