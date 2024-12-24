import streamlit as st
from script_generator import script_generator
from animation_creator import create_animation
from voice_generator import generate_voice
from video_assembler import assemble_video

st.title("EduVidGen: AI Educational Video Creator")

topic = st.text_input("Enter the topic:")
complexity = st.text_input("Enter the complexity level:")

if st.button("Generate Script"):
    prompt = f"Generate an educational script on {topic} with complexity level {complexity}."
    script = script_generator.generate_script(prompt)
    st.text_area("Generated Script", script, height=300)
    st.session_state['script'] = script  # Save the script in session state

if st.button("Generate Video"):
    script = st.session_state.get('script', '')
    if script:
        create_animation(script)
        generate_voice(script)
        assemble_video('media/videos/animation/1440x1080/60/MyScene.mp4', 'output.wav', 'final_video.mp4')
        st.video('final_video.mp4')
    else:
        st.error("Please generate a script first.")
