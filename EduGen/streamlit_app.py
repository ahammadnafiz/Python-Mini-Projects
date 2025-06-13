import streamlit as st
import json
from script_generator import script_generator
from manim_code_generator import manim_code_generator
from animation_creator import create_animation_from_code

st.set_page_config(
    page_title="EduGen: AI Math Video Generator",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 EduGen: AI-Powered Math Video Generator")
st.markdown("Generate educational math videos using AI and Manim animations!")

# Sidebar for controls
with st.sidebar:
    st.header("📝 Content Generation")
    
    # Math topic input
    topic = st.text_input(
        "Enter the mathematical topic:",
        placeholder="e.g., Pythagorean Theorem, Quadratic Equations, Derivatives"
    )
    
    # Complexity level
    complexity = st.selectbox(
        "Select complexity level:",
        ["beginner", "intermediate", "advanced"]
    )
    
    # Mathematical domain
    domain = st.selectbox(
        "Select mathematical domain:",
        ["algebra", "geometry", "calculus", "statistics", "trigonometry", "general"]
    )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🎯 Step 1: Generate Educational Content")
    
    if st.button("🚀 Generate Structured Content", type="primary"):
        if topic:
            with st.spinner("Generating educational content..."):
                # Create enhanced prompt
                enhanced_prompt = f"""
                Create an educational explanation for: {topic}
                Complexity level: {complexity}
                Mathematical domain: {domain}
                
                Focus on creating clear, step-by-step explanations suitable for video format.
                """
                
                structured_content = script_generator.generate_script(enhanced_prompt)
                
                if structured_content:
                    st.session_state['structured_content'] = structured_content
                    st.success("✅ Content generated successfully!")
                    
                    # Display the structured content
                    st.subheader("📋 Generated Content Structure")
                    
                    # Title and intro
                    st.write(f"**Title:** {structured_content.get('title', 'N/A')}")
                    st.write(f"**Domain:** {structured_content.get('mathematical_domain', 'N/A')}")
                    st.write(f"**Complexity:** {structured_content.get('complexity_level', 'N/A')}")
                    st.write(f"**Duration:** {structured_content.get('total_duration', 'N/A')} seconds")
                    
                    with st.expander("📖 Introduction"):
                        st.write(structured_content.get('introduction', 'N/A'))
                    
                    # Explanation steps
                    with st.expander("🎬 Animation Steps"):
                        steps = structured_content.get('explanation_steps', [])
                        for step in steps:
                            st.write(f"**Step {step.get('step', 'N/A')}** ({step.get('duration', 'N/A')}s)")
                            st.write(f"*Narration:* {step.get('narration', 'N/A')}")
                            st.write(f"*Visual:* {step.get('visual_description', 'N/A')}")
                            if step.get('key_equation'):
                                st.latex(step.get('key_equation', ''))
                            st.write("---")
                    
                    with st.expander("📚 Summary"):
                        st.write(structured_content.get('summary', 'N/A'))
                    
                    # Raw JSON for debugging
                    with st.expander("🔧 Raw JSON (Debug)"):
                        st.json(structured_content)
                else:
                    st.error("❌ Failed to generate content. Please try again.")
        else:
            st.warning("⚠️ Please enter a mathematical topic first.")

with col2:
    st.header("🎨 Step 2: Generate Manim Animation")
    if st.button("🎬 Generate Manim Code", type="primary"):
        if 'structured_content' in st.session_state:
            with st.spinner("Generating Manim animation code..."):
                manim_code = manim_code_generator.generate_3b1b_manim_code(
                    st.session_state['structured_content']
                )
                
                if manim_code:
                    st.session_state['manim_code'] = manim_code
                    st.success("✅ Manim code generated successfully!")
                    
                    # Display the generated code
                    with st.expander("🐍 Generated Manim Code"):
                        st.code(manim_code, language='python')
                else:
                    st.error("❌ Failed to generate Manim code. Please try again.")
        else:
            st.warning("⚠️ Please generate structured content first.")

# Full-width section for video generation
st.header("🎥 Step 3: Create Final Video")

col3 = st.columns([1])[0]

with col3:
    if st.button("🎨 Render Animation", type="secondary"):
        if 'manim_code' in st.session_state:
            with st.spinner("Rendering Manim animation... This may take a few minutes."):
                video_path = create_animation_from_code(st.session_state['manim_code'])
                
                if video_path:
                    st.session_state['video_path'] = video_path
                    st.success(f"✅ Animation rendered: {video_path}")
                    
                    # Display the video
                    st.header("🎉 Generated Animation")
                    st.video(video_path)
                    
                    # Download button
                    with open(video_path, 'rb') as video_file:
                        st.download_button(
                            label="📥 Download Animation",
                            data=video_file.read(),
                            file_name="math_animation.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error("❌ Failed to render animation.")
        else:
            st.warning("⚠️ Please generate Manim code first.")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit, Manim, and AI*")
