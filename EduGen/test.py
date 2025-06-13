#!/usr/bin/env python3
"""
Test script for EduGen AI-Powered Math Video Generator
"""

import json
from script_generator import script_generator
from manim_code_generator import manim_code_generator

def test_content_generation():
    """Test the educational content generator"""
    print("🧪 Testing Educational Content Generator...")
    
    test_prompt = "Explain the concept of derivatives in calculus for beginners"
    
    try:
        structured_content = script_generator.generate_script(test_prompt)
        
        if structured_content:
            print("✅ Content generation successful!")
            print(f"📝 Title: {structured_content.get('title', 'N/A')}")
            print(f"🎯 Domain: {structured_content.get('mathematical_domain', 'N/A')}")
            print(f"📊 Complexity: {structured_content.get('complexity_level', 'N/A')}")
            print(f"⏱️ Duration: {structured_content.get('total_duration', 'N/A')} seconds")
            print(f"🎬 Steps: {len(structured_content.get('explanation_steps', []))}")
            
            # Pretty print the structure
            print("\n📋 Full Structure:")
            print(json.dumps(structured_content, indent=2))
            
            return structured_content
        else:
            print("❌ Content generation failed!")
            return None
            
    except Exception as e:
        print(f"❌ Error in content generation: {e}")
        return None

def test_manim_code_generation(structured_content):
    """Test the Manim code generator"""
    print("\n🎨 Testing Manim Code Generator...")
    
    if not structured_content:
        print("❌ No structured content provided")
        return None
    
    try:
        manim_code = manim_code_generator.generate_3b1b_manim_code(structured_content)
        
        if manim_code:
            print("✅ Manim code generation successful!")
            print(f"📏 Code length: {len(manim_code)} characters")
            
            # Check for basic Manim components
            components = []
            if 'from manim import *' in manim_code:
                components.append("✓ Proper imports")
            if 'class' in manim_code and 'Scene' in manim_code:
                components.append("✓ Scene class")
            if 'def construct' in manim_code:
                components.append("✓ Construct method")
            if 'self.play' in manim_code:
                components.append("✓ Animations")
            if 'MathTex' in manim_code or 'Text' in manim_code:
                components.append("✓ Text/Math objects")
            
            print("🔧 Code components:")
            for component in components:
                print(f"  {component}")
            
            print("\n🐍 Generated Code Preview (first 500 chars):")
            print("-" * 50)
            print(manim_code[:500] + "..." if len(manim_code) > 500 else manim_code)
            print("-" * 50)
            
            return manim_code
        else:
            print("❌ Manim code generation failed!")
            return None
            
    except Exception as e:
        print(f"❌ Error in Manim code generation: {e}")
        return None

def test_full_pipeline():
    """Test the complete pipeline"""
    print("🚀 Testing Full EduGen Pipeline")
    print("=" * 50)
    
    # Test content generation
    structured_content = test_content_generation()
    
    if structured_content:
        # Test Manim code generation
        manim_code = test_manim_code_generation(structured_content)
        
        if manim_code:
            print("\n🎉 Full pipeline test successful!")
            print("✅ Stage 1: Educational content generated")
            print("✅ Stage 2: Manim code generated")
            print("\n📝 Next steps:")
            print("  1. Run the Streamlit app: streamlit run streamlit_app.py")
            print("  2. Try different mathematical topics")
            print("  3. Generate and render complete videos")
            
            return True
        else:
            print("\n⚠️ Pipeline partially successful (Stage 1 only)")
            return False
    else:
        print("\n❌ Pipeline test failed at Stage 1")
        return False

if __name__ == "__main__":
    print("🎓 EduGen Test Suite")
    print("==================")
    
    # Check if environment is set up
    import os
    if not os.getenv('GROQ_API_KEY'):
        print("⚠️ Warning: GROQ_API_KEY not found in environment")
        print("Please set up your .env file with API keys")
        print("Example:")
        print("GROQ_API_KEY=your_api_key_here")
        exit(1)
    
    # Run the full pipeline test
    success = test_full_pipeline()
    
    if success:
        print("\n🎯 Test Summary: SUCCESS")
        exit(0)
    else:
        print("\n❌ Test Summary: FAILED")
        exit(1)