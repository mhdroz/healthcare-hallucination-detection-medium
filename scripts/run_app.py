#!/usr/bin/env python3
"""
Script to run the Healthcare RAG FastAPI web application.
"""
import os
import sys
from pathlib import Path

def main():
    """Run the FastAPI web application."""
    print("🚀 Starting Healthcare RAG FastAPI Application")
    print("=" * 50)
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Please set these in your .env file or environment")
        return
    
    # Check for index
    index_path = os.getenv("INDEX_PATH", "./data/indices/medical_index")
    if not os.path.exists(index_path):
        print(f"❌ Medical index not found at: {index_path}")
        print("💡 Please run 'python scripts/build_index.py' first to create an index")
        return
    
    print("✅ Environment check passed")
    print(f"📊 Using index: {index_path}")
    print("🌐 Starting FastAPI server...")
    print("   Web Interface: http://localhost:8000")
    print("   API Docs: http://localhost:8000/api/docs")
    print("   ReDoc: http://localhost:8000/api/redoc")
    print("   Press Ctrl+C to stop")
    print("=" * 50)
    
    # Change to web_app directory and run
    web_app_dir = Path(__file__).parent.parent / "web_app"
    os.chdir(web_app_dir)
    
    # Import and run the app
    try:
        import uvicorn
        
        # Development configuration
        debug_mode = os.getenv("FASTAPI_DEBUG", "false").lower() == "true"
        port = int(os.getenv("PORT", 8000))
        
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=debug_mode,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() Path

# Add web_app to path
sys.path.append(str(Path(__file__).parent.parent / "web_app"))

def main():
    """Run the web application."""
    print("🚀 Starting Healthcare RAG Web Application")
    print("=" * 50)
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Please set these in your .env file or environment")
        return
    
    # Check for index
    index_path = os.getenv("INDEX_PATH", "./data/indices/medical_index")
    if not os.path.exists(index_path):
        print(f"❌ Medical index not found at: {index_path}")
        print("💡 Please run 'python scripts/build_index.py' first to create an index")
        return
    
    print("✅ Environment check passed")
    print(f"📊 Using index: {index_path}")
    print("🌐 Starting web server...")
    print("   Open: http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print("=" * 50)
    
    # Import and run the app
    try:
        from app import app, initialize_models
        
        # Initialize models
        initialize_models()
        
        # Run the app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()