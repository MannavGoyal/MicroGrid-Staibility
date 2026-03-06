"""Flask application entry point."""
import sys
import os
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from flask import Flask
from flask_cors import CORS


def create_app(config=None):
    """Create and configure Flask application."""
    app = Flask(__name__)
    
    # Enable CORS for frontend communication
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Load configuration
    if config:
        app.config.update(config)
    else:
        app.config.update({
            'DEBUG': True,
            'TESTING': False,
            'MAX_CONTENT_LENGTH': 100 * 1024 * 1024,  # 100MB max upload
        })
    
    # Register blueprints
    from src.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    @app.route('/')
    def index():
        """Root endpoint with API information."""
        return {
            'service': 'Microgrid Stability Enhancement API',
            'version': '1.0.0',
            'status': 'running',
            'endpoints': {
                'health': '/health',
                'api_base': '/api',
                'documentation': '/api/docs',
                'categories': {
                    'models': '/api/models/*',
                    'data': '/api/data/*',
                    'simulation': '/api/simulation/*',
                    'analysis': '/api/analysis/*',
                    'experiments': '/api/experiments/*',
                    'results': '/api/results/*',
                    'config': '/api/config/*',
                    'system': '/api/system/*'
                }
            },
            'documentation_url': 'See API_DOCUMENTATION.md for complete API reference'
        }, 200
    
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        return {'status': 'healthy', 'service': 'microgrid-backend'}, 200
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
