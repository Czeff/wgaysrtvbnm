# Embroidery Pattern Generator

## Overview

This is a Flask-based web application that converts uploaded images into embroidery patterns. The application processes various image formats (PNG, JPG, JPEG, SVG) and generates embroidery patterns with stitch information, color palettes, and realistic embroidery visualization. The app creates DST files internally and provides users with professional-quality embroidery previews that show how the pattern will look when stitched.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Framework**: Flask (Python web framework)
- **Application Structure**: Modular design with separate route handling and business logic
- **Processing Engine**: Custom `EmbroideryProcessor` class for image-to-embroidery conversion
- **File Handling**: Secure file upload with validation and temporary storage

### Frontend Architecture
- **Template Engine**: Jinja2 (Flask's default templating)
- **UI Framework**: Bootstrap with dark theme
- **JavaScript**: Vanilla JS for upload handling and form validation
- **Styling**: Custom CSS for embroidery-specific UI components

### Key Design Decisions
1. **Separation of Concerns**: Routes, business logic, and application configuration are separated into different modules
2. **File Processing Pipeline**: Multi-step conversion process from raster/vector images to embroidery patterns
3. **Responsive Design**: Bootstrap-based UI that works across devices
4. **Security**: Secure filename handling and file type validation

## Key Components

### Core Files
1. **app.py**: Flask application factory and configuration
2. **routes.py**: URL routing and request handling
3. **embroidery_processor.py**: Image processing and embroidery pattern generation
4. **main.py**: Application entry point

### Frontend Components
1. **templates/index.html**: Main upload interface
2. **templates/result.html**: Pattern preview and download page
3. **static/css/custom.css**: Embroidery-specific styling
4. **static/js/upload.js**: File validation and upload handling

### Processing Pipeline
1. **File Upload**: Secure file handling with type and size validation
2. **Image Processing**: SVG conversion for raster images, direct processing for SVG with unit conversion
3. **Pattern Generation**: Conversion to embroidery format with stitch information
4. **Preview Generation**: Realistic embroidery visualization with fabric texture, thread effects, and hoop borders
5. **DST File Generation**: Internal creation of machine-readable embroidery files

## Data Flow

1. **Upload**: User uploads image file through web interface
2. **Validation**: File type, size, and security checks
3. **Processing**: Image vectorization (if needed) and embroidery pattern generation
4. **Output**: Pattern preview, statistics, and downloadable files
5. **Cleanup**: Temporary files are managed in upload/output folders

## External Dependencies

### Python Libraries
- **Flask**: Web framework and templating
- **Pillow (PIL)**: Image processing
- **pyembroidery**: Embroidery format handling
- **cairosvg**: SVG processing
- **werkzeug**: WSGI utilities and security

### Frontend Dependencies
- **Bootstrap**: UI framework (via CDN)
- **Font Awesome**: Icons (via CDN)
- **Custom CSS/JS**: Application-specific styling and behavior

### Image Processing Tools
- **potrace**: Vector tracing (external subprocess)
- **autotrace**: Alternative vectorization tool
- **ImageMagick**: Image format conversion

## Deployment Strategy

### Configuration
- **Environment-based**: Secret keys and configuration via environment variables
- **File Storage**: Local filesystem for uploads and generated files
- **WSGI**: ProxyFix middleware for deployment behind reverse proxy

### Directory Structure
```
├── app.py              # Flask application
├── routes.py           # URL routing
├── embroidery_processor.py  # Core processing logic
├── main.py             # Entry point
├── templates/          # HTML templates
├── static/             # CSS, JS, images
├── uploads/            # Temporary file storage
└── output/             # Generated pattern files
```

### Scalability Considerations
- **File Limits**: 16MB upload limit configured
- **Processing**: Synchronous processing suitable for small-scale deployment
- **Storage**: Local filesystem storage (can be extended to cloud storage)
- **Logging**: Debug logging configured for development

### Security Features
- **File Validation**: Strict file type and size checking
- **Secure Filenames**: Werkzeug secure filename handling
- **Session Management**: Flask session with configurable secret key
- **Input Sanitization**: Form data validation and sanitization

The application follows a traditional web application architecture with clear separation between presentation, business logic, and data handling layers, making it maintainable and extensible for future enhancements.