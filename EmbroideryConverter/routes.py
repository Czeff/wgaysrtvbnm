import os
import logging
from flask import render_template, request, flash, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
from app import app
from embroidery_processor import EmbroideryProcessor

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with file upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if file was actually selected
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        # Check file type
        if not allowed_file(file.filename):
            flash('File type not supported. Please upload PNG, JPG, JPEG, or SVG files.', 'error')
            return redirect(request.url)
        
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Save uploaded file
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        logging.debug(f"File uploaded: {upload_path}")
        
        # Process the file
        processor = EmbroideryProcessor(app.config['OUTPUT_FOLDER'])
        result = processor.process_file(upload_path)
        
        if result['success']:
            # Clean up uploaded file
            os.remove(upload_path)
            
            return render_template('result.html', 
                                 preview_image=result['preview_image'],
                                 original_filename=filename,
                                 processing_info=result['info'])
        else:
            # Clean up uploaded file
            if os.path.exists(upload_path):
                os.remove(upload_path)
            
            flash(f'Processing failed: {result["error"]}', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        flash('An error occurred during file processing. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/preview/<filename>')
def preview_image(filename):
    """Serve preview images"""
    try:
        return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))
    except Exception as e:
        logging.error(f"Preview error: {str(e)}")
        return "Image not found", 404

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))
